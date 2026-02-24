"""Provision and deploy a Vast.ai GPU instance for ai-t9 training.

Called by both ``vast_orchestrate.py`` (--detach mode) and
``vast_manager.py`` (auto-restart/manual restart).
"""

from __future__ import annotations

import io
import json
import shlex
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from vast.api import create_instance, destroy_instance, search_offers, wait_for_instance
from vast.ssh import connect_ssh, scp_upload, ssh_run


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ProvisionConfig:
    """All settings needed to provision and start a training run."""

    # GPU offer constraints
    gpu: str = "RTX_3090"
    num_gpus: int = 1
    min_vram: int = 16
    max_price: float = 1.0
    cuda_version: str = "12.0"
    disk_gb: int = 30

    # Instance flavour
    interruptable: bool = False
    bid_price: float | None = None  # None → use offer min_bid

    # Docker image
    image: str = "pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel"

    # Package install mode
    install: str = "wheel"  # "wheel" | "skip"

    # S3 artifact keys (already uploaded before provisioning)
    config_s3_key: str | None = None
    wheel_s3_key: str | None = None
    supervisor_s3_key: str | None = None  # key of scripts/vast_supervisor.py in S3

    # Training CLI overrides
    run_steps: list[str] = field(default_factory=list)
    skip_steps: list[str] = field(default_factory=list)

    # Environment variables forwarded to the remote instance
    forward_env: dict[str, str] = field(default_factory=dict)

    # SSH tuning
    ssh_max_retries: int = 8
    stabilize_wait: int = 30


# ---------------------------------------------------------------------------
# Step 1: find an offer
# ---------------------------------------------------------------------------


def find_best_offer(cfg: ProvisionConfig) -> dict:
    """Search for a GPU offer matching *cfg*. Raises ``RuntimeError`` if none found."""
    offers = search_offers(
        cfg.gpu,
        cfg.min_vram,
        cfg.max_price,
        cfg.cuda_version,
        num_gpus=cfg.num_gpus,
    )
    if not offers:
        raise RuntimeError(
            f"No matching offers for gpu={cfg.gpu!r} ×{cfg.num_gpus} "
            f"max_price=${cfg.max_price}/hr min_vram={cfg.min_vram}GB "
            f"cuda>={cfg.cuda_version}"
        )
    return offers[0]


# ---------------------------------------------------------------------------
# Step 2: create instance and wait
# ---------------------------------------------------------------------------


def create_and_wait(
    offer: dict,
    cfg: ProvisionConfig,
    log: Callable[[str], None] = print,
) -> tuple[int, str, int]:
    """Create an instance from *offer* and wait for it to start.

    Returns ``(instance_id, ssh_host, ssh_port)``.
    """
    bid_price: float | None = None
    use_interruptable = cfg.interruptable or cfg.bid_price is not None
    if use_interruptable:
        if cfg.bid_price is not None:
            bid_price = cfg.bid_price
        elif offer.get("min_bid") is not None:
            bid_price = float(offer["min_bid"])
        else:
            raise RuntimeError(
                "--interruptable requested but offer has no min_bid; "
                "set bid_price explicitly in ProvisionConfig"
            )

    note = f" (interruptable, bid=${bid_price:.3f}/hr)" if bid_price is not None else ""
    log(
        f"Creating instance from offer {offer['id']}  "
        f"GPU={offer['gpu_name']} ×{offer['num_gpus']}  "
        f"${offer['dph_base']:.3f}/hr{note}"
    )
    instance_id = create_instance(
        offer["id"],
        image=cfg.image,
        disk_gb=cfg.disk_gb,
        bid_price=bid_price,
    )
    log(f"Instance {instance_id} created. Waiting for it to start…")
    info = wait_for_instance(instance_id)
    ssh_host = info.get("ssh_host") or info.get("public_ipaddr")
    ssh_port = int(info.get("ssh_port", 22))
    log(f"Instance ready: {ssh_host}:{ssh_port}")
    return instance_id, ssh_host, ssh_port


# ---------------------------------------------------------------------------
# Step 3: deploy and start supervisor via tmux
# ---------------------------------------------------------------------------


def deploy_and_start(
    instance_id: int,
    ssh_host: str,
    ssh_port: int,
    cfg: ProvisionConfig,
    s3: Any,
    bucket: str,
    prefix: str,
    job_id: str,
    log: Callable[[str], None] = print,
) -> None:
    """SSH into the instance, upload artifacts, and start the supervisor in tmux."""
    import time

    if cfg.stabilize_wait > 0:
        log(f"Waiting {cfg.stabilize_wait}s for instance to stabilise…")
        time.sleep(cfg.stabilize_wait)

    log("Connecting via SSH…")
    client = connect_ssh(ssh_host, ssh_port, max_retries=cfg.ssh_max_retries)

    try:
        # ---- Install wheel from S3 -----------------------------------------
        if cfg.install == "wheel" and cfg.wheel_s3_key:
            wheel_filename = cfg.wheel_s3_key.split("/")[-1]
            remote_wheel = f"/tmp/{wheel_filename}"
            log(f"Downloading wheel from S3 ({cfg.wheel_s3_key}) → {remote_wheel}")
            # Stream S3 object into a local tmp file then SCP
            with tempfile.NamedTemporaryFile(suffix=".whl", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                obj = s3.get_object(Bucket=bucket, Key=cfg.wheel_s3_key)
                tmp.write(obj["Body"].read())
            scp_upload(client, tmp_path, remote_wheel)
            tmp_path.unlink(missing_ok=True)
            log("Installing ai-t9 from wheel…")
            exit_code = ssh_run(
                client,
                f'pip install --quiet "{remote_wheel}[train,data]" 2>&1 | tail -5',
            )
            if exit_code != 0:
                raise RuntimeError("Failed to install ai-t9 from wheel on remote")

        # ---- Upload supervisor.py from S3 ----------------------------------
        if cfg.supervisor_s3_key:
            log(f"Downloading supervisor from S3 ({cfg.supervisor_s3_key})")
            obj = s3.get_object(Bucket=bucket, Key=cfg.supervisor_s3_key)
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(obj["Body"].read())
            scp_upload(client, tmp_path, "/root/supervisor.py")
            tmp_path.unlink(missing_ok=True)

        # ---- Upload config YAML from S3 ------------------------------------
        if cfg.config_s3_key:
            log(f"Downloading config from S3 ({cfg.config_s3_key})")
            obj = s3.get_object(Bucket=bucket, Key=cfg.config_s3_key)
            with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(obj["Body"].read())
            scp_upload(client, tmp_path, "/root/train_config.yaml")
            tmp_path.unlink(missing_ok=True)

        # ---- Build supervisor CLI args -------------------------------------
        run_cmd = "ai-t9-run /root/train_config.yaml"
        for s in cfg.run_steps:
            run_cmd += f" --step {shlex.quote(s)}"
        for s in cfg.skip_steps:
            run_cmd += f" --skip {shlex.quote(s)}"

        sup_cmd = " ".join([
            "python /root/supervisor.py",
            f"--job-id {shlex.quote(job_id)}",
            f"--instance-id {instance_id}",
            f"--s3-bucket {shlex.quote(bucket)}",
            f"--jobs-prefix {shlex.quote(prefix)}",
            f"--run-cmd {shlex.quote(run_cmd)}",
        ])

        # Write a startup shell script so we avoid nested quoting in tmux.
        # shlex.quote is safe for bash `export KEY=VALUE` context.
        env_lines = "\n".join(
            f"export {k}={shlex.quote(v)}" for k, v in cfg.forward_env.items()
        )
        start_script = (
            "#!/bin/bash\nset -e\n"
            + (env_lines + "\n" if env_lines else "")
            + f"exec {sup_cmd} >> /var/log/supervisor.log 2>&1\n"
        )
        with tempfile.NamedTemporaryFile(suffix=".sh", delete=False, mode="w") as f:
            f.write(start_script)
            tmp_script = Path(f.name)
        scp_upload(client, tmp_script, "/root/start_supervisor.sh")
        tmp_script.unlink(missing_ok=True)
        ssh_run(client, "chmod +x /root/start_supervisor.sh")

        tmux_cmd = "tmux new-session -d -s supervisor /root/start_supervisor.sh"
        log(f"Starting supervisor in tmux: {sup_cmd}")
        exit_code = ssh_run(client, tmux_cmd)
        if exit_code != 0:
            raise RuntimeError(f"Failed to start tmux supervisor (exit {exit_code})")

        log("Supervisor started successfully.")

    finally:
        client.close()


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------


def provision_and_start(
    cfg: ProvisionConfig,
    s3: Any,
    bucket: str,
    prefix: str,
    job_id: str,
    log: Callable[[str], None] = print,
) -> tuple[int, str, int]:
    """Find an offer, create the instance, deploy, and start supervisor.

    Returns ``(instance_id, ssh_host, ssh_port)``.
    Raises on any failure.
    """
    log(f"Searching for '{cfg.gpu}' ×{cfg.num_gpus} offers…")
    offer = find_best_offer(cfg)

    instance_id, ssh_host, ssh_port = create_and_wait(offer, cfg, log=log)

    try:
        deploy_and_start(
            instance_id, ssh_host, ssh_port,
            cfg, s3, bucket, prefix, job_id,
            log=log,
        )
    except Exception as exc:
        log(f"Deploy failed: {exc}. Destroying instance {instance_id}…")
        try:
            destroy_instance(instance_id)
        except Exception as destroy_exc:
            log(f"Warning: destroy failed: {destroy_exc}")
        raise

    return instance_id, ssh_host, ssh_port
