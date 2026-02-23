#!/usr/bin/env python3
"""Orchestrate ai-t9 training on Vast.ai GPU instances.

This script uses the ``vastai`` CLI (subprocess) to:
  1. Search for a suitable GPU instance
  2. Provision it
  3. Upload the YAML config and run ``ai-t9-run`` over SSH
  4. Wait for training to complete
  5. Download artifacts to a local directory
  6. Destroy the instance

Prerequisites:
    pip install vastai paramiko
    vastai apikey set <your-key>

Usage:
    # Full pipeline using a config file
    python scripts/vast_orchestrate.py configs/vast-large.yaml

    # GPU-only training (vocab + pairs already in S3)
    python scripts/vast_orchestrate.py configs/gpu-train-only.yaml

    # Override GPU type and budget
    python scripts/vast_orchestrate.py configs/vast-large.yaml --gpu H100 --max-price 3.0

    # Dry-run: print the best offer without provisioning
    python scripts/vast_orchestrate.py configs/vast-large.yaml --dry-run

    # Reuse an already-provisioned instance
    python scripts/vast_orchestrate.py configs/vast-large.yaml --instance-id 12345678

    # Skip specific steps on the remote (e.g. corpus already in S3)
    python scripts/vast_orchestrate.py configs/vast-large.yaml --skip corpus --skip vocab
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import paramiko, paramiko.ssh_exception

DEFAULT_OUTPUT_DIR = Path("data")

# ---------------------------------------------------------------------------
# vastai CLI helpers (subprocess-based — vastai SDK is not stable)
# ---------------------------------------------------------------------------


def _vastai(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a vastai CLI command and return the CompletedProcess."""
    cmd = ["vastai", *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def search_offers(
    gpu_name: str,
    min_vram_gb: int = 16,
    max_price_per_hour: float = 10.0,
    cuda_version: str = "12.0",
    limit: int = 5,
) -> list[dict]:
    """Search for GPU offers matching the given constraints.

    Returns a list of offer dicts sorted by dph_base (cheapest first).
    """
    query = (
        f"gpu_name={gpu_name} "
        f"gpu_ram>={min_vram_gb} "
        f"cuda_vers>={cuda_version} "
        f"dph_base<={max_price_per_hour} "
        f"num_gpus=1 "
        f"inet_down>1000 " # Ensure good download speed for datasets/artifacts
        f"inet_up>600 "    # Ensure good upload speed for checkpoints/artifacts, since they are not as big, we can be a bit more lenient here
        f"disk_space>100 " # Ensure enough disk space for datasets and checkpoints (especially if using a larger batch size) (TODO: calculate this based on config)
        f"reliability>0.95"
    )
    result = _vastai(
        "search", "offers",
        "--order", "dph_base asc",
        "--limit", str(limit),
        "--raw",
        query,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return []


def create_instance(
    offer_id: int,
    image: str = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
) -> int:
    """Create an instance from an offer ID.  Returns the instance ID."""
    result = _vastai(
        "create", "instance", str(offer_id),
        "--image", image,
        "--disk", "30",
        "--raw",
    )
    data = json.loads(result.stdout)
    return int(data["new_contract"])


def get_instance_info(instance_id: int) -> dict:
    """Return info dict for a running instance."""
    result = _vastai("show", "instance", str(instance_id), "--raw")
    return json.loads(result.stdout)


def wait_for_instance(
    instance_id: int, timeout: int = 300, poll_interval: int = 10
) -> dict:
    """Block until the instance is running, returning its info dict."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = get_instance_info(instance_id)
        status = info.get("actual_status", "")
        if status == "running":
            return info
        print(
            f"  Instance {instance_id} status: {status} — waiting…",
            end="\r",
            flush=True,
        )
        time.sleep(poll_interval)
    raise TimeoutError(
        f"Instance {instance_id} did not become ready within {timeout}s"
    )


def destroy_instance(instance_id: int) -> None:
    """Terminate and destroy an instance."""
    _vastai("destroy", "instance", str(instance_id))
    print(f"Instance {instance_id} destroyed.")


# ---------------------------------------------------------------------------
# SSH helpers (paramiko)
# ---------------------------------------------------------------------------


def _ssh_client(host: str, port: int, user: str = "root"):
    """Open an SSH connection to the instance."""
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    key_paths = [
        Path.home() / ".ssh" / "id_ed25519",
        Path.home() / ".ssh" / "id_rsa",
    ]
    pkey = None
    for kp in key_paths:
        if kp.exists():
            try:
                pkey = paramiko.Ed25519Key.from_private_key_file(str(kp))
                break
            except Exception:
                try:
                    pkey = paramiko.RSAKey.from_private_key_file(str(kp))
                    break
                except Exception:
                    pass
    
    # While we did already wait a bit earlier, if the instance is still not fully ready, the SSH connection might fail. To mitigate this, we can add a retry loop here as well.
    for attempt in range(5):
        try:
            client.connect(host, port=port, username=user, pkey=pkey, timeout=30)
            return client
        except paramiko.ssh_exception.NoValidConnectionsError as e:
            print(f"SSH connection attempt {attempt+1}/5 failed: {e}")
            time.sleep(5)

    return client


def ssh_run(
    client: paramiko.SSHClient, command: str, env: dict[str, str] | None = None
) -> int:
    """Run a command over SSH, streaming stdout/stderr.  Returns exit code."""
    env_prefix = ""
    if env:
        env_prefix = " ".join(f'{k}="{v}"' for k, v in env.items()) + " "
    full_cmd = env_prefix + command
    _stdin, stdout, _stderr = client.exec_command(full_cmd, get_pty=True)
    import select

    channel = stdout.channel
    while not channel.exit_status_ready():
        rready, _, _ = select.select([channel], [], [], 0.5)
        if rready:
            data = channel.recv(4096)
            if data:
                sys.stdout.buffer.write(data)
                sys.stdout.flush()
    # Drain remaining output
    while True:
        data = channel.recv(4096)
        if not data:
            break
        sys.stdout.buffer.write(data)
    sys.stdout.flush()
    return channel.recv_exit_status()


def scp_upload(client, local_path: Path, remote_path: str) -> None:
    """Upload a file over SFTP."""
    sftp = client.open_sftp()
    sftp.put(str(local_path), remote_path)
    sftp.close()


def scp_download(client, remote_path: str, local_path: Path) -> None:
    """Download a file over SFTP."""
    sftp = client.open_sftp()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote_path, str(local_path))
    sftp.close()


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Orchestrate ai-t9 training on Vast.ai using a YAML config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/vast_orchestrate.py configs/vast-large.yaml\n"
            "  python scripts/vast_orchestrate.py configs/vast-large.yaml --gpu H100\n"
            "  python scripts/vast_orchestrate.py configs/gpu-train-only.yaml --skip corpus\n"
        ),
    )

    # Required: YAML config
    parser.add_argument(
        "config",
        help="Path to YAML training config file",
    )

    # Instance selection
    parser.add_argument(
        "--gpu",
        default="RTX_3090",
        help="GPU name filter for offer search (default: RTX_3090)",
    )
    parser.add_argument(
        "--min-vram",
        type=int,
        default=16,
        help="Minimum VRAM in GB (default: 16)",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=1.0,
        help="Max $/hour (default: 1.0)",
    )
    parser.add_argument(
        "--image",
        default="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
        help="Docker image for the instance",
    )
    parser.add_argument(
        "--instance-id",
        type=int,
        default=None,
        help="Reuse an existing instance instead of provisioning",
    )
    parser.add_argument(
        "--no-destroy",
        action="store_true",
        help="Do not destroy the instance after training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the cheapest offer and exit",
    )

    # Step overrides (forwarded to ai-t9-run)
    parser.add_argument(
        "--step",
        action="append",
        metavar="NAME",
        help="Forward --step to ai-t9-run (can repeat)",
    )
    parser.add_argument(
        "--skip",
        action="append",
        metavar="NAME",
        help="Forward --skip to ai-t9-run (can repeat)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Local directory to download artifacts to (default: data/)",
    )

    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        return 1

    # ---- Collect env vars to forward -----------------------------------
    # S3 credentials are forwarded so ${VAR} references in the YAML resolve
    # on the remote instance.
    env_keys = [
        "AI_T9_S3_ENDPOINT",
        "AI_T9_S3_BUCKET",
        "AI_T9_S3_ACCESS_KEY",
        "AI_T9_S3_SECRET_KEY",
        "AI_T9_S3_REGION",
    ]
    forward_env = {k: os.environ[k] for k in env_keys if os.environ.get(k)}

    # ---- Dry run -------------------------------------------------------
    if args.dry_run:
        print(
            f"Searching for '{args.gpu}' offers "
            f"(max ${args.max_price}/hr, ≥{args.min_vram} GB VRAM)…"
        )
        offers = search_offers(args.gpu, args.min_vram, args.max_price)
        if not offers:
            print("No matching offers found.")
            return 1
        best = offers[0]
        print(f"\nBest offer:")
        print(f"  ID:        {best.get('id')}")
        print(f"  GPU:       {best.get('gpu_name')} × {best.get('num_gpus')}")
        print(f"  VRAM:      {best.get('gpu_ram', 0):.0f} GB")
        print(f"  Price:     ${best.get('dph_base', 0):.3f}/hr")
        print(f"  Location:  {best.get('geolocation', 'unknown')}")
        return 0

    # ---- Provision or reuse instance -----------------------------------
    instance_id = args.instance_id
    created = False
    if instance_id is None:
        print(f"Searching for '{args.gpu}' offers…")
        offers = search_offers(args.gpu, args.min_vram, args.max_price)
        if not offers:
            print(
                "ERROR: No matching offers. Try relaxing --gpu, --min-vram, "
                "or --max-price.",
                file=sys.stderr,
            )
            return 1
        best = offers[0]
        print(
            f"Best offer: ID={best['id']} GPU={best['gpu_name']} "
            f"${best['dph_base']:.3f}/hr"
        )
        print("Creating instance…")
        instance_id = create_instance(best["id"], image=args.image)
        created = True
        print(f"Instance {instance_id} created. Waiting for it to start…")

    try:
        info = wait_for_instance(instance_id)
        ssh_host = info.get("ssh_host") or info.get("public_ipaddr")
        ssh_port = int(info.get("ssh_port", 22))
        print(f"\nInstance ready: {ssh_host}:{ssh_port}")

        # Delay a bit to let the instance stabilize (at least some instances seem to need to wait a bit)
        print("Waiting a bit for the instance to stabilize…")
        time.sleep(10)

        # ---- Connect ---------------------------------------------------
        print("Connecting via SSH…")
        client = _ssh_client(ssh_host, ssh_port)

        # ---- Install ai-t9 --------------------------------------------
        print("Installing ai-t9[train,data]…")
        exit_code = ssh_run(
            client,
            'pip install --quiet "git+https://github.com/ElectrodeYT/ai-t9.git#egg=ai-t9[train,data]" pyyaml 2>&1 | tail -5',
        )
        if exit_code != 0:
            print("ERROR: Failed to install ai-t9", file=sys.stderr)
            if created and not args.no_destroy:
                destroy_instance(instance_id)
            return 1

        # ---- Upload config ---------------------------------------------
        remote_config = "/root/train_config.yaml"
        print(f"Uploading {config_path} → {remote_config}")
        scp_upload(client, config_path, remote_config)

        # ---- Build the ai-t9-run command -------------------------------
        run_cmd = f"ai-t9-run {remote_config}"
        if args.step:
            for s in args.step:
                run_cmd += f" --step {s}"
        if args.skip:
            for s in args.skip:
                run_cmd += f" --skip {s}"

        # ---- Run training ----------------------------------------------
        print(f"Running: {run_cmd}")
        exit_code = ssh_run(client, run_cmd, env=forward_env)
        if exit_code != 0:
            print(
                f"\nERROR: ai-t9-run exited with code {exit_code}",
                file=sys.stderr,
            )
            if not args.no_destroy and created:
                destroy_instance(instance_id)
            return exit_code

        # ---- Download artifacts ----------------------------------------
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        print(f"\nDownloading artifacts to {out}/…")
        # The remote output_dir is read from the config; default is "data"
        remote_data = "/root/data"
        for name in [
            "model.npz",
            "bigram.npz",
            "vocab.json",
            "dict.json",
        ]:
            remote = f"{remote_data}/{name}"
            local = out / name
            try:
                scp_download(client, remote, local)
                size_mb = local.stat().st_size / 1e6
                print(f"  {name}: {size_mb:.1f} MB")
            except Exception:
                print(f"  {name}: not found on remote, skipping")

        client.close()

    finally:
        if created and not args.no_destroy:
            print("\nDestroying instance…")
            destroy_instance(instance_id)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
