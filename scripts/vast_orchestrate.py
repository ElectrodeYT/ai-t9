#!/usr/bin/env python3
"""Orchestrate ai-t9 training on Vast.ai GPU instances.

Uses the ``vastai`` CLI (via subprocess) to:
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

    # Use a custom Docker image (package pre-installed — no wheel upload needed)
    python scripts/vast_orchestrate.py configs/vast-large.yaml \\
        --image ai-t9-trainer:latest --install skip

    # Interruptable (spot-like) instance — cheaper, may be preempted
    python scripts/vast_orchestrate.py configs/vast-large.yaml --interruptable

    # Interruptable with automatic respawn on interruption (requires S3 for state)
    python scripts/vast_orchestrate.py configs/vast-large.yaml --interruptable --retries 3

    # Multi-GPU instance
    python scripts/vast_orchestrate.py configs/vast-large.yaml --num-gpus 2 --max-price 2.0

    # Detach mode: provision instance, start supervisor in tmux, exit immediately
    # (requires S3 env vars; job is then managed by vast_manager.py)
    python scripts/vast_orchestrate.py configs/vast-large.yaml --detach
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Ensure the scripts/ directory is on sys.path so that `vast.*` sub-modules
# can be imported regardless of the working directory.
sys.path.insert(0, str(Path(__file__).parent))

from vast.api import (
    create_instance,
    destroy_instance,
    search_offers,
    wait_for_instance,
)
from vast.ssh import connect_ssh, scp_download, scp_upload, ssh_run

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("data")

# A known-good PyTorch image.  Override with --image if you need a different
# CUDA or PyTorch version.
_DEFAULT_IMAGE = "pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel"

# Environment variables forwarded verbatim to the remote instance so that
# ${VAR} references in YAML configs resolve correctly.
_FORWARD_ENV_KEYS = [
    "AI_T9_S3_ENDPOINT",
    "AI_T9_S3_BUCKET",
    "AI_T9_S3_ACCESS_KEY",
    "AI_T9_S3_SECRET_KEY",
    "AI_T9_S3_REGION",
    # HuggingFace auth — needed for gated datasets
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_local_wheel() -> Path:
    """Build a wheel from the local project source. Returns the .whl path.

    The wheel is placed in a temporary directory that persists for the
    lifetime of the process; the caller does not need to clean it up.

    Raises ``RuntimeError`` if the build fails.
    """
    project_root = Path(__file__).parent.parent
    wheel_dir = Path(tempfile.mkdtemp(prefix="ai_t9_wheel_"))
    result = subprocess.run(
        [
            sys.executable, "-m", "pip", "wheel",
            ".",
            "--no-deps",
            "--wheel-dir", str(wheel_dir),
            "--quiet",
        ],
        cwd=str(project_root),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Wheel build failed (exit {result.returncode}):\n"
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    wheels = list(wheel_dir.glob("ai_t9-*.whl"))
    if not wheels:
        raise RuntimeError(f"No ai_t9-*.whl found after build in {wheel_dir}")
    return wheels[0]


def _read_remote_output_dir(config_path: Path) -> str:
    """Parse ``output_dir`` from the YAML config, defaulting to ``'data'``."""
    try:
        import yaml  # type: ignore

        with config_path.open() as f:
            data = yaml.safe_load(f)
        return data.get("output_dir", "data") if isinstance(data, dict) else "data"
    except Exception:
        return "data"


# ---------------------------------------------------------------------------
# Detach mode
# ---------------------------------------------------------------------------


def _run_detach(args, config_path: Path, wheel_path: Path | None, forward_env: dict) -> int:
    """Provision instance, upload artifacts to S3, start supervisor in tmux, exit.

    Returns exit code (0 on success, 1 on failure).
    """
    # Validate S3 env vars
    required_s3 = ["AI_T9_S3_BUCKET", "AI_T9_S3_ACCESS_KEY", "AI_T9_S3_SECRET_KEY"]
    missing = [k for k in required_s3 if not os.environ.get(k)]
    if missing:
        print(f"ERROR: --detach requires env vars: {', '.join(missing)}", file=sys.stderr)
        return 1

    # Lazy import of S3/job modules so non-detach users don't need boto3
    try:
        import boto3  # type: ignore
    except ImportError:
        print("ERROR: --detach requires boto3. Install with: pip install boto3", file=sys.stderr)
        return 1

    from vast.jobs import make_job_id, save_job, Job
    from vast.provision import ProvisionConfig, provision_and_start

    bucket = os.environ["AI_T9_S3_BUCKET"]
    prefix = os.environ.get("AI_T9_JOBS_PREFIX", "jobs/")

    # Build S3 client
    endpoint = os.environ.get("AI_T9_S3_ENDPOINT")
    s3_kwargs: dict = {
        "aws_access_key_id": os.environ.get("AI_T9_S3_ACCESS_KEY"),
        "aws_secret_access_key": os.environ.get("AI_T9_S3_SECRET_KEY"),
        "region_name": os.environ.get("AI_T9_S3_REGION", "us-east-1"),
    }
    if endpoint:
        s3_kwargs["endpoint_url"] = endpoint
    s3 = boto3.client("s3", **s3_kwargs)

    job_id = make_job_id()
    print(f"Detach mode: job_id={job_id}")

    def _s3_key(filename: str) -> str:
        return f"{prefix.rstrip('/')}/{job_id}/{filename}"

    # Upload config YAML
    config_key = _s3_key("train_config.yaml")
    print(f"Uploading config → s3://{bucket}/{config_key}")
    s3.put_object(
        Bucket=bucket, Key=config_key,
        Body=config_path.read_bytes(), ContentType="text/yaml",
    )

    # Upload wheel
    wheel_key: str | None = None
    if wheel_path is not None:
        wheel_key = _s3_key(f"wheel/{wheel_path.name}")
        print(f"Uploading wheel → s3://{bucket}/{wheel_key}")
        s3.put_object(
            Bucket=bucket, Key=wheel_key,
            Body=wheel_path.read_bytes(), ContentType="application/octet-stream",
        )

    # Upload supervisor.py
    supervisor_src = Path(__file__).parent / "vast_supervisor.py"
    if not supervisor_src.exists():
        print(f"ERROR: {supervisor_src} not found", file=sys.stderr)
        return 1
    supervisor_key = _s3_key("supervisor.py")
    print(f"Uploading supervisor → s3://{bucket}/{supervisor_key}")
    s3.put_object(
        Bucket=bucket, Key=supervisor_key,
        Body=supervisor_src.read_bytes(), ContentType="text/x-python",
    )

    # Create job manifest
    job = Job(
        job_id=job_id,
        status="pending",
        gpu=args.gpu,
        num_gpus=args.num_gpus,
        min_vram=args.min_vram,
        max_price=args.max_price,
        cuda_version=args.cuda_version,
        disk_gb=args.disk,
        interruptable=args.interruptable or args.bid_price is not None,
        bid_price=args.bid_price,
        image=args.image,
        install=args.install,
        config_s3_key=config_key,
        wheel_s3_key=wheel_key,
        supervisor_s3_key=supervisor_key,
        run_steps=args.step or [],
        skip_steps=args.skip or [],
    )
    job.add_event("created", "Job created via --detach mode")
    save_job(s3, bucket, prefix, job)

    # Provision and start
    cfg = ProvisionConfig(
        gpu=args.gpu,
        num_gpus=args.num_gpus,
        min_vram=args.min_vram,
        max_price=args.max_price,
        cuda_version=args.cuda_version,
        disk_gb=args.disk,
        interruptable=args.interruptable or args.bid_price is not None,
        bid_price=args.bid_price,
        image=args.image,
        install=args.install,
        config_s3_key=config_key,
        wheel_s3_key=wheel_key,
        supervisor_s3_key=supervisor_key,
        run_steps=args.step or [],
        skip_steps=args.skip or [],
        forward_env=forward_env,
        ssh_max_retries=args.ssh_retries,
        stabilize_wait=args.stabilize_wait,
    )

    try:
        instance_id, ssh_host, ssh_port = provision_and_start(
            cfg, s3, bucket, prefix, job_id,
            log=print,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        job.status = "failed"
        job.add_event("failed", str(exc))
        save_job(s3, bucket, prefix, job)
        return 1

    job.instance_id = instance_id
    job.ssh_host = ssh_host
    job.ssh_port = ssh_port
    job.status = "running"
    job.add_event("running", f"Supervisor started on instance {instance_id} at {ssh_host}:{ssh_port}")
    save_job(s3, bucket, prefix, job)

    print(f"\nJob {job_id} started in detach mode.")
    print(f"  Instance: {instance_id}  SSH: {ssh_host}:{ssh_port}")
    print(f"  Monitor: vast_manager.py  or  aws s3 cp s3://{bucket}/{prefix}{job_id}/heartbeat.json -")
    return 0


# ---------------------------------------------------------------------------
# Main
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

    # Required positional
    parser.add_argument("config", help="Path to YAML training config file")

    # Instance selection
    parser.add_argument(
        "--gpu",
        default="RTX_3090",
        help="GPU name filter for offer search (default: RTX_3090)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to search for (default: 1)",
    )
    parser.add_argument(
        "--min-vram",
        type=int,
        default=16,
        help="Minimum VRAM in GB per GPU (default: 16)",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=1.0,
        help="Max $/hour (default: 1.0)",
    )
    parser.add_argument(
        "--cuda-version",
        default="12.0",
        help="Minimum CUDA version for offer search (default: 12.0)",
    )
    parser.add_argument(
        "--disk",
        type=int,
        default=30,
        help="Disk size in GB to allocate for the instance (default: 30)",
    )
    parser.add_argument(
        "--image",
        default=_DEFAULT_IMAGE,
        help=f"Docker image for the instance (default: {_DEFAULT_IMAGE})",
    )
    parser.add_argument(
        "--interruptable",
        action="store_true",
        help=(
            "Rent as an interruptable (spot-like) instance by bidding the offer's "
            "min_bid price. Cheaper but may be preempted. "
            "Use --retries to automatically respawn on interruption."
        ),
    )
    parser.add_argument(
        "--bid-price",
        type=float,
        default=None,
        metavar="$/HR",
        help=(
            "Override the bid price for an interruptable instance (implies "
            "--interruptable). Defaults to the offer's min_bid when --interruptable "
            "is used without an explicit price."
        ),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help=(
            "Number of times to retry by provisioning a new instance on SSH "
            "disconnection (default: 0). Useful with --interruptable and S3 "
            "state so partially-completed work is not lost between attempts."
        ),
    )
    parser.add_argument(
        "--instance-id",
        type=int,
        default=None,
        help="Reuse an existing instance instead of provisioning a new one",
    )
    parser.add_argument(
        "--no-destroy",
        action="store_true",
        help="Do not destroy the instance after training finishes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the cheapest matching offer and exit without provisioning",
    )

    # Package installation
    parser.add_argument(
        "--install",
        choices=["wheel", "skip"],
        default="wheel",
        help=(
            "How to install ai-t9 on the remote instance. "
            "'wheel' (default) builds a wheel from local source and uploads it — "
            "always installs the exact working-tree version without needing PyPI. "
            "'skip' assumes the Docker image already has the package installed."
        ),
    )

    # SSH tuning
    parser.add_argument(
        "--ssh-retries",
        type=int,
        default=8,
        help="Number of SSH connection attempts before giving up (default: 8)",
    )
    parser.add_argument(
        "--stabilize-wait",
        type=int,
        default=30,
        help="Seconds to wait after instance is 'running' before SSH (default: 30)",
    )

    # Step overrides forwarded to ai-t9-run
    parser.add_argument(
        "--step",
        action="append",
        metavar="NAME",
        help="Forward --step to ai-t9-run (can be repeated)",
    )
    parser.add_argument(
        "--skip",
        action="append",
        metavar="NAME",
        help="Forward --skip to ai-t9-run (can be repeated)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Local directory to download artifacts to (default: data/)",
    )

    # Detach mode
    parser.add_argument(
        "--detach",
        action="store_true",
        help=(
            "Provision the instance, upload the supervisor and config to S3, "
            "start the supervisor in a tmux session, then exit immediately. "
            "Training is managed autonomously by vast_manager.py. "
            "Requires AI_T9_S3_* environment variables."
        ),
    )

    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}", file=sys.stderr)
        return 1

    # Collect env vars to forward to the remote instance
    forward_env = {k: os.environ[k] for k in _FORWARD_ENV_KEYS if os.environ.get(k)}

    # ---- Build wheel before touching the instance --------------------------
    # Do this early so a build failure doesn't cost money.
    wheel_path: Path | None = None
    if args.install == "wheel":
        print("Building local wheel…")
        try:
            wheel_path = _build_local_wheel()
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        print(f"  Built: {wheel_path.name}")

    # ---- Dry run -----------------------------------------------------------
    if args.dry_run:
        interruptable_str = " interruptable" if args.interruptable else ""
        print(
            f"Searching for '{args.gpu}' ×{args.num_gpus}{interruptable_str} offers "
            f"(max ${args.max_price}/hr, ≥{args.min_vram} GB VRAM/GPU, "
            f"CUDA ≥{args.cuda_version})…"
        )
        offers = search_offers(
            args.gpu, args.min_vram, args.max_price, args.cuda_version,
            num_gpus=args.num_gpus,
        )
        if not offers:
            print("No matching offers found.")
            return 1
        best = offers[0]
        print("\nBest offer:")
        print(f"  ID:           {best.get('id')}")
        print(f"  GPU:          {best.get('gpu_name')} × {best.get('num_gpus')}")
        print(f"  VRAM:         {best.get('gpu_ram', 0):.0f} GB/GPU  "
              f"({best.get('gpu_ram', 0) * best.get('num_gpus', 1):.0f} GB total)")
        print(f"  Price:        ${best.get('dph_base', 0):.3f}/hr (on-demand)")
        if best.get("min_bid") is not None:
            print(f"  Min bid:      ${best['min_bid']:.3f}/hr (interruptable)")
        print(f"  Location:     {best.get('geolocation', 'unknown')}")
        print(f"  Reliability:  {best.get('reliability', 0):.1%}")
        return 0

    # ---- Detach mode -------------------------------------------------------
    if args.detach:
        return _run_detach(args, config_path, wheel_path, forward_env)

    # ---- Provision + train (with optional retries on interruption) ---------
    attempts_total = args.retries + 1

    for attempt in range(1, attempts_total + 1):
        if attempt > 1:
            print(f"\n--- Retry {attempt - 1}/{args.retries} ---")

        # Provision a new instance, or reuse the caller-supplied one on attempt 1.
        created = False
        if attempt == 1 and args.instance_id is not None:
            instance_id = args.instance_id
        else:
            interruptable_str = " interruptable" if args.interruptable else ""
            print(f"Searching for '{args.gpu}' ×{args.num_gpus}{interruptable_str} offers…")
            offers = search_offers(
                args.gpu, args.min_vram, args.max_price, args.cuda_version,
                num_gpus=args.num_gpus,
            )
            if not offers:
                print(
                    "ERROR: No matching offers found. Try relaxing --gpu, --min-vram, "
                    "--max-price, --cuda-version, or --num-gpus.",
                    file=sys.stderr,
                )
                return 1
            best = offers[0]
            print(
                f"Best offer: ID={best['id']}  GPU={best['gpu_name']} ×{best['num_gpus']}  "
                f"${best['dph_base']:.3f}/hr"
            )
            # Resolve bid price: explicit --bid-price > offer min_bid > none
            bid_price: float | None = None
            use_interruptable = args.interruptable or args.bid_price is not None
            if use_interruptable:
                if args.bid_price is not None:
                    bid_price = args.bid_price
                elif best.get("min_bid") is not None:
                    bid_price = float(best["min_bid"])
                else:
                    print(
                        "ERROR: --interruptable requested but this offer has no min_bid. "
                        "Specify --bid-price explicitly.",
                        file=sys.stderr,
                    )
                    return 1

            print("Creating instance…")
            try:
                instance_id = create_instance(
                    best["id"],
                    image=args.image,
                    disk_gb=args.disk,
                    bid_price=bid_price,
                )
            except RuntimeError as exc:
                print(f"ERROR: {exc}", file=sys.stderr)
                return 1
            created = True
            interruptable_note = f" (interruptable, bid=${bid_price:.3f}/hr)" if bid_price is not None else ""
            print(f"Instance {instance_id} created{interruptable_note}. Waiting for it to start…")

        interrupted = False
        try:
            info = wait_for_instance(instance_id)
            ssh_host = info.get("ssh_host") or info.get("public_ipaddr")
            ssh_port = int(info.get("ssh_port", 22))
            print(f"Instance ready: {ssh_host}:{ssh_port}")

            # Give the instance a moment to finish initialising before SSH
            if args.stabilize_wait > 0:
                print(f"Waiting {args.stabilize_wait}s for instance to stabilise…")
                time.sleep(args.stabilize_wait)

            # ---- Connect ---------------------------------------------------
            print("Connecting via SSH…")
            client = connect_ssh(ssh_host, ssh_port, max_retries=args.ssh_retries)

            # ---- Install ai-t9 ---------------------------------------------
            if args.install == "wheel":
                remote_wheel = f"/tmp/{wheel_path.name}"
                print(f"Uploading wheel → {remote_wheel}")
                scp_upload(client, wheel_path, remote_wheel)
                print("Installing ai-t9 from wheel…")
                exit_code = ssh_run(
                    client,
                    f'pip install --quiet "{remote_wheel}[train,data]" 2>&1 | tail -5',
                )
                if exit_code != 0:
                    print("ERROR: Failed to install ai-t9 from wheel", file=sys.stderr)
                    return 1
            else:
                print("Skipping package installation (--install skip)")

            # ---- Upload config ---------------------------------------------
            remote_config = "/root/train_config.yaml"
            print(f"Uploading {config_path} → {remote_config}")
            scp_upload(client, config_path, remote_config)

            # ---- Build the ai-t9-run command --------------------------------
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
                return exit_code

            # ---- Download artifacts ----------------------------------------
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)

            remote_output_dir = "/root/" + _read_remote_output_dir(config_path).lstrip("/")
            print(f"\nDownloading artifacts from {remote_output_dir}/ to {out}/…")

            artifacts = ["model.npz", "bigram.npz", "vocab.json", "dict.json"]
            missing: list[str] = []
            for name in artifacts:
                remote = f"{remote_output_dir}/{name}"
                local = out / name
                try:
                    scp_download(client, remote, local)
                    size_mb = local.stat().st_size / 1e6
                    print(f"  {name}: {size_mb:.1f} MB")
                except IOError:
                    print(f"  {name}: not found on remote, skipping")
                    missing.append(name)
                except Exception as exc:
                    print(f"  {name}: download failed — {exc}", file=sys.stderr)
                    missing.append(name)

            client.close()

            if missing:
                print(f"\nNote: {len(missing)} artifact(s) were not downloaded: {', '.join(missing)}")

        except ConnectionError as exc:
            interrupted = True
            print(f"\nSSH connection lost: {exc}", file=sys.stderr)
            if attempt < attempts_total:
                print(
                    f"Instance may have been interrupted. "
                    f"Retrying with a new instance (attempt {attempt + 1}/{attempts_total})…"
                )
            else:
                if attempts_total > 1:
                    print(f"All {attempts_total} attempt(s) exhausted.", file=sys.stderr)
                # Fall through to finally, then return below.

        finally:
            if created:
                # Always destroy interrupted instances (they are unusable).
                # For successful/failed-non-interrupt runs, respect --no-destroy.
                if interrupted or not args.no_destroy:
                    print("\nDestroying instance…")
                    destroy_instance(instance_id)

        if interrupted:
            if attempt >= attempts_total:
                return 1
            continue  # retry

        # Training completed successfully — exit the retry loop.
        break

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
