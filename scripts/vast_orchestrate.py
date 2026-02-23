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
        print(
            f"Searching for '{args.gpu}' offers "
            f"(max ${args.max_price}/hr, ≥{args.min_vram} GB VRAM, "
            f"CUDA ≥{args.cuda_version})…"
        )
        offers = search_offers(
            args.gpu, args.min_vram, args.max_price, args.cuda_version
        )
        if not offers:
            print("No matching offers found.")
            return 1
        best = offers[0]
        print("\nBest offer:")
        print(f"  ID:       {best.get('id')}")
        print(f"  GPU:      {best.get('gpu_name')} × {best.get('num_gpus')}")
        print(f"  VRAM:     {best.get('gpu_ram', 0):.0f} GB")
        print(f"  Price:    ${best.get('dph_base', 0):.3f}/hr")
        print(f"  Location: {best.get('geolocation', 'unknown')}")
        return 0

    # ---- Provision or reuse instance ---------------------------------------
    instance_id = args.instance_id
    created = False
    if instance_id is None:
        print(f"Searching for '{args.gpu}' offers…")
        offers = search_offers(
            args.gpu, args.min_vram, args.max_price, args.cuda_version
        )
        if not offers:
            print(
                "ERROR: No matching offers found. Try relaxing --gpu, --min-vram, "
                "--max-price, or --cuda-version.",
                file=sys.stderr,
            )
            return 1
        best = offers[0]
        print(
            f"Best offer: ID={best['id']} GPU={best['gpu_name']} "
            f"${best['dph_base']:.3f}/hr"
        )
        print("Creating instance…")
        try:
            instance_id = create_instance(best["id"], image=args.image, disk_gb=args.disk)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        created = True
        print(f"Instance {instance_id} created. Waiting for it to start…")

    try:
        info = wait_for_instance(instance_id)
        ssh_host = info.get("ssh_host") or info.get("public_ipaddr")
        ssh_port = int(info.get("ssh_port", 22))
        print(f"Instance ready: {ssh_host}:{ssh_port}")

        # Give the instance a moment to finish initialising before SSH
        if args.stabilize_wait > 0:
            print(f"Waiting {args.stabilize_wait}s for instance to stabilise…")
            time.sleep(args.stabilize_wait)

        # ---- Connect -------------------------------------------------------
        print("Connecting via SSH…")
        try:
            client = connect_ssh(ssh_host, ssh_port, max_retries=args.ssh_retries)
        except ConnectionError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        # ---- Install ai-t9 ------------------------------------------------
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

        # ---- Upload config -------------------------------------------------
        remote_config = "/root/train_config.yaml"
        print(f"Uploading {config_path} → {remote_config}")
        scp_upload(client, config_path, remote_config)

        # ---- Build the ai-t9-run command -----------------------------------
        run_cmd = f"ai-t9-run {remote_config}"
        if args.step:
            for s in args.step:
                run_cmd += f" --step {s}"
        if args.skip:
            for s in args.skip:
                run_cmd += f" --skip {s}"

        # ---- Run training --------------------------------------------------
        print(f"Running: {run_cmd}")
        exit_code = ssh_run(client, run_cmd, env=forward_env)
        if exit_code != 0:
            print(
                f"\nERROR: ai-t9-run exited with code {exit_code}",
                file=sys.stderr,
            )
            return exit_code

        # ---- Download artifacts --------------------------------------------
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Determine the remote output directory from the config
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
                # File simply wasn't produced (e.g. ngram skipped) — not an error
                print(f"  {name}: not found on remote, skipping")
                missing.append(name)
            except Exception as exc:
                print(f"  {name}: download failed — {exc}", file=sys.stderr)
                missing.append(name)

        client.close()

        if missing:
            print(f"\nNote: {len(missing)} artifact(s) were not downloaded: {', '.join(missing)}")

    finally:
        if created and not args.no_destroy:
            print("\nDestroying instance…")
            destroy_instance(instance_id)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
