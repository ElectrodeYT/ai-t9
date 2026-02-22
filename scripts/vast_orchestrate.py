#!/usr/bin/env python3
"""Orchestrate ai-t9 training on Vast.ai GPU instances.

This script uses the ``vastai`` CLI (subprocess) to:
  1. Search for a suitable GPU instance
  2. Provision it
  3. Upload ``vast_train.sh`` and run it over SSH
  4. Wait for training to complete
  5. Download artifacts to the local ``data/`` directory
  6. Destroy the instance

Prerequisites:
    pip install vastai paramiko
    vastai apikey set <your-key>

Required environment variables (forwarded to vast_train.sh):
    AI_T9_S3_ENDPOINT, AI_T9_S3_BUCKET, AI_T9_S3_ACCESS_KEY, AI_T9_S3_SECRET_KEY

Usage:
    # Find and use cheapest H100, train for 5 epochs, download result
    python scripts/vast_orchestrate.py

    # Override GPU and training hyperparameters
    python scripts/vast_orchestrate.py --gpu RTX3090 --epochs 10 --embed-dim 128

    # Dry-run: print the best offer without provisioning
    python scripts/vast_orchestrate.py --dry-run

    # Reuse an already-provisioned instance (skip search/create)
    python scripts/vast_orchestrate.py --instance-id 12345678

All flags:
    python scripts/vast_orchestrate.py --help
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
TRAIN_SCRIPT = SCRIPT_DIR / "vast_train.sh"
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
    # Build a query string for vastai's search syntax
    query = (
        f"gpu_name={gpu_name} "
        f"gpu_ram>={min_vram_gb} "
        f"cuda_vers>={cuda_version} "
        f"dph_base<={max_price_per_hour} "
        f"num_gpus=1 "
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


def create_instance(offer_id: int, image: str = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime") -> int:
    """Create an instance from an offer ID.

    Returns the instance ID.
    """
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


def wait_for_instance(instance_id: int, timeout: int = 300, poll_interval: int = 10) -> dict:
    """Block until the instance is running, returning its info dict."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = get_instance_info(instance_id)
        status = info.get("actual_status", "")
        if status == "running":
            return info
        print(f"  Instance {instance_id} status: {status} — waiting…", end="\r", flush=True)
        time.sleep(poll_interval)
    raise TimeoutError(f"Instance {instance_id} did not become ready within {timeout}s")


def destroy_instance(instance_id: int) -> None:
    """Terminate and destroy an instance."""
    _vastai("destroy", "instance", str(instance_id))
    print(f"Instance {instance_id} destroyed.")


# ---------------------------------------------------------------------------
# SSH helpers (paramiko)
# ---------------------------------------------------------------------------

def _ssh_client(host: str, port: int, user: str = "root"):
    """Open an SSH connection to the instance.

    Tries the default SSH key (``~/.ssh/id_rsa`` or ``~/.ssh/id_ed25519``).
    """
    import paramiko  # pip install ai-t9[vast]
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
    client.connect(host, port=port, username=user, pkey=pkey, timeout=30)
    return client


def ssh_run(client, command: str, env: dict[str, str] | None = None) -> int:
    """Run a command over SSH, streaming stdout/stderr.  Returns the exit code."""
    env_prefix = ""
    if env:
        env_prefix = " ".join(f'{k}="{v}"' for k, v in env.items()) + " "
    full_cmd = env_prefix + command
    stdin, stdout, stderr = client.exec_command(full_cmd, get_pty=True)
    # Stream output
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
        description="Orchestrate ai-t9 training on Vast.ai"
    )

    # Instance selection
    parser.add_argument("--gpu",             default="RTX_3090",  help="GPU name filter for offer search (default: RTX_3090)")
    parser.add_argument("--min-vram",        type=int, default=16, help="Minimum VRAM in GB (default: 16)")
    parser.add_argument("--max-price",       type=float, default=1.0, help="Max $ per hour (default: 1.0)")
    parser.add_argument("--image",           default="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
                        help="Docker image for the instance")
    parser.add_argument("--instance-id",     type=int, default=None,
                        help="Reuse an existing instance instead of provisioning a new one")
    parser.add_argument("--no-destroy",      action="store_true",
                        help="Do not destroy the instance after training (useful for debugging)")
    parser.add_argument("--dry-run",         action="store_true",
                        help="Print the cheapest matching offer and exit without provisioning")

    # Training hyperparameters (forwarded to vast_train.sh via env vars)
    parser.add_argument("--epochs",          default="5")
    parser.add_argument("--embed-dim",       default="64")
    parser.add_argument("--batch-size",      default="8192")
    parser.add_argument("--accumulate",      default="4")
    parser.add_argument("--temperature",     default="0.07")
    parser.add_argument("--weight-decay",    default="0.0001")
    parser.add_argument("--warmup-frac",     default="0.05")
    parser.add_argument("--lr",              default="0.001")
    parser.add_argument("--context-window",  default="3")
    parser.add_argument("--model-type",      default="char-ngram", choices=["dual-encoder", "char-ngram"])
    parser.add_argument("--shard-size",      default="10000000")
    parser.add_argument("--skip-prep",       action="store_true", help="Skip pair precomputation (reuse existing pairs from S3)")
    parser.add_argument("--skip-train",      action="store_true", help="Only run prep, skip training")
    parser.add_argument("--save-ngram",      action="store_true", help="Train and upload bigram model")

    # Download
    parser.add_argument("--output-dir",      default=str(DEFAULT_OUTPUT_DIR),
                        help="Local directory to download artifacts to (default: data/)")

    args = parser.parse_args(argv)

    # ---- Validate required S3 env vars ------------------------------------
    required_env = [
        "AI_T9_S3_ENDPOINT", "AI_T9_S3_BUCKET",
        "AI_T9_S3_ACCESS_KEY", "AI_T9_S3_SECRET_KEY",
    ]
    missing = [k for k in required_env if not os.environ.get(k)]
    if missing:
        print(f"ERROR: Missing required environment variables: {', '.join(missing)}", file=sys.stderr)
        return 1

    # ---- Build train env vars ----------------------------------------------
    train_env = {k: os.environ[k] for k in required_env}
    train_env.update({
        "EPOCHS": args.epochs,
        "EMBED_DIM": args.embed_dim,
        "BATCH_SIZE": args.batch_size,
        "ACCUMULATE": args.accumulate,
        "TEMPERATURE": args.temperature,
        "WEIGHT_DECAY": args.weight_decay,
        "WARMUP_FRAC": args.warmup_frac,
        "LR": args.lr,
        "CONTEXT_WINDOW": args.context_window,
        "MODEL_TYPE": args.model_type,
        "SHARD_SIZE": args.shard_size,
        "SKIP_PREP": "1" if args.skip_prep else "0",
        "SKIP_TRAIN": "1" if args.skip_train else "0",
        "SAVE_NGRAM": "1" if args.save_ngram else "0",
    })
    if os.environ.get("VOCAB_REMOTE"):
        train_env["VOCAB_REMOTE"] = os.environ["VOCAB_REMOTE"]
    if os.environ.get("PAIRS_REMOTE"):
        train_env["PAIRS_REMOTE"] = os.environ["PAIRS_REMOTE"]
    if os.environ.get("MODEL_REMOTE"):
        train_env["MODEL_REMOTE"] = os.environ["MODEL_REMOTE"]
    if os.environ.get("NGRAM_REMOTE"):
        train_env["NGRAM_REMOTE"] = os.environ["NGRAM_REMOTE"]
    if os.environ.get("CORPUS_REMOTE"):
        train_env["CORPUS_REMOTE"] = os.environ["CORPUS_REMOTE"]

    # ---- Dry run ----------------------------------------------------------
    if args.dry_run:
        print(f"Searching for '{args.gpu}' offers (max ${args.max_price}/hr, ≥{args.min_vram} GB VRAM)…")
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

    # ---- Provision or reuse instance --------------------------------------
    instance_id = args.instance_id
    created = False
    if instance_id is None:
        print(f"Searching for '{args.gpu}' offers…")
        offers = search_offers(args.gpu, args.min_vram, args.max_price)
        if not offers:
            print("ERROR: No matching offers found. Try relaxing --gpu, --min-vram, or --max-price.", file=sys.stderr)
            return 1
        best = offers[0]
        print(f"Best offer: ID={best['id']} GPU={best['gpu_name']} ${best['dph_base']:.3f}/hr")
        print("Creating instance…")
        instance_id = create_instance(best["id"], image=args.image)
        created = True
        print(f"Instance {instance_id} created. Waiting for it to start…")

    try:
        info = wait_for_instance(instance_id)
        ssh_host = info.get("ssh_host") or info.get("public_ipaddr")
        ssh_port = int(info.get("ssh_port", 22))
        print(f"\nInstance ready: {ssh_host}:{ssh_port}")

        # ---- Connect via SSH -------------------------------------------
        print("Connecting via SSH…")
        client = _ssh_client(ssh_host, ssh_port)

        # ---- Upload training script ------------------------------------
        print("Uploading vast_train.sh…")
        scp_upload(client, TRAIN_SCRIPT, "/root/vast_train.sh")
        ssh_run(client, "chmod +x /root/vast_train.sh")

        # Install aws CLI if needed
        print("Installing aws CLI…")
        ssh_run(client, "which aws || (apt-get update -qq && apt-get install -y awscli 2>&1 | tail -5)")

        # ---- Run training script ---------------------------------------
        print("Starting training…")
        exit_code = ssh_run(client, "bash /root/vast_train.sh", env=train_env)
        if exit_code != 0:
            print(f"\nERROR: vast_train.sh exited with code {exit_code}", file=sys.stderr)
            if not args.no_destroy and created:
                destroy_instance(instance_id)
            return exit_code

        # ---- Download artifacts ----------------------------------------
        if not args.skip_train:
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            print(f"\nDownloading artifacts to {out}/…")
            for name in ["model.npz", "bigram.npz", "vocab/vocab.json"]:
                remote = f"/root/ai_t9_workdir/{name}"
                local = out / Path(name).name
                try:
                    scp_download(client, remote, local)
                    size_mb = local.stat().st_size / 1e6
                    print(f"  {Path(name).name}: {size_mb:.1f} MB")
                except FileNotFoundError:
                    print(f"  {Path(name).name}: not found, skipping")

        client.close()

    finally:
        # Always destroy if we created the instance (unless --no-destroy)
        if created and not args.no_destroy:
            print("\nDestroying instance…")
            destroy_instance(instance_id)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
