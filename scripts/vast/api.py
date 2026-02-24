"""Vast.ai CLI wrappers (subprocess-based).

Uses the ``vastai`` CLI rather than the SDK, which is not stable.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time


def _vastai(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    """Run a vastai CLI command and return the CompletedProcess."""
    cmd = ["vastai", *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def search_offers(
    gpu_name: str,
    min_vram_gb: int = 16,
    max_price_per_hour: float = 10.0,
    cuda_version: str = "12.0",
    min_disk_gb: int = 100,
    num_gpus: int = 1,
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
        f"num_gpus={num_gpus} "
        f"inet_down>1000 "   # Good download speed for datasets/artifacts
        f"inet_up>600 "      # Good upload speed for checkpoints
        f"disk_space>={min_disk_gb} "
        f"reliability>0.95"
    )
    result = _vastai(
        "search", "offers",
        "--order", "dph_base asc",
        "--limit", str(limit),
        "--raw",
        query,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return []


def create_instance(
    offer_id: int,
    image: str,
    disk_gb: int = 30,
    bid_price: float | None = None,
) -> int:
    """Create an instance from an offer ID. Returns the new instance ID.

    When *bid_price* is set the instance is created as an interruptable
    (spot-like) bid.  Omit it for an on-demand instance.
    """
    args = [
        "create", "instance", str(offer_id),
        "--image", image,
        "--disk", str(disk_gb),
        "--raw",
    ]
    if bid_price is not None:
        args.extend(["--bid_price", str(bid_price)])
    result = _vastai(*args)
    if result.returncode != 0:
        raise RuntimeError(
            f"vastai create instance failed (exit {result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Unexpected response from 'vastai create instance': {result.stdout!r}"
        ) from exc
    if "new_contract" not in data:
        raise RuntimeError(
            f"'new_contract' key missing from create response: {data}"
        )
    return int(data["new_contract"])


def get_instance_info(instance_id: int) -> dict:
    """Return the info dict for a running instance."""
    result = _vastai("show", "instance", str(instance_id), "--raw")
    if result.returncode != 0:
        raise RuntimeError(
            f"vastai show instance failed (exit {result.returncode}): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Unexpected response from 'vastai show instance': {result.stdout!r}"
        ) from exc


def wait_for_instance(
    instance_id: int, timeout: int = 300, poll_interval: int = 10
) -> dict:
    """Block until the instance is running. Returns its info dict."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            info = get_instance_info(instance_id)
        except RuntimeError as exc:
            print(f"  Warning: could not get instance info: {exc}", flush=True)
            time.sleep(poll_interval)
            continue
        status = info.get("actual_status", "")
        if status == "running":
            print()  # clear the \r line
            return info
        print(
            f"  Instance {instance_id} status: {status!r} — waiting…",
            end="\r",
            flush=True,
        )
        time.sleep(poll_interval)
    raise TimeoutError(
        f"Instance {instance_id} did not become ready within {timeout}s"
    )


def destroy_instance(instance_id: int) -> None:
    """Terminate and destroy an instance. Warns but does not raise on failure."""
    result = _vastai("destroy", "instance", str(instance_id))
    if result.returncode != 0:
        print(
            f"  Warning: 'vastai destroy instance' exited {result.returncode}: "
            f"{result.stderr.strip() or result.stdout.strip()}",
            file=sys.stderr,
        )
    else:
        print(f"Instance {instance_id} destroyed.")
