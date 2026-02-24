#!/usr/bin/env python3
"""Supervisor that runs on a Vast.ai GPU instance inside a tmux session.

Self-contained: uses only stdlib + boto3 (no ai_t9 imports).
Uploaded to S3 at job creation and re-downloaded on every respawn.

Usage (started by provision.py via tmux):
    python /root/supervisor.py \
        --job-id train-20240215-abc123 \
        --instance-id 98765432 \
        --s3-bucket my-bucket \
        --jobs-prefix jobs/ \
        --run-cmd "ai-t9-run /root/train_config.yaml"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _make_s3() -> tuple:
    """Build a boto3 S3 client from environment variables.

    Returns (client, bucket).
    """
    import boto3  # type: ignore

    endpoint = os.environ.get("AI_T9_S3_ENDPOINT")
    access_key = os.environ.get("AI_T9_S3_ACCESS_KEY")
    secret_key = os.environ.get("AI_T9_S3_SECRET_KEY")
    region = os.environ.get("AI_T9_S3_REGION", "us-east-1")
    bucket = os.environ.get("AI_T9_S3_BUCKET", "")

    kwargs: dict = {
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
        "region_name": region,
    }
    if endpoint:
        kwargs["endpoint_url"] = endpoint

    client = boto3.client("s3", **kwargs)
    return client, bucket


def _s3_key(prefix: str, job_id: str, filename: str) -> str:
    return f"{prefix.rstrip('/')}/{job_id}/{filename}"


def _update_manifest_status(s3, bucket: str, prefix: str, job_id: str, status: str) -> None:
    """Read the manifest, update its status field, and write it back."""
    key = _s3_key(prefix, job_id, "manifest.json")
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        manifest = json.loads(response["Body"].read())
    except Exception as exc:
        print(f"[supervisor] WARNING: could not read manifest: {exc}", flush=True)
        manifest = {"job_id": job_id}

    manifest["status"] = status
    manifest["updated_at"] = _now_iso()

    event = {"time": _now_iso(), "type": status, "msg": f"supervisor: process exited with status={status}"}
    manifest.setdefault("events", []).append(event)

    body = json.dumps(manifest, indent=2).encode()
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    except Exception as exc:
        print(f"[supervisor] WARNING: could not update manifest: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Heartbeat thread
# ---------------------------------------------------------------------------


class HeartbeatThread(threading.Thread):
    """Writes a heartbeat.json to S3 every *interval* seconds."""

    def __init__(
        self,
        s3,
        bucket: str,
        prefix: str,
        job_id: str,
        instance_id: int,
        pid: int,
        interval: int = 60,
    ) -> None:
        super().__init__(daemon=True)
        self._s3 = s3
        self._bucket = bucket
        self._prefix = prefix
        self._job_id = job_id
        self._instance_id = instance_id
        self._pid = pid
        self._interval = interval
        self._stop_event = threading.Event()
        self.returncode: int | None = None

    def _write(self) -> None:
        data = {
            "timestamp": _now_iso(),
            "instance_id": self._instance_id,
            "pid": self._pid,
            "returncode": self.returncode,
        }
        key = _s3_key(self._prefix, self._job_id, "heartbeat.json")
        try:
            self._s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=json.dumps(data).encode(),
                ContentType="application/json",
            )
        except Exception as exc:
            print(f"[supervisor] WARNING: heartbeat write failed: {exc}", flush=True)

    def run(self) -> None:
        while not self._stop_event.wait(timeout=self._interval):
            self._write()

    def stop(self) -> None:
        """Signal the thread to stop and write one final heartbeat."""
        self._stop_event.set()
        self.join(timeout=10)
        # Always write a final heartbeat so the returncode is recorded.
        self._write()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="ai-t9 on-instance supervisor")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--instance-id", type=int, required=True)
    parser.add_argument("--s3-bucket", default=None, help="Override AI_T9_S3_BUCKET env var")
    parser.add_argument("--jobs-prefix", default="jobs/")
    parser.add_argument("--run-cmd", required=True, help="Command to execute (passed to shell)")
    parser.add_argument("--heartbeat-interval", type=int, default=60)
    args = parser.parse_args()

    job_id = args.job_id
    instance_id = args.instance_id
    prefix = args.jobs_prefix

    print(f"[supervisor] Starting job {job_id} on instance {instance_id}", flush=True)
    print(f"[supervisor] Run command: {args.run_cmd}", flush=True)

    # ---- Build S3 client ---------------------------------------------------
    try:
        s3, bucket = _make_s3()
        if args.s3_bucket:
            bucket = args.s3_bucket
    except Exception as exc:
        print(f"[supervisor] FATAL: could not create S3 client: {exc}", flush=True)
        return 1

    # ---- Launch training subprocess ----------------------------------------
    print(f"[supervisor] Launching: {args.run_cmd}", flush=True)
    proc = subprocess.Popen(
        args.run_cmd,
        shell=True,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    # ---- Start heartbeat thread ---------------------------------------------
    hb = HeartbeatThread(
        s3=s3,
        bucket=bucket,
        prefix=prefix,
        job_id=job_id,
        instance_id=instance_id,
        pid=proc.pid,
        interval=args.heartbeat_interval,
    )
    hb.start()
    print(f"[supervisor] Heartbeat thread started (interval={args.heartbeat_interval}s)", flush=True)

    # ---- Wait for subprocess -----------------------------------------------
    proc.wait()
    returncode = proc.returncode
    print(f"[supervisor] Process exited with returncode={returncode}", flush=True)

    # ---- Stop heartbeat and record final status ----------------------------
    hb.returncode = returncode
    hb.stop()

    final_status = "completed" if returncode == 0 else "failed"
    print(f"[supervisor] Writing final manifest status={final_status}", flush=True)
    _update_manifest_status(s3, bucket, prefix, job_id, final_status)

    return returncode


if __name__ == "__main__":
    sys.exit(main())
