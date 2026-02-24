"""Job data model and S3 persistence for the Vast.ai training manager.

boto3 is imported lazily so this module can be imported on machines without
it installed (e.g. during unit testing or on the GPU instance itself).
"""

from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _s3_key(prefix: str, job_id: str, filename: str) -> str:
    """Return ``prefix/job_id/filename`` with correct slash handling."""
    prefix = prefix.rstrip("/")
    return f"{prefix}/{job_id}/{filename}"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class JobEvent:
    time: str
    type: str  # e.g. "created", "running", "interrupted", "respawned", "stopped"
    msg: str

    def to_dict(self) -> dict:
        return {"time": self.time, "type": self.type, "msg": self.msg}

    @classmethod
    def from_dict(cls, d: dict) -> "JobEvent":
        return cls(time=d["time"], type=d["type"], msg=d.get("msg", ""))


@dataclass
class Job:
    # Identity
    job_id: str
    status: str  # pending | running | completed | failed | stopped | interrupted

    # Instance info (filled after provisioning)
    instance_id: int | None = None
    ssh_host: str | None = None
    ssh_port: int | None = None

    # GPU / offer constraints
    gpu: str = "RTX_3090"
    num_gpus: int = 1
    min_vram: int = 16
    max_price: float = 1.0
    cuda_version: str = "12.0"
    disk_gb: int = 30

    # Instance flavour
    interruptable: bool = False
    bid_price: float | None = None

    # Docker image + install mode
    image: str = "pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel"
    install: str = "wheel"  # "wheel" | "skip"

    # S3 artifact keys (relative to bucket root)
    config_s3_key: str | None = None
    wheel_s3_key: str | None = None
    supervisor_s3_key: str | None = None

    # Training step overrides forwarded to ai-t9-run
    run_steps: list[str] = field(default_factory=list)
    skip_steps: list[str] = field(default_factory=list)

    # Timestamps + counters
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    restart_count: int = 0

    # Append-only event log
    events: list[JobEvent] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def add_event(self, type_: str, msg: str) -> None:
        self.events.append(JobEvent(time=_now_iso(), type=type_, msg=msg))
        self.updated_at = _now_iso()

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "job_id": self.job_id,
            "status": self.status,
            "instance_id": self.instance_id,
            "ssh_host": self.ssh_host,
            "ssh_port": self.ssh_port,
            "gpu": self.gpu,
            "num_gpus": self.num_gpus,
            "min_vram": self.min_vram,
            "max_price": self.max_price,
            "cuda_version": self.cuda_version,
            "disk_gb": self.disk_gb,
            "interruptable": self.interruptable,
            "bid_price": self.bid_price,
            "image": self.image,
            "install": self.install,
            "config_s3_key": self.config_s3_key,
            "wheel_s3_key": self.wheel_s3_key,
            "supervisor_s3_key": self.supervisor_s3_key,
            "run_steps": self.run_steps,
            "skip_steps": self.skip_steps,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "restart_count": self.restart_count,
            "events": [e.to_dict() for e in self.events],
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        events = [JobEvent.from_dict(e) for e in d.pop("events", [])]
        obj = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        obj.events = events
        return obj


# ---------------------------------------------------------------------------
# Job ID generation
# ---------------------------------------------------------------------------


def make_job_id() -> str:
    """Return a unique job ID like ``train-20240215-a3f2b1``."""
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"train-{date}-{suffix}"


# ---------------------------------------------------------------------------
# S3 I/O
# ---------------------------------------------------------------------------


def save_job(s3: Any, bucket: str, prefix: str, job: Job) -> None:
    """Serialise *job* and write it to S3 as ``manifest.json``."""
    key = _s3_key(prefix, job.job_id, "manifest.json")
    body = json.dumps(job.to_dict(), indent=2).encode()
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")


def load_job(s3: Any, bucket: str, prefix: str, job_id: str) -> Job:
    """Read and deserialise a job manifest from S3."""
    key = _s3_key(prefix, job_id, "manifest.json")
    response = s3.get_object(Bucket=bucket, Key=key)
    data = json.loads(response["Body"].read())
    return Job.from_dict(data)


def list_jobs(s3: Any, bucket: str, prefix: str) -> list[Job]:
    """Return all jobs found under *prefix* by scanning for ``manifest.json``."""
    prefix_norm = prefix.rstrip("/") + "/"
    paginator = s3.get_paginator("list_objects_v2")
    jobs: list[Job] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix_norm, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            # cp["Prefix"] = "jobs/train-20240215-abc123/"
            sub = cp["Prefix"]
            job_id = sub.rstrip("/").split("/")[-1]
            try:
                jobs.append(load_job(s3, bucket, prefix, job_id))
            except Exception:
                pass  # skip corrupt / incomplete entries
    # Sort newest first
    jobs.sort(key=lambda j: j.created_at, reverse=True)
    return jobs


def save_heartbeat(s3: Any, bucket: str, prefix: str, job_id: str, data: dict) -> None:
    """Write *data* as ``heartbeat.json`` under the job prefix."""
    key = _s3_key(prefix, job_id, "heartbeat.json")
    body = json.dumps(data).encode()
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")


def load_heartbeat(s3: Any, bucket: str, prefix: str, job_id: str) -> dict | None:
    """Return the heartbeat dict or ``None`` if it doesn't exist yet."""
    key = _s3_key(prefix, job_id, "heartbeat.json")
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(response["Body"].read())
    except Exception:
        return None
