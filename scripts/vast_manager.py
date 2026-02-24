#!/usr/bin/env python3
"""Autonomous Vast.ai Training Manager.

Always-on VPS daemon that automatically restarts preempted GPU instances
and exposes a simple web UI for monitoring and control.

S3 is the single source of truth. The manager never SSHes into a running
instance — it reads only heartbeat.json and manifest.json from S3.

Environment variables required:
    AI_T9_S3_ENDPOINT      e.g. https://s3.example.com  (omit for AWS)
    AI_T9_S3_BUCKET        bucket name
    AI_T9_S3_ACCESS_KEY    access key
    AI_T9_S3_SECRET_KEY    secret key
    AI_T9_S3_REGION        region (default: us-east-1)
    AI_T9_JOBS_PREFIX      S3 prefix for jobs (default: jobs/)

Optional env (forwarded to GPU instances):
    HF_TOKEN / HUGGING_FACE_HUB_TOKEN

Usage:
    python scripts/vast_manager.py --port 7860
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
import tempfile
import threading
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).parent))

from vast.api import destroy_instance, get_instance_info
from vast.jobs import (
    Job,
    JobEvent,
    list_jobs,
    load_heartbeat,
    load_job,
    make_job_id,
    save_job,
)
from vast.provision import ProvisionConfig, provision_and_start

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_IMAGE = "pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel"

_FORWARD_ENV_KEYS = [
    "AI_T9_S3_ENDPOINT",
    "AI_T9_S3_BUCKET",
    "AI_T9_S3_ACCESS_KEY",
    "AI_T9_S3_SECRET_KEY",
    "AI_T9_S3_REGION",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
]

# Seconds of heartbeat silence before we check the instance is alive
_HEARTBEAT_STALE_SECONDS = 300

# Seconds after job creation before we expect a heartbeat
_STARTUP_GRACE_SECONDS = 600

_SUPERVISOR_PATH = Path(__file__).parent / "vast_supervisor.py"

# ---------------------------------------------------------------------------
# Dashboard HTML
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ai-t9 Training Manager</title>
<style>
  *{box-sizing:border-box}
  body{font-family:system-ui,sans-serif;margin:0;background:#0f1117;color:#e2e8f0}
  header{background:#1a1d2e;padding:1rem 1.5rem;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #2d3748}
  h1{margin:0;font-size:1.25rem;color:#a78bfa}
  #new-btn{background:#7c3aed;color:#fff;border:none;padding:.5rem 1rem;border-radius:.375rem;cursor:pointer;font-size:.875rem}
  #new-btn:hover{background:#6d28d9}
  main{padding:1.5rem;max-width:960px;margin:0 auto}
  .card{background:#1e2235;border:1px solid #2d3748;border-radius:.5rem;margin-bottom:1rem;padding:1rem}
  .card-header{display:flex;align-items:center;gap:.75rem;flex-wrap:wrap}
  .job-id{font-family:monospace;font-size:.875rem;color:#94a3b8}
  .badge{display:inline-block;padding:.2rem .6rem;border-radius:9999px;font-size:.75rem;font-weight:600}
  .badge-running{background:#065f46;color:#6ee7b7}
  .badge-completed{background:#1e3a5f;color:#60a5fa}
  .badge-failed{background:#7f1d1d;color:#fca5a5}
  .badge-interrupted{background:#78350f;color:#fcd34d}
  .badge-stopped{background:#374151;color:#9ca3af}
  .badge-pending{background:#4a2a00;color:#fbbf24}
  .meta{font-size:.8rem;color:#64748b;margin-top:.5rem}
  .progress{font-size:.8rem;color:#a78bfa;margin-top:.25rem}
  .actions{margin-top:.75rem;display:flex;gap:.5rem}
  .btn{border:none;padding:.375rem .75rem;border-radius:.25rem;cursor:pointer;font-size:.8rem}
  .btn-stop{background:#7f1d1d;color:#fca5a5}
  .btn-stop:hover{background:#991b1b}
  .btn-restart{background:#1e3a5f;color:#60a5fa}
  .btn-restart:hover{background:#1e40af}
  .events-toggle{background:none;border:none;color:#64748b;cursor:pointer;font-size:.75rem;padding:0;margin-top:.5rem}
  .events-toggle:hover{color:#94a3b8}
  .events{display:none;margin-top:.5rem;border-top:1px solid #2d3748;padding-top:.5rem}
  .events.open{display:block}
  .event{font-size:.75rem;margin:.2rem 0;padding:.2rem .5rem;border-left:2px solid #4b5563}
  .event-time{color:#64748b;margin-right:.5rem}
  .event-type{font-weight:600;margin-right:.5rem}
  /* Modal */
  .overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:100;align-items:center;justify-content:center}
  .overlay.open{display:flex}
  .modal{background:#1a1d2e;border:1px solid #2d3748;border-radius:.5rem;padding:1.5rem;width:min(600px,95vw);max-height:90vh;overflow-y:auto}
  .modal h2{margin:0 0 1rem;color:#a78bfa;font-size:1.1rem}
  label{display:block;font-size:.8rem;color:#94a3b8;margin:.75rem 0 .25rem}
  input,select,textarea{width:100%;background:#0f1117;border:1px solid #2d3748;color:#e2e8f0;padding:.5rem;border-radius:.25rem;font-size:.875rem}
  textarea{font-family:monospace;height:160px;resize:vertical}
  .form-row{display:grid;grid-template-columns:1fr 1fr;gap:.75rem}
  .modal-actions{display:flex;gap:.75rem;margin-top:1.25rem;justify-content:flex-end}
  .btn-cancel{background:#374151;color:#e2e8f0}
  .btn-cancel:hover{background:#4b5563}
  .btn-submit{background:#7c3aed;color:#fff}
  .btn-submit:hover{background:#6d28d9}
  .spinner{display:inline-block;width:1rem;height:1rem;border:2px solid #a78bfa;border-top-color:transparent;border-radius:50%;animation:spin .6s linear infinite;vertical-align:middle;margin-right:.4rem}
  @keyframes spin{to{transform:rotate(360deg)}}
  #status-bar{font-size:.75rem;color:#64748b;text-align:right;margin-bottom:.75rem}
  #jobs-empty{color:#64748b;text-align:center;padding:2rem}
</style>
</head>
<body>
<header>
  <h1>ai-t9 Training Manager</h1>
  <button id="new-btn" onclick="openModal()">+ New Job</button>
</header>
<main>
  <div id="status-bar">Loading…</div>
  <div id="jobs-list"><div id="jobs-empty">No jobs yet.</div></div>
</main>

<!-- New Job Modal -->
<div class="overlay" id="modal">
<div class="modal">
  <h2>Create Training Job</h2>
  <label>Config YAML <small>(paste contents)</small></label>
  <textarea id="f-config" placeholder="output_dir: data&#10;corpus: ..."></textarea>
  <div class="form-row">
    <div>
      <label>GPU</label>
      <input id="f-gpu" value="RTX_3090">
    </div>
    <div>
      <label>Num GPUs</label>
      <input id="f-num-gpus" type="number" value="1" min="1">
    </div>
  </div>
  <div class="form-row">
    <div>
      <label>Max $/hr</label>
      <input id="f-price" type="number" value="1.0" step="0.1" min="0.1">
    </div>
    <div>
      <label>Min VRAM (GB/GPU)</label>
      <input id="f-vram" type="number" value="16" min="4">
    </div>
  </div>
  <div class="form-row">
    <div>
      <label>Docker Image</label>
      <input id="f-image" value="pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel">
    </div>
    <div>
      <label>Install</label>
      <select id="f-install">
        <option value="wheel">wheel (build + upload)</option>
        <option value="skip">skip (pre-baked image)</option>
      </select>
    </div>
  </div>
  <div class="form-row">
    <div>
      <label>Run steps <small>(comma-separated, blank=all)</small></label>
      <input id="f-steps" placeholder="corpus,vocab,train">
    </div>
    <div>
      <label>Skip steps <small>(comma-separated)</small></label>
      <input id="f-skip" placeholder="">
    </div>
  </div>
  <label>
    <input type="checkbox" id="f-interruptable"> Interruptable instance
  </label>
  <label>Wheel file <small>(optional .whl — leave blank to build from source)</small></label>
  <input type="file" id="f-wheel" accept=".whl">
  <div class="modal-actions">
    <button class="btn btn-cancel" onclick="closeModal()">Cancel</button>
    <button class="btn btn-submit" id="submit-btn" onclick="submitJob()">Create Job</button>
  </div>
</div>
</div>

<script>
const API = '';
let refreshTimer = null;

function statusBadge(s) {
  const cls = {
    running:'running', completed:'completed', failed:'failed',
    interrupted:'interrupted', stopped:'stopped', pending:'pending'
  }[s] || 'stopped';
  return `<span class="badge badge-${cls}">${s}</span>`;
}

function fmtTime(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleString();
}

function renderCheckpoint(cp) {
  if (!cp) return '';
  const parts = [];
  if (cp.epoch !== undefined) parts.push(`epoch ${cp.epoch}`);
  if (cp.shard !== undefined) parts.push(`shard ${cp.shard}`);
  if (cp.step !== undefined) parts.push(`step ${cp.step}`);
  return parts.length ? `<div class="progress">Checkpoint: ${parts.join(' / ')}</div>` : '';
}

function renderEvents(events, id) {
  if (!events || !events.length) return '';
  const rows = events.map(e =>
    `<div class="event"><span class="event-time">${e.time}</span><span class="event-type">${e.type}</span>${e.msg}</div>`
  ).join('');
  return `
    <button class="events-toggle" onclick="toggleEvents('${id}')">▸ Events (${events.length})</button>
    <div class="events" id="events-${id}">${rows}</div>`;
}

function toggleEvents(id) {
  const el = document.getElementById('events-' + id);
  el.classList.toggle('open');
  el.previousElementSibling.textContent = el.classList.contains('open')
    ? '▾ Events (' + el.children.length + ')'
    : '▸ Events (' + el.children.length + ')';
}

function renderCard(job) {
  const canStop = ['running','pending'].includes(job.status);
  const canRestart = ['completed','failed','stopped','interrupted'].includes(job.status);
  const restarts = job.restart_count ? ` · restarts: ${job.restart_count}` : '';
  const interruptable = job.interruptable ? ' · interruptable' : '';
  return `
  <div class="card">
    <div class="card-header">
      ${statusBadge(job.status)}
      <span class="job-id">${job.job_id}</span>
    </div>
    <div class="meta">
      GPU: ${job.gpu} ×${job.num_gpus} · $${job.max_price}/hr${interruptable}${restarts}
      · created: ${fmtTime(job.created_at)}
    </div>
    ${renderCheckpoint(job.checkpoint_ptr)}
    ${renderEvents(job.events, job.job_id)}
    <div class="actions">
      ${canStop ? `<button class="btn btn-stop" onclick="jobAction('${job.job_id}','stop')">Stop</button>` : ''}
      ${canRestart ? `<button class="btn btn-restart" onclick="jobAction('${job.job_id}','restart')">Restart</button>` : ''}
    </div>
  </div>`;
}

async function loadJobs() {
  try {
    const resp = await fetch(API + '/api/jobs');
    const jobs = await resp.json();
    const el = document.getElementById('jobs-list');
    const bar = document.getElementById('status-bar');
    bar.textContent = 'Last refresh: ' + new Date().toLocaleTimeString();
    if (!jobs.length) {
      el.innerHTML = '<div id="jobs-empty">No jobs yet.</div>';
    } else {
      el.innerHTML = jobs.map(renderCard).join('');
    }
  } catch(e) {
    document.getElementById('status-bar').textContent = 'Error loading jobs: ' + e;
  }
}

async function jobAction(id, action) {
  if (!confirm(`${action} job ${id}?`)) return;
  await fetch(`${API}/api/jobs/${id}/${action}`, {method:'POST'});
  loadJobs();
}

function openModal() {
  document.getElementById('modal').classList.add('open');
}
function closeModal() {
  document.getElementById('modal').classList.remove('open');
  document.getElementById('submit-btn').textContent = 'Create Job';
}

async function submitJob() {
  const btn = document.getElementById('submit-btn');
  btn.innerHTML = '<span class="spinner"></span>Creating… (may take ~2 min)';
  btn.disabled = true;

  const configYaml = document.getElementById('f-config').value.trim();
  if (!configYaml) { alert('Config YAML is required'); btn.innerHTML='Create Job'; btn.disabled=false; return; }

  const wheelFile = document.getElementById('f-wheel').files[0];
  let wheelB64 = null;
  if (wheelFile) {
    const ab = await wheelFile.arrayBuffer();
    wheelB64 = btoa(String.fromCharCode(...new Uint8Array(ab)));
  }

  const body = {
    config_yaml: configYaml,
    gpu: document.getElementById('f-gpu').value.trim() || 'RTX_3090',
    num_gpus: parseInt(document.getElementById('f-num-gpus').value) || 1,
    max_price: parseFloat(document.getElementById('f-price').value) || 1.0,
    min_vram: parseInt(document.getElementById('f-vram').value) || 16,
    image: document.getElementById('f-image').value.trim() || 'pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel',
    install: document.getElementById('f-install').value,
    interruptable: document.getElementById('f-interruptable').checked,
    run_steps: document.getElementById('f-steps').value.split(',').map(s=>s.trim()).filter(Boolean),
    skip_steps: document.getElementById('f-skip').value.split(',').map(s=>s.trim()).filter(Boolean),
    wheel_b64: wheelB64,
    wheel_filename: wheelFile ? wheelFile.name : null,
  };

  try {
    const resp = await fetch(API + '/api/jobs', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    const result = await resp.json();
    if (!resp.ok) {
      alert('Error: ' + (result.error || JSON.stringify(result)));
    } else {
      closeModal();
      loadJobs();
    }
  } catch(e) {
    alert('Request failed: ' + e);
  } finally {
    btn.innerHTML = 'Create Job';
    btn.disabled = false;
  }
}

// Initial load + auto-refresh every 30 s
loadJobs();
refreshTimer = setInterval(loadJobs, 30000);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# S3 utilities
# ---------------------------------------------------------------------------


def _s3_key(prefix: str, job_id: str, filename: str) -> str:
    return f"{prefix.rstrip('/')}/{job_id}/{filename}"


def _make_s3():
    """Build a boto3 S3 client from environment variables."""
    import boto3  # type: ignore

    endpoint = os.environ.get("AI_T9_S3_ENDPOINT")
    access_key = os.environ.get("AI_T9_S3_ACCESS_KEY")
    secret_key = os.environ.get("AI_T9_S3_SECRET_KEY")
    region = os.environ.get("AI_T9_S3_REGION", "us-east-1")

    kwargs: dict = {
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
        "region_name": region,
    }
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    return boto3.client("s3", **kwargs)


def _upload_text(s3, bucket: str, key: str, text: str, content_type: str = "text/plain") -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=text.encode(), ContentType=content_type)


def _upload_bytes(s3, bucket: str, key: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    s3.put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class Manager:
    def __init__(
        self,
        s3,
        bucket: str,
        jobs_prefix: str,
        poll_interval: int = 120,
    ) -> None:
        self.s3 = s3
        self.bucket = bucket
        self.jobs_prefix = jobs_prefix
        self.poll_interval = poll_interval
        self._lock = threading.Lock()

        # Env vars forwarded to GPU instances
        self._forward_env: dict[str, str] = {
            k: os.environ[k] for k in _FORWARD_ENV_KEYS if os.environ.get(k)
        }

    # ------------------------------------------------------------------ #
    # Job creation
    # ------------------------------------------------------------------ #

    def create_job(self, body: dict) -> Job:
        """Create, provision, and start a new training job."""
        job_id = make_job_id()
        prefix = self.jobs_prefix

        # ---- Upload config YAML -----------------------------------------
        config_yaml: str = body.get("config_yaml", "")
        config_key = _s3_key(prefix, job_id, "train_config.yaml")
        _upload_text(self.s3, self.bucket, config_key, config_yaml, "text/yaml")
        print(f"[manager] Uploaded config → {config_key}")

        # ---- Upload wheel (if provided) ----------------------------------
        wheel_key: str | None = None
        wheel_b64: str | None = body.get("wheel_b64")
        wheel_filename: str | None = body.get("wheel_filename")
        if wheel_b64 and wheel_filename:
            wheel_bytes = base64.b64decode(wheel_b64)
            wheel_key = _s3_key(prefix, job_id, f"wheel/{wheel_filename}")
            _upload_bytes(self.s3, self.bucket, wheel_key, wheel_bytes)
            print(f"[manager] Uploaded wheel → {wheel_key}")
        elif body.get("install", "wheel") == "wheel":
            # Build wheel from source
            print("[manager] Building local wheel…")
            wheel_path = _build_local_wheel()
            wheel_key = _s3_key(prefix, job_id, f"wheel/{wheel_path.name}")
            _upload_bytes(self.s3, self.bucket, wheel_key, wheel_path.read_bytes())
            print(f"[manager] Uploaded built wheel → {wheel_key}")

        # ---- Upload supervisor.py ----------------------------------------
        supervisor_key = _s3_key(prefix, job_id, "supervisor.py")
        _upload_bytes(
            self.s3, self.bucket, supervisor_key,
            _SUPERVISOR_PATH.read_bytes(), "text/x-python"
        )
        print(f"[manager] Uploaded supervisor → {supervisor_key}")

        # ---- Create Job record ------------------------------------------
        job = Job(
            job_id=job_id,
            status="pending",
            gpu=body.get("gpu", "RTX_3090"),
            num_gpus=int(body.get("num_gpus", 1)),
            min_vram=int(body.get("min_vram", 16)),
            max_price=float(body.get("max_price", 1.0)),
            cuda_version=body.get("cuda_version", "12.0"),
            disk_gb=int(body.get("disk_gb", 30)),
            interruptable=bool(body.get("interruptable", False)),
            bid_price=body.get("bid_price"),
            image=body.get("image", _DEFAULT_IMAGE),
            install=body.get("install", "wheel"),
            config_s3_key=config_key,
            wheel_s3_key=wheel_key,
            supervisor_s3_key=supervisor_key,
            run_steps=body.get("run_steps", []),
            skip_steps=body.get("skip_steps", []),
        )
        job.add_event("created", f"Job created via API")
        save_job(self.s3, self.bucket, prefix, job)

        # ---- Provision + start ------------------------------------------
        try:
            instance_id, ssh_host, ssh_port = provision_and_start(
                self._make_provision_cfg(job),
                self.s3, self.bucket, prefix, job_id,
                log=lambda msg: print(f"[manager][{job_id}] {msg}"),
            )
            job.instance_id = instance_id
            job.ssh_host = ssh_host
            job.ssh_port = ssh_port
            job.status = "running"
            job.add_event("running", f"Provisioned instance {instance_id} at {ssh_host}:{ssh_port}")
        except Exception as exc:
            tb = traceback.format_exc()
            print(f"[manager] ERROR provisioning {job_id}: {exc}\n{tb}")
            job.status = "failed"
            job.add_event("failed", f"Provisioning failed: {exc}")

        save_job(self.s3, self.bucket, prefix, job)
        return job

    # ------------------------------------------------------------------ #
    # Stop / Restart
    # ------------------------------------------------------------------ #

    def stop_job(self, job: Job) -> None:
        if job.instance_id is not None:
            try:
                destroy_instance(job.instance_id)
            except Exception as exc:
                print(f"[manager] Warning: destroy {job.instance_id} failed: {exc}")
        job.status = "stopped"
        job.add_event("stopped", "Stopped by user")
        save_job(self.s3, self.bucket, self.jobs_prefix, job)

    def restart_job(self, job: Job) -> None:
        if job.instance_id is not None:
            try:
                destroy_instance(job.instance_id)
            except Exception as exc:
                print(f"[manager] Warning: destroy {job.instance_id} failed: {exc}")
            job.instance_id = None
        self._respawn(job, reason="manual restart")

    def _respawn(self, job: Job, reason: str = "preempted") -> None:
        job.restart_count += 1
        job.status = "pending"
        job.add_event("interrupted", f"Respawning: {reason} (attempt {job.restart_count})")
        save_job(self.s3, self.bucket, self.jobs_prefix, job)

        try:
            instance_id, ssh_host, ssh_port = provision_and_start(
                self._make_provision_cfg(job),
                self.s3, self.bucket, self.jobs_prefix, job.job_id,
                log=lambda msg: print(f"[manager][{job.job_id}] {msg}"),
            )
            job.instance_id = instance_id
            job.ssh_host = ssh_host
            job.ssh_port = ssh_port
            job.status = "running"
            job.add_event("running", f"Respawned on instance {instance_id}")
        except Exception as exc:
            print(f"[manager] ERROR respawning {job.job_id}: {exc}")
            job.status = "failed"
            job.add_event("failed", f"Respawn failed: {exc}")

        save_job(self.s3, self.bucket, self.jobs_prefix, job)

    def _make_provision_cfg(self, job: Job) -> ProvisionConfig:
        return ProvisionConfig(
            gpu=job.gpu,
            num_gpus=job.num_gpus,
            min_vram=job.min_vram,
            max_price=job.max_price,
            cuda_version=job.cuda_version,
            disk_gb=job.disk_gb,
            interruptable=job.interruptable,
            bid_price=job.bid_price,
            image=job.image,
            install=job.install,
            config_s3_key=job.config_s3_key,
            wheel_s3_key=job.wheel_s3_key,
            supervisor_s3_key=job.supervisor_s3_key,
            run_steps=job.run_steps,
            skip_steps=job.skip_steps,
            forward_env=self._forward_env,
        )

    # ------------------------------------------------------------------ #
    # Polling
    # ------------------------------------------------------------------ #

    def poll_once(self) -> None:
        """Check all running jobs and respawn stale/gone instances."""
        try:
            jobs = list_jobs(self.s3, self.bucket, self.jobs_prefix)
        except Exception as exc:
            print(f"[manager] poll_once: list_jobs failed: {exc}")
            return

        for job in jobs:
            if job.status != "running":
                continue
            try:
                self._check_job(job)
            except Exception as exc:
                print(f"[manager] poll_once: error checking {job.job_id}: {exc}")

    def _check_job(self, job: Job) -> None:
        from datetime import datetime, timezone

        hb = load_heartbeat(self.s3, self.bucket, self.jobs_prefix, job.job_id)

        # If supervisor wrote a final returncode, the manifest was already updated
        if hb and hb.get("returncode") is not None:
            print(f"[manager] {job.job_id}: final returncode={hb['returncode']} already recorded")
            return

        # Grace period — instance may still be starting
        try:
            created = datetime.fromisoformat(job.created_at)
            age_s = (datetime.now(timezone.utc) - created).total_seconds()
        except Exception:
            age_s = 9999

        if hb is None and age_s < _STARTUP_GRACE_SECONDS:
            print(f"[manager] {job.job_id}: no heartbeat yet (age={age_s:.0f}s, still starting)")
            return

        if hb is not None:
            # Check freshness
            try:
                ts = datetime.fromisoformat(hb["timestamp"])
                hb_age = (datetime.now(timezone.utc) - ts).total_seconds()
            except Exception:
                hb_age = 9999

            if hb_age <= _HEARTBEAT_STALE_SECONDS:
                return  # healthy

            print(f"[manager] {job.job_id}: heartbeat stale ({hb_age:.0f}s old), checking instance…")
        else:
            print(f"[manager] {job.job_id}: no heartbeat after {age_s:.0f}s, checking instance…")

        # Verify instance still exists
        instance_alive = False
        if job.instance_id is not None:
            try:
                info = get_instance_info(job.instance_id)
                instance_alive = info.get("actual_status") == "running"
            except Exception:
                instance_alive = False

        if instance_alive:
            print(f"[manager] {job.job_id}: instance {job.instance_id} still running, leaving it")
            return

        print(f"[manager] {job.job_id}: instance gone — respawning")
        self._respawn(job, reason="instance preempted or lost")

    def _load_checkpoint_ptr(self, job: Job) -> dict | None:
        """Read checkpoints/ptr.json from S3 for UI display."""
        try:
            key = "checkpoints/ptr.json"
            resp = self.s3.get_object(Bucket=self.bucket, Key=key)
            return json.loads(resp["Body"].read())
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Poll thread
# ---------------------------------------------------------------------------


class PollThread(threading.Thread):
    def __init__(self, manager: Manager, interval: int = 120) -> None:
        super().__init__(daemon=True)
        self._manager = manager
        self._interval = interval
        self._stop = threading.Event()

    def run(self) -> None:
        print(f"[poll] Started (interval={self._interval}s)")
        while not self._stop.wait(timeout=self._interval):
            print("[poll] Checking running jobs…")
            self._manager.poll_once()

    def stop(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------


def _make_handler(manager: Manager):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # suppress default access log
            pass

        def _send_json(self, data: Any, status: int = 200) -> None:
            body = json.dumps(data, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, html: str) -> None:
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                self._send_html(_DASHBOARD_HTML)
            elif path == "/api/jobs":
                try:
                    jobs = list_jobs(manager.s3, manager.bucket, manager.jobs_prefix)
                    self._send_json([j.to_dict() for j in jobs])
                except Exception as exc:
                    self._send_json({"error": str(exc)}, 500)
            elif path.startswith("/api/jobs/") and not path.endswith("/stop") and not path.endswith("/restart"):
                job_id = path[len("/api/jobs/"):]
                try:
                    job = load_job(manager.s3, manager.bucket, manager.jobs_prefix, job_id)
                    d = job.to_dict()
                    d["checkpoint_ptr"] = manager._load_checkpoint_ptr(job)
                    self._send_json(d)
                except Exception as exc:
                    self._send_json({"error": str(exc)}, 404)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            path = self.path.split("?")[0]
            if path == "/api/jobs":
                try:
                    body = self._read_body()
                    # Run in thread so the HTTP response isn't blocked for 2 min
                    result: dict = {}
                    error: list[str] = []

                    def _create():
                        try:
                            job = manager.create_job(body)
                            result["job"] = job.to_dict()
                        except Exception as exc:
                            error.append(str(exc))

                    t = threading.Thread(target=_create, daemon=True)
                    t.start()
                    t.join(timeout=300)  # wait up to 5 min

                    if error:
                        self._send_json({"error": error[0]}, 500)
                    elif "job" in result:
                        self._send_json(result["job"], 201)
                    else:
                        self._send_json({"error": "timeout"}, 504)
                except Exception as exc:
                    self._send_json({"error": str(exc)}, 500)

            elif path.startswith("/api/jobs/"):
                rest = path[len("/api/jobs/"):]
                parts = rest.rsplit("/", 1)
                if len(parts) != 2:
                    self.send_response(404)
                    self.end_headers()
                    return
                job_id, action = parts
                if action not in ("stop", "restart"):
                    self.send_response(404)
                    self.end_headers()
                    return
                try:
                    job = load_job(manager.s3, manager.bucket, manager.jobs_prefix, job_id)
                except Exception as exc:
                    self._send_json({"error": str(exc)}, 404)
                    return
                try:
                    if action == "stop":
                        manager.stop_job(job)
                    elif action == "restart":
                        threading.Thread(
                            target=manager.restart_job, args=(job,), daemon=True
                        ).start()
                    self._send_json({"ok": True})
                except Exception as exc:
                    self._send_json({"error": str(exc)}, 500)
            else:
                self.send_response(404)
                self.end_headers()

    return Handler


# ---------------------------------------------------------------------------
# Wheel builder (reused from orchestrate)
# ---------------------------------------------------------------------------


def _build_local_wheel() -> Path:
    import subprocess

    project_root = Path(__file__).parent.parent
    wheel_dir = Path(tempfile.mkdtemp(prefix="ai_t9_wheel_"))
    result = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", ".", "--no-deps",
         "--wheel-dir", str(wheel_dir), "--quiet"],
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


# ---------------------------------------------------------------------------
# Systemd unit helper
# ---------------------------------------------------------------------------

_SYSTEMD_UNIT = """\
[Unit]
Description=ai-t9 Training Manager
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory={workdir}
EnvironmentFile={workdir}/.env
ExecStart={python} {script} --port {port}
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ai-t9 autonomous training manager")
    parser.add_argument("--port", type=int, default=7860, help="HTTP port (default: 7860)")
    parser.add_argument("--poll-interval", type=int, default=120, help="Poll interval in seconds (default: 120)")
    parser.add_argument("--print-systemd-unit", action="store_true", help="Print a systemd unit file template and exit")
    args = parser.parse_args(argv)

    if args.print_systemd_unit:
        print(_SYSTEMD_UNIT.format(
            workdir=Path(__file__).parent.parent,
            python=sys.executable,
            script=Path(__file__).resolve(),
            port=args.port,
        ))
        return 0

    # ---- Validate environment ------------------------------------------
    required = ["AI_T9_S3_BUCKET", "AI_T9_S3_ACCESS_KEY", "AI_T9_S3_SECRET_KEY"]
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        print(f"ERROR: Missing required env vars: {', '.join(missing)}", file=sys.stderr)
        print(
            "Set AI_T9_S3_ENDPOINT, AI_T9_S3_BUCKET, AI_T9_S3_ACCESS_KEY, "
            "AI_T9_S3_SECRET_KEY, AI_T9_S3_REGION",
            file=sys.stderr,
        )
        return 1

    if not shutil.which("vastai"):
        print("ERROR: 'vastai' not found on PATH. Install with: pip install vastai", file=sys.stderr)
        return 1

    if not _SUPERVISOR_PATH.exists():
        print(f"ERROR: Supervisor script not found at {_SUPERVISOR_PATH}", file=sys.stderr)
        return 1

    bucket = os.environ["AI_T9_S3_BUCKET"]
    prefix = os.environ.get("AI_T9_JOBS_PREFIX", "jobs/")

    # ---- Build S3 client -----------------------------------------------
    try:
        s3 = _make_s3()
    except Exception as exc:
        print(f"ERROR: Could not create S3 client: {exc}", file=sys.stderr)
        return 1

    print(f"[manager] S3 bucket={bucket!r} prefix={prefix!r}")

    # ---- Start manager + poll thread ------------------------------------
    manager = Manager(s3=s3, bucket=bucket, jobs_prefix=prefix, poll_interval=args.poll_interval)
    poll_thread = PollThread(manager, interval=args.poll_interval)
    poll_thread.start()

    # ---- Start HTTP server ---------------------------------------------
    handler = _make_handler(manager)
    server = HTTPServer(("0.0.0.0", args.port), handler)
    print(f"[manager] Listening on http://0.0.0.0:{args.port}")
    print("[manager] Hint: run with --print-systemd-unit to get a systemd service template")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[manager] Shutting down…")
        poll_thread.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
