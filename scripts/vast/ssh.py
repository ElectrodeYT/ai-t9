"""SSH/SFTP helpers for connecting to Vast.ai instances."""

from __future__ import annotations

import select
import shlex
import sys
import time
from pathlib import Path

import paramiko
import paramiko.ssh_exception

# Key classes tried in preference order when loading an SSH private key.
# DSSKey was removed in paramiko 3+, so we guard it.
_KEY_CLASSES = [
    paramiko.Ed25519Key,
    paramiko.RSAKey,
    paramiko.ECDSAKey,
    *([paramiko.DSSKey] if hasattr(paramiko, "DSSKey") else []),
]

# Default private key file locations, tried in order.
_DEFAULT_KEY_PATHS = [
    Path.home() / ".ssh" / "id_ed25519",
    Path.home() / ".ssh" / "id_ecdsa",
    Path.home() / ".ssh" / "id_rsa",
]


def _load_private_key(path: Path) -> paramiko.PKey:
    """Attempt to load a private key from *path*, trying each key type in turn.

    Raises ``ValueError`` with a combined message if all types fail.
    """
    errors: list[str] = []
    for cls in _KEY_CLASSES:
        try:
            return cls.from_private_key_file(str(path))
        except Exception as exc:
            errors.append(f"{cls.__name__}: {exc}")
    raise ValueError(
        f"Could not load private key from {path}:\n" + "\n".join(errors)
    )


def connect_ssh(
    host: str,
    port: int,
    user: str = "root",
    max_retries: int = 8,
    retry_delay: int = 10,
    connect_timeout: int = 30,
) -> paramiko.SSHClient:
    """Open an authenticated SSH connection to a Vast.ai instance.

    Tries each candidate key in ``~/.ssh`` in turn, retrying the connection on
    transient errors with a fixed delay between attempts.

    Raises ``ConnectionError`` if all attempts are exhausted.
    """
    pkey: paramiko.PKey | None = None
    for kp in _DEFAULT_KEY_PATHS:
        if not kp.exists():
            continue
        try:
            pkey = _load_private_key(kp)
            break
        except ValueError:
            pass  # Try the next candidate

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    _RETRYABLE = (
        paramiko.ssh_exception.NoValidConnectionsError,
        paramiko.ssh_exception.SSHException,
        OSError,
    )

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            client.connect(
                host,
                port=port,
                username=user,
                pkey=pkey,
                timeout=connect_timeout,
                # Only fall back to the SSH agent / ~/.ssh if we found no key ourselves
                look_for_keys=(pkey is None),
                allow_agent=(pkey is None),
            )
            return client
        except _RETRYABLE as exc:
            last_exc = exc
            print(
                f"  SSH attempt {attempt}/{max_retries} failed: {exc}",
                flush=True,
            )
            if attempt < max_retries:
                time.sleep(retry_delay)

    raise ConnectionError(
        f"SSH to {host}:{port} failed after {max_retries} attempts: {last_exc}"
    )


def ssh_run(
    client: paramiko.SSHClient,
    command: str,
    env: dict[str, str] | None = None,
) -> int:
    """Run a shell command over SSH, streaming its output. Returns the exit code.

    Environment variables in *env* are prepended to the command using
    ``shlex.quote`` so values with spaces or special characters are safe.
    """
    if env:
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " "
    else:
        env_prefix = ""

    full_cmd = env_prefix + command
    _stdin, stdout, _stderr = client.exec_command(full_cmd, get_pty=True)
    channel = stdout.channel

    while not channel.exit_status_ready():
        rready, _, _ = select.select([channel], [], [], 0.5)
        if rready:
            data = channel.recv(4096)
            if data:
                sys.stdout.buffer.write(data)
                sys.stdout.flush()

    # Drain any remaining buffered output
    while True:
        data = channel.recv(4096)
        if not data:
            break
        sys.stdout.buffer.write(data)
    sys.stdout.flush()

    return channel.recv_exit_status()


def scp_upload(client: paramiko.SSHClient, local_path: Path, remote_path: str) -> None:
    """Upload a local file to *remote_path* over SFTP."""
    sftp = client.open_sftp()
    try:
        sftp.put(str(local_path), remote_path)
    finally:
        sftp.close()


def scp_download(
    client: paramiko.SSHClient, remote_path: str, local_path: Path
) -> None:
    """Download *remote_path* to *local_path* over SFTP."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    sftp = client.open_sftp()
    try:
        sftp.get(remote_path, str(local_path))
    finally:
        sftp.close()
