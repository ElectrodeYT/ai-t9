"""CLI: S3-compatible bucket management for ai-t9 training data.

Configuration (set once, e.g. in your shell profile or Modal secrets)::

    export AI_T9_S3_ENDPOINT="https://<account>.r2.cloudflarestorage.com"  # or AWS/B2/MinIO
    export AI_T9_S3_BUCKET="my-ai-t9-bucket"
    export AI_T9_S3_ACCESS_KEY="..."
    export AI_T9_S3_SECRET_KEY="..."
    export AI_T9_S3_REGION="auto"                                           # optional

Recommended backend: Cloudflare R2.  Unlike AWS S3, R2 has no egress fees and
lower GiB/month storage costs.  On Modal, mount the bucket via
``modal.CloudBucketMount`` (+ ``read_only=True`` for GPU training containers)
rather than using boto3 inside the function — the mounted path then behaves
like a normal filesystem and requires no bucket credentials at inference time.

File management::

    ai-t9-data ls                           # list all objects
    ai-t9-data ls corpuses/                 # list objects under a prefix
    ai-t9-data upload data/pairs.npz pairs/pairs.npz
    ai-t9-data download pairs/pairs.npz data/pairs.npz
    ai-t9-data rm pairs/old-pairs.npz
    ai-t9-data stat models/model.npz        # show size, etag, last-modified

Download a HuggingFace dataset as a corpus file.

Two variants cover different network situations:

``fetch-hf`` — suitable when running locally or on a cheap CPU instance:
  Streams the dataset row-by-row and pushes it to the bucket via a multipart
  upload using boto3.  Nothing is buffered to disk and RAM usage is ~5 MB
  (one S3 part buffer).  Requires the S3 environment variables above.

    ai-t9-data fetch-hf wikitext wikitext-103-raw-v1 train corpuses/wiki.txt
    ai-t9-data fetch-hf openwebtext plain train corpuses/owt.txt --max-lines 5000000

``fetch-hf-local`` — intended for Modal Functions with a CloudBucketMount:
  Streams dataset rows and writes them line-by-line to a **local path**.
  When the destination is a CloudBucketMount path (e.g. ``/data/corpuses/wiki.txt``)
  the write is sequential and compatible with Mountpoint semantics.  No boto3
  or environment variables are required — the modal container handles auth.

    # Inside a Modal function with volumes={"/data": modal.CloudBucketMount(...)}
    ai-t9-data fetch-hf-local wikitext wikitext-103-raw-v1 train /data/corpuses/wiki.txt

Conventions for bucket layout::

    corpuses/     plain-text corpus files  (*.txt, one utterance per line)
    vocab/        vocab.json artifacts
    pairs/        precomputed training pairs  (*.npz, from ai-t9-train --save-pairs)
    models/       trained model weights  (*.npz)
    ngrams/       bigram models  (*.json)

Install the required extra before using::

    pip install -e ".[data]"    # adds boto3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

_ENV_ENDPOINT   = "AI_T9_S3_ENDPOINT"
_ENV_BUCKET     = "AI_T9_S3_BUCKET"
_ENV_ACCESS_KEY = "AI_T9_S3_ACCESS_KEY"
_ENV_SECRET_KEY = "AI_T9_S3_SECRET_KEY"
_ENV_REGION     = "AI_T9_S3_REGION"


def _require_boto3():
    try:
        import boto3
        return boto3
    except ImportError:
        raise ImportError(
            "boto3 is required for bucket operations: pip install ai-t9[data]  "
            "or  pip install boto3"
        )


def _get_config() -> dict:
    """Read S3 connection settings from environment variables.

    All five variables must be present; raises SystemExit with a clear message
    if any are missing.
    """
    missing = [v for v in (_ENV_ENDPOINT, _ENV_BUCKET, _ENV_ACCESS_KEY, _ENV_SECRET_KEY)
               if not os.environ.get(v)]
    if missing:
        print(
            "ERROR: the following environment variables are not set:\n"
            + "\n".join(f"  {v}" for v in missing)
            + "\n\nSee 'ai-t9-data --help' for configuration instructions.",
            file=sys.stderr,
        )
        sys.exit(1)
    return {
        "endpoint_url":          os.environ[_ENV_ENDPOINT],
        "bucket":                os.environ[_ENV_BUCKET],
        "aws_access_key_id":     os.environ[_ENV_ACCESS_KEY],
        "aws_secret_access_key": os.environ[_ENV_SECRET_KEY],
        "region_name":           os.environ.get(_ENV_REGION, "auto"),
    }


def _make_client(boto3, cfg: dict):
    """Create a boto3 S3 client from a config dict."""
    return boto3.client(
        "s3",
        endpoint_url=cfg["endpoint_url"],
        aws_access_key_id=cfg["aws_access_key_id"],
        aws_secret_access_key=cfg["aws_secret_access_key"],
        region_name=cfg["region_name"],
    )


# ---------------------------------------------------------------------------
# Sub-command implementations
# ---------------------------------------------------------------------------

def cmd_ls(client, bucket: str, prefix: str = "") -> int:
    """List objects in the bucket, optionally filtered by prefix."""
    paginator = client.get_paginator("list_objects_v2")
    total = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            size_kb = obj["Size"] / 1024
            modified = obj["LastModified"].strftime("%Y-%m-%d %H:%M")
            print(f"{modified}  {size_kb:>10.1f} KB  {obj['Key']}")
            total += 1
    if total == 0:
        print(f"(no objects found under prefix '{prefix}')")
    return 0


def cmd_upload(client, bucket: str, local: str, remote: str, verbose: bool = True) -> int:
    """Upload a local file to the bucket with a progress bar."""
    local_path = Path(local)
    if not local_path.exists():
        print(f"ERROR: local file not found: {local_path}", file=sys.stderr)
        return 1
    size = local_path.stat().st_size
    if verbose:
        print(f"Uploading {local_path}  ({size / 1e6:.1f} MB)  →  {remote}")
    try:
        from tqdm import tqdm

        with tqdm(total=size, unit="B", unit_scale=True, unit_divisor=1024,
                  desc=remote, leave=True) as bar:
            client.upload_file(
                str(local_path), bucket, remote,
                Callback=lambda n: bar.update(n),
            )
    except ImportError:
        client.upload_file(str(local_path), bucket, remote)
    if verbose:
        print(f"  Done.")
    return 0


def cmd_download(client, bucket: str, remote: str, local: str, verbose: bool = True) -> int:
    """Download a bucket object to a local file with a progress bar."""
    try:
        meta = client.head_object(Bucket=bucket, Key=remote)
    except client.exceptions.ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("404", "NoSuchKey"):
            print(f"ERROR: object not found in bucket: {remote}", file=sys.stderr)
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    size = meta["ContentLength"]
    local_path = Path(local)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"Downloading {remote}  ({size / 1e6:.1f} MB)  →  {local_path}")
    try:
        from tqdm import tqdm

        with tqdm(total=size, unit="B", unit_scale=True, unit_divisor=1024,
                  desc=local_path.name, leave=True) as bar:
            client.download_file(
                bucket, remote, str(local_path),
                Callback=lambda n: bar.update(n),
            )
    except ImportError:
        client.download_file(bucket, remote, str(local_path))
    if verbose:
        print(f"  Done.")
    return 0


def cmd_rm(client, bucket: str, remote: str) -> int:
    """Delete an object from the bucket."""
    client.delete_object(Bucket=bucket, Key=remote)
    print(f"Deleted: {remote}")
    return 0


def cmd_stat(client, bucket: str, remote: str) -> int:
    """Print metadata for a single bucket object."""
    try:
        meta = client.head_object(Bucket=bucket, Key=remote)
    except client.exceptions.ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("404", "NoSuchKey"):
            print(f"ERROR: object not found: {remote}", file=sys.stderr)
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Key:           {remote}")
    print(f"Size:          {meta['ContentLength'] / 1e6:.2f} MB  ({meta['ContentLength']:,} bytes)")
    print(f"Last-Modified: {meta['LastModified']}")
    print(f"ETag:          {meta['ETag']}")
    print(f"Content-Type:  {meta.get('ContentType', 'unknown')}")
    return 0


def cmd_fetch_hf(
    client,
    bucket: str,
    dataset: str,
    config: str,
    split: str,
    remote: str,
    field: str = "text",
    max_lines: int | None = None,
    verbose: bool = True,
) -> int:
    """Stream a HuggingFace dataset and upload it as a plain-text corpus file.

    Streams the dataset row-by-row (no local temp file required) and writes
    it directly to the bucket via a multipart upload.  Each non-empty value
    from ``field`` is written as one line, lowercased and stripped.

    Args:
        dataset:    HuggingFace dataset name, e.g. ``"wikitext"``
        config:     Dataset config/subset, e.g. ``"wikitext-103-raw-v1"``
        split:      Dataset split, e.g. ``"train"``
        remote:     Destination key in the bucket, e.g. ``"corpuses/wiki.txt"``
        field:      Text field to extract from each example (default: ``"text"``)
        max_lines:  Stop after this many lines (default: no limit)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required: pip install datasets", file=sys.stderr)
        return 1

    # Enable hf_transfer for faster downloads
    import os
    os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')

    if verbose:
        limit_str = f", limit={max_lines:,}" if max_lines else ""
        print(f"Fetching HuggingFace dataset: {dataset}/{config} split={split}{limit_str}")
        print(f"  Streaming → {bucket}/{remote}")

    # Initiate S3 multipart upload
    mpu = client.create_multipart_upload(Bucket=bucket, Key=remote,
                                          ContentType="text/plain; charset=utf-8")
    upload_id = mpu["UploadId"]
    parts = []
    part_number = 1
    _MIN_PART = 5 * 1024 * 1024   # S3 minimum: 5 MB per part (except last)
    buf: list[bytes] = []
    buf_size = 0
    lines_written = 0

    def _flush(final: bool = False) -> None:
        nonlocal part_number, buf_size
        if not buf:
            return
        data = b"".join(buf)
        resp = client.upload_part(
            Bucket=bucket, Key=remote, UploadId=upload_id,
            PartNumber=part_number, Body=data,
        )
        parts.append({"PartNumber": part_number, "ETag": resp["ETag"]})
        part_number += 1
        buf.clear()
        buf_size = 0

    try:
        ds = load_dataset(dataset, config, split=split, streaming=True, trust_remote_code=False)
        try:
            from tqdm import tqdm
            ds = tqdm(ds, desc="rows", unit="row", leave=False)
        except ImportError:
            pass

        for example in ds:
            if max_lines is not None and lines_written >= max_lines:
                break
            value = example.get(field, "") or ""
            line = value.strip().lower()
            if not line:
                continue
            encoded = (line + "\n").encode("utf-8")
            buf.append(encoded)
            buf_size += len(encoded)
            lines_written += 1
            if buf_size >= _MIN_PART:
                _flush()

        _flush(final=True)

        # Complete the multipart upload
        client.complete_multipart_upload(
            Bucket=bucket, Key=remote, UploadId=upload_id,
            MultipartUpload={"Parts": parts},
        )
        if verbose:
            print(f"  Written {lines_written:,} lines → {remote}")
        return 0

    except Exception as exc:
        # Abort on any error to avoid leaving incomplete multipart uploads
        # (they accrue storage charges on most providers).
        client.abort_multipart_upload(Bucket=bucket, Key=remote, UploadId=upload_id)
        print(f"ERROR: upload aborted: {exc}", file=sys.stderr)
        return 1

def cmd_fetch_hf_local(
    dataset: str,
    config: str,
    split: str,
    local_path: str,
    field: str = "text",
    max_lines: int | None = None,
    verbose: bool = True,
) -> int:
    """Stream a HuggingFace dataset and write it as a plain-text corpus file
    to a local or mounted file path.

    Intended for use inside a Modal Function that has the target bucket mounted
    via ``modal.CloudBucketMount``.  Each non-empty value from ``field`` is
    written as one line (lowercased, stripped) using a plain sequential
    ``open()`` + ``write()`` — fully compatible with Mountpoint's write
    semantics (no append-mode, no seeks, new file only).

    RAM usage is O(1) — rows are written immediately and not buffered.
    No boto3 or S3 credentials are required.

    Example Modal Function usage::

        import modal
        app = modal.App()
        r2 = modal.CloudBucketMount(
            "my-bucket",
            bucket_endpoint_url="https://<account>.r2.cloudflarestorage.com",
            secret=modal.Secret.from_name("r2-credentials"),
        )

        @app.function(volumes={"/data": r2}, timeout=3600 * 12)
        def ingest():
            import subprocess
            subprocess.run([
                "ai-t9-data", "fetch-hf-local",
                "wikitext", "wikitext-103-raw-v1", "train",
                "/data/corpuses/wikitext.txt",
            ], check=True)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required: pip install datasets", file=sys.stderr)
        return 1

    # Enable hf_transfer for faster downloads
    import os
    os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')

    local = Path(local_path)
    local.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        limit_str = f", limit={max_lines:,}" if max_lines else ""
        print(f"Fetching HuggingFace dataset: {dataset}/{config} split={split}{limit_str}")
        print(f"  Writing \u2192 {local}")

    lines_written = 0
    try:
        ds = load_dataset(dataset, config, split=split, streaming=True, trust_remote_code=False)
        try:
            from tqdm import tqdm
            ds = tqdm(ds, desc="rows", unit="row", leave=False)
        except ImportError:
            pass

        # Open in write (truncate) mode — append mode is not supported by
        # Mountpoint / CloudBucketMount.  The file must not already exist on
        # the mount; create it fresh each run.
        with open(local, "w", encoding="utf-8") as f:
            for example in ds:
                if max_lines is not None and lines_written >= max_lines:
                    break
                value = example.get(field, "") or ""
                line = value.strip().lower()
                if not line:
                    continue
                f.write(line + "\n")
                lines_written += 1

        if verbose:
            print(f"  Written {lines_written:,} lines \u2192 {local}")
        return 0

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-t9-data",
        description="Manage ai-t9 training data in an S3-compatible bucket.",
        epilog=(
            "Configuration is read from environment variables:\n"
            f"  {_ENV_ENDPOINT}    bucket endpoint URL\n"
            f"  {_ENV_BUCKET}      bucket name\n"
            f"  {_ENV_ACCESS_KEY}  access key\n"
            f"  {_ENV_SECRET_KEY}  secret key\n"
            f"  {_ENV_REGION}      region (default: auto)\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ls
    p_ls = sub.add_parser("ls", help="List bucket objects")
    p_ls.add_argument("prefix", nargs="?", default="", help="Key prefix filter")

    # upload
    p_up = sub.add_parser("upload", help="Upload a local file to the bucket")
    p_up.add_argument("local",  help="Local file path")
    p_up.add_argument("remote", help="Destination key in the bucket")

    # download
    p_dl = sub.add_parser("download", help="Download a bucket object to a local file")
    p_dl.add_argument("remote", help="Source key in the bucket")
    p_dl.add_argument("local",  help="Local destination path")

    # rm
    p_rm = sub.add_parser("rm", help="Delete an object from the bucket")
    p_rm.add_argument("remote", help="Key to delete")

    # stat
    p_stat = sub.add_parser("stat", help="Show metadata for a bucket object")
    p_stat.add_argument("remote", help="Key to inspect")

    # fetch-hf
    p_hf = sub.add_parser(
        "fetch-hf",
        help="Stream a HuggingFace dataset and upload it as a corpus file",
    )
    p_hf.add_argument("dataset", help="HuggingFace dataset name, e.g. 'wikitext'")
    p_hf.add_argument("config",  help="Dataset config/subset, e.g. 'wikitext-103-raw-v1'")
    p_hf.add_argument("split",   help="Dataset split, e.g. 'train'")
    p_hf.add_argument("remote",  help="Destination key in bucket, e.g. 'corpuses/wiki.txt'")
    p_hf.add_argument(
        "--field", default="text",
        help="Name of the text field in each dataset example (default: text)",
    )
    p_hf.add_argument(
        "--max-lines", type=int, default=None,
        help="Stop after this many lines (default: no limit)",
    )

    # fetch-hf-local
    p_hfl = sub.add_parser(
        "fetch-hf-local",
        help="Stream a HuggingFace dataset and write it to a local/mounted file path "
             "(use inside Modal Functions with CloudBucketMount; no boto3 required)",
    )
    p_hfl.add_argument("dataset",    help="HuggingFace dataset name, e.g. 'wikitext'")
    p_hfl.add_argument("config",     help="Dataset config/subset, e.g. 'wikitext-103-raw-v1'")
    p_hfl.add_argument("split",      help="Dataset split, e.g. 'train'")
    p_hfl.add_argument("local_path", help="Destination file path (local or CloudBucketMount)")
    p_hfl.add_argument(
        "--field", default="text",
        help="Name of the text field in each dataset example (default: text)",
    )
    p_hfl.add_argument(
        "--max-lines", type=int, default=None,
        help="Stop after this many lines (default: no limit)",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # fetch-hf-local needs no S3 credentials or boto3 — handle before config.
    if args.command == "fetch-hf-local":
        return cmd_fetch_hf_local(
            dataset=args.dataset,
            config=args.config,
            split=args.split,
            local_path=args.local_path,
            field=args.field,
            max_lines=args.max_lines,
        )

    boto3 = _require_boto3()
    cfg = _get_config()
    client = _make_client(boto3, cfg)
    bucket = cfg["bucket"]

    if args.command == "ls":
        return cmd_ls(client, bucket, args.prefix)
    if args.command == "upload":
        return cmd_upload(client, bucket, args.local, args.remote)
    if args.command == "download":
        return cmd_download(client, bucket, args.remote, args.local)
    if args.command == "rm":
        return cmd_rm(client, bucket, args.remote)
    if args.command == "stat":
        return cmd_stat(client, bucket, args.remote)
    if args.command == "fetch-hf":
        return cmd_fetch_hf(
            client, bucket,
            dataset=args.dataset,
            config=args.config,
            split=args.split,
            remote=args.remote,
            field=args.field,
            max_lines=args.max_lines,
        )
    print(f"Unknown command: {args.command}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
