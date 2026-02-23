"""Modal App: Run ai-t9 training on cloud GPUs.

This script wraps the existing ai-t9 CLIs (``ai-t9-build-vocab``,
``ai-t9-train``, ``ai-t9-data``) so training can run on Modal's serverless
GPU infrastructure while keeping the local workflow unchanged.

Architecture overview::

    ┌────────────────────────────────────────────────────────────────┐
    │  R2 bucket (CloudBucketMount, read-only on GPU containers)    │
    │    corpuses/wiki.txt   corpuses/brown.txt   ...               │
    └────────────────────────┬───────────────────────────────────────┘
                             │ mounted at /corpus
    ┌────────────────────────▼───────────────────────────────────────┐
    │  prep  (CPU, long timeout)                                    │
    │    • ai-t9-build-vocab --corpus /corpus/... --output /vol/    │
    │    • ai-t9-train --save-pairs /vol/pairs/pairs                │
    │          --shard-size 10000000 --pairs-only                   │
    └────────────────────────┬───────────────────────────────────────┘
                             │ pairs/pairs_*.npz + vocab.json on Volume
    ┌────────────────────────▼───────────────────────────────────────┐
    │  train  (GPU, modal.Retries for preemption)                   │
    │    • ai-t9-train --pairs-dir /vol/pairs/ --output /vol/       │
    │         --accumulate-grad-batches 4                           │
    │    • optionally trains bigram model                           │
    └────────────────────────┬───────────────────────────────────────┘
                             │ model.npz + bigram.npz on Volume
    ┌────────────────────────▼───────────────────────────────────────┐
    │  download  (local entrypoint)                                 │
    │    • copies artifacts from Volume to local data/              │
    └───────────────────────────────────────────────────────────────┘

Prerequisites:

    pip install modal
    modal setup

    # Create a Modal secret named "r2-credentials" with keys:
    #   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
    # (only needed if using R2 corpus storage; not needed for Volume-only flow)

    # Create a Modal volume:
    modal volume create ai-t9-data

Usage::

    # Full pipeline: prep data on CPU → train on GPU → download artifacts
    modal run modal_app.py --prep --train

    # Train only (pairs already precomputed on the Volume)
    modal run modal_app.py --train

    # Prep only (precompute pairs, skip training)
    modal run modal_app.py --prep

    # Override training hyperparameters
    modal run modal_app.py --train --epochs 10 --embed-dim 128 --batch-size 8192 --gpu A100

    # Use a different corpus prefix in the R2 bucket
    modal run modal_app.py --prep --corpus-prefix corpuses/

    # Download artifacts from the Volume without running anything
    modal run modal_app.py --download

    # Long-running training in the background (survives terminal disconnect)
    modal run --detach modal_app.py --train --epochs 20 --gpu H100

All flags::

    modal run modal_app.py --help
"""

from __future__ import annotations

import sys

import modal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_NAME = "ai-t9-train"
VOLUME_NAME = "ai-t9-data"
VOLUME_PATH = "/vol"
CORPUS_MOUNT_PATH = "/corpus"
R2_SECRET_NAME = "r2-credentials"

# Default training hyperparameters (mirror ai-t9-train defaults but sized up
# for cloud GPU runs).
DEFAULT_EPOCHS = 3
DEFAULT_EMBED_DIM = 64
DEFAULT_CONTEXT_WINDOW = 3
DEFAULT_LR = 0.001
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_WARMUP_FRAC = 0.05
DEFAULT_TEMPERATURE = 0.07
DEFAULT_BATCH_SIZE = 8192
DEFAULT_ACCUMULATE = 4          # effective batch = 8192 × 4 = 32768 on an A100
DEFAULT_CLIP_GRAD_NORM = 1.0
DEFAULT_SEED = 42
DEFAULT_SHARD_SIZE = 10_000_000  # 10M pairs per shard

PAIRS_DIR = f"{VOLUME_PATH}/pairs"

# ---------------------------------------------------------------------------
# Modal resources
# ---------------------------------------------------------------------------

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Base image: bakes in pip dependencies and entry-point scripts.
# Only rebuilt when the pip package list or pyproject.toml changes — NOT on
# every source edit.  Source changes are picked up at invocation time via
# _source_mount below.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.24",
        "nltk>=3.8",
        "datasets>=2.0",
        "hf-transfer>=0.1.8",
        "torch>=2.0",
        "tqdm>=4.0",
    )
    .add_local_dir("src", "/root/src", copy=True)
    .add_local_file("pyproject.toml", "/root/pyproject.toml", copy=True)
    .add_local_file("README.md", "/root/README.md", copy=True)
    .run_commands("pip install --no-deps -e /root")
    .add_local_python_source("ai_t9")
)

app = modal.App(APP_NAME, image=image)


# ---------------------------------------------------------------------------
# Helper: optional R2 CloudBucketMount
# ---------------------------------------------------------------------------

def _r2_mount(bucket_name: str, endpoint_url: str) -> modal.CloudBucketMount:
    """Build a read-only CloudBucketMount for Cloudflare R2.

    Requires a Modal secret named ``r2-credentials`` with keys
    ``AWS_ACCESS_KEY_ID`` and ``AWS_SECRET_ACCESS_KEY``.
    """
    return modal.CloudBucketMount(
        bucket_name,
        bucket_endpoint_url=endpoint_url,
        secret=modal.Secret.from_name(R2_SECRET_NAME),
        read_only=True,
    )


# ---------------------------------------------------------------------------
# Prep function (CPU — build vocab + precompute sharded pairs)
# ---------------------------------------------------------------------------

@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=3600 * 6,       # 6 hours (large corpus ingestion can be slow)
    memory=8192,            # 8 GB RAM for large vocabs
)
def prep(
    corpus_path: str = f"{CORPUS_MOUNT_PATH}/corpuses",
    dictionary_path: str | None = None,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    max_words: int = 500_000,
    min_count: int = 20,
    shard_size: int = DEFAULT_SHARD_SIZE,
    use_volume_corpus: bool = False,
) -> str:
    """Build vocabulary, dictionary, and precompute sharded training pairs.

    When ``use_volume_corpus`` is True, reads corpus files from the Volume
    (``/vol/corpuses/``) instead of the R2 mount. Useful when you've already
    uploaded corpus files to the Volume via ``modal volume put``.

    Pairs are written as shards to ``/vol/pairs/pairs_000.npz``, etc. so the
    training function can stream them one shard at a time, keeping GPU VRAM
    usage independent of corpus size.

    Returns a summary string.
    """
    import subprocess
    from pathlib import Path

    actual_corpus = f"{VOLUME_PATH}/corpuses" if use_volume_corpus else corpus_path

    # -- Build vocab + dict ------------------------------------------------
    vocab_cmd = [
        "ai-t9-build-vocab",
        "--corpus", actual_corpus,
        "--output", VOLUME_PATH,
        "--max-words", str(max_words),
        "--min-count", str(min_count),
    ]
    if dictionary_path:
        vocab_cmd.extend(["--dictionary", dictionary_path])

    print(f"=== Building vocabulary from {actual_corpus} ===")
    subprocess.run(vocab_cmd, check=True)

    # -- Precompute sharded pairs ------------------------------------------
    pairs_prefix = f"{PAIRS_DIR}/pairs"
    pairs_cmd = [
        "ai-t9-train",
        "--vocab", f"{VOLUME_PATH}/vocab.json",
        "--corpus", actual_corpus,
        "--save-pairs", pairs_prefix,
        "--shard-size", str(shard_size),
        "--pairs-only",
        "--context-window", str(context_window),
        "--output", "/dev/null",   # not used with --pairs-only
    ]
    print(f"\n=== Precomputing training pairs (shard size: {shard_size:,}) ===")
    subprocess.run(pairs_cmd, check=True)

    volume.commit()

    # Summarise what was written.
    artifacts = []
    for name in ("vocab.json", "dict.json"):
        p = Path(VOLUME_PATH) / name
        if p.exists():
            size_mb = p.stat().st_size / 1e6
            artifacts.append(f"  {name}: {size_mb:.1f} MB")
    pairs_dir = Path(PAIRS_DIR)
    if pairs_dir.is_dir():
        shards = sorted(pairs_dir.glob("pairs_*.npz"))
        total_mb = sum(p.stat().st_size for p in shards) / 1e6
        artifacts.append(f"  pairs/: {len(shards)} shard(s), {total_mb:.1f} MB total")
    summary = "Prep complete. Volume contents:\n" + "\n".join(artifacts)
    print(f"\n{summary}")
    return summary


# ---------------------------------------------------------------------------
# Train class (GPU — uses @app.cls so GPU type can be overridden at runtime
# via Trainer.with_options(gpu="A100"))
# ---------------------------------------------------------------------------

@app.cls(
    volumes={VOLUME_PATH: volume},
    gpu="L4",               # default; overridden via with_options(gpu=...)
    timeout=3600 * 3,       # 3 hours
    retries=modal.Retries(max_retries=3, initial_delay=0.0),
)
class Trainer:
    @modal.method()
    def run(
        self,
        epochs: int = DEFAULT_EPOCHS,
        embed_dim: int = DEFAULT_EMBED_DIM,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        lr: float = DEFAULT_LR,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        warmup_frac: float = DEFAULT_WARMUP_FRAC,
        temperature: float = DEFAULT_TEMPERATURE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        accumulate_grad_batches: int = DEFAULT_ACCUMULATE,
        clip_grad_norm: float = DEFAULT_CLIP_GRAD_NORM,
        seed: int = DEFAULT_SEED,
        save_ngram: bool = True,
    ) -> str:
        """Train the DualEncoder on GPU from sharded precomputed pairs.

        Reads ``vocab.json`` and ``pairs/pairs_*.npz`` from the Volume, writes
        ``model.npz`` back. Does not generate the bigram model.

        Returns a summary string.
        """
        import subprocess
        from pathlib import Path

        # Reload volume to pick up any recent prep output
        volume.reload()

        effective_batch = batch_size * accumulate_grad_batches
        print(f"Effective batch size: {effective_batch:,} ({batch_size:,} × {accumulate_grad_batches} accumulation steps)")

        # Check if bigram training is possible (requires corpus files)
        corpus_dir = Path(VOLUME_PATH) / "corpuses"
        has_corpus = corpus_dir.is_dir() and any(corpus_dir.glob("*.txt"))
        if save_ngram and not has_corpus:
            print("Note: No corpus files on Volume; skipping bigram model.")
            save_ngram = False

        # Resolve pairs input: prefer sharded directory, fall back to single file.
        pairs_dir = Path(PAIRS_DIR)
        has_shards = pairs_dir.is_dir() and any(pairs_dir.glob("pairs_*.npz"))
        single_pairs = Path(VOLUME_PATH) / "pairs.npz"

        if has_shards:
            pairs_args = ["--pairs-dir", str(pairs_dir)]
        elif single_pairs.exists():
            print(f"Note: No sharded pairs found in {PAIRS_DIR}; using single file {single_pairs}.")
            pairs_args = ["--load-pairs", str(single_pairs)]
        else:
            raise FileNotFoundError(
                f"No training pairs found. Run prep first (expects {PAIRS_DIR}/pairs_*.npz "
                f"or {single_pairs})."
            )

        # Step 1: Train the embedding model.
        train_cmd = [
            "ai-t9-train",
            "--vocab", f"{VOLUME_PATH}/vocab.json",
            *pairs_args,
            "--output", f"{VOLUME_PATH}/model.npz",
            "--epochs", str(epochs),
            "--embed-dim", str(embed_dim),
            "--context-window", str(context_window),
            "--lr", str(lr),
            "--weight-decay", str(weight_decay),
            "--warmup-frac", str(warmup_frac),
            "--temperature", str(temperature),
            "--batch-size", str(batch_size),
            "--accumulate-grad-batches", str(accumulate_grad_batches),
            "--clip-grad-norm", str(clip_grad_norm),
            "--seed", str(seed),
            "--device", "cuda",
            "--debug",
        ]
        print("=== Training Model ===")
        subprocess.run(train_cmd, check=True)
        volume.commit()

        artifacts: list[str] = []
        for name in ("model.npz", "bigram.npz", "vocab.json", "dict.json"):
            p = Path(VOLUME_PATH) / name
            if p.exists():
                size_mb = p.stat().st_size / 1e6
                artifacts.append(f"  {name}: {size_mb:.1f} MB")
        summary = "Training complete. Volume contents:\n" + "\n".join(artifacts)
        print(f"\n{summary}")
        return summary


# ---------------------------------------------------------------------------
# Ingest function (CPU — fetch HuggingFace dataset to Volume)
# ---------------------------------------------------------------------------

@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=3600 * 12,       # 12 hours for very large datasets
    memory=4096,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def ingest_hf(
    dataset: str = "wikitext",
    config: str = "wikitext-103-raw-v1",
    split: str = "train",
    filename: str = "wikitext.txt",
    field: str = "text",
    max_lines: int | None = None,
) -> str:
    """Stream a HuggingFace dataset and write it to the Volume as a corpus file.

    The file is written to ``/vol/corpuses/<filename>``.
    Uses the ``fetch-hf-local`` subcommand internally so no boto3 is needed.
    """
    import subprocess

    dest = f"{VOLUME_PATH}/corpuses/{filename}"

    cmd = [
        "ai-t9-data", "fetch-hf-local",
        dataset, config, split, dest,
        "--field", field,
    ]
    if max_lines is not None:
        cmd.extend(["--max-lines", str(max_lines)])

    print(f"=== Ingesting {dataset}/{config} split={split} → {dest} ===")
    subprocess.run(cmd, check=True)

    volume.commit()

    from pathlib import Path
    p = Path(dest)
    size_mb = p.stat().st_size / 1e6 if p.exists() else 0
    summary = f"Ingested {dataset}/{config} → {filename} ({size_mb:.1f} MB)"
    print(summary)
    return summary


# ---------------------------------------------------------------------------
# Upload corpus files to Volume from local machine
# ---------------------------------------------------------------------------

@app.function(
    volumes={VOLUME_PATH: volume},
    timeout=600,
)
def list_volume() -> str:
    """List all files on the Volume (for debugging)."""
    import os

    volume.reload()
    lines = []
    for root, dirs, files in os.walk(VOLUME_PATH):
        for f in sorted(files):
            fp = os.path.join(root, f)
            size_mb = os.path.getsize(fp) / 1e6
            rel = os.path.relpath(fp, VOLUME_PATH)
            lines.append(f"  {rel}: {size_mb:.1f} MB")
    if not lines:
        return "(Volume is empty)"
    return "Volume contents:\n" + "\n".join(sorted(lines))


# ---------------------------------------------------------------------------
# Local entrypoint (orchestrator)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    # Workflow control
    prep: bool = False,
    train: bool = False,
    download: bool = False,
    list_files: bool = False,
    # Ingest HuggingFace dataset
    ingest: str = "",
    ingest_config: str = "",
    ingest_split: str = "train",
    ingest_filename: str = "",
    ingest_field: str = "text",
    ingest_max_lines: int = 0,
    # Prep options
    use_volume_corpus: bool = False,
    max_words: int = 500_000,
    min_count: int = 20,
    shard_size: int = DEFAULT_SHARD_SIZE,
    # Training hyperparameters
    epochs: int = DEFAULT_EPOCHS,
    embed_dim: int = DEFAULT_EMBED_DIM,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    warmup_frac: float = DEFAULT_WARMUP_FRAC,
    temperature: float = DEFAULT_TEMPERATURE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    accumulate_grad_batches: int = DEFAULT_ACCUMULATE,
    clip_grad_norm: float = DEFAULT_CLIP_GRAD_NORM,
    seed: int = DEFAULT_SEED,
    no_ngram: bool = False,
    # GPU selection
    gpu: str = "L4",
    # Download destination
    output_dir: str = "modal_data",
):
    """Orchestrate ai-t9 training on Modal.

    Pass a flag to enable a step in the workflow:

      --ingest      Run ingest; add dataset to Volume.
      --prep        Run prep; build vocab/dict/pairs from corpus on Volume.
      --train       Run train; train model from precomputed pairs, then download.
      --download    Download artifacts from the Volume.

    Other flags:

      --list-files       List files on the Volume and exit.

    Ingest a HuggingFace dataset to the Volume:

      --ingest wikitext --ingest-config wikitext-103-raw-v1

    Training flags are forwarded to ``ai-t9-train``:

      --epochs, --embed-dim, --context-window, --lr, --weight-decay,
      --warmup-frac, --temperature, --batch-size, --accumulate-grad-batches,
      --clip-grad-norm, --seed, --no-ngram

    GPU selection:

      --gpu L4           Cheap ($0.60/hr), good for iteration, but slow for training
      --gpu A10          Midrange ($1.10/hr), good for larger models, faster than L4
      --gpu A100         Fast, 40 GB VRAM
      --gpu A100-80GB    Large vocab / big embed_dim
      --gpu H100         Fastest available
    """
    from pathlib import Path
    import subprocess

    # -- List files --------------------------------------------------------
    if list_files:
        print(list_volume.remote())
        return

    # If we are missing arguments for any workflow, print an error and exit.
    if not (ingest or prep or train or download):
        print("ERROR: No workflow specified. Use --ingest, --prep, --train, or --download.", file=sys.stderr)
        raise SystemExit(1)

    # -- Separate workflows ------------------------------------------------
    if ingest:
        if not ingest_config:
            print("ERROR: --ingest-config CONFIG required when using --ingest")
            raise SystemExit(1)
        result = ingest_hf.remote(
            dataset=ingest,
            config=ingest_config,
            split=ingest_split,
            filename=ingest_filename or f"{ingest.replace('/', '_')}.txt",
            field=ingest_field,
            max_lines=ingest_max_lines if ingest_max_lines > 0 else None,
        )
        print(result)
        return

    if prep:
        result = prep.remote(
            context_window=context_window,
            max_words=max_words,
            min_count=min_count,
            shard_size=shard_size,
            use_volume_corpus=use_volume_corpus,
        )
        print(result)
        return

    if train:
        # Override the GPU at invocation time via with_options on the Cls
        trainer = Trainer.with_options(gpu=gpu)()
        result = trainer.run.remote(
            epochs=epochs,
            embed_dim=embed_dim,
            context_window=context_window,
            lr=lr,
            weight_decay=weight_decay,
            warmup_frac=warmup_frac,
            temperature=temperature,
            batch_size=batch_size,
            accumulate_grad_batches=accumulate_grad_batches,
            clip_grad_norm=clip_grad_norm,
            seed=seed,
            save_ngram=not no_ngram,
        )
        print(result)
        print("Done training.")
        return

    if download:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        artifact_names = ["vocab.json", "dict.json", "model.npz", "bigram.npz"]
        print(f"\nDownloading artifacts to {out}/")
        for name in artifact_names:
            remote_path = name
            local_path = out / name
            try:
                subprocess.run(
                    ["modal", "volume", "get", VOLUME_NAME, remote_path, str(local_path), "--force"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                size_mb = local_path.stat().st_size / 1e6
                print(f"  {name}: {size_mb:.1f} MB")
            except subprocess.CalledProcessError as e:
                print(f"  {name}: (not found on volume, skipping)")
                print(f"    Error details: {e.stderr.strip()}")
                print(f"    StdOut: {e.stdout.strip()}")
        print("Done downloading artifacts.")
        return

    print("Done running workflow.")
