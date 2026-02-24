"""CLI: Run a complete ai-t9 training pipeline from a YAML config.

This is the primary entry point for automated training runs.  It reads a
single YAML config file that declares datasets, hyperparameters, and
pipeline steps — then executes everything end-to-end.

Usage::

    # Run full pipeline
    ai-t9-run configs/default.yaml

    # Run only specific steps
    ai-t9-run configs/default.yaml --step corpus --step vocab

    # Skip specific steps (e.g. reuse existing corpus)
    ai-t9-run configs/default.yaml --skip corpus

Pipeline steps (in order):

    corpus   Fetch/combine all dataset sources into a single corpus file.
    vocab    Build vocabulary and T9 dictionary from the corpus.
    pairs    Precompute (context, target) training pairs.
    train    Train the dual-encoder model from pairs.
    ngram    Train a bigram language model from the corpus.

S3 integration:

    When ``s3.upload: true`` is set in the config, artifacts are uploaded
    after each step.  When a required file is missing locally and S3 is
    configured, the runner auto-downloads it — so you can skip earlier
    steps on GPU instances that only need to run ``train``.

See ``configs/default.yaml`` for a documented example configuration.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path


def _log(msg: str) -> None:
    """Print a timestamped log line."""
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------


def _s3_client(s3_cfg):
    """Create a boto3 S3 client from the config's S3 settings."""
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=s3_cfg.endpoint,
        aws_access_key_id=s3_cfg.access_key,
        aws_secret_access_key=s3_cfg.secret_key,
        region_name=s3_cfg.region,
        # SigV4 must be explicit for custom S3-compatible endpoints (e.g.
        # Cloudflare R2). Without it, botocore's signing heuristics for
        # non-AWS endpoints can produce requests that R2 rejects, manifesting
        # as a confusing "NoSuchBucket" rather than an auth error.
        config=Config(signature_version="s3v4"),
    )


def _s3_upload(s3_cfg, local_path: Path, remote_key: str) -> None:
    """Upload a single file to the S3 bucket."""
    client = _s3_client(s3_cfg)
    client.upload_file(str(local_path), s3_cfg.bucket, remote_key)
    _log(f"  ↑ s3://{s3_cfg.bucket}/{remote_key}")


def _s3_download(s3_cfg, remote_key: str, local_path: Path) -> None:
    """Download a single file from the S3 bucket."""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    client = _s3_client(s3_cfg)
    client.download_file(s3_cfg.bucket, remote_key, str(local_path))
    _log(f"  ↓ s3://{s3_cfg.bucket}/{remote_key} → {local_path}")


def _s3_sync_from(s3_cfg, remote_prefix: str, local_dir: Path) -> int:
    """Download all objects under a prefix to a local directory.

    Returns the number of files downloaded.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    client = _s3_client(s3_cfg)
    paginator = client.get_paginator("list_objects_v2")
    count = 0
    for page in paginator.paginate(Bucket=s3_cfg.bucket, Prefix=remote_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            # Compute relative path from prefix
            rel = key[len(remote_prefix) :].lstrip("/")
            if not rel:
                # The prefix itself is a file (not a directory prefix)
                rel = Path(key).name
            dest = local_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(s3_cfg.bucket, key, str(dest))
            count += 1
            _log(f"  ↓ {key} → {dest}")
    return count


def _ensure_file(
    local: Path, s3_cfg, remote_key: str, description: str
) -> bool:
    """Ensure a local file exists, downloading from S3 if necessary.

    Returns True if the file exists (or was successfully downloaded).
    """
    if local.exists():
        return True
    if s3_cfg.enabled:
        _log(f"  {description} not found locally — downloading from S3…")
        try:
            _s3_download(s3_cfg, remote_key, local)
            return local.exists()
        except Exception as exc:
            _log(f"  WARNING: S3 download failed: {exc}")
    return False


class _S3CheckpointManager:
    """Atomic two-slot S3 checkpoint protocol.

    Layout in the bucket::

        {prefix}/ckpt_0.pt    — slot 0 checkpoint file
        {prefix}/ckpt_1.pt    — slot 1 checkpoint file
        {prefix}/ptr.json     — pointer: {"slot": 0|1, "epoch": N, "shard": N, "step": N}

    Save sequence (guarantees the last *fully committed* checkpoint is always
    recoverable, even if the instance is interrupted mid-save):

        1. Upload new checkpoint to the *inactive* slot.
        2. Write + upload ``ptr.json`` pointing to that slot  ← the "commit".

    If the instance is interrupted between steps 1 and 2, the pointer still
    points to the *previous* valid slot, so ``download_latest`` returns that.
    If interrupted during step 1, the pointer still points to the safe slot.
    """

    def __init__(self, s3_cfg, remote_prefix: str) -> None:
        self._s3_cfg = s3_cfg
        self._prefix = remote_prefix.rstrip("/")
        self._active_slot: int = 0  # slot currently referenced by ptr.json

    def _ptr_key(self) -> str:
        return f"{self._prefix}/ptr.json"

    def _slot_key(self, slot: int) -> str:
        return f"{self._prefix}/ckpt_{slot}.pt"

    def download_latest(self, local_dir: Path) -> "Path | None":
        """Download the latest committed checkpoint.

        Returns the local path of the checkpoint file, or ``None`` if no
        committed checkpoint exists in S3.
        """
        import json

        local_dir.mkdir(parents=True, exist_ok=True)
        ptr_local = local_dir / "ptr.json"

        try:
            _s3_download(self._s3_cfg, self._ptr_key(), ptr_local)
        except Exception:
            return None  # No checkpoint in S3 yet

        try:
            with open(ptr_local) as f:
                ptr = json.load(f)
            slot = int(ptr["slot"])
        except Exception:
            return None

        ckpt_local = local_dir / f"ckpt_{slot}.pt"
        try:
            _s3_download(self._s3_cfg, self._slot_key(slot), ckpt_local)
        except Exception:
            return None

        self._active_slot = slot
        return ckpt_local

    def upload(self, local_path: Path, epoch: int, shard: int, step: int) -> None:
        """Upload *local_path* to the inactive slot then atomically commit."""
        import json

        inactive = 1 - self._active_slot

        # Step 1: upload checkpoint to the inactive slot.
        _s3_upload(self._s3_cfg, local_path, self._slot_key(inactive))

        # Step 2: write + upload pointer (the "commit").
        ptr = {"slot": inactive, "epoch": epoch, "shard": shard, "step": step}
        ptr_local = local_path.parent / "ptr.json"
        ptr_local.write_text(__import__("json").dumps(ptr))
        _s3_upload(self._s3_cfg, ptr_local, self._ptr_key())

        self._active_slot = inactive


def _ensure_pairs_dir(
    local_dir: Path, s3_cfg, remote_prefix: str
) -> bool:
    """Ensure the pairs directory has .npz files, syncing from S3 if needed."""
    local_dir.mkdir(parents=True, exist_ok=True)
    if list(local_dir.glob("pairs_*.npz")) or list(local_dir.glob("pairs.npz")):
        return True
    if s3_cfg.enabled:
        _log("  Pairs not found locally — downloading from S3…")
        try:
            n = _s3_sync_from(s3_cfg, remote_prefix, local_dir)
            return n > 0
        except Exception as exc:
            _log(f"  WARNING: S3 sync failed: {exc}")
    return False


# ---------------------------------------------------------------------------
# Corpus ingestion helpers
# ---------------------------------------------------------------------------


def _ingest_local(src, out_file) -> int:
    """Ingest a local file or directory of *.txt files."""
    path = Path(src.path)
    if path.is_dir():
        files = sorted(path.glob("*.txt"))
    elif path.is_file():
        files = [path]
    else:
        _log(f"    WARNING: path not found: {path}")
        return 0

    lines = 0
    for f in files:
        with f.open(encoding="utf-8", errors="ignore") as inp:
            for line in inp:
                stripped = line.strip().lower()
                if stripped:
                    out_file.write(stripped + "\n")
                    lines += 1
    return lines


def _ingest_hf(src, out_file) -> int:
    """Stream a HuggingFace dataset and write lines to the corpus file."""
    import os

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from datasets import load_dataset

    ds = load_dataset(
        src.name,
        src.config,
        split=src.split,
        streaming=True,
        trust_remote_code=False,
    )

    lines = 0
    try:
        from tqdm import tqdm

        ds = tqdm(ds, desc=f"    {src.name}", unit="row", leave=False)
    except ImportError:
        pass

    for example in ds:
        if src.max_lines and lines >= src.max_lines:
            break
        value = example.get(src.column, "") or ""
        text = value.strip().lower()
        if text:
            out_file.write(text + "\n")
            lines += 1

    return lines


def _ingest_s3(src, s3_cfg, workdir: Path, out_file) -> int:
    """Download a file from S3 and write its contents to the corpus."""
    if not s3_cfg.enabled:
        _log("    WARNING: S3 not configured, skipping S3 dataset")
        return 0

    tmp = workdir / f"_s3_tmp_{Path(src.key).name}"
    try:
        _s3_download(s3_cfg, src.key, tmp)
    except Exception as exc:
        _log(f"    WARNING: S3 download failed for {src.key}: {exc}")
        return 0

    lines = 0
    with open(tmp, encoding="utf-8", errors="ignore") as inp:
        for line in inp:
            stripped = line.strip().lower()
            if stripped:
                out_file.write(stripped + "\n")
                lines += 1

    tmp.unlink(missing_ok=True)
    return lines


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def step_corpus(cfg, workdir: Path) -> Path:
    """Combine all dataset sources into a single corpus file.

    Returns the path to the combined corpus.
    """
    corpus_path = workdir / "corpus.txt"
    _log(f"=== Step: corpus → {corpus_path} ===")

    sources = cfg.datasets
    if not sources:
        if corpus_path.exists():
            _log("  No datasets configured; reusing existing corpus.txt")
            return corpus_path
        _log("ERROR: No datasets configured and no existing corpus found")
        sys.exit(1)

    lines_total = 0
    with open(corpus_path, "w", encoding="utf-8") as out:
        for i, src in enumerate(sources):
            label = src.path or src.name or src.key or "?"
            _log(f"  [{i + 1}/{len(sources)}] {src.type}: {label}")

            if src.type == "local":
                n = _ingest_local(src, out)
            elif src.type == "huggingface":
                n = _ingest_hf(src, out)
            elif src.type == "s3":
                n = _ingest_s3(src, cfg.s3, workdir, out)
            else:
                _log(f"    WARNING: unknown dataset type '{src.type}', skipping")
                continue

            lines_total += n
            _log(f"    {n:,} lines  (total: {lines_total:,})")

    size_mb = corpus_path.stat().st_size / 1e6
    _log(f"  Combined corpus: {lines_total:,} lines, {size_mb:.1f} MB → {corpus_path}")

    if cfg.s3.upload and cfg.s3.enabled:
        remote = cfg.s3.paths.corpus.rstrip("/") + "/corpus.txt"
        _s3_upload(cfg.s3, corpus_path, remote)

    return corpus_path


def step_vocab(cfg, workdir: Path, corpus_path: Path) -> None:
    """Build vocabulary and T9 dictionary from the corpus."""
    _log("=== Step: vocab ===")

    from ai_t9._scripts.build_vocab import _count_words_from_file
    from ai_t9.dictionary import T9Dictionary, load_wordlist
    from ai_t9.model.vocab import Vocabulary

    counter: Counter = Counter()
    _count_words_from_file(corpus_path, counter, verbose=True)
    _log(
        f"  {len(counter):,} unique words, {sum(counter.values()):,} total tokens"
    )

    vocab = Vocabulary.build_from_counts(
        counter,
        max_words=cfg.vocab.max_words,
        min_count=cfg.vocab.min_count,
    )

    vocab_path = workdir / "vocab.json"
    vocab.save(vocab_path)
    _log(f"  Vocabulary: {vocab.size} words → {vocab_path}")

    # Optional verified wordlist
    wordlist: set[str] | None = None
    if cfg.dictionary:
        dict_file = Path(cfg.dictionary)
        if dict_file.exists():
            wordlist = load_wordlist(dict_file)
            _log(f"  Wordlist: {len(wordlist):,} words from {dict_file}")
            old_size = vocab.size
            vocab = vocab.merge_wordlist(wordlist)
            if vocab.size > old_size:
                added = vocab.size - old_size
                _log(f"  Merged {added:,} wordlist-only words (new size: {vocab.size})")
                vocab.save(vocab_path)
        else:
            _log(f"  WARNING: dictionary file not found: {dict_file}")

    # Build T9 dictionary index
    dictionary = T9Dictionary.build(vocab, wordlist=wordlist, verbose=True)
    dict_path = workdir / "dict.json"
    dictionary.save(dict_path)
    _log(f"  T9 dictionary → {dict_path}")

    if cfg.s3.upload and cfg.s3.enabled:
        _s3_upload(cfg.s3, vocab_path, cfg.s3.paths.vocab)
        _s3_upload(cfg.s3, dict_path, cfg.s3.paths.dict)


def step_pairs(cfg, workdir: Path, corpus_path: Path) -> Path:
    """Precompute (context, target) training pairs from the corpus.

    Returns the path to the pairs directory.
    """
    _log("=== Step: pairs ===")

    from ai_t9.model.trainer import _corpus_file_sentence_ids, save_pairs
    from ai_t9.model.vocab import Vocabulary

    vocab_path = workdir / "vocab.json"
    if not _ensure_file(vocab_path, cfg.s3, cfg.s3.paths.vocab, "vocab.json"):
        _log("ERROR: vocab.json not found. Run the 'vocab' step first.")
        sys.exit(1)

    vocab = Vocabulary.load(vocab_path)
    _log(f"  Loaded vocabulary ({vocab.size} words)")

    if not corpus_path.exists():
        _log(f"ERROR: corpus not found at {corpus_path}. Run the 'corpus' step first.")
        sys.exit(1)

    sentences = _corpus_file_sentence_ids(corpus_path, vocab)
    _log(f"  {len(sentences):,} sentences → pair precomputation")

    pairs_dir = workdir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = pairs_dir / "pairs.npz"

    n = save_pairs(
        sentences,
        context_window=cfg.model.context_window,
        vocab_size=vocab.size,
        path=pairs_path,
        verbose=True,
        max_shard_pairs=cfg.pairs.shard_size,
    )
    _log(f"  {n:,} pairs precomputed → {pairs_dir}")

    if cfg.s3.upload and cfg.s3.enabled:
        remote_prefix = cfg.s3.paths.pairs.rstrip("/")
        for f in pairs_dir.glob("*.npz"):
            _s3_upload(cfg.s3, f, f"{remote_prefix}/{f.name}")

    return pairs_dir


def step_train(cfg, workdir: Path) -> None:
    """Train the dual-encoder model from precomputed pairs."""
    _log("=== Step: train ===")

    from ai_t9.model.trainer import DualEncoderTrainer
    from ai_t9.model.vocab import Vocabulary

    # Ensure vocab exists
    vocab_path = workdir / "vocab.json"
    if not _ensure_file(vocab_path, cfg.s3, cfg.s3.paths.vocab, "vocab.json"):
        _log("ERROR: vocab.json not found. Run the 'vocab' step first.")
        sys.exit(1)

    vocab = Vocabulary.load(vocab_path)
    _log(f"  Vocabulary: {vocab.size} words")

    # Ensure pairs exist
    pairs_dir = workdir / "pairs"
    if not _ensure_pairs_dir(pairs_dir, cfg.s3, cfg.s3.paths.pairs):
        _log("ERROR: No pairs found. Run the 'pairs' step first.")
        sys.exit(1)

    # Build trainer
    t = cfg.training
    trainer = DualEncoderTrainer(
        vocab=vocab,
        embed_dim=cfg.model.embed_dim,
        context_window=cfg.model.context_window,
        lr=t.lr,
        weight_decay=t.weight_decay,
        warmup_frac=t.warmup_frac,
        min_lr_frac=t.min_lr_frac,
        batch_size=t.batch_size,
        accumulate_grad_batches=t.accumulate_grad_batches,
        clip_grad_norm=t.clip_grad_norm,
        seed=t.seed,
        device=t.device,
        objective=t.objective,
        n_negatives=t.n_negatives,
        temperature=t.temperature,
    )

    # ---- Checkpoint setup ------------------------------------------------
    # Local checkpoint directory is always workdir/checkpoints/.
    # When S3 is configured we use the two-slot atomic protocol to keep the
    # last fully committed checkpoint safe even under mid-save interruptions.
    ckpt_dir = workdir / "checkpoints"
    ckpt_path = ckpt_dir / "checkpoint.pt"
    ckpt_manager: _S3CheckpointManager | None = None

    if cfg.s3.enabled:
        remote_ckpt_prefix = cfg.s3.paths.checkpoint.rstrip("/")
        ckpt_manager = _S3CheckpointManager(cfg.s3, remote_ckpt_prefix)
        _log("  Checking S3 for existing checkpoint…")
        downloaded = ckpt_manager.download_latest(ckpt_dir)
        if downloaded:
            _log(f"  Restored checkpoint from S3 → {downloaded}")
            ckpt_path = downloaded
        else:
            _log("  No checkpoint found in S3 — starting fresh")

    # Also honour the legacy config field (local path only).
    if not ckpt_path.exists() and t.checkpoint and Path(t.checkpoint).exists():
        ckpt_path = Path(t.checkpoint)

    if ckpt_path.exists():
        _log(f"  Loading checkpoint: {ckpt_path}")
        trainer.load_checkpoint(ckpt_path)

    # Callback: after every local save, upload to S3 via the two-slot manager.
    def _on_checkpoint(path: Path) -> None:
        if ckpt_manager is None:
            return
        try:
            import torch as _torch
            data = _torch.load(path, map_location="cpu", weights_only=False)
            epoch = data.get("epoch", 0)
            shard = data.get("shard", -1)
            step = data.get("global_step", 0)
        except Exception:
            epoch = shard = step = 0
        try:
            ckpt_manager.upload(path, epoch=epoch, shard=shard, step=step)
        except Exception as exc:
            _log(f"  WARNING: S3 checkpoint upload failed: {exc}")

    on_checkpoint = _on_checkpoint if ckpt_manager is not None else None

    # ---- Train -----------------------------------------------------------
    shards = sorted(pairs_dir.glob("pairs_*.npz"))
    if shards:
        _log(f"  Training from {len(shards)} shard(s)")
        trainer.train_from_pairs_dir(
            pairs_dir,
            epochs=t.epochs,
            verbose=True,
            checkpoint_path=ckpt_path,
            on_checkpoint=on_checkpoint,
        )
    else:
        pairs_file = pairs_dir / "pairs.npz"
        if pairs_file.exists():
            _log(f"  Training from {pairs_file}")
            trainer.train_from_pairs_file(
                pairs_file,
                epochs=t.epochs,
                verbose=True,
                checkpoint_path=ckpt_path,
                on_checkpoint=on_checkpoint,
            )
        else:
            _log("ERROR: No pairs files found in pairs directory")
            sys.exit(1)

    model_path = workdir / "model.npz"
    trainer.save_numpy(model_path)
    _log(f"  Model saved → {model_path}")

    if cfg.s3.upload and cfg.s3.enabled:
        _s3_upload(cfg.s3, model_path, cfg.s3.paths.model)


def step_ngram(cfg, workdir: Path, corpus_path: Path) -> None:
    """Train a bigram language model from the corpus."""
    _log("=== Step: ngram ===")

    if not cfg.ngram:
        _log("  Skipped (ngram: false in config)")
        return

    from ai_t9.model.trainer import _corpus_file_sentence_ids
    from ai_t9.model.vocab import Vocabulary
    from ai_t9.ngram import BigramScorer

    # Ensure vocab
    vocab_path = workdir / "vocab.json"
    if not _ensure_file(vocab_path, cfg.s3, cfg.s3.paths.vocab, "vocab.json"):
        _log("ERROR: vocab.json not found. Run the 'vocab' step first.")
        sys.exit(1)

    if not corpus_path.exists():
        _log(f"ERROR: corpus not found at {corpus_path}. Run the 'corpus' step first.")
        sys.exit(1)

    vocab = Vocabulary.load(vocab_path)
    scorer = BigramScorer(vocab)

    sents = _corpus_file_sentence_ids(corpus_path, vocab)
    _log(f"  Training on {len(sents):,} sentences")
    for sent in sents:
        scorer.train_on_ids(sent)

    ngram_path = workdir / "bigram.npz"
    actual_path = scorer.save(ngram_path)
    _log(f"  Bigram model → {actual_path}")

    if cfg.s3.upload and cfg.s3.enabled:
        _s3_upload(cfg.s3, Path(actual_path), cfg.s3.paths.ngram)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a complete ai-t9 training pipeline from a YAML config",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Steps (executed in order):\n"
            "  corpus   Combine/fetch datasets into one corpus file\n"
            "  vocab    Build vocabulary and T9 dictionary\n"
            "  pairs    Precompute training pairs\n"
            "  train    Train the dual-encoder model\n"
            "  ngram    Train bigram language model\n"
            "\n"
            "Examples:\n"
            "  ai-t9-run configs/default.yaml\n"
            "  ai-t9-run configs/default.yaml --skip corpus\n"
            "  ai-t9-run configs/default.yaml --step train --step ngram\n"
        ),
    )
    parser.add_argument("config", help="Path to YAML config file")
    parser.add_argument(
        "--step",
        action="append",
        metavar="NAME",
        help="Run only these steps (can be repeated)",
    )
    parser.add_argument(
        "--skip",
        action="append",
        metavar="NAME",
        help="Skip these steps (can be repeated)",
    )
    args = parser.parse_args(argv)

    from ai_t9.config import load_config

    cfg = load_config(args.config)

    # Resolve which steps to run
    steps = list(cfg.steps)
    if args.step:
        steps = list(args.step)
    if args.skip:
        steps = [s for s in steps if s not in args.skip]

    # Banner
    _log(f"╔══ ai-t9 training run: {cfg.name} ══╗")
    _log(f"  Steps:  {', '.join(steps)}")
    _log(f"  Output: {cfg.output_dir}")
    if cfg.s3.enabled:
        _log(f"  S3:     {cfg.s3.endpoint} / {cfg.s3.bucket} (upload={cfg.s3.upload})")
    else:
        _log("  S3:     not configured")
    _log("")

    workdir = Path(cfg.output_dir)
    workdir.mkdir(parents=True, exist_ok=True)

    # The corpus file path is shared between steps
    corpus_path = workdir / "corpus.txt"

    t_start = time.monotonic()

    if "corpus" in steps:
        corpus_path = step_corpus(cfg, workdir)

    if "vocab" in steps:
        step_vocab(cfg, workdir, corpus_path)

    if "pairs" in steps:
        step_pairs(cfg, workdir, corpus_path)

    if "train" in steps:
        step_train(cfg, workdir)

    if "ngram" in steps:
        step_ngram(cfg, workdir, corpus_path)

    elapsed = time.monotonic() - t_start
    m, s = divmod(int(elapsed), 60)
    h, m = divmod(m, 60)
    time_str = f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")
    _log(f"╚══ All steps complete ({time_str}) ══╝")
    return 0


if __name__ == "__main__":
    sys.exit(main())
