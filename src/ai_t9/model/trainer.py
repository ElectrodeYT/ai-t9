"""PyTorch training for the DualEncoder model.

This module is intentionally isolated from the inference path — PyTorch is an
optional dependency (pip install ai-t9[train]).  The output is a .npz file
that the pure-NumPy encoders can load without any ML framework.

Training objectives are pluggable (see ``objectives.py``):

  ``sgns``  (default) — Skip-Gram Negative Sampling (fastText / Word2Vec style)
    For each (context_words, target_word) pair:
      L = −log σ(c · w⁺) − (1/k) Σᵢ log σ(−c · wᵢ⁻)
    Negatives are drawn from a frequency-weighted distribution (f^0.75).
    Cost: O(B × k) per step — linear in batch size.

  ``clip`` — CLIP-style in-batch negatives with symmetric cross-entropy
    Builds a (B, B) similarity matrix; each context identifies its target and
    vice versa.  Cost: O(B²) per step — quadratic in batch size.

Custom objectives can be added by subclassing ``TrainingObjective`` in
``objectives.py`` and passing an instance to ``DualEncoderTrainer``.

Optimiser: AdamW with cosine LR decay and linear warmup.

torch.compile() (PyTorch 2.0+) handles graph capture and kernel fusion.
Mixed precision (BF16 / FP16 + GradScaler) is applied automatically on CUDA.
"""

from __future__ import annotations

import io
import math
import random
import re
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from .vocab import Vocabulary
from .dual_encoder import DualEncoder
from .objectives import TrainingObjective


def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for training: pip install ai-t9[train]  "
            "or  pip install torch"
        )


def _resolve_device(torch, preference: str) -> "torch.device":
    """Pick the best available compute device.

    preference values:
      "auto"  – CUDA if available, then MPS (Apple Silicon), then CPU
      "cuda"  – CUDA; raises if not available
      "mps"   – MPS; raises if not available
      "cpu"   – always CPU
    """
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device found.")
        return torch.device("cuda")
    if preference == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but not available.")
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _auto_batch_size(
    torch, device, objective: "TrainingObjective | None" = None,
) -> int:
    """Pick a throughput-optimal power-of-2 batch size.

    For linear-cost objectives (SGNS, the default), throughput scales linearly
    with batch size so we use large batches.  For quadratic-cost objectives
    (CLIP), the objective's :meth:`max_batch_size` caps the result to avoid
    the O(B²) memory/compute cliff.

    Falls back to 4096 on non-CUDA devices.
    """
    if not (str(device).startswith("cuda") and torch.cuda.is_available()):
        return 4096

    try:
        free, _total = torch.cuda.mem_get_info(device)
    except AttributeError:
        free = torch.cuda.get_device_properties(device).total_memory

    free_gib = free / 1024 ** 3
    if free_gib >= 40:
        cap = 32768     # data-centre (A100 / H100)
    elif free_gib >= 12:
        cap = 16384     # high-end consumer (RTX 4080+, 3090, …)
    else:
        cap = 8192      # mid-range consumer

    # Let the objective impose its own VRAM-based limit (e.g. CLIP's O(B²)).
    if objective is not None:
        obj_max = objective.max_batch_size(free)
        if obj_max is not None:
            cap = min(cap, obj_max)

    if cap < 2048:
        return 2048
    return 1 << int(math.log2(cap))


def _resolve_batch_size(
    batch_size: int,
    torch,
    device,
    objective: "TrainingObjective | None" = None,
    verbose: bool = False,
) -> int:
    """Return *batch_size* unchanged if positive, otherwise auto-detect."""
    if batch_size > 0:
        return batch_size
    auto = _auto_batch_size(torch, device, objective=objective)
    if verbose:
        print(f"  Auto batch size: {auto:,}")
    return auto


# ---------------------------------------------------------------------------
# Corpus iterator helpers
# ---------------------------------------------------------------------------

# Split on sentence-ending punctuation followed by whitespace.
# This prevents context windows from spanning sentence boundaries when a
# corpus file contains multiple sentences per line (paragraphs, etc.).
_SENT_BOUNDARY_RE = re.compile(r"[.!?]+\s+")


def _corpus_file_sentence_ids(path: Path, vocab: Vocabulary) -> list[list[int]]:
    """Read a plain-text file and convert to per-sentence word-ID lists.

    Lines are split on sentence-ending punctuation so that context windows
    never cross sentence boundaries — e.g. a paragraph on one line is split
    into individual sentences before pair generation.

    UNK tokens (ID 0) are excluded so they never appear as training
    targets or pollute context windows.
    """
    sentences = []
    unk = vocab.UNK_ID
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip().lower()
            if not line:
                continue
            for sent_text in _SENT_BOUNDARY_RE.split(line):
                words = [w for w in sent_text.split() if w.isalpha()]
                ids = [vocab.word_to_id(w) for w in words]
                ids = [wid for wid in ids if wid != unk]
                if len(ids) >= 2:
                    sentences.append(ids)
    return sentences


def _precompute_pairs(
    sentences: list[list[int]],
    context_window: int,
    verbose: bool = False,
    subsample_probs: "np.ndarray | None" = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-compute all (context, target) training pairs as flat NumPy arrays.

    Called once before training.  The resulting arrays are shuffled per epoch
    via NumPy index permutation (fast, releases the GIL).

    Pairs where the target is UNK (ID 0) are skipped — they teach nothing
    useful and waste gradient signal.  (UNK tokens should already be stripped
    from sentences by the corpus loaders, but this is a safety net.)

    Uses vectorised NumPy operations per sentence (stride-trick context
    windows + boolean masking) instead of element-by-element Python writes.

    Args:
        subsample_probs: optional float32 (vocab_size,) array where entry i is
            the keep-probability for word_id i.  When supplied, each word
            occurrence is independently kept with that probability before pairs
            are generated (Word2Vec-style frequent-word subsampling).

    Returns:
        ctx_arr: int64 (n_pairs, context_window) — zero-padded on the left
        pos_arr: int64 (n_pairs,)
    """
    _UNK = 0
    if verbose:
        print("Precomputing training pairs...")

    ctx_chunks: list[np.ndarray] = []
    pos_chunks: list[np.ndarray] = []
    total_pairs = 0
    n_sents = len(sentences)
    report_interval = max(1, n_sents // 100)

    rng = np.random.default_rng()

    for si, sent in enumerate(sentences):
        if len(sent) < 2:
            continue
        arr = np.array(sent, dtype=np.int64)

        # Frequent-word subsampling: stochastically discard word occurrences
        # with probability 1 - subsample_probs[word_id].  This reduces the
        # dominance of high-frequency function words (the, a, is, …) and
        # provides the same speedup and quality improvement as Word2Vec's -sample.
        if subsample_probs is not None:
            keep = rng.random(len(arr)) < subsample_probs[arr]
            arr = arr[keep]
            if len(arr) < 2:
                continue

        n = len(arr)

        # Boolean mask: which targets (positions 1..n-1) are non-UNK?
        targets = arr[1:]
        mask = targets != _UNK
        if not mask.any():
            continue

        # Prepend context_window zeros so that early positions get left-padding,
        # then use a strided view to extract all context windows at once.
        padded = np.empty(context_window + n, dtype=np.int64)
        padded[:context_window] = 0
        padded[context_window:] = arr

        # windows[t] == padded[t : t + context_window] — left-padded context
        # for the target at sentence index t.
        windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=context_window)

        # Select only valid (non-UNK target) rows.  Fancy indexing copies
        # data, which also safely detaches from the strided view.
        valid_t = np.where(mask)[0] + 1  # target positions in arr
        ctx_chunks.append(windows[valid_t].copy())
        pos_chunks.append(arr[valid_t])
        total_pairs += len(valid_t)

        if verbose and si % report_interval == 0:
            frac = (si + 1) / n_sents
            bar_w = 20
            filled = int(bar_w * frac)
            bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
            print(
                f"\r  |{bar}| {si + 1}/{n_sents} sentences ({total_pairs:,} pairs)",
                end="", flush=True,
            )

    if verbose:
        print(f"\r  {total_pairs:,} pairs from {n_sents:,} sentences" + " " * 30)

    if ctx_chunks:
        return np.concatenate(ctx_chunks), np.concatenate(pos_chunks)
    return np.zeros((0, context_window), dtype=np.int64), np.zeros(0, dtype=np.int64)


def save_pairs(
    sentences: list[list[int]],
    context_window: int,
    vocab_size: int,
    path: str | Path,
    verbose: bool = False,
    max_shard_pairs: int | None = None,
    subsample_probs: "np.ndarray | None" = None,
) -> int:
    """Precompute training pairs from sentences and persist them to .npz file(s).

    When ``max_shard_pairs`` is None (default), writes a single file at ``path``.
    When set, writes sharded files named ``path_000.npz``, ``path_001.npz``, …
    each containing at most ``max_shard_pairs`` rows.  Unlike the single-file
    path, sharding is done in a **streaming** fashion — only one shard's worth
    of pairs is held in memory at a time, making it safe for arbitrarily large
    corpora.

    The file(s) store arrays as int32 (half the size of int64) and embed
    ``context_window``, ``vocab_size``, and ``n_pairs`` as metadata so
    ``load_pairs()`` can detect stale files and the LR scheduler can size
    itself accurately without loading all pair data.

    Args:
        subsample_probs: optional (vocab_size,) keep-probability array for
            Word2Vec-style frequent-word subsampling.  Built by
            ``_BaseTrainer._compute_subsample_probs()``.

    Returns the total number of pairs written.
    """
    if max_shard_pairs is None:
        # Single-file path: materialise all pairs then write once.
        ctx_arr, pos_arr = _precompute_pairs(
            sentences, context_window, verbose=verbose, subsample_probs=subsample_probs,
        )
        n_total = len(pos_arr)
        _write_pairs_npz(ctx_arr, pos_arr, context_window, vocab_size, path)
        if verbose:
            p = Path(path) if str(path).endswith(".npz") else Path(str(path) + ".npz")
            print(f"  Saved {n_total:,} pairs → {p}  ({p.stat().st_size / 1e6:.1f} MB)")
        return n_total

    # Sharded path: stream sentence-by-sentence and flush shards as they fill.
    # This keeps at most max_shard_pairs pairs in memory at a time, making it
    # suitable for corpora too large to fit in RAM.
    path = Path(path)
    stem = path.stem if path.suffix == ".npz" else path.name
    parent = path.parent
    _UNK = 0
    rng = np.random.default_rng()

    shard = 0
    ctx_buf: list[np.ndarray] = []
    pos_buf: list[np.ndarray] = []
    buf_size = 0
    total = 0
    n_sents = len(sentences)
    report_interval = max(1, n_sents // 100)

    if verbose:
        print("Precomputing training pairs (streaming shards)...")

    for si, sent in enumerate(sentences):
        if len(sent) < 2:
            continue
        arr = np.array(sent, dtype=np.int64)

        if subsample_probs is not None:
            keep = rng.random(len(arr)) < subsample_probs[arr]
            arr = arr[keep]
            if len(arr) < 2:
                continue

        n = len(arr)
        targets = arr[1:]
        mask = targets != _UNK
        if not mask.any():
            continue

        padded = np.empty(context_window + n, dtype=np.int64)
        padded[:context_window] = 0
        padded[context_window:] = arr
        windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=context_window)
        valid_t = np.where(mask)[0] + 1
        ctx_buf.append(windows[valid_t].copy())
        pos_buf.append(arr[valid_t])
        buf_size += len(valid_t)

        # Flush completed shards from the buffer.
        while buf_size >= max_shard_pairs:
            ctx_all = np.concatenate(ctx_buf)
            pos_all = np.concatenate(pos_buf)
            shard_path = parent / f"{stem}_{shard:03d}.npz"
            _write_pairs_npz(
                ctx_all[:max_shard_pairs], pos_all[:max_shard_pairs],
                context_window, vocab_size, shard_path,
            )
            if verbose:
                print(f"  Shard {shard}: {max_shard_pairs:,} pairs → {shard_path}")
            total += max_shard_pairs
            shard += 1
            remainder_ctx = ctx_all[max_shard_pairs:]
            remainder_pos = pos_all[max_shard_pairs:]
            ctx_buf = [remainder_ctx] if len(remainder_ctx) > 0 else []
            pos_buf = [remainder_pos] if len(remainder_pos) > 0 else []
            buf_size = len(remainder_pos)

        if verbose and si % report_interval == 0:
            frac = (si + 1) / n_sents
            bar_w = 20
            filled = int(bar_w * frac)
            bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
            print(
                f"\r  |{bar}| {si + 1}/{n_sents} sentences ({total + buf_size:,} pairs)",
                end="", flush=True,
            )

    # Flush any remaining pairs as the final (partial) shard.
    if buf_size > 0:
        ctx_all = np.concatenate(ctx_buf)
        pos_all = np.concatenate(pos_buf)
        shard_path = parent / f"{stem}_{shard:03d}.npz"
        _write_pairs_npz(ctx_all, pos_all, context_window, vocab_size, shard_path)
        if verbose:
            print(f"\r  Shard {shard}: {len(pos_all):,} pairs → {shard_path}" + " " * 20)
        total += len(pos_all)
    elif verbose:
        print()

    return total


def _write_pairs_npz(
    ctx_arr: np.ndarray,
    pos_arr: np.ndarray,
    context_window: int,
    vocab_size: int,
    path: str | Path,
) -> None:
    """Write a pairs array pair to a .npz file via an in-memory buffer.

    Writing via BytesIO avoids random seeks, which is incompatible with S3
    CloudBucketMounts (Mountpoint only supports sequential writes).
    The ``n_pairs`` scalar is stored as metadata so shard pair counts can be
    read without decompressing the full arrays (used for LR schedule sizing).
    """
    buf = io.BytesIO()
    np.savez(
        buf,
        ctx=ctx_arr.astype(np.int32),
        pos=pos_arr.astype(np.int32),
        vocab_size=np.array(vocab_size, dtype=np.int64),
        context_window=np.array(context_window, dtype=np.int64),
        n_pairs=np.array(len(pos_arr), dtype=np.int64),
    )
    buf.seek(0)
    path = Path(path)
    if not str(path).endswith(".npz"):
        path = Path(str(path) + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        shutil.copyfileobj(buf, f)


def _get_shard_n_pairs(path: str | Path) -> int:
    """Return the pair count stored in a shard's metadata without loading arrays.

    NumPy's NpzFile is lazy — only the requested entry is decompressed.
    Reading the tiny ``n_pairs`` scalar is therefore very fast even for
    large shards.  Falls back to loading ``pos`` if the key is absent
    (old-format shards written before this metadata was added).
    """
    data = np.load(path)
    if "n_pairs" in data:
        return int(data["n_pairs"])
    return len(data["pos"])  # backward-compat: load full array


def load_pairs(
    path: str | Path,
    context_window: int | None = None,
    vocab_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load precomputed training pairs from a .npz file.

    Validates ``context_window`` and ``vocab_size`` against embedded metadata
    when provided, raising ``ValueError`` if they don't match — catching the
    common mistake of reusing pairs built from a different vocab or window.

    Returns ``(ctx_arr, pos_arr)`` as int64 arrays ready for training.
    """
    data = np.load(path)
    if context_window is not None:
        saved_cw = int(data["context_window"])
        if saved_cw != context_window:
            raise ValueError(
                f"Pairs file context_window={saved_cw} does not match "
                f"requested context_window={context_window}. "
                "Regenerate the pairs file with matching settings."
            )
    if vocab_size is not None:
        saved_vs = int(data["vocab_size"])
        if saved_vs != vocab_size:
            raise ValueError(
                f"Pairs file vocab_size={saved_vs} does not match "
                f"current vocab_size={vocab_size}. "
                "Regenerate the pairs file with the matching vocab."
            )
    return data["ctx"].astype(np.int64), data["pos"].astype(np.int64)


# ---------------------------------------------------------------------------
# LR schedule helper
# ---------------------------------------------------------------------------

def _make_lr_lambda(total_steps: int, warmup_steps: int, min_lr_frac: float):
    """Return a LambdaLR schedule: linear warmup then cosine decay."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_frac, cosine)
    return lr_lambda


def _log_training_complete(t_start: float) -> None:
    """Print a 'Training complete' summary line with wall-clock duration."""
    total_time = time.monotonic() - t_start
    m, s = divmod(int(total_time), 60)
    h, m = divmod(m, 60)
    time_str = f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")
    print(f"  Training complete in {time_str}")


# ---------------------------------------------------------------------------
# Shared trainer base class
# ---------------------------------------------------------------------------

class _BaseTrainer:
    """Shared training infrastructure for DualEncoderTrainer and CharNgramDualEncoderTrainer.

    Handles corpus loading, device setup, AMP, the epoch/batch training loop,
    shard prefetching, optimizer/scheduler management, and timing output.

    Subclasses must implement:
        _build_model(torch, device)  → (model, compiled_model)
        save_numpy(path)
        get_encoder()

    Subclasses may override:
        _before_training()           — called before training starts
        _verbose_extra_fields()      — extra fields for the verbose header line
    """

    def __init__(
        self,
        vocab: Vocabulary,
        embed_dim: int = 64,
        context_window: int = 3,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        warmup_frac: float = 0.05,
        min_lr_frac: float = 0.01,
        batch_size: int = 0,
        accumulate_grad_batches: int = 1,
        clip_grad_norm: float = 1.0,
        seed: int = 42,
        device: str = "auto",
        debug: bool = False,
        subsample_threshold: float = 1e-4,
    ) -> None:
        self._vocab = vocab
        self._embed_dim = embed_dim
        self._context_window = context_window
        self._lr = lr
        self._weight_decay = weight_decay
        self._warmup_frac = warmup_frac
        self._min_lr_frac = min_lr_frac
        self._batch_size = batch_size
        self._accumulate = max(1, accumulate_grad_batches)
        self._clip_grad_norm = clip_grad_norm
        self._seed = seed
        self._device_pref = device
        self._debug = debug
        self._subsample_threshold = subsample_threshold
        self._model = None
        self._epoch = 0
        self._global_step = 0
        self._optimizer = None
        self._scheduler = None
        self._scaler = None
        self._objective: TrainingObjective | None = None
        self._current_shard: int = -1
        self._current_shard_order: list | None = None
        self._checkpoint_data: dict | None = None

    def _compute_subsample_probs(self) -> "np.ndarray | None":
        """Compute per-word keep-probabilities for frequent-word subsampling.

        For each word w with corpus frequency f(w), the keep probability is::

            P(keep w) = min(1.0, sqrt(t / f(w)))

        where t = subsample_threshold (default 1e-4).  Words with f(w) <= t are
        always kept; frequent words are stochastically down-sampled.  This is
        Word2Vec's ``-sample`` mechanism, which reduces the dominance of function
        words and speeds up training by generating fewer pairs.

        Returns None when subsample_threshold <= 0 (subsampling disabled).
        """
        t = self._subsample_threshold
        if t <= 0:
            return None
        counts = self._vocab.counts
        total = max(sum(counts), 1)
        probs = np.ones(self._vocab.size, dtype=np.float32)
        for wid in range(1, self._vocab.size):  # skip UNK
            freq = counts[wid] / total
            if freq > 0:
                probs[wid] = min(1.0, math.sqrt(t / freq))
        probs[0] = 0.0  # UNK always discarded (already excluded elsewhere)
        return probs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_from_sentences(self, sentences: list[list[int]], epochs: int = 3, verbose: bool = True, checkpoint_path: str | Path | None = None) -> None:
        """Train from a list of sentences (list of word ID lists)."""
        torch = _require_torch()
        self._before_training()
        self._train(sentences, epochs=epochs, torch=torch, verbose=verbose, checkpoint_path=checkpoint_path)

    def train_from_file(self, corpus_path: str | Path, epochs: int = 3, verbose: bool = True, checkpoint_path: str | Path | None = None) -> None:
        """Train on a plain-text file (one sentence per line)."""
        self.train_from_files([Path(corpus_path)], epochs=epochs, verbose=verbose, checkpoint_path=checkpoint_path)

    def train_from_files(self, paths: list[str | Path], epochs: int = 3, verbose: bool = True, checkpoint_path: str | Path | None = None) -> None:
        """Train on multiple plain-text corpus files (combined into one pass)."""
        torch = _require_torch()
        t0 = time.monotonic()
        sentences: list[list[int]] = []
        if verbose:
            print("Loading corpus...")
        total_sentences = 0
        for i, path in enumerate(paths):
            path = Path(path)
            file_sents = _corpus_file_sentence_ids(path, self._vocab)
            total_sentences += len(file_sents)
            sentences.extend(file_sents)
            if verbose:
                elapsed = time.monotonic() - t0
                print(f"\r  Loaded {i+1}/{len(paths)} files ({total_sentences:,} sentences)  ({elapsed:.2f}s)", end="", flush=True)
        if verbose:
            print()
            print(f"  Total: {len(sentences):,} sentences")
        self._before_training()
        self._train(sentences, epochs=epochs, torch=torch, verbose=verbose, checkpoint_path=checkpoint_path)

    def train_from_pairs_file(self, pairs_path: str | Path, epochs: int = 3, verbose: bool = True, checkpoint_path: str | Path | None = None, on_checkpoint=None) -> None:
        """Train directly from a precomputed pairs .npz file."""
        torch = _require_torch()
        if verbose:
            print(f"Loading precomputed pairs from {pairs_path}…")
        ctx_np, pos_np = load_pairs(
            pairs_path,
            context_window=self._context_window,
            vocab_size=self._vocab.size,
        )
        if verbose:
            print(f"  {len(pos_np):,} pairs loaded")
        self._before_training()
        self._train_from_arrays(ctx_np, pos_np, epochs=epochs, torch=torch, verbose=verbose, checkpoint_path=checkpoint_path, on_checkpoint=on_checkpoint)

    def train_from_pairs_dir(
        self,
        pairs_dir: str | Path,
        pattern: str = "pairs_*.npz",
        epochs: int = 3,
        prefetch: bool = True,
        verbose: bool = True,
        checkpoint_path: str | Path | None = None,
        on_checkpoint=None,
    ) -> None:
        """Train from a directory of sharded pairs .npz files.

        Shards are shuffled each epoch to improve gradient diversity.
        When ``prefetch=True``, the next shard is loaded on a background
        thread while the GPU trains on the current shard, overlapping I/O
        and compute.

        Args:
            pairs_dir:       Directory containing shard files.
            pattern:         Glob pattern to match shard files (default ``pairs_*.npz``).
            epochs:          Number of full passes over all shards.
            prefetch:        Overlap CPU I/O with GPU compute via background thread.
            verbose:         Print progress.
            checkpoint_path: Local path to save checkpoint after each shard/epoch.
            on_checkpoint:   Optional ``(path: Path) -> None`` called after each
                             local checkpoint save (e.g. to upload to S3).
        """
        torch = _require_torch()
        shard_paths = sorted(Path(pairs_dir).glob(pattern))
        if not shard_paths:
            raise FileNotFoundError(f"No files matching '{pattern}' in {pairs_dir}")
        if verbose:
            print(f"Found {len(shard_paths)} shard(s) in {pairs_dir}")
        self._before_training()
        self._train_from_shards(
            shard_paths, epochs=epochs, torch=torch, prefetch=prefetch, verbose=verbose,
            checkpoint_path=checkpoint_path, on_checkpoint=on_checkpoint,
        )

    # ------------------------------------------------------------------
    # Hooks for subclasses
    # ------------------------------------------------------------------

    def _before_training(self) -> None:
        """Called once before the training loop starts.

        Override to perform setup that must happen before the model is built
        (e.g., building the n-gram vocabulary in CharNgramDualEncoderTrainer).
        """
        pass

    def _verbose_extra_fields(self) -> dict[str, object]:
        """Return extra key:value pairs for the verbose training header line.

        Override in subclasses to add model-specific info (e.g., n_ngrams).
        """
        return {}

    def _build_model(self, torch, device) -> "tuple[torch.nn.Module, torch.nn.Module]":
        """Build and return (model, compiled_model) on the given device.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def save_checkpoint(self, path: str | Path) -> None:
        """Save training state atomically (write temp + rename) to *path*."""
        torch = _require_torch()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "epoch": self._epoch,
            "shard": self._current_shard,
            "global_step": self._global_step,
            "model_state_dict": self._model.state_dict() if self._model is not None else None,
            "optimizer_state_dict": self._optimizer.state_dict() if self._optimizer is not None else None,
            "scheduler_state_dict": self._scheduler.state_dict() if self._scheduler is not None else None,
            "scaler_state_dict": self._scaler.state_dict() if self._scaler is not None else None,
            "shard_order_names": (
                [Path(p).name for p in self._current_shard_order]
                if self._current_shard_order is not None else None
            ),
            "vocab_size": self._vocab.size,
            "embed_dim": self._embed_dim,
        }
        tmp_path = path.with_suffix(".tmp")
        torch.save(state, tmp_path)
        tmp_path.replace(path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Load training state from *path* into this trainer.

        State is applied lazily: model/optimiser weights are restored inside
        ``_train_from_arrays`` / ``_train_from_shards`` once the model has been
        built.  This method just reads the file and caches it.
        """
        torch = _require_torch()
        path = Path(path)
        if not path.exists():
            return
        data = torch.load(path, map_location="cpu", weights_only=False)
        if "vocab_size" in data and data["vocab_size"] != self._vocab.size:
            raise ValueError(
                f"Checkpoint vocab_size={data['vocab_size']} does not match "
                f"current vocab_size={self._vocab.size}. "
                "Use a checkpoint built from the same vocabulary."
            )
        self._checkpoint_data = data
        self._epoch = data.get("epoch", 0)
        self._global_step = data.get("global_step", 0)

    # ------------------------------------------------------------------
    # Internal training
    # ------------------------------------------------------------------

    def _train(self, sentences, epochs, torch, verbose, checkpoint_path=None):
        # _before_training() is called by the public API methods (train_from_*)
        # before reaching here — do NOT call it again.
        subsample_probs = self._compute_subsample_probs()
        ctx_np, pos_np = _precompute_pairs(
            sentences, self._context_window, verbose=verbose,
            subsample_probs=subsample_probs,
        )
        self._train_from_arrays(ctx_np, pos_np, epochs=epochs, torch=torch, verbose=verbose, checkpoint_path=checkpoint_path)

    def _setup_device_and_amp(self, torch):
        """Set up device, CUDA optimisations, and AMP dtype/scaler.

        Returns (device, is_cuda, amp_dtype, scaler).
        """
        device = _resolve_device(torch, self._device_pref)
        is_cuda = str(device).startswith("cuda")
        random.seed(self._seed)
        np.random.seed(self._seed)
        torch.manual_seed(self._seed)

        if is_cuda and torch.cuda.is_available() and torch.cuda.get_device_properties(device).major >= 8:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        amp_dtype = scaler = None
        if is_cuda:
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                amp_dtype = torch.float16
                scaler = torch.amp.GradScaler("cuda")

        return device, is_cuda, amp_dtype, scaler

    def _train_from_arrays(self, ctx_np, pos_np, epochs, torch, verbose, checkpoint_path=None, on_checkpoint=None):
        """Core epoch/batch training loop operating on preloaded numpy pair arrays."""
        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            _tqdm = None

        assert self._objective is not None, (
            "Training objective not set — subclass must build it in _before_training()"
        )

        device, is_cuda, amp_dtype, scaler = self._setup_device_and_amp(torch)
        self._objective.setup(device)

        # Build the model FIRST so its VRAM footprint is reflected when
        # auto-sizing the batch (the model consumes a significant fraction
        # of GPU memory before pairs are transferred).
        t_build = time.monotonic()
        model, compiled = self._build_model(torch, device)
        self._model = model
        model.train()
        if verbose:
            print(f"  [{time.strftime('%H:%M:%S')}] Model ready in {time.monotonic() - t_build:.1f}s", flush=True)

        self._batch_size = _resolve_batch_size(
            self._batch_size, torch, device,
            objective=self._objective, verbose=verbose,
        )

        # embed_fn for objectives that need to embed extra word IDs (e.g. SGNS
        # negative sampling).  Compiled separately from the forward pass so the
        # hot negative-embedding path benefits from kernel fusion.
        _raw_embed_fn = getattr(model, 'embed_tgt_words', None)
        if _raw_embed_fn is not None and hasattr(torch, "compile"):
            try:
                embed_fn = torch.compile(
                    _raw_embed_fn, mode="max-autotune-no-cudagraphs"
                )
            except Exception:
                embed_fn = _raw_embed_fn
        else:
            embed_fn = _raw_embed_fn

        # Restore model from checkpoint
        if self._checkpoint_data is not None and self._checkpoint_data.get('model_state_dict'):
            self._model.load_state_dict(self._checkpoint_data['model_state_dict'])

        n_pairs = len(pos_np)
        n_batches = n_pairs // self._batch_size
        effective_batch = self._batch_size * self._accumulate
        total_steps = math.ceil(n_batches / self._accumulate) * epochs
        warmup_steps = int(total_steps * self._warmup_frac)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, _make_lr_lambda(total_steps, warmup_steps, self._min_lr_frac)
        )

        self._optimizer = optimizer
        self._scheduler = scheduler
        self._scaler = scaler

        # Restore optimizer/scheduler/scaler from checkpoint
        if self._checkpoint_data is not None:
            cd = self._checkpoint_data
            if cd.get('optimizer_state_dict'):
                self._optimizer.load_state_dict(cd['optimizer_state_dict'])
            if cd.get('scheduler_state_dict'):
                self._scheduler.load_state_dict(cd['scheduler_state_dict'])
            if cd.get('scaler_state_dict') and self._scaler is not None:
                self._scaler.load_state_dict(cd['scaler_state_dict'])

        if verbose:
            print(f"  [{time.strftime('%H:%M:%S')}] Transferring {(ctx_np.nbytes + pos_np.nbytes) / 1e6:.0f} MB of pairs to {device}…", flush=True)
        t_xfer = time.monotonic()
        ctx_dev = torch.from_numpy(ctx_np).to(device)
        pos_dev = torch.from_numpy(pos_np).to(device)
        del ctx_np, pos_np  # free CPU copies
        if verbose:
            print(f"  [{time.strftime('%H:%M:%S')}] Transfer done in {time.monotonic() - t_xfer:.1f}s", flush=True)

        if verbose:
            if is_cuda:
                vram_alloc_mb = torch.cuda.memory_allocated(device) / 1e6
                vram_total_mb = torch.cuda.get_device_properties(device).total_memory / 1e6
                mem_str = f"VRAM: {vram_alloc_mb:.0f} / {vram_total_mb:.0f} MB"
            else:
                mem_str = f"data: {(ctx_dev.nbytes + pos_dev.nbytes) / 1e6:.0f} MB"
            compiled_str = "on" if compiled is not model else "off"
            amp_str = str(amp_dtype).replace("torch.", "") if amp_dtype is not None else "fp32"
            extra = self._verbose_extra_fields()
            extra_str = ("  |  " + "  |  ".join(f"{k}: {v}" for k, v in extra.items())) if extra else ""
            print(
                f"Device: {device}  |  pairs: {n_pairs:,}  |  "
                f"vocab: {self._vocab.size}  |  embed_dim: {self._embed_dim}"
                f"{extra_str}  |  objective: {self._objective.label}  |  "
                f"batch: {self._batch_size}  |  accumulate: {self._accumulate}  |  "
                f"effective_batch: {effective_batch}  |  "
                f"torch.compile: {compiled_str}  |  amp: {amp_str}  |  {mem_str}"
            )

        if verbose:
            print(f"  [{time.strftime('%H:%M:%S')}] Starting training  "
                  "(first batch triggers torch.compile kernel build — may take several minutes)", flush=True)

        t_train = time.monotonic()
        global_step = self._global_step

        # Fix epoch range: capture start/end before the loop so the tqdm
        # description and verbose print always show the correct denominator.
        start_epoch = self._epoch + 1
        end_epoch = self._epoch + epochs

        for epoch in range(start_epoch, end_epoch + 1):
            self._epoch = epoch
            # Synchronise before starting the timer so async CUDA ops from any
            # prior work are flushed and don't inflate this epoch's elapsed time.
            if is_cuda:
                torch.cuda.synchronize(device)
            t_epoch = time.monotonic()
            # Accumulate loss on GPU to avoid a blocking GPU→CPU sync every batch.
            # A single .item() call per epoch (at reporting time) is all we need.
            running_loss = torch.zeros(1, device=device)
            perm = torch.randperm(n_pairs, device=device)
            display_step = 0

            batch_range: object = range(n_batches)
            if _tqdm is not None:
                batch_range = _tqdm(
                    batch_range,
                    desc=f"Epoch {epoch}/{end_epoch}",
                    unit="batch",
                    leave=False,
                )

            optimizer.zero_grad(set_to_none=True)
            t_loop = time.monotonic()

            for b in batch_range:
                start = b * self._batch_size
                idx = perm[start:start + self._batch_size]

                if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                    torch.compiler.cudagraph_mark_step_begin()
                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=amp_dtype is not None):
                    _pos_batch = pos_dev[idx]
                    ctx_vecs, pos_vecs = compiled(ctx_dev[idx], _pos_batch)
                    loss = self._objective.compute_loss(
                        ctx_vecs, pos_vecs,
                        embed_fn=embed_fn,
                        pos_ids=_pos_batch,
                    ) / self._accumulate

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                running_loss.add_(loss.detach() * self._accumulate)

                is_last_in_accum = ((b + 1) % self._accumulate == 0 or b == n_batches - 1)
                if is_last_in_accum:
                    if self._clip_grad_norm > 0:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), self._clip_grad_norm)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    display_step += 1
                    # Sync for tqdm only every ~20 optimizer steps rather than
                    # every batch — avoids stalling the GPU pipeline.
                    if _tqdm is not None and display_step % 20 == 0:
                        elapsed_loop = time.monotonic() - t_loop
                        p_s = (b + 1) * self._batch_size / elapsed_loop if elapsed_loop > 0 else 0
                        batch_range.set_postfix({
                            "loss": f"{running_loss.item() / (b + 1):.4f}",
                            "p/s": f"{p_s:,.0f}",
                        })

            if is_cuda:
                torch.cuda.synchronize(device)
            elapsed = time.monotonic() - t_epoch
            avg_loss = running_loss.item() / max(n_batches, 1)  # one sync per epoch
            pairs_per_sec = n_pairs / elapsed if elapsed > 0 else 0
            if verbose:
                print(
                    f"  Epoch {epoch}/{end_epoch}  loss={avg_loss:.4f}  "
                    f"time={elapsed:.1f}s  ({pairs_per_sec:,.0f} pairs/s)"
                )

            self._global_step = global_step
            if checkpoint_path:
                self.save_checkpoint(checkpoint_path)
                if on_checkpoint is not None:
                    on_checkpoint(Path(checkpoint_path))

        self._global_step = global_step

        if verbose:
            _log_training_complete(t_train)

    def _train_from_shards(self, shard_paths, epochs, torch, prefetch, verbose, checkpoint_path=None, on_checkpoint=None):
        """Epoch loop over sharded pairs files with optional prefetch and checkpointing.

        Checkpointing protocol:
          - ``checkpoint_path`` is the local ``.pt`` file updated after every shard.
          - ``on_checkpoint(path)`` is called immediately after each local save (e.g.
            to upload to S3).
          - On resume, the trainer's ``_epoch`` / ``_checkpoint_data`` (set by
            ``load_checkpoint``) determine where to restart.  The saved shard order
            is restored for partial-epoch resumption so the skipped shards are
            correctly identified.
        """
        assert self._objective is not None

        device, is_cuda, amp_dtype, scaler = self._setup_device_and_amp(torch)
        self._scaler = scaler
        self._objective.setup(device)

        # Build the model FIRST so its VRAM footprint is visible when
        # auto-sizing the batch.
        t_build = time.monotonic()
        model, compiled = self._build_model(torch, device)
        self._model = model
        model.train()
        _raw_embed_fn = getattr(model, 'embed_tgt_words', None)
        if _raw_embed_fn is not None and hasattr(torch, "compile"):
            try:
                embed_fn = torch.compile(
                    _raw_embed_fn, mode="max-autotune-no-cudagraphs"
                )
            except Exception:
                embed_fn = _raw_embed_fn
        else:
            embed_fn = _raw_embed_fn
        if verbose:
            print(f"  [{time.strftime('%H:%M:%S')}] Model ready in {time.monotonic() - t_build:.1f}s", flush=True)

        self._batch_size = _resolve_batch_size(
            self._batch_size, torch, device,
            objective=self._objective, verbose=verbose,
        )

        # Restore model weights from checkpoint (must happen before optimizer)
        if self._checkpoint_data is not None and self._checkpoint_data.get('model_state_dict'):
            model.load_state_dict(self._checkpoint_data['model_state_dict'])

        # Estimate total training steps using the stored n_pairs metadata from
        # every shard — this is cheap (reads only a tiny scalar per file) and
        # correctly accounts for the smaller final shard so the LR cosine
        # decay ends precisely at the last optimizer step.
        if verbose:
            print(f"  [{time.strftime('%H:%M:%S')}] Counting pairs across {len(shard_paths)} shard(s)…", flush=True)
        t_count = time.monotonic()
        n_pairs_total = sum(_get_shard_n_pairs(p) for p in shard_paths)
        if verbose:
            print(f"  [{time.strftime('%H:%M:%S')}] Total pairs: {n_pairs_total:,}  "
                  f"(counted in {time.monotonic() - t_count:.1f}s)", flush=True)
        # Load the first shard so it's ready when the training loop starts.
        if verbose:
            print(f"  [{time.strftime('%H:%M:%S')}] Pre-loading first shard…", flush=True)
        t_shard0 = time.monotonic()
        first_ctx, first_pos = load_pairs(
            shard_paths[0], context_window=self._context_window, vocab_size=self._vocab.size,
        )
        if verbose:
            print(f"  [{time.strftime('%H:%M:%S')}] First shard: {len(first_pos):,} pairs  "
                  f"(loaded in {time.monotonic() - t_shard0:.1f}s)", flush=True)
        n_pairs_estimate = n_pairs_total * epochs
        total_steps = max(1, n_pairs_estimate // (self._batch_size * self._accumulate))
        warmup_steps = int(total_steps * self._warmup_frac)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, _make_lr_lambda(total_steps, warmup_steps, self._min_lr_frac)
        )
        self._optimizer = optimizer
        self._scheduler = scheduler

        # Restore optimizer/scheduler/scaler from checkpoint
        if self._checkpoint_data is not None:
            cd = self._checkpoint_data
            if cd.get('optimizer_state_dict'):
                optimizer.load_state_dict(cd['optimizer_state_dict'])
            if cd.get('scheduler_state_dict'):
                scheduler.load_state_dict(cd['scheduler_state_dict'])
            if cd.get('scaler_state_dict') and scaler is not None:
                scaler.load_state_dict(cd['scaler_state_dict'])

        # Determine resume point from checkpoint
        ckpt_shard = -1 if self._checkpoint_data is None else self._checkpoint_data.get("shard", -1)
        # _epoch set by load_checkpoint. If last checkpoint was a full epoch (shard==-1),
        # start the next epoch; if mid-epoch, resume the same epoch.
        start_epoch = self._epoch + 1 if ckpt_shard == -1 else self._epoch
        if start_epoch > epochs:
            if verbose:
                print(f"  Checkpoint at epoch {self._epoch}/{epochs} — training already complete.", flush=True)
            return

        # If resuming mid-epoch, restore the exact shard order so we can skip done shards.
        resume_shard_idx = 0
        ckpt_shard_order: list | None = None
        if self._checkpoint_data is not None and ckpt_shard >= 0:
            resume_shard_idx = ckpt_shard + 1
            saved_names = self._checkpoint_data.get("shard_order_names")
            if saved_names:
                name_to_path = {p.name: p for p in shard_paths}
                ckpt_shard_order = [name_to_path[n] for n in saved_names if n in name_to_path]

        if verbose:
            print(f"  [{time.strftime('%H:%M:%S')}] Starting training  "
                  f"(first batch triggers torch.compile kernel build — may take several minutes)", flush=True)

        t_train = time.monotonic()
        global_step = self._global_step

        for epoch in range(start_epoch, epochs + 1):
            # Restore or generate shard order for this epoch.
            if epoch == start_epoch and ckpt_shard_order is not None:
                shard_order = ckpt_shard_order
            else:
                shard_order = list(shard_paths)
                random.shuffle(shard_order)

            self._current_shard_order = shard_order

            # Skip shards already completed in a partial epoch resume.
            skip_count = resume_shard_idx if epoch == start_epoch else 0
            effective_order = shard_order[skip_count:]

            if skip_count > 0 and verbose:
                print(f"  [{time.strftime('%H:%M:%S')}] Epoch {epoch}/{epochs}  "
                      f"resuming from shard {skip_count + 1}/{len(shard_order)} (checkpoint restore)", flush=True)

            epoch_loss = 0.0
            epoch_batches = 0
            prefetch_result: list = [None]
            prefetch_thread: threading.Thread | None = None

            def _load_shard(path, out):
                out[0] = load_pairs(path, context_window=self._context_window, vocab_size=self._vocab.size)

            for si_rel, shard_path in enumerate(effective_order):
                si = skip_count + si_rel  # absolute shard index in full epoch order

                # Wait for prefetch if active, otherwise load directly.
                if prefetch_thread is not None:
                    prefetch_thread.join()
                    ctx_np, pos_np = prefetch_result[0]
                else:
                    t_load = time.monotonic()
                    ctx_np, pos_np = load_pairs(shard_path, context_window=self._context_window, vocab_size=self._vocab.size)
                    if verbose:
                        print(f"  [{time.strftime('%H:%M:%S')}] Shard loaded in {time.monotonic() - t_load:.1f}s", flush=True)

                if verbose:
                    print(f"  [{time.strftime('%H:%M:%S')}] Epoch {epoch}/{epochs}  "
                          f"shard {si + 1}/{len(shard_order)}  {len(pos_np):,} pairs", flush=True)

                # Start prefetch of next shard.
                if prefetch and si_rel + 1 < len(effective_order):
                    prefetch_result = [None]
                    prefetch_thread = threading.Thread(
                        target=_load_shard, args=(effective_order[si_rel + 1], prefetch_result), daemon=True
                    )
                    prefetch_thread.start()
                else:
                    prefetch_thread = None

                t_shard = time.monotonic()
                sl, sb, global_step = self._train_shard_core(
                    ctx_np, pos_np, compiled, optimizer, scheduler,
                    device, global_step, amp_dtype, scaler, embed_fn,
                    verbose=verbose,
                    epoch_desc=f"Epoch {epoch}/{epochs} shard {si + 1}/{len(shard_order)}",
                )
                epoch_loss += sl
                epoch_batches += sb
                if verbose:
                    shard_elapsed = time.monotonic() - t_shard
                    pairs_per_sec = len(pos_np) / shard_elapsed if shard_elapsed > 0 else 0
                    print(f"  [{time.strftime('%H:%M:%S')}] Shard done in {shard_elapsed:.1f}s  "
                          f"({pairs_per_sec:,.0f} pairs/s)", flush=True)

                # Save checkpoint after each shard.
                self._current_shard = si
                self._epoch = epoch
                self._global_step = global_step
                if checkpoint_path:
                    self.save_checkpoint(checkpoint_path)
                    if on_checkpoint is not None:
                        on_checkpoint(Path(checkpoint_path))

            # Epoch complete — shard=-1 signals a full epoch boundary.
            self._current_shard = -1
            self._epoch = epoch
            self._global_step = global_step
            if checkpoint_path:
                self.save_checkpoint(checkpoint_path)
                if on_checkpoint is not None:
                    on_checkpoint(Path(checkpoint_path))

            if verbose:
                print(f"  Epoch {epoch}/{epochs}  loss={epoch_loss / max(epoch_batches, 1):.4f}")

        if verbose:
            _log_training_complete(t_train)

    def _train_shard_core(
        self, ctx_np, pos_np, compiled, optimizer, scheduler,
        device, global_step, amp_dtype, scaler, embed_fn,
        verbose=False, epoch_desc="",
    ):
        """Train one shard of pairs. Returns (loss_float, n_batches, global_step)."""
        import torch
        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            _tqdm = None

        is_cuda = str(device).startswith("cuda")
        if is_cuda:
            ctx_dev = torch.from_numpy(ctx_np).pin_memory().to(device, non_blocking=True)
            pos_dev = torch.from_numpy(pos_np).pin_memory().to(device, non_blocking=True)
            torch.cuda.synchronize(device)
        else:
            ctx_dev = torch.from_numpy(ctx_np).to(device)
            pos_dev = torch.from_numpy(pos_np).to(device)

        n_pairs = len(pos_np)
        n_batches = n_pairs // self._batch_size
        perm = torch.randperm(n_pairs, device=device)
        running_loss = torch.zeros(1, device=device)
        optimizer.zero_grad(set_to_none=True)
        display_step = 0

        batch_range: object = range(n_batches)
        if _tqdm is not None:
            batch_range = _tqdm(
                batch_range,
                desc=epoch_desc,
                unit="batch",
                leave=False,
            )

        t_loop = time.monotonic()

        for b in batch_range:
            start = b * self._batch_size
            idx = perm[start:start + self._batch_size]

            if hasattr(torch.compiler, "cudagraph_mark_step_begin"):
                torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                _pos_batch = pos_dev[idx]
                ctx_vecs, pos_vecs = compiled(ctx_dev[idx], _pos_batch)
                loss = self._objective.compute_loss(
                    ctx_vecs, pos_vecs,
                    embed_fn=embed_fn,
                    pos_ids=_pos_batch,
                ) / self._accumulate

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            running_loss.add_(loss.detach() * self._accumulate)

            is_last = ((b + 1) % self._accumulate == 0 or b == n_batches - 1)
            if is_last:
                if self._clip_grad_norm > 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._clip_grad_norm)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                display_step += 1
                if _tqdm is not None and display_step % 20 == 0:
                    elapsed_loop = time.monotonic() - t_loop
                    p_s = (b + 1) * self._batch_size / elapsed_loop if elapsed_loop > 0 else 0
                    batch_range.set_postfix({
                        "loss": f"{running_loss.item() / (b + 1):.4f}",
                        "p/s": f"{p_s:,.0f}",
                    })

        del ctx_dev, pos_dev
        return running_loss.item(), n_batches, global_step


# ---------------------------------------------------------------------------
# DualEncoderTrainer
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CharNgramDualEncoderTrainer
# ---------------------------------------------------------------------------

class DualEncoderTrainer(_BaseTrainer):
    """Train a DualEncoder using pluggable training objectives.

    The training data pipeline uses (context_word_ids, target_word_id) pairs.
    Words are represented as mean-pooled character n-gram embeddings,
    decoupling model size from vocab size.

    The context encoder is a GRU (hidden_size = embed_dim) with per-slot
    positional embeddings added to the word embeddings before the GRU.  This
    replaces the old mean-pooling encoder.

    When the ``"sgns"`` objective is selected, T9 ambiguity groups are
    precomputed from the vocabulary and supplied as hard negatives, directly
    training the model to discriminate between T9-ambiguous words.

    Usage::

        trainer = DualEncoderTrainer(vocab, embed_dim=128)
        trainer.train_from_files(corpus_files, epochs=5)
        trainer.save_numpy("data/model.npz")
    """

    def __init__(
        self,
        vocab: Vocabulary,
        embed_dim: int = 64,
        context_window: int = 3,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        warmup_frac: float = 0.05,
        min_lr_frac: float = 0.01,
        batch_size: int = 0,
        accumulate_grad_batches: int = 1,
        clip_grad_norm: float = 1.0,
        seed: int = 42,
        device: str = "auto",
        debug: bool = False,
        ns: tuple[int, ...] = (2, 3),
        objective: str | TrainingObjective = "sgns",
        n_negatives: int = 15,
        hard_neg_frac: float = 0.5,
        temperature: float = 0.07,
        subsample_threshold: float = 1e-4,
    ) -> None:
        super().__init__(
            vocab, embed_dim, context_window, lr, weight_decay, warmup_frac,
            min_lr_frac, batch_size, accumulate_grad_batches,
            clip_grad_norm, seed, device, debug,
            subsample_threshold=subsample_threshold,
        )
        self._ns = ns
        self._ngram_to_id: dict[str, int] | None = None
        self._objective_spec = objective
        self._n_negatives = n_negatives
        self._hard_neg_frac = hard_neg_frac
        self._temperature = temperature
        self._t9_groups: "dict[int, list[int]] | None" = None

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _before_training(self) -> None:
        self._build_ngram_vocab_if_needed()
        self._build_t9_groups_if_needed()
        self._build_objective_if_needed()

    def _verbose_extra_fields(self) -> dict[str, object]:
        if self._ngram_to_id is None:
            return {}
        return {"n_ngrams": len(self._ngram_to_id) + 1}

    # ------------------------------------------------------------------
    # T9 hard negative groups
    # ------------------------------------------------------------------

    def _build_t9_groups_if_needed(self) -> None:
        """Precompute T9 ambiguity groups for hard negative mining.

        Groups map each word_id to the list of other word_ids that share the
        same T9 digit sequence (e.g. "home", "good", "gone", "hood" all map
        to "4663").  Passed to SGNSObjective so that half the negatives per
        positive are drawn from the target's T9 siblings.
        """
        if self._t9_groups is not None:
            return
        from ..t9_map import word_to_digits

        digit_to_wids: dict[str, list[int]] = {}
        for wid in range(1, self._vocab.size):  # skip UNK (id 0)
            word = self._vocab.id_to_word(wid)
            digits = word_to_digits(word)
            if digits is None:
                continue
            if digits not in digit_to_wids:
                digit_to_wids[digits] = []
            digit_to_wids[digits].append(wid)

        t9_groups: dict[int, list[int]] = {}
        for wids in digit_to_wids.values():
            if len(wids) < 2:
                continue
            for wid in wids:
                t9_groups[wid] = [w for w in wids if w != wid]

        n_ambiguous = sum(1 for v in t9_groups.values() if v)
        print(
            f"  T9 hard negatives: {n_ambiguous:,} words have ambiguous T9 siblings "
            f"(hard_neg_frac={self._hard_neg_frac})"
        )
        self._t9_groups = t9_groups

    # ------------------------------------------------------------------
    # Objective building
    # ------------------------------------------------------------------

    def _build_objective_if_needed(self) -> None:
        """Construct the training objective from ``self._objective_spec``."""
        if self._objective is not None:
            return
        from .objectives import SGNSObjective, CLIPObjective, OBJECTIVES

        spec = self._objective_spec
        if isinstance(spec, TrainingObjective):
            self._objective = spec
        elif spec == "sgns":
            self._objective = SGNSObjective(
                counts=self._vocab.counts,
                k=self._n_negatives,
                t9_groups=self._t9_groups,
                hard_neg_frac=self._hard_neg_frac,
            )
        elif spec == "clip":
            self._objective = CLIPObjective(temperature=self._temperature)
        else:
            raise ValueError(
                f"Unknown objective: {spec!r}. "
                f"Available: {list(OBJECTIVES)}"
            )

    # ------------------------------------------------------------------
    # N-gram vocabulary helpers
    # ------------------------------------------------------------------

    def _build_ngram_vocab_if_needed(self) -> None:
        if self._ngram_to_id is not None:
            return
        from .dual_encoder import build_ngram_vocab
        # Skip word_id=0 (<unk>): its n-grams would never receive gradient
        # (UNK is excluded from training targets) and would waste n-gram slots.
        words = [self._vocab.id_to_word(i) for i in range(1, self._vocab.size)]
        self._ngram_to_id = build_ngram_vocab(words, ns=self._ns)

    def _build_word_ngram_table(self, torch) -> "tuple[torch.Tensor, torch.Tensor]":
        """Precompute a padded (vocab_size, max_ngrams) word→n-gram lookup table."""
        from .dual_encoder import _char_ngrams
        vocab_size = self._vocab.size
        ng2id = self._ngram_to_id

        rows: list[list[int]] = []
        for wid in range(vocab_size):
            word = self._vocab.id_to_word(wid)
            ids = [ng2id.get(g, 0) for g in _char_ngrams(word, self._ns)]
            ids = [i for i in ids if i != 0] or [0]
            rows.append(ids)

        max_n = max(len(r) for r in rows)
        table = np.zeros((vocab_size, max_n), dtype=np.int64)
        counts = np.zeros(vocab_size, dtype=np.float32)
        for i, row in enumerate(rows):
            table[i, :len(row)] = row
            counts[i] = len(row)

        return (
            torch.from_numpy(table),
            torch.from_numpy(counts).unsqueeze(1),  # (vocab_size, 1)
        )

    # ------------------------------------------------------------------
    # Model building
    # ------------------------------------------------------------------

    def _build_model(self, torch, device):
        """Build the _CharNgramModel and move it to device."""
        vocab_size = self._vocab.size
        embed_dim = self._embed_dim
        n_ngrams = len(self._ngram_to_id) + 1

        t0 = time.monotonic()
        print(f"  [{time.strftime('%H:%M:%S')}] Building n-gram word table ({vocab_size} words, {n_ngrams} n-grams)…", flush=True)
        ngram_table, ngram_counts = self._build_word_ngram_table(torch)
        print(f"  [{time.strftime('%H:%M:%S')}] N-gram table built in {time.monotonic() - t0:.1f}s  "
              f"(table shape: {list(ngram_table.shape)})", flush=True)
        ngram_table = ngram_table.to(device)
        ngram_counts = ngram_counts.to(device)

        nn = torch.nn

        context_window = self._context_window

        class _CharNgramModel(nn.Module):
            def __init__(self_):
                super().__init__()
                self_.ctx_embed = nn.Embedding(n_ngrams, embed_dim, padding_idx=0)
                self_.wrd_embed = nn.Embedding(n_ngrams, embed_dim, padding_idx=0)
                # Positional embeddings: one vector per context slot.
                self_.pos_embed = nn.Embedding(context_window, embed_dim)
                # GRU context encoder: hidden_size = embed_dim.
                self_.gru = nn.GRU(embed_dim, embed_dim, batch_first=True)
                nn.init.xavier_uniform_(self_.ctx_embed.weight)
                nn.init.xavier_uniform_(self_.wrd_embed.weight)
                nn.init.xavier_uniform_(self_.pos_embed.weight)
                # Orthogonal init for GRU is a common best practice.
                for name, p in self_.gru.named_parameters():
                    if "weight" in name:
                        nn.init.orthogonal_(p)
                    elif "bias" in name:
                        nn.init.zeros_(p)
                with torch.no_grad():
                    self_.ctx_embed.weight[0] = 0
                    self_.wrd_embed.weight[0] = 0
                self_.register_buffer("_ng_table", ngram_table)
                self_.register_buffer("_ng_counts", ngram_counts)
                # Pre-computed position indices — avoids torch.arange() in forward.
                # persistent=False: deterministic from context_window, not part
                # of state_dict, so old checkpoints load without errors.
                self_.register_buffer(
                    "_slot_ids",
                    torch.arange(context_window).unsqueeze(0),  # (1, W)
                    persistent=False,
                )

            def _embed_words(self_, embed_layer, word_ids):
                """Mean-pool n-gram embeddings, then L2-normalise per word.

                Deduplicates word_ids before the embedding_bag call so that
                each unique word is embedded exactly once, then gathers
                results back to the original shape.  With large batches where
                B*k >> vocab_size (e.g. 490k neg IDs over a 50k vocab) the
                same word would otherwise be embedded ~10x redundantly —
                deduplication is the dominant speed win on the negative
                sampling path.
                """
                F = torch.nn.functional
                unique_ids, inverse = torch.unique(word_ids, return_inverse=True)
                ng_ids = self_._ng_table[unique_ids]          # (n_unique, max_ngrams)
                # Fused sum: avoids creating (n_unique, max_ngrams, dim) tensor.
                summed = F.embedding_bag(
                    ng_ids, embed_layer.weight,
                    mode="sum", padding_idx=0,
                )                                             # (n_unique, dim)
                counts = self_._ng_counts[unique_ids]         # (n_unique, 1)
                unique_vecs = F.normalize(summed / counts, dim=-1)
                return unique_vecs[inverse]                   # (n, dim)

            def embed_tgt_words(self_, word_ids):
                """Embed word IDs through the target (word) embedding table."""
                return self_._embed_words(self_.wrd_embed, word_ids)

            def forward(self_, ctx_ids, pos_ids):
                """
                ctx_ids : (batch, context_window) int — word IDs, 0 = padding
                pos_ids : (batch,) int — target word IDs
                """
                F = torch.nn.functional
                batch = ctx_ids.size(0)

                # ── Context encoding ──────────────────────────────────
                # 1. Embed each context word (L2-normalised per word).
                flat_ctx = ctx_ids.reshape(-1)                       # (B*W,)
                ctx_word_vecs = self_._embed_words(
                    self_.ctx_embed, flat_ctx
                ).reshape(batch, context_window, embed_dim)          # (B, W, dim)

                # 2. Add positional embeddings (slot_ids is a pre-built buffer).
                pos_vecs_ctx = self_.pos_embed(self_._slot_ids)      # (1, W, dim)
                ctx_input = ctx_word_vecs + pos_vecs_ctx             # (B, W, dim)

                # 3. Normalise each position's (word + pos) vector to unit norm.
                #    This decouples the GRU input scale from the relative
                #    magnitudes of word embeddings (unit-norm) vs positional
                #    embeddings (unconstrained), preventing either signal from
                #    dominating the other as training progresses.
                ctx_input = F.normalize(ctx_input, dim=-1)           # (B, W, dim)

                # 4. Mask padding positions: zero out the normalised padding
                #    vectors so the GRU treats them as empty slots.
                pad_mask = (ctx_ids != 0).float().unsqueeze(-1)      # (B, W, 1)
                ctx_input = ctx_input * pad_mask                     # (B, W, dim)

                # 4. GRU: use final hidden state as context vector.
                _, h_n = self_.gru(ctx_input)                        # h_n: (1, B, dim)
                ctx_vecs = F.normalize(h_n.squeeze(0), dim=-1)       # (B, dim)

                # ── Positive word embedding ───────────────────────────
                pos_word_vecs = self_._embed_words(self_.wrd_embed, pos_ids)  # (B, dim)

                return ctx_vecs, pos_word_vecs

        model = _CharNgramModel().to(device)
        compiled = model
        if hasattr(torch, "compile"):
            try:
                t1 = time.monotonic()
                print(f"  [{time.strftime('%H:%M:%S')}] torch.compile (max-autotune-no-cudagraphs)…", flush=True)
                compiled = torch.compile(model, mode="max-autotune-no-cudagraphs")
                print(f"  [{time.strftime('%H:%M:%S')}] torch.compile registered in {time.monotonic() - t1:.1f}s"
                      "  (kernel compilation deferred to first batch)", flush=True)
            except Exception:
                pass

        return model, compiled

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save_numpy(self, path) -> None:
        """Export trained weights to .npz for NumPy inference."""
        if self._model is None:
            raise RuntimeError("No trained model — call train_from_* first.")
        from .dual_encoder import DualEncoder
        m = self._model
        ctx = m.ctx_embed.weight.detach().cpu().numpy()
        wrd = m.wrd_embed.weight.detach().cpu().numpy()
        gru_weights = (
            m.gru.weight_ih_l0.detach().cpu().numpy(),
            m.gru.weight_hh_l0.detach().cpu().numpy(),
            m.gru.bias_ih_l0.detach().cpu().numpy(),
            m.gru.bias_hh_l0.detach().cpu().numpy(),
        )
        pos_embed = m.pos_embed.weight.detach().cpu().numpy()
        encoder = DualEncoder(
            ctx, wrd, self._ngram_to_id, self._vocab,
            ns=self._ns,
            gru_weights=gru_weights,
            pos_embed=pos_embed,
            context_window=self._context_window,
        )
        encoder.save(path)
        print(f"Saved model to {path}  ({Path(path).stat().st_size / 1e6:.1f} MB)")

    def get_encoder(self):
        """Return a DualEncoder with the current trained weights."""
        if self._model is None:
            raise RuntimeError("No trained model — call train_from_* first.")
        from .dual_encoder import DualEncoder
        m = self._model
        ctx = m.ctx_embed.weight.detach().cpu().numpy()
        wrd = m.wrd_embed.weight.detach().cpu().numpy()
        gru_weights = (
            m.gru.weight_ih_l0.detach().cpu().numpy(),
            m.gru.weight_hh_l0.detach().cpu().numpy(),
            m.gru.bias_ih_l0.detach().cpu().numpy(),
            m.gru.bias_hh_l0.detach().cpu().numpy(),
        )
        pos_embed = m.pos_embed.weight.detach().cpu().numpy()
        return DualEncoder(
            ctx, wrd, self._ngram_to_id, self._vocab,
            ns=self._ns,
            gru_weights=gru_weights,
            pos_embed=pos_embed,
            context_window=self._context_window,
        )
