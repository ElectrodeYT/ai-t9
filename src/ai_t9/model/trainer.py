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

def _corpus_file_sentence_ids(path: Path, vocab: Vocabulary) -> list[list[int]]:
    """Read a plain-text file (one sentence per line) and convert to word IDs.

    UNK tokens (ID 0) are excluded so they never appear as training
    targets or pollute context windows.
    """
    sentences = []
    unk = vocab.UNK_ID
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            words = line.strip().lower().split()
            ids = [vocab.word_to_id(w) for w in words if w.isalpha()]
            ids = [wid for wid in ids if wid != unk]
            if len(ids) >= 2:
                sentences.append(ids)
    return sentences


def _precompute_pairs(
    sentences: list[list[int]],
    context_window: int,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-compute all (context, target) training pairs as flat NumPy arrays.

    Called once before training.  The resulting arrays are shuffled per epoch
    via NumPy index permutation (fast, releases the GIL).

    Pairs where the target is UNK (ID 0) are skipped — they teach nothing
    useful and waste gradient signal.  (UNK tokens should already be stripped
    from sentences by the corpus loaders, but this is a safety net.)

    Uses vectorised NumPy operations per sentence (stride-trick context
    windows + boolean masking) instead of element-by-element Python writes.

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

    for si, sent in enumerate(sentences):
        n = len(sent)
        if n < 2:
            continue
        arr = np.array(sent, dtype=np.int64)

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
) -> int:
    """Precompute training pairs from sentences and persist them to .npz file(s).

    When ``max_shard_pairs`` is None (default), writes a single file at ``path``.
    When set, writes sharded files named ``path_000.npz``, ``path_001.npz``, …
    each containing at most ``max_shard_pairs`` rows — useful for corpora too
    large to fit in RAM or GPU VRAM at once.

    The file(s) store arrays as int32 (half the size of int64) and embed
    ``context_window`` and ``vocab_size`` as metadata so ``load_pairs()``
    can detect stale files built from a different vocab or window setting.

    Returns the total number of pairs written.
    """
    ctx_arr, pos_arr = _precompute_pairs(sentences, context_window, verbose=verbose)
    n_total = len(pos_arr)

    if max_shard_pairs is None:
        # Single-file path (original behaviour).
        _write_pairs_npz(ctx_arr, pos_arr, context_window, vocab_size, path)
        if verbose:
            p = Path(path) if str(path).endswith(".npz") else Path(str(path) + ".npz")
            print(f"  Saved {n_total:,} pairs → {p}  ({p.stat().st_size / 1e6:.1f} MB)")
        return n_total

    # Sharded path.
    path = Path(path)
    stem = path.stem if path.suffix == ".npz" else path.name
    parent = path.parent
    shard = 0
    written = 0
    while written < n_total:
        end = min(written + max_shard_pairs, n_total)
        shard_path = parent / f"{stem}_{shard:03d}.npz"
        _write_pairs_npz(
            ctx_arr[written:end], pos_arr[written:end],
            context_window, vocab_size, shard_path,
        )
        if verbose:
            print(f"  Shard {shard}: {end - written:,} pairs → {shard_path}")
        written = end
        shard += 1
    return n_total


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
    """
    buf = io.BytesIO()
    np.savez(
        buf,
        ctx=ctx_arr.astype(np.int32),
        pos=pos_arr.astype(np.int32),
        vocab_size=np.array(vocab_size, dtype=np.int64),
        context_window=np.array(context_window, dtype=np.int64),
    )
    buf.seek(0)
    path = Path(path)
    if not str(path).endswith(".npz"):
        path = Path(str(path) + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        shutil.copyfileobj(buf, f)


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
        self._model = None
        self._epoch = 0
        self._global_step = 0
        self._optimizer = None
        self._scheduler = None
        self._scaler = None
        self._objective: TrainingObjective | None = None

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

    def train_from_pairs_file(self, pairs_path: str | Path, epochs: int = 3, verbose: bool = True, checkpoint_path: str | Path | None = None) -> None:
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
        self._train_from_arrays(ctx_np, pos_np, epochs=epochs, torch=torch, verbose=verbose, checkpoint_path=checkpoint_path)

    def train_from_pairs_dir(
        self,
        pairs_dir: str | Path,
        pattern: str = "pairs_*.npz",
        epochs: int = 3,
        prefetch: bool = True,
        verbose: bool = True,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        """Train from a directory of sharded pairs .npz files.

        Shards are shuffled each epoch to improve gradient diversity.
        When ``prefetch=True``, the next shard is loaded on a background
        thread while the GPU trains on the current shard, overlapping I/O
        and compute.

        Args:
            pairs_dir: Directory containing shard files.
            pattern:   Glob pattern to match shard files (default ``pairs_*.npz``).
            epochs:    Number of full passes over all shards.
            prefetch:  Overlap CPU I/O with GPU compute via background thread.
            verbose:   Print progress.
        """
        torch = _require_torch()
        shard_paths = sorted(Path(pairs_dir).glob(pattern))
        if not shard_paths:
            raise FileNotFoundError(f"No files matching '{pattern}' in {pairs_dir}")
        if verbose:
            print(f"Found {len(shard_paths)} shard(s) in {pairs_dir}")
        self._before_training()
        self._train_from_shards(shard_paths, epochs=epochs, torch=torch, prefetch=prefetch, verbose=verbose)

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

    # ------------------------------------------------------------------
    # Internal training
    # ------------------------------------------------------------------

    def _train(self, sentences, epochs, torch, verbose, checkpoint_path=None):
        ctx_np, pos_np = _precompute_pairs(sentences, self._context_window, verbose=verbose)
        self._before_training()
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

    def _train_from_arrays(self, ctx_np, pos_np, epochs, torch, verbose, checkpoint_path=None):
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
        self._batch_size = _resolve_batch_size(
            self._batch_size, torch, device,
            objective=self._objective, verbose=verbose,
        )

        t_build = time.monotonic()
        model, compiled = self._build_model(torch, device)
        self._model = model
        model.train()
        if verbose:
            print(f"  [{time.strftime('%H:%M:%S')}] Model ready in {time.monotonic() - t_build:.1f}s", flush=True)

        # embed_fn for objectives that need to embed extra word IDs (e.g. SGNS
        # negative sampling).  Uses the *uncompiled* model so the call works
        # regardless of torch.compile graph boundaries.
        embed_fn = getattr(model, 'embed_tgt_words', None)

        # Load checkpoint if available
        if hasattr(self, '_checkpoint_data') and self._checkpoint_data:
            self._model.load_state_dict(self._checkpoint_data['model_state_dict'])
            self._epoch = self._checkpoint_data.get('epoch', 0)
            self._global_step = self._checkpoint_data.get('global_step', 0)

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

        # Load optimizer/scheduler states if checkpoint
        if hasattr(self, '_checkpoint_data') and self._checkpoint_data:
            if 'optimizer_state_dict' in self._checkpoint_data and self._checkpoint_data['optimizer_state_dict']:
                self._optimizer.load_state_dict(self._checkpoint_data['optimizer_state_dict'])
            if 'scheduler_state_dict' in self._checkpoint_data and self._checkpoint_data['scheduler_state_dict']:
                self._scheduler.load_state_dict(self._checkpoint_data['scheduler_state_dict'])
            if 'scaler_state_dict' in self._checkpoint_data and self._checkpoint_data['scaler_state_dict'] and self._scaler:
                self._scaler.load_state_dict(self._checkpoint_data['scaler_state_dict'])

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

        for epoch in range(self._epoch + 1, self._epoch + epochs + 1):
            self._epoch = epoch
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
                    desc=f"Epoch {epoch}/{self._epoch + epochs - 1}",
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
                    ctx_vecs, pos_vecs = compiled(ctx_dev[idx], pos_dev[idx])
                    loss = self._objective.compute_loss(
                        ctx_vecs, pos_vecs, embed_fn=embed_fn,
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

            elapsed = time.monotonic() - t_epoch
            avg_loss = running_loss.item() / max(n_batches, 1)  # one sync per epoch
            pairs_per_sec = n_pairs / elapsed if elapsed > 0 else 0
            if verbose:
                print(
                    f"  Epoch {epoch}/{self._epoch + epochs - 1}  loss={avg_loss:.4f}  "
                    f"time={elapsed:.1f}s  ({pairs_per_sec:,.0f} pairs/s)"
                )

            if checkpoint_path:
                self.save_checkpoint(checkpoint_path)

        self._global_step = global_step

        if verbose:
            _log_training_complete(t_train)

    def _train_from_shards(self, shard_paths, epochs, torch, prefetch, verbose):
        """Epoch loop over sharded pairs files with optional prefetch."""
        assert self._objective is not None

        device, is_cuda, amp_dtype, scaler = self._setup_device_and_amp(torch)
        self._objective.setup(device)
        self._batch_size = _resolve_batch_size(
            self._batch_size, torch, device,
            objective=self._objective, verbose=verbose,
        )

        t_build = time.monotonic()
        model, compiled = self._build_model(torch, device)
        self._model = model
        model.train()
        embed_fn = getattr(model, 'embed_tgt_words', None)
        if verbose:
            print(f"  [{time.strftime('%H:%M:%S')}] Model ready in {time.monotonic() - t_build:.1f}s", flush=True)

        # Estimate total steps from the first shard.
        if verbose:
            print(f"  [{time.strftime('%H:%M:%S')}] Loading first shard for step estimate…", flush=True)
        t_shard0 = time.monotonic()
        first_ctx, first_pos = load_pairs(
            shard_paths[0], context_window=self._context_window, vocab_size=self._vocab.size,
        )
        if verbose:
            print(f"  [{time.strftime('%H:%M:%S')}] First shard: {len(first_pos):,} pairs  "
                  f"(loaded in {time.monotonic() - t_shard0:.1f}s)", flush=True)
        n_pairs_estimate = len(first_pos) * len(shard_paths) * epochs
        total_steps = max(1, n_pairs_estimate // (self._batch_size * self._accumulate))
        warmup_steps = int(total_steps * self._warmup_frac)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, _make_lr_lambda(total_steps, warmup_steps, self._min_lr_frac)
        )

        if verbose:
            print(f"  [{time.strftime('%H:%M:%S')}] Starting training  "
                  f"(first batch triggers torch.compile kernel build — may take several minutes)", flush=True)

        t_train = time.monotonic()
        global_step = 0

        for epoch in range(1, epochs + 1):
            shard_order = list(shard_paths)
            random.shuffle(shard_order)
            epoch_loss = 0.0
            epoch_batches = 0

            prefetch_result: list = [None]
            prefetch_thread: threading.Thread | None = None

            def _load_shard(path, out):
                out[0] = load_pairs(path, context_window=self._context_window, vocab_size=self._vocab.size)

            for si, shard_path in enumerate(shard_order):
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
                if prefetch and si + 1 < len(shard_order):
                    prefetch_result = [None]
                    prefetch_thread = threading.Thread(
                        target=_load_shard, args=(shard_order[si + 1], prefetch_result), daemon=True
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
                ctx_vecs, pos_vecs = compiled(ctx_dev[idx], pos_dev[idx])
                loss = self._objective.compute_loss(
                    ctx_vecs, pos_vecs, embed_fn=embed_fn,
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
                    torch.nn.utils.clip_grad_norm_(compiled.parameters(), self._clip_grad_norm)
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

    The training objective is configurable:

    - ``"sgns"`` (default): Skip-Gram Negative Sampling — O(B×k), fast and
      well-suited for word embedding quality.
    - ``"clip"``: CLIP-style in-batch negatives — O(B²), useful for research.
    - Any :class:`TrainingObjective` instance for custom objectives.

    Usage::

        trainer = DualEncoderTrainer(vocab, embed_dim=64)
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
        temperature: float = 0.07,
    ) -> None:
        super().__init__(
            vocab, embed_dim, context_window, lr, weight_decay, warmup_frac,
            min_lr_frac, batch_size, accumulate_grad_batches,
            clip_grad_norm, seed, device, debug,
        )
        self._ns = ns
        self._ngram_to_id: dict[str, int] | None = None
        self._objective_spec = objective
        self._n_negatives = n_negatives
        self._temperature = temperature

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _before_training(self) -> None:
        self._build_ngram_vocab_if_needed()
        self._build_objective_if_needed()

    def _verbose_extra_fields(self) -> dict[str, object]:
        if self._ngram_to_id is None:
            return {}
        return {"n_ngrams": len(self._ngram_to_id) + 1}

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
                counts=self._vocab._counts,
                k=self._n_negatives,
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
        words = [self._vocab.id_to_word(i) for i in range(self._vocab.size)]
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

        class _CharNgramModel(nn.Module):
            def __init__(self_):
                super().__init__()
                self_.ctx_embed = nn.Embedding(n_ngrams, embed_dim, padding_idx=0)
                self_.wrd_embed = nn.Embedding(n_ngrams, embed_dim, padding_idx=0)
                nn.init.xavier_uniform_(self_.ctx_embed.weight)
                nn.init.xavier_uniform_(self_.wrd_embed.weight)
                with torch.no_grad():
                    self_.ctx_embed.weight[0] = 0
                    self_.wrd_embed.weight[0] = 0
                self_.register_buffer("_ng_table", ngram_table)
                self_.register_buffer("_ng_counts", ngram_counts)

            def _embed_words(self_, embed_layer, word_ids):
                """Mean-pool n-gram embeddings, then L2-normalise per word."""
                F = torch.nn.functional
                ng_ids = self_._ng_table[word_ids]           # (n, max_ngrams)
                vecs = embed_layer(ng_ids)                    # (n, max_ngrams, dim)
                counts = self_._ng_counts[word_ids]          # (n, 1)
                summed = vecs.sum(dim=1)                      # (n, dim)
                return F.normalize(summed / counts, dim=-1)   # (n, dim)

            def embed_tgt_words(self_, word_ids):
                """Embed word IDs through the target (word) embedding table.

                This is used by objectives that need to score additional words
                (e.g. SGNS negative sampling).  Uses the same n-gram mean-pool
                + L2-normalise path as the forward pass.
                """
                return self_._embed_words(self_.wrd_embed, word_ids)

            def forward(self_, ctx_ids, pos_ids):
                F = torch.nn.functional
                batch = ctx_ids.size(0)

                # Context: embed each word, mean-pool across context window, normalise.
                flat_ctx = ctx_ids.reshape(-1)
                ctx_word_vecs = self_._embed_words(
                    self_.ctx_embed, flat_ctx
                ).reshape(batch, -1, embed_dim)
                ctx_vecs = F.normalize(ctx_word_vecs.mean(1), dim=-1)   # (batch, dim)

                # Positive: embed target words, normalise.
                pos_vecs = self_._embed_words(self_.wrd_embed, pos_ids)  # (batch, dim)

                return ctx_vecs, pos_vecs

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
        """Export trained n-gram embeddings to .npz for inference."""
        if self._model is None:
            raise RuntimeError("No trained model — call train_from_* first.")
        from .dual_encoder import DualEncoder
        ctx = self._model.ctx_embed.weight.detach().cpu().numpy()
        wrd = self._model.wrd_embed.weight.detach().cpu().numpy()
        encoder = DualEncoder(ctx, wrd, self._ngram_to_id, self._vocab, ns=self._ns)
        encoder.save(path)
        print(f"Saved char-ngram model to {path}  ({Path(path).stat().st_size / 1e6:.1f} MB)")

    def get_encoder(self):
        """Return a CharNgramDualEncoder with the current trained weights."""
        if self._model is None:
            raise RuntimeError("No trained model — call train_from_* first.")
        from .dual_encoder import DualEncoder
        ctx = self._model.ctx_embed.weight.detach().cpu().numpy()
        wrd = self._model.wrd_embed.weight.detach().cpu().numpy()
        return DualEncoder(ctx, wrd, self._ngram_to_id, self._vocab, ns=self._ns)
