"""PyTorch training for the DualEncoder and CharNgramDualEncoder models.

This module is intentionally isolated from the inference path — PyTorch is an
optional dependency (pip install ai-t9[train]).  The output is a .npz file
that the pure-NumPy encoders can load without any ML framework.

Training objective: in-batch negative sampling (SimCLR / CLIP style)
  For each (context_words, target_word) pair drawn from the corpus:
    - Encode all contexts and targets in the batch → ctx_vecs, pos_vecs (B, dim)
    - Compute a (B, B) similarity matrix: logits = ctx_vecs @ pos_vecs.T / temperature
    - The diagonal entries are the correct (ctx_i, word_i) pairings
    - Loss: cross-entropy over rows (each context must identify its target word)

  This gives B-1 negatives per positive instead of the previous 20 fixed
  negatives, providing ~200× more gradient signal at the same compute cost
  when using large batches (B=4096+).

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


def _auto_batch_size(torch, device) -> int:
    """Pick the largest power-of-2 batch size whose (B,B) logits fit in VRAM.

    The dominant memory consumer for in-batch negative training is the
    (B, B) logits matrix and its gradients.  This function allocates ~50%
    of available GPU memory for that matrix, leaving room for model
    parameters, optimizer state, and training data.

    Falls back to 2048 on non-CUDA devices.
    """
    if not (str(device).startswith("cuda") and torch.cuda.is_available()):
        return 2048

    try:
        free, _total = torch.cuda.mem_get_info(device)
    except AttributeError:
        free = torch.cuda.get_device_properties(device).total_memory

    # Budget ~50% of free VRAM for the logits matrix + gradients.
    # Remaining goes to model params, optimizer state, data, CUDA overhead.
    # I genuinely do not know the best value for this, or if it is even possible to 
    # calculate this rather than just having to test it.
    logit_budget = int(free * 0.016)

    # During forward + backward the logits matrix occupies:
    #   (B, B) in AMP dtype (~2 bytes) + gradient (~2 bytes)
    #   + cross-entropy softmax intermediates (~4 bytes float32)
    #   ≈ 8 bytes per element total.
    max_b = int(math.sqrt(logit_budget / 8))

    if max_b < 2048:
        return 2048
    batch_size = 1 << int(math.log2(max_b))
    return min(batch_size, 131072)


def _resolve_batch_size(
    batch_size: int, torch, device, verbose: bool = False,
) -> int:
    """Return *batch_size* unchanged if positive, otherwise auto-detect."""
    if batch_size > 0:
        return batch_size
    auto = _auto_batch_size(torch, device)
    if verbose:
        print(f"  Auto batch size: {auto:,}")
    return auto


# ---------------------------------------------------------------------------
# Corpus iterator helpers
# ---------------------------------------------------------------------------

def _brown_sentence_ids(vocab: Vocabulary) -> list[list[int]]:
    """Return all Brown corpus sentences as lists of word IDs.

    UNK tokens (ID 0) are excluded so they never appear as training
    targets or pollute context windows.
    """
    import nltk
    try:
        nltk.data.find("corpora/brown")
    except LookupError:
        nltk.download("brown", quiet=True)
    from nltk.corpus import brown
    sentences = []
    unk = vocab.UNK_ID
    for sent in brown.sents():
        ids = [vocab.word_to_id(w.lower()) for w in sent if w.isalpha()]
        ids = [wid for wid in ids if wid != unk]
        if len(ids) >= 2:
            sentences.append(ids)
    return sentences


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
        strides = (padded.strides[0], padded.strides[0])
        windows = np.lib.stride_tricks.as_strided(
            padded, shape=(n, context_window), strides=strides,
        )

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


# ---------------------------------------------------------------------------
# DualEncoderTrainer
# ---------------------------------------------------------------------------

class DualEncoderTrainer:
    """Train a DualEncoder from a text corpus using in-batch negative sampling.

    Uses a standard nn.Module with torch.compile() for efficient GPU training.

    Training objective: in-batch negatives (SimCLR / CLIP style).
    Each batch of B pairs produces a (B, B) similarity matrix; the diagonal
    entries are positives and all off-diagonal entries are negatives, giving
    B-1 negatives per positive.  Cross-entropy is computed over rows.

    Optimiser: AdamW with cosine LR decay and linear warmup.

    Usage::

        vocab   = Vocabulary.build_from_nltk()
        trainer = DualEncoderTrainer(vocab, embed_dim=64)
        trainer.train_from_nltk(epochs=3)
        trainer.save_numpy("model.npz")

    The resulting .npz can be loaded with ``DualEncoder.load(path, vocab)``
    without PyTorch.
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
        temperature: float = 0.07,
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
        self._temperature = temperature
        self._batch_size = batch_size
        self._accumulate = max(1, accumulate_grad_batches)
        self._clip_grad_norm = clip_grad_norm
        self._seed = seed
        self._device_pref = device
        self._debug = debug
        self._model = None  # set after first train call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_from_nltk(self, epochs: int = 3, verbose: bool = True) -> None:
        """Train on the NLTK Brown corpus (auto-downloaded)."""
        torch = _require_torch()
        t0 = time.monotonic()
        if verbose:
            print("Loading Brown corpus…")
        sentences = _brown_sentence_ids(self._vocab)
        if verbose:
            print(f"  {len(sentences):,} sentences loaded  ({time.monotonic()-t0:.2f}s)")
        self._train(sentences, epochs=epochs, torch=torch, verbose=verbose)

    def train_from_file(self, corpus_path: str | Path, epochs: int = 3, verbose: bool = True) -> None:
        """Train on a plain-text file (one sentence per line)."""
        self.train_from_files([Path(corpus_path)], epochs=epochs, verbose=verbose)

    def train_from_files(self, paths: list[str | Path], epochs: int = 3, verbose: bool = True) -> None:
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
        self._train(sentences, epochs=epochs, torch=torch, verbose=verbose)

    def train_from_pairs_file(self, pairs_path: str | Path, epochs: int = 3, verbose: bool = True) -> None:
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
        self._train_from_arrays(ctx_np, pos_np, epochs=epochs, torch=torch, verbose=verbose)

    def train_from_pairs_dir(
        self,
        pairs_dir: str | Path,
        pattern: str = "pairs_*.npz",
        epochs: int = 3,
        prefetch: bool = True,
        verbose: bool = True,
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
        self._train_from_shards(shard_paths, epochs=epochs, torch=torch, prefetch=prefetch, verbose=verbose)

    def save_numpy(self, path: str | Path) -> None:
        """Export trained embeddings to .npz (NumPy format) for inference."""
        if self._model is None:
            raise RuntimeError("No trained model — call train_from_* first.")
        ctx = self._model.ctx_embed.weight.detach().cpu().numpy()
        wrd = self._model.wrd_embed.weight.detach().cpu().numpy()
        encoder = DualEncoder(ctx, wrd, self._vocab)
        encoder.save(path)
        print(f"Saved model to {path}  ({Path(path).stat().st_size / 1e6:.1f} MB)")

    def get_encoder(self) -> DualEncoder:
        """Return a DualEncoder with the current trained weights (no file I/O)."""
        if self._model is None:
            raise RuntimeError("No trained model — call train_from_* first.")
        ctx = self._model.ctx_embed.weight.detach().cpu().numpy()
        wrd = self._model.wrd_embed.weight.detach().cpu().numpy()
        return DualEncoder(ctx, wrd, self._vocab)

    # ------------------------------------------------------------------
    # Internal training
    # ------------------------------------------------------------------

    def _train(self, sentences, epochs, torch, verbose):
        ctx_np, pos_np = _precompute_pairs(sentences, self._context_window, verbose=verbose)
        self._train_from_arrays(ctx_np, pos_np, epochs=epochs, torch=torch, verbose=verbose)

    def _train_from_shards(self, shard_paths, epochs, torch, prefetch, verbose):
        """Epoch loop over sharded pairs files."""
        device = _resolve_device(torch, self._device_pref)
        is_cuda = str(device).startswith("cuda")
        torch.manual_seed(self._seed)

        if is_cuda and torch.cuda.is_available() and torch.cuda.get_device_properties(device).major >= 8:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        amp_dtype = None
        scaler = None
        if is_cuda:
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                amp_dtype = torch.float16
                scaler = torch.cuda.amp.GradScaler()

        self._batch_size = _resolve_batch_size(
            self._batch_size, torch, device, verbose=verbose,
        )

        vocab_size = self._vocab.size
        embed_dim = self._embed_dim

        nn = torch.nn

        class _Model(nn.Module):
            def __init__(self_):
                super().__init__()
                self_.ctx_embed = nn.Embedding(vocab_size, embed_dim)
                self_.wrd_embed = nn.Embedding(vocab_size, embed_dim)
                nn.init.xavier_uniform_(self_.ctx_embed.weight)
                nn.init.xavier_uniform_(self_.wrd_embed.weight)

            def forward(self_, ctx_ids, pos_ids, temperature):
                F = torch.nn.functional
                ctx_vecs = F.normalize(self_.ctx_embed(ctx_ids).mean(1), dim=-1)
                pos_vecs = F.normalize(self_.wrd_embed(pos_ids), dim=-1)
                logits = ctx_vecs @ pos_vecs.T / temperature
                return logits

        model = _Model().to(device)
        self._model = model
        compiled = model
        if hasattr(torch, "compile"):
            try:
                compiled = torch.compile(model, mode="max-autotune")
            except Exception:
                pass

        # Compute total steps as sum of (n_batches_per_shard) × epochs.
        # Use the first shard to estimate; real step count tracked during training.
        first_ctx, first_pos = load_pairs(shard_paths[0], context_window=self._context_window, vocab_size=vocab_size)
        n_pairs_estimate = len(first_pos) * len(shard_paths) * epochs
        n_batches_estimate = max(1, n_pairs_estimate // self._batch_size)
        total_steps = n_batches_estimate
        warmup_steps = int(total_steps * self._warmup_frac)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, _make_lr_lambda(total_steps, warmup_steps, self._min_lr_frac)
        )
        ce_loss = nn.CrossEntropyLoss()
        temperature = self._temperature

        global_step = 0
        t_train = time.monotonic()

        for epoch in range(1, epochs + 1):
            shard_order = list(shard_paths)
            random.shuffle(shard_order)
            epoch_loss = 0.0
            epoch_batches = 0

            # Prefetch state
            prefetch_result: list = [None]
            prefetch_thread: threading.Thread | None = None

            def _load_shard(path, out):
                out[0] = load_pairs(path, context_window=self._context_window, vocab_size=vocab_size)

            for si, shard_path in enumerate(shard_order):
                # Wait for prefetch if active, otherwise load directly.
                if prefetch_thread is not None:
                    prefetch_thread.join()
                    ctx_np, pos_np = prefetch_result[0]
                else:
                    ctx_np, pos_np = load_pairs(shard_path, context_window=self._context_window, vocab_size=vocab_size)

                # Start prefetch of next shard.
                if prefetch and si + 1 < len(shard_order):
                    prefetch_result = [None]
                    prefetch_thread = threading.Thread(
                        target=_load_shard, args=(shard_order[si + 1], prefetch_result), daemon=True
                    )
                    prefetch_thread.start()
                else:
                    prefetch_thread = None

                shard_loss, shard_batches, global_step = self._train_shard(
                    ctx_np, pos_np, compiled, optimizer, scheduler, ce_loss,
                    temperature, device, global_step, verbose=(verbose and si == 0),
                    amp_dtype=amp_dtype, scaler=scaler,
                )
                epoch_loss += shard_loss
                epoch_batches += shard_batches

            avg_loss = epoch_loss / max(epoch_batches, 1)
            if verbose:
                print(f"  Epoch {epoch}/{epochs}  loss={avg_loss:.4f}")

        if verbose:
            total_time = time.monotonic() - t_train
            m, s = divmod(int(total_time), 60)
            h, m = divmod(m, 60)
            time_str = f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")
            print(f"  Training complete in {time_str}")

    def _train_shard(self, ctx_np, pos_np, compiled, optimizer, scheduler, ce_loss,
                     temperature, device, global_step, verbose=False,
                     amp_dtype=None, scaler=None):
        """Train a single shard of pairs. Returns (shard_loss_float, n_batches, global_step)."""
        import torch
        is_cuda = str(device).startswith("cuda")
        if is_cuda:
            ctx_dev = torch.from_numpy(ctx_np).pin_memory().to(device, non_blocking=True)
            pos_dev = torch.from_numpy(pos_np).pin_memory().to(device, non_blocking=True)
            torch.cuda.synchronize(device)
        else:
            ctx_dev = torch.from_numpy(ctx_np).to(device)
            pos_dev = torch.from_numpy(pos_np).to(device)
        n_pairs = len(pos_np)
        n_batches = max(1, n_pairs // self._batch_size)
        perm = torch.randperm(n_pairs, device=device)
        labels_full = torch.arange(self._batch_size, device=device)
        running_loss = torch.zeros(1, device=device)
        optimizer.zero_grad(set_to_none=True)

        for b in range(n_batches):
            start = b * self._batch_size
            end = min(start + self._batch_size, n_pairs)
            idx = perm[start:end]

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = compiled(ctx_dev[idx], pos_dev[idx], temperature)
                labels = labels_full[:end - start]
                loss = ce_loss(logits, labels) / self._accumulate

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
                    torch.nn.utils.clip_grad_norm_(compiled.parameters(), self._clip_grad_norm)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        del ctx_dev, pos_dev
        return running_loss.item(), n_batches, global_step

    def _train_from_arrays(self, ctx_np, pos_np, epochs, torch, verbose):
        """Core training loop operating on pre-loaded numpy pair arrays."""
        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            _tqdm = None

        device = _resolve_device(torch, self._device_pref)
        is_cuda = str(device).startswith("cuda")
        torch.manual_seed(self._seed)

        if is_cuda and torch.cuda.is_available() and torch.cuda.get_device_properties(device).major >= 8:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        amp_dtype = None
        scaler = None
        if is_cuda:
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                amp_dtype = torch.float16
                scaler = torch.cuda.amp.GradScaler()

        self._batch_size = _resolve_batch_size(
            self._batch_size, torch, device, verbose=verbose,
        )

        vocab_size = self._vocab.size
        embed_dim = self._embed_dim
        temperature = self._temperature

        nn = torch.nn

        class _DualEncoderModel(nn.Module):
            def __init__(self_):
                super().__init__()
                self_.ctx_embed = nn.Embedding(vocab_size, embed_dim)
                self_.wrd_embed = nn.Embedding(vocab_size, embed_dim)
                nn.init.xavier_uniform_(self_.ctx_embed.weight)
                nn.init.xavier_uniform_(self_.wrd_embed.weight)

            def forward(self_, ctx_ids, pos_ids):
                F = torch.nn.functional
                # Context vector: mean-pool then L2-normalise → (batch, dim)
                ctx_vecs = F.normalize(self_.ctx_embed(ctx_ids).mean(1), dim=-1)
                # Positive word vectors → (batch, dim)
                pos_vecs = F.normalize(self_.wrd_embed(pos_ids), dim=-1)
                # (batch, batch) similarity matrix; diagonal = correct pairs
                return ctx_vecs @ pos_vecs.T / temperature

        model = _DualEncoderModel().to(device)
        self._model = model

        compiled = model
        if hasattr(torch, "compile"):
            try:
                compiled = torch.compile(model, mode="max-autotune")
            except Exception:
                pass

        n_pairs = len(pos_np)
        n_batches = max(1, n_pairs // self._batch_size)
        effective_batch = self._batch_size * self._accumulate
        total_steps = math.ceil(n_batches / self._accumulate) * epochs
        warmup_steps = int(total_steps * self._warmup_frac)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, _make_lr_lambda(total_steps, warmup_steps, self._min_lr_frac)
        )
        ce_loss = nn.CrossEntropyLoss()

        ctx_dev = torch.from_numpy(ctx_np).to(device)
        pos_dev = torch.from_numpy(pos_np).to(device)
        del ctx_np, pos_np  # free CPU copies

        if verbose:
            if is_cuda:
                vram_alloc_mb = torch.cuda.memory_allocated(device) / 1e6
                vram_total_mb = torch.cuda.get_device_properties(device).total_memory / 1e6
                vram_str = f"VRAM: {vram_alloc_mb:.0f} / {vram_total_mb:.0f} MB"
            else:
                vram_str = f"data: {(ctx_dev.nbytes + pos_dev.nbytes) / 1e6:.0f} MB"
            compiled_str = "on" if compiled is not model else "off"
            amp_str = str(amp_dtype).replace("torch.", "") if amp_dtype is not None else "fp32"
            print(
                f"Device: {device}  |  pairs: {n_pairs:,}  |  "
                f"vocab: {vocab_size}  |  embed_dim: {embed_dim}  |  "
                f"temperature: {temperature}  |  "
                f"batch: {self._batch_size}  |  accumulate: {self._accumulate}  |  "
                f"effective_batch: {effective_batch}  |  "
                f"torch.compile: {compiled_str}  |  amp: {amp_str}  |  {vram_str}"
            )

        # Pre-allocate the full-batch labels tensor once; slice for the last
        # (potentially smaller) batch.  Avoids a torch.arange() allocation on
        # every iteration of the hot loop.
        labels_full = torch.arange(self._batch_size, device=device)

        t_train = time.monotonic()
        global_step = 0

        for epoch in range(1, epochs + 1):
            t_epoch = time.monotonic()
            # Accumulate loss on GPU to avoid a blocking GPU→CPU sync every batch.
            # A single .item() call per epoch (at reporting time) is all we need.
            running_loss = torch.zeros(1, device=device)
            perm = torch.randperm(n_pairs, device=device)
            display_step = 0

            batch_range: object = range(n_batches)
            if _tqdm is not None and verbose:
                batch_range = _tqdm(
                    batch_range,
                    desc=f"Epoch {epoch}/{epochs}",
                    unit="batch",
                    leave=False,
                )

            optimizer.zero_grad(set_to_none=True)

            for b in batch_range:
                start = b * self._batch_size
                end = min(start + self._batch_size, n_pairs)
                idx = perm[start:end]

                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=amp_dtype is not None):
                    logits = compiled(ctx_dev[idx], pos_dev[idx])
                    labels = labels_full[:end - start]
                    loss = ce_loss(logits, labels) / self._accumulate

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
                    if _tqdm is not None and verbose and display_step % 20 == 0:
                        batch_range.set_postfix(
                            loss=f"{running_loss.item() / (b + 1):.4f}"
                        )

            elapsed = time.monotonic() - t_epoch
            avg_loss = running_loss.item() / max(n_batches, 1)  # one sync per epoch
            pairs_per_sec = n_pairs / elapsed if elapsed > 0 else 0
            if verbose:
                print(
                    f"  Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  "
                    f"time={elapsed:.1f}s  ({pairs_per_sec:,.0f} pairs/s)"
                )

        if verbose:
            total_time = time.monotonic() - t_train
            m, s = divmod(int(total_time), 60)
            h, m = divmod(m, 60)
            time_str = f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")
            print(f"  Training complete in {time_str}")


# ---------------------------------------------------------------------------
# CharNgramDualEncoderTrainer
# ---------------------------------------------------------------------------

class CharNgramDualEncoderTrainer:
    """Train a CharNgramDualEncoder using in-batch negative sampling.

    The training data pipeline is identical to DualEncoderTrainer — the same
    (context_word_ids, target_word_id) pairs, the same in-batch negatives and
    cross-entropy loss.  The key architectural difference is in the forward pass:
    each word ID is first expanded to its character n-gram IDs, then those
    n-gram embeddings are mean-pooled to produce the word vector.

    Usage::

        trainer = CharNgramDualEncoderTrainer(vocab, embed_dim=64)
        trainer.train_from_files(corpus_files, epochs=5)
        trainer.save_numpy("data/model_ngram.npz")
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
        temperature: float = 0.07,
        batch_size: int = 0,
        accumulate_grad_batches: int = 1,
        clip_grad_norm: float = 1.0,
        seed: int = 42,
        device: str = "auto",
        debug: bool = False,
        ns: tuple[int, ...] = (2, 3),
    ) -> None:
        self._vocab = vocab
        self._embed_dim = embed_dim
        self._context_window = context_window
        self._lr = lr
        self._weight_decay = weight_decay
        self._warmup_frac = warmup_frac
        self._min_lr_frac = min_lr_frac
        self._temperature = temperature
        self._batch_size = batch_size
        self._accumulate = max(1, accumulate_grad_batches)
        self._clip_grad_norm = clip_grad_norm
        self._seed = seed
        self._device_pref = device
        self._debug = debug
        self._ns = ns
        self._model = None
        self._ngram_to_id: dict[str, int] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_from_nltk(self, epochs: int = 3, verbose: bool = True) -> None:
        torch = _require_torch()
        if verbose:
            print("Loading Brown corpus…")
        sentences = _brown_sentence_ids(self._vocab)
        if verbose:
            print(f"  {len(sentences):,} sentences loaded")
        self._train(sentences, epochs=epochs, torch=torch, verbose=verbose)

    def train_from_files(self, paths: list, epochs: int = 3, verbose: bool = True) -> None:
        torch = _require_torch()
        sentences: list[list[int]] = []
        if verbose:
            print("Loading corpus...")
        for i, path in enumerate(paths):
            file_sents = _corpus_file_sentence_ids(Path(path), self._vocab)
            sentences.extend(file_sents)
            if verbose:
                print(f"\r  Loaded {i+1}/{len(paths)} files ({len(sentences):,} sentences)", end="", flush=True)
        if verbose:
            print()
        self._train(sentences, epochs=epochs, torch=torch, verbose=verbose)

    def train_from_pairs_file(self, pairs_path, epochs: int = 3, verbose: bool = True) -> None:
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
        self._build_ngram_vocab_if_needed()
        self._train_from_arrays(ctx_np, pos_np, epochs=epochs, torch=torch, verbose=verbose)

    def train_from_pairs_dir(
        self,
        pairs_dir: str | Path,
        pattern: str = "pairs_*.npz",
        epochs: int = 3,
        prefetch: bool = True,
        verbose: bool = True,
    ) -> None:
        """Train from a directory of sharded pairs .npz files."""
        torch = _require_torch()
        shard_paths = sorted(Path(pairs_dir).glob(pattern))
        if not shard_paths:
            raise FileNotFoundError(f"No files matching '{pattern}' in {pairs_dir}")
        if verbose:
            print(f"Found {len(shard_paths)} shard(s) in {pairs_dir}")
        self._build_ngram_vocab_if_needed()
        # Reuse the shard training logic from DualEncoderTrainer by building the
        # model first, then delegating to a common _train_from_shards helper.
        # For simplicity in this implementation, load and concatenate all shards
        # per epoch (sufficient for typical shard counts of <100).
        self._train_shards_sequential(shard_paths, epochs=epochs, torch=torch, prefetch=prefetch, verbose=verbose)

    def save_numpy(self, path) -> None:
        """Export trained n-gram embeddings to .npz for inference."""
        if self._model is None:
            raise RuntimeError("No trained model — call train_from_* first.")
        from .char_ngram_encoder import CharNgramDualEncoder
        ctx = self._model.ctx_embed.weight.detach().cpu().numpy()
        wrd = self._model.wrd_embed.weight.detach().cpu().numpy()
        encoder = CharNgramDualEncoder(ctx, wrd, self._ngram_to_id, self._vocab, ns=self._ns)
        encoder.save(path)
        print(f"Saved char-ngram model to {path}  ({Path(path).stat().st_size / 1e6:.1f} MB)")

    def get_encoder(self):
        """Return a CharNgramDualEncoder with the current trained weights."""
        if self._model is None:
            raise RuntimeError("No trained model — call train_from_* first.")
        from .char_ngram_encoder import CharNgramDualEncoder
        ctx = self._model.ctx_embed.weight.detach().cpu().numpy()
        wrd = self._model.wrd_embed.weight.detach().cpu().numpy()
        return CharNgramDualEncoder(ctx, wrd, self._ngram_to_id, self._vocab, ns=self._ns)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_ngram_vocab_if_needed(self) -> None:
        if self._ngram_to_id is not None:
            return
        from .char_ngram_encoder import build_ngram_vocab
        words = [self._vocab.id_to_word(i) for i in range(self._vocab.size)]
        self._ngram_to_id = build_ngram_vocab(words, ns=self._ns)

    def _build_word_ngram_table(self, torch) -> "torch.Tensor":
        """Precompute a padded (vocab_size, max_ngrams) word→n-gram lookup table."""
        from .char_ngram_encoder import _char_ngrams
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
        for i, row in enumerate(rows):
            table[i, :len(row)] = row

        return torch.from_numpy(table)

    def _train(self, sentences, epochs, torch, verbose):
        ctx_np, pos_np = _precompute_pairs(sentences, self._context_window, verbose=verbose)
        self._build_ngram_vocab_if_needed()
        self._train_from_arrays(ctx_np, pos_np, epochs=epochs, torch=torch, verbose=verbose)

    def _train_shards_sequential(self, shard_paths, epochs, torch, prefetch, verbose):
        """Train over shards. Loads each shard per epoch."""
        self._build_ngram_vocab_if_needed()
        device = _resolve_device(torch, self._device_pref)
        is_cuda = str(device).startswith("cuda")
        torch.manual_seed(self._seed)

        if is_cuda and torch.cuda.is_available() and torch.cuda.get_device_properties(device).major >= 8:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        amp_dtype = None
        scaler = None
        if is_cuda:
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                amp_dtype = torch.float16
                scaler = torch.cuda.amp.GradScaler()

        self._batch_size = _resolve_batch_size(
            self._batch_size, torch, device, verbose=verbose,
        )

        model, compiled = self._build_model(torch, device)
        self._model = model

        n_pairs_estimate = sum(
            len(load_pairs(p, context_window=self._context_window, vocab_size=self._vocab.size)[1])
            for p in shard_paths[:1]  # use first shard as estimate
        ) * len(shard_paths) * epochs
        total_steps = max(1, n_pairs_estimate // (self._batch_size * self._accumulate))
        warmup_steps = int(total_steps * self._warmup_frac)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, _make_lr_lambda(total_steps, warmup_steps, self._min_lr_frac)
        )
        ce_loss = torch.nn.CrossEntropyLoss()
        temperature = self._temperature

        t_train = time.monotonic()
        global_step = 0

        for epoch in range(1, epochs + 1):
            order = list(shard_paths)
            random.shuffle(order)
            epoch_loss = 0.0
            epoch_batches = 0

            prefetch_result: list = [None]
            prefetch_thread: threading.Thread | None = None

            def _load(path, out):
                out[0] = load_pairs(path, context_window=self._context_window, vocab_size=self._vocab.size)

            for si, shard_path in enumerate(order):
                if prefetch_thread is not None:
                    prefetch_thread.join()
                    ctx_np, pos_np = prefetch_result[0]
                else:
                    ctx_np, pos_np = load_pairs(shard_path, context_window=self._context_window, vocab_size=self._vocab.size)

                if prefetch and si + 1 < len(order):
                    prefetch_result = [None]
                    prefetch_thread = threading.Thread(target=_load, args=(order[si + 1], prefetch_result), daemon=True)
                    prefetch_thread.start()
                else:
                    prefetch_thread = None

                sl, sb, global_step = self._train_shard_arrays(
                    ctx_np, pos_np, compiled, optimizer, scheduler, ce_loss, temperature, device, global_step,
                    amp_dtype=amp_dtype, scaler=scaler,
                )
                epoch_loss += sl
                epoch_batches += sb

            if verbose:
                avg = epoch_loss / max(epoch_batches, 1)
                print(f"  Epoch {epoch}/{epochs}  loss={avg:.4f}")

        if verbose:
            total_time = time.monotonic() - t_train
            m, s = divmod(int(total_time), 60)
            h, m = divmod(m, 60)
            time_str = f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")
            print(f"  Training complete in {time_str}")

    def _train_shard_arrays(self, ctx_np, pos_np, compiled, optimizer, scheduler, ce_loss, temperature, device, global_step,
                            amp_dtype=None, scaler=None):
        import torch
        is_cuda = str(device).startswith("cuda")
        if is_cuda:
            ctx_dev = torch.from_numpy(ctx_np).pin_memory().to(device, non_blocking=True)
            pos_dev = torch.from_numpy(pos_np).pin_memory().to(device, non_blocking=True)
            torch.cuda.synchronize(device)
        else:
            ctx_dev = torch.from_numpy(ctx_np).to(device)
            pos_dev = torch.from_numpy(pos_np).to(device)
        n_pairs = len(pos_np)
        n_batches = max(1, n_pairs // self._batch_size)
        perm = torch.randperm(n_pairs, device=device)
        labels_full = torch.arange(self._batch_size, device=device)
        running_loss = torch.zeros(1, device=device)
        optimizer.zero_grad(set_to_none=True)

        for b in range(n_batches):
            start = b * self._batch_size
            end = min(start + self._batch_size, n_pairs)
            idx = perm[start:end]

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = compiled(ctx_dev[idx], pos_dev[idx])
                labels = labels_full[:end - start]
                loss = ce_loss(logits, labels) / self._accumulate

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

        del ctx_dev, pos_dev
        return running_loss.item(), n_batches, global_step

    def _build_model(self, torch, device):
        """Build the _CharNgramModel and move it to device."""
        vocab_size = self._vocab.size
        embed_dim = self._embed_dim
        n_ngrams = len(self._ngram_to_id) + 1
        temperature = self._temperature

        ngram_table = self._build_word_ngram_table(torch)
        ngram_table = ngram_table.to(device)

        nn = torch.nn

        class _CharNgramModel(nn.Module):
            def __init__(self_):
                super().__init__()
                self_.ctx_embed = nn.EmbeddingBag(n_ngrams, embed_dim, mode='mean', padding_idx=0)
                self_.wrd_embed = nn.EmbeddingBag(n_ngrams, embed_dim, mode='mean', padding_idx=0)
                nn.init.xavier_uniform_(self_.ctx_embed.weight)
                nn.init.xavier_uniform_(self_.wrd_embed.weight)
                with torch.no_grad():
                    self_.ctx_embed.weight[0] = 0
                    self_.wrd_embed.weight[0] = 0
                self_.register_buffer("_ng_table", ngram_table)

            def _embed_words(self_, embed_layer, word_ids):
                """Mean-pool n-gram embeddings via EmbeddingBag, then L2-normalise."""
                F = torch.nn.functional
                ng_ids = self_._ng_table[word_ids]           # (n, max_ngrams)
                word_vecs = embed_layer(ng_ids)               # (n, dim) — fused
                return F.normalize(word_vecs, dim=-1)          # (n, dim)

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

                # (batch, batch) similarity matrix; diagonal = correct pairs.
                return ctx_vecs @ pos_vecs.T / temperature

        model = _CharNgramModel().to(device)
        compiled = model
        if hasattr(torch, "compile"):
            try:
                compiled = torch.compile(model, mode="max-autotune")
            except Exception:
                pass

        return model, compiled

    def _train_from_arrays(self, ctx_np, pos_np, epochs, torch, verbose):
        """Core training loop for the char-ngram model."""
        try:
            from tqdm import tqdm as _tqdm
        except ImportError:
            _tqdm = None

        device = _resolve_device(torch, self._device_pref)
        is_cuda = str(device).startswith("cuda")
        torch.manual_seed(self._seed)

        if is_cuda and torch.cuda.is_available() and torch.cuda.get_device_properties(device).major >= 8:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        amp_dtype = None
        scaler = None
        if is_cuda:
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                amp_dtype = torch.float16
                scaler = torch.cuda.amp.GradScaler()

        self._batch_size = _resolve_batch_size(
            self._batch_size, torch, device, verbose=verbose,
        )

        model, compiled = self._build_model(torch, device)
        self._model = model

        n_pairs = len(pos_np)
        n_batches = max(1, n_pairs // self._batch_size)
        effective_batch = self._batch_size * self._accumulate
        total_steps = math.ceil(n_batches / self._accumulate) * epochs
        warmup_steps = int(total_steps * self._warmup_frac)
        temperature = self._temperature

        optimizer = torch.optim.AdamW(model.parameters(), lr=self._lr, weight_decay=self._weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, _make_lr_lambda(total_steps, warmup_steps, self._min_lr_frac)
        )
        ce_loss = torch.nn.CrossEntropyLoss()

        ctx_dev = torch.from_numpy(ctx_np).to(device)
        pos_dev = torch.from_numpy(pos_np).to(device)
        del ctx_np, pos_np

        if verbose:
            vocab_size = self._vocab.size
            n_ngrams = len(self._ngram_to_id) + 1
            if is_cuda:
                vram_mb = torch.cuda.memory_allocated(device) / 1e6
                vram_total = torch.cuda.get_device_properties(device).total_memory / 1e6
                mem_str = f"VRAM: {vram_mb:.0f} / {vram_total:.0f} MB"
            else:
                mem_str = f"data: {(ctx_dev.nbytes + pos_dev.nbytes) / 1e6:.0f} MB"
            compiled_str = "on" if compiled is not model else "off"
            amp_str = str(amp_dtype).replace("torch.", "") if amp_dtype else "fp32"
            print(
                f"Device: {device}  |  pairs: {n_pairs:,}  |  vocab: {vocab_size}  |  "
                f"n_ngrams: {n_ngrams}  |  embed_dim: {self._embed_dim}  |  "
                f"temperature: {temperature}  |  batch: {self._batch_size}  |  "
                f"accumulate: {self._accumulate}  |  effective_batch: {effective_batch}  |  "
                f"torch.compile: {compiled_str}  |  amp: {amp_str}  |  {mem_str}"
            )

        labels_full = torch.arange(self._batch_size, device=device)

        t_train = time.monotonic()
        global_step = 0

        for epoch in range(1, epochs + 1):
            t_epoch = time.monotonic()
            running_loss = torch.zeros(1, device=device)
            perm = torch.randperm(n_pairs, device=device)
            display_step = 0

            batch_range: object = range(n_batches)
            if _tqdm is not None and verbose:
                batch_range = _tqdm(
                    batch_range,
                    desc=f"Epoch {epoch}/{epochs}",
                    unit="batch",
                    leave=False,
                )

            optimizer.zero_grad(set_to_none=True)

            for b in batch_range:
                start = b * self._batch_size
                end = min(start + self._batch_size, n_pairs)
                idx = perm[start:end]

                with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                    enabled=amp_dtype is not None):
                    logits = compiled(ctx_dev[idx], pos_dev[idx])
                    labels = labels_full[:end - start]
                    loss = ce_loss(logits, labels) / self._accumulate

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
                    if _tqdm is not None and verbose and display_step % 20 == 0:
                        batch_range.set_postfix(
                            loss=f"{running_loss.item() / (b + 1):.4f}"
                        )

            elapsed = time.monotonic() - t_epoch
            avg_loss = running_loss.item() / max(n_batches, 1)
            pairs_sec = n_pairs / elapsed if elapsed > 0 else 0
            if verbose:
                print(
                    f"  Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  "
                    f"time={elapsed:.1f}s  ({pairs_sec:,.0f} pairs/s)"
                )

        if verbose:
            total_time = time.monotonic() - t_train
            m, s = divmod(int(total_time), 60)
            h, m = divmod(m, 60)
            time_str = f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")
            print(f"  Training complete in {time_str}")
