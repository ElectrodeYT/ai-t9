"""PyTorch training for the DualEncoder model.

This module is intentionally isolated from the inference path — PyTorch is an
optional dependency (pip install ai-t9[train]).  The output is a .npz file
that the pure-NumPy DualEncoder can load without any ML framework.

Training objective: candidate-only negative sampling
  For each (context_words, target_word) pair drawn from the corpus:
    - Score the positive (target) word against the context embedding
    - Sample `neg_samples` random negatives *on the GPU* with torch.randint
    - Apply binary cross-entropy (positive=1, negatives=0)

GPU utilisation design
-----------------------
1. Pre-computed pairs — all (context, target) pairs are materialised as flat
   NumPy arrays *once* before training.  Each epoch shuffles a lightweight
   index array in-place (no allocation); batches are gathered via small
   fancy-index slices (~50 KB each) rather than copying the full dataset.
   This eliminates GIL contention and the ~500 MB per-epoch copy+pin that
   previously saturated the CPU.

2. On-GPU negative sampling — torch.randint generates all negative IDs
   directly on the GPU; only (ctx_ids, pos_ids) are transferred from CPU,
   roughly 20× less data per batch than before.

3. Pinned memory + non-blocking transfers (CUDA only) — double-buffered
   pre-allocated pinned tensors avoid per-batch cudaHostAlloc/Free OS calls;
   batch data is DMA-transferred asynchronously.

4. Automatic mixed precision (AMP) — on CUDA, forward passes run under
   torch.amp.autocast, casting torch.bmm to float16.  GradScaler is
   intentionally omitted because this model produces no float16 gradients
   (only bmm is autocasted; everything else stays float32), and removing it
   eliminates a per-batch CPU↔GPU sync caused by its internal .item() call
   (CUDA defaults to spin-wait synchronisation, which pins a CPU core).

5. GPU-side loss accumulation — training loss is accumulated in a device
   tensor and only transferred to CPU for progress-bar redraws (~4×/s),
   avoiding ~1000 per-batch spin-wait sync points per second.

6. TF32 matmul (Ampere+ GPUs) — float32 matrix multiplications use
   TensorFloat-32 for ~3× speed at effectively no accuracy cost.
"""

from __future__ import annotations

import math
import random
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


# ---------------------------------------------------------------------------
# PyTorch model (training-time only)
# ---------------------------------------------------------------------------

class _TorchDualEncoder:
    """Two nn.Embedding tables with on-GPU negative sampling.

    The key method is score_with_negatives(), which generates all negative
    candidate IDs with a single torch.randint() call on the GPU, avoiding
    any Python loop for negatives.
    """

    def __init__(self, vocab_size: int, embed_dim: int, neg_samples: int, torch, device):
        self._vocab_size = vocab_size
        self._neg_samples = neg_samples
        nn = torch.nn
        self.ctx_embed = nn.Embedding(vocab_size, embed_dim).to(device)
        self.wrd_embed = nn.Embedding(vocab_size, embed_dim).to(device)
        nn.init.xavier_uniform_(self.ctx_embed.weight)
        nn.init.xavier_uniform_(self.wrd_embed.weight)

    def parameters(self):
        return list(self.ctx_embed.parameters()) + list(self.wrd_embed.parameters())

    def score_with_negatives(self, ctx_ids, pos_ids, torch):
        """Compute scores for positives and GPU-generated negatives.

        Args:
            ctx_ids: LongTensor (batch, ctx_len) — context word IDs
            pos_ids: LongTensor (batch,)         — positive target word IDs

        Returns:
            logits: FloatTensor (batch, 1 + neg_samples)
            labels: FloatTensor (batch, 1 + neg_samples) — 1.0 at index 0 only
        """
        F = torch.nn.functional
        device = ctx_ids.device
        batch = ctx_ids.size(0)

        # Context vector: mean-pool then L2-normalise  → (batch, dim)
        ctx_vecs = F.normalize(self.ctx_embed(ctx_ids).mean(dim=1), dim=-1)

        # Positive scores  → (batch, 1)
        pos_vecs   = F.normalize(self.wrd_embed(pos_ids), dim=-1)
        pos_scores = (ctx_vecs * pos_vecs).sum(-1, keepdim=True)

        # Negative IDs generated entirely on the GPU — no Python loop
        neg_ids   = torch.randint(1, self._vocab_size, (batch, self._neg_samples), device=device)
        neg_vecs  = F.normalize(self.wrd_embed(neg_ids), dim=-1)          # (batch, neg, dim)
        neg_scores = torch.bmm(neg_vecs, ctx_vecs.unsqueeze(-1)).squeeze(-1)  # (batch, neg)

        logits = torch.cat([pos_scores, neg_scores], dim=1)               # (batch, 1+neg)
        labels = torch.zeros_like(logits)
        labels[:, 0] = 1.0
        return logits, labels


# ---------------------------------------------------------------------------
# Corpus iterator helpers
# ---------------------------------------------------------------------------

def _brown_sentence_ids(vocab: Vocabulary) -> list[list[int]]:
    """Return all Brown corpus sentences as lists of word IDs."""
    import nltk
    try:
        nltk.data.find("corpora/brown")
    except LookupError:
        nltk.download("brown", quiet=True)
    from nltk.corpus import brown
    sentences = []
    for sent in brown.sents():
        ids = [vocab.word_to_id(w.lower()) for w in sent if w.isalpha()]
        if len(ids) >= 2:
            sentences.append(ids)
    return sentences


def _corpus_file_sentence_ids(path: Path, vocab: Vocabulary) -> list[list[int]]:
    """Read a plain-text file (one sentence per line) and convert to word IDs."""
    sentences = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            words = line.strip().lower().split()
            ids = [vocab.word_to_id(w) for w in words if w.isalpha()]
            if len(ids) >= 2:
                sentences.append(ids)
    return sentences


def _precompute_pairs(
    sentences: list[list[int]],
    context_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-compute all (context, target) training pairs as flat NumPy arrays.

    Called once before training.  The resulting arrays are shuffled per epoch
    via NumPy index permutation (fast, releases the GIL).

    Returns:
        ctx_arr: int64 (n_pairs, context_window) — zero-padded on the left
        pos_arr: int64 (n_pairs,)
    """
    n_pairs = sum(max(0, len(s) - 1) for s in sentences)
    ctx_arr = np.zeros((n_pairs, context_window), dtype=np.int64)
    pos_arr = np.empty(n_pairs, dtype=np.int64)
    idx = 0
    for sent in sentences:
        for t in range(1, len(sent)):
            ctx_start = max(0, t - context_window)
            src = sent[ctx_start:t]
            ctx_arr[idx, context_window - len(src):] = src
            pos_arr[idx] = sent[t]
            idx += 1
    return ctx_arr, pos_arr


def _format_eta(seconds: float) -> str:
    """Format seconds into a compact human-readable duration."""
    if seconds < 0:
        return "?"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DualEncoderTrainer:
    """Train a DualEncoder from a text corpus using candidate-only neg sampling.

    Usage::

        vocab   = Vocabulary.build_from_nltk()
        trainer = DualEncoderTrainer(vocab, embed_dim=64)
        trainer.train_from_nltk(epochs=3)
        trainer.save_numpy("model.npz")

    The resulting .npz can be loaded with ``DualEncoder.load(path, vocab)``
    without PyTorch.

    Set ``debug=True`` to print a timestamped breakdown of every setup phase
    and the first few batches, which is useful for identifying startup bottlenecks.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        embed_dim: int = 64,
        context_window: int = 3,
        neg_samples: int = 20,
        lr: float = 0.005,
        batch_size: int = 2048,
        prefetch_batches: int = 8,
        seed: int = 42,
        device: str = "auto",
        debug: bool = False,
    ) -> None:
        self._vocab = vocab
        self._embed_dim = embed_dim
        self._context_window = context_window
        self._neg_samples = neg_samples
        self._lr = lr
        self._batch_size = batch_size
        self._prefetch_batches = prefetch_batches
        self._seed = seed
        self._device_pref = device
        self._debug = debug
        self._device = None   # resolved when torch is available at train time
        self._model: _TorchDualEncoder | None = None
        self._torch = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_from_nltk(
        self,
        epochs: int = 3,
        verbose: bool = True,
    ) -> None:
        """Train on the NLTK Brown corpus (auto-downloaded)."""
        torch = _require_torch()
        t0 = time.monotonic()
        if verbose:
            print("Loading Brown corpus…")
        sentences = _brown_sentence_ids(self._vocab)
        if verbose:
            print(f"  {len(sentences):,} sentences loaded  ({time.monotonic()-t0:.2f}s)")
        self._train(sentences, epochs=epochs, torch=torch, verbose=verbose)

    def train_from_file(
        self,
        corpus_path: str | Path,
        epochs: int = 3,
        verbose: bool = True,
    ) -> None:
        """Train on a plain-text file (one sentence per line)."""
        self.train_from_files([Path(corpus_path)], epochs=epochs, verbose=verbose)

    def train_from_files(
        self,
        paths: list[str | Path],
        epochs: int = 3,
        verbose: bool = True,
    ) -> None:
        """Train on multiple plain-text corpus files (combined into one pass).

        Files are loaded in the order given; their sentences are concatenated
        before training begins.  The per-epoch shuffle then mixes them together.
        """
        torch = _require_torch()
        t0 = time.monotonic()
        sentences: list[list[int]] = []
        for path in paths:
            path = Path(path)
            file_sents = _corpus_file_sentence_ids(path, self._vocab)
            if verbose:
                print(f"  {path.name}: {len(file_sents):,} sentences  "
                      f"({time.monotonic()-t0:.2f}s)")
            sentences.extend(file_sents)
        if verbose:
            print(f"  Total: {len(sentences):,} sentences")
        self._train(sentences, epochs=epochs, torch=torch, verbose=verbose)

    def save_numpy(self, path: str | Path) -> None:
        """Export trained embeddings to .npz (NumPy format) for inference."""
        if self._model is None:
            raise RuntimeError("No trained model — call train_from_nltk() first.")
        ctx = self._model.ctx_embed.weight.detach().cpu().numpy()
        wrd = self._model.wrd_embed.weight.detach().cpu().numpy()
        encoder = DualEncoder(ctx, wrd, self._vocab)
        encoder.save(path)
        print(f"Saved model to {path}  ({Path(path).stat().st_size / 1e6:.1f} MB)")

    def get_encoder(self) -> DualEncoder:
        """Return a DualEncoder with the current trained weights (no file I/O)."""
        if self._model is None:
            raise RuntimeError("No trained model — call train_from_nltk() first.")
        ctx = self._model.ctx_embed.weight.detach().cpu().numpy()
        wrd = self._model.wrd_embed.weight.detach().cpu().numpy()
        return DualEncoder(ctx, wrd, self._vocab)

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def _phase(self, label: str, t_ref: float) -> None:
        """Print a timestamped phase marker (only when debug=True)."""
        if self._debug:
            print(f"  [+{time.monotonic() - t_ref:.3f}s] {label}")

    # ------------------------------------------------------------------
    # Internal training loop
    # ------------------------------------------------------------------

    def _train(
        self,
        sentences: list[list[int]],
        epochs: int,
        torch,
        verbose: bool,
    ) -> None:
        t_ref = time.monotonic()

        # ---- Phase 1: device resolution --------------------------------------
        device = _resolve_device(torch, self._device_pref)
        self._device = device
        self._torch = torch
        self._phase("device resolved", t_ref)

        torch.manual_seed(self._seed)
        random.seed(self._seed)
        np_rng = np.random.default_rng(self._seed)

        # ---- Phase 2: model creation + .to(device) --------------------------
        # NOTE: On CUDA the first .to(device) call triggers driver initialisation
        # which can block for 10–30 s.  This is a one-time cost per process.
        vocab_size = self._vocab.size
        model = _TorchDualEncoder(
            vocab_size, self._embed_dim, self._neg_samples, torch, device
        )
        self._model = model
        self._phase(f"model created and moved to {device}", t_ref)

        # ---- Phase 3: optimizer + loss fn -----------------------------------
        optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)
        bce = torch.nn.BCEWithLogitsLoss()
        self._phase("optimizer ready", t_ref)

        # ---- Phase 4: pre-compute all training pairs as flat arrays ----------
        # Done once; per-epoch shuffling uses a lightweight index permutation
        # and pairs are gathered per-batch (only ~50 KB each), avoiding the
        # ~500 MB full-array copy + pin that previously pinned the CPU at 100%.
        ctx_all, pos_all = _precompute_pairs(sentences, self._context_window)
        n_pairs = len(pos_all)
        n_batches = math.ceil(n_pairs / self._batch_size)
        self._phase(f"pairs pre-computed ({n_pairs:,} pairs)", t_ref)

        # AMP autocast (CUDA only).  Under autocast only torch.bmm is cast to
        # float16; everything else (embedding lookup, normalize, element-wise
        # ops, BCE loss) stays float32.  Because no float16 gradients are
        # produced, GradScaler is unnecessary — and removing it eliminates the
        # per-batch CPU↔GPU sync from its internal found_inf.item() call
        # (CUDA uses spin-wait synchronisation by default, which pins a CPU
        # core at 100 % for every sync point).
        is_cuda = str(device).startswith("cuda")
        use_amp = is_cuda

        # TF32 matmul on Ampere+ GPUs (~3× faster float32 ops, negligible
        # accuracy difference for embedding training).
        if is_cuda and hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        # Pre-allocate double-buffered pinned transfer buffers (CUDA only).
        # Avoids ~2×n_batches cudaHostAlloc/Free OS calls per epoch.
        # Double-buffering ensures the DMA from buffer A is finished before
        # we overwrite it (an entire forward+backward pass intervenes).
        if is_cuda:
            _pin_ctx = tuple(
                torch.empty(self._batch_size, self._context_window,
                            dtype=torch.int64).pin_memory()
                for _ in range(2)
            )
            _pin_pos = tuple(
                torch.empty(self._batch_size, dtype=torch.int64).pin_memory()
                for _ in range(2)
            )

        if verbose:
            print(
                f"Device: {device}  |  training pairs: {n_pairs:,}  |  "
                f"vocab: {vocab_size}  |  embed_dim: {self._embed_dim}  |  "
                f"neg_samples: {self._neg_samples} (on GPU)  |  "
                f"batch: {self._batch_size}  |  AMP: {'on' if use_amp else 'off'}"
            )

        # Reusable index array — shuffled in-place each epoch (no allocation).
        perm = np.arange(n_pairs, dtype=np.int64)

        # GPU-side loss accumulator — avoids per-batch CPU↔GPU sync from
        # .item().  Only synced to CPU for progress-bar redraws (~4×/s).
        loss_accum = torch.zeros((), device=device)

        # ---- Training epochs ------------------------------------------------
        total_train_batches = n_batches * epochs
        train_batches_done = 0
        t_train = time.monotonic()
        last_progress_t = 0.0
        _PROGRESS_INTERVAL = 0.25       # seconds between progress-bar redraws

        for epoch in range(1, epochs + 1):
            t_epoch = time.monotonic()
            loss_accum.zero_()

            # In-place shuffle of index array — O(n) with no allocation,
            # and NumPy releases the GIL during the shuffle.
            np_rng.shuffle(perm)
            self._phase(f"epoch {epoch} shuffle done", t_ref)

            for b in range(n_batches):
                start = b * self._batch_size
                end = min(start + self._batch_size, n_pairs)
                idx = perm[start:end]               # O(1) view into perm
                batch_len = end - start

                # Copy into pre-pinned double-buffer, then async-DMA to GPU.
                # No per-batch cudaHostAlloc, no per-batch .item() sync.
                if is_cuda:
                    buf_i = b & 1
                    _pin_ctx[buf_i][:batch_len].copy_(
                        torch.from_numpy(ctx_all[idx]))
                    _pin_pos[buf_i][:batch_len].copy_(
                        torch.from_numpy(pos_all[idx]))
                    ctx_t = _pin_ctx[buf_i][:batch_len].to(
                        device, non_blocking=True)
                    pos_t = _pin_pos[buf_i][:batch_len].to(
                        device, non_blocking=True)
                else:
                    ctx_t = torch.from_numpy(
                        np.ascontiguousarray(ctx_all[idx])).to(device)
                    pos_t = torch.from_numpy(
                        np.ascontiguousarray(pos_all[idx])).to(device)

                # ---- debug: time first batch in detail ----------------------
                if self._debug and b == 0 and epoch == 1:
                    self._phase(f"  first batch on device ({batch_len} pairs)", t_ref)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                        logits, labels = model.score_with_negatives(ctx_t, pos_t, torch)
                        loss = bce(logits, labels)
                    self._phase("  first forward pass done", t_ref)
                    loss.backward()
                    self._phase("  first backward pass done", t_ref)
                    optimizer.step()
                    self._phase("  first optimizer step done", t_ref)
                    loss_accum += loss.detach()
                    train_batches_done += 1
                    continue
                # -------------------------------------------------------------

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    logits, labels = model.score_with_negatives(ctx_t, pos_t, torch)
                    loss = bce(logits, labels)
                loss.backward()
                optimizer.step()

                loss_accum += loss.detach()
                train_batches_done += 1

                # ---- inline progress bar ------------------------------------
                now = time.monotonic()
                if verbose and (
                    now - last_progress_t >= _PROGRESS_INTERVAL
                    or b == n_batches - 1
                ):
                    last_progress_t = now
                    done = b + 1
                    frac = done / n_batches
                    bar_w = 20
                    filled = int(bar_w * frac)
                    bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
                    avg_loss = loss_accum.item() / done  # only sync point

                    epoch_elapsed = now - t_epoch
                    epoch_eta = epoch_elapsed / done * (n_batches - done)

                    train_elapsed = now - t_train
                    total_eta = (
                        train_elapsed / train_batches_done
                        * (total_train_batches - train_batches_done)
                    )

                    print(
                        f"\r  Epoch {epoch}/{epochs}  |{bar}| "
                        f"{done}/{n_batches}  "
                        f"loss={avg_loss:.4f}  "
                        f"ETA: {_format_eta(epoch_eta)}  "
                        f"[total: {_format_eta(total_eta)}]"
                        "\033[K",
                        end="", flush=True,
                    )

            elapsed = time.monotonic() - t_epoch
            if verbose:
                total_loss = loss_accum.item()
                avg_loss = total_loss / max(n_batches, 1)
                pairs_per_sec = n_pairs / elapsed if elapsed > 0 else 0
                # Overwrite the progress bar with the final epoch summary.
                print(
                    f"\r  Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  "
                    f"time={elapsed:.1f}s  ({pairs_per_sec:,.0f} pairs/s)"
                    "\033[K"
                )

        if verbose:
            total_time = time.monotonic() - t_train
            print(f"  Training complete in {_format_eta(total_time)}")
