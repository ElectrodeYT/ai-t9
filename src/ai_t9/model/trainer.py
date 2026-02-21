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
   NumPy arrays *once* before training.  Each epoch shuffles via a NumPy
   permutation (which releases the GIL) and converts to a pinned-memory
   tensor in one step.  Batches are then simple O(1) tensor slices — no
   Python generator or per-batch array construction, eliminating the main
   GIL-contention bottleneck that kept the GPU starved for data.

2. On-GPU negative sampling — torch.randint generates all negative IDs
   directly on the GPU; only (ctx_ids, pos_ids) are transferred from CPU,
   roughly 20× less data per batch than before.

3. Pinned memory + non-blocking transfers (CUDA only) — the full shuffled
   epoch tensor is pinned once; batch slices inherit the pinned status
   and transfer via DMA without per-batch OS allocation overhead.

4. Automatic mixed precision (AMP) — on CUDA, forward passes run in float16
   via torch.amp.autocast, roughly doubling throughput for embedding lookups
   and matrix operations.  A GradScaler prevents float16 gradient underflow.

5. TF32 matmul (Ampere+ GPUs) — float32 matrix multiplications use
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
        # Done once; per-epoch shuffling uses numpy permutation (releases GIL).
        ctx_all, pos_all = _precompute_pairs(sentences, self._context_window)
        n_pairs = len(pos_all)
        n_batches = math.ceil(n_pairs / self._batch_size)
        self._phase(f"pairs pre-computed ({n_pairs:,} pairs)", t_ref)

        # AMP (automatic mixed precision) — CUDA only.  The GradScaler acts as
        # a no-op when enabled=False, so no if/else branches are needed below.
        is_cuda = str(device).startswith("cuda")
        use_amp = is_cuda
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # TF32 matmul on Ampere+ GPUs (~3× faster float32 ops, negligible
        # accuracy difference for embedding training).
        if is_cuda and hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        if verbose:
            print(
                f"Device: {device}  |  training pairs: {n_pairs:,}  |  "
                f"vocab: {vocab_size}  |  embed_dim: {self._embed_dim}  |  "
                f"neg_samples: {self._neg_samples} (on GPU)  |  "
                f"batch: {self._batch_size}  |  AMP: {'on' if use_amp else 'off'}"
            )

        # ---- Training epochs ------------------------------------------------
        total_train_batches = n_batches * epochs
        train_batches_done = 0
        t_train = time.monotonic()
        last_progress_t = 0.0
        _PROGRESS_INTERVAL = 0.25       # seconds between progress-bar redraws

        for epoch in range(1, epochs + 1):
            t_epoch = time.monotonic()
            total_loss = 0.0

            # Shuffle pair order via index permutation, then build a
            # (possibly pinned) tensor for the whole epoch.  Slicing this
            # tensor per batch is O(1) — no Python loops, no GIL pressure.
            perm = np_rng.permutation(n_pairs)
            ctx_epoch = torch.from_numpy(ctx_all[perm])
            pos_epoch = torch.from_numpy(pos_all[perm])
            if is_cuda:
                ctx_epoch = ctx_epoch.pin_memory()
                pos_epoch = pos_epoch.pin_memory()
            self._phase(f"epoch {epoch} shuffle+pin done", t_ref)

            for b in range(n_batches):
                start = b * self._batch_size
                end = min(start + self._batch_size, n_pairs)

                ctx_t = ctx_epoch[start:end].to(device, non_blocking=True)
                pos_t = pos_epoch[start:end].to(device, non_blocking=True)

                # ---- debug: time first batch in detail ----------------------
                if self._debug and b == 0 and epoch == 1:
                    self._phase(f"  first batch on device ({end - start} pairs)", t_ref)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                        logits, labels = model.score_with_negatives(ctx_t, pos_t, torch)
                        loss = bce(logits, labels)
                    self._phase("  first forward pass done", t_ref)
                    scaler.scale(loss).backward()
                    self._phase("  first backward pass done", t_ref)
                    scaler.step(optimizer)
                    scaler.update()
                    self._phase("  first optimizer step done", t_ref)
                    total_loss += loss.item()
                    train_batches_done += 1
                    continue
                # -------------------------------------------------------------

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    logits, labels = model.score_with_negatives(ctx_t, pos_t, torch)
                    loss = bce(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()
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
                    avg_loss = total_loss / done

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
