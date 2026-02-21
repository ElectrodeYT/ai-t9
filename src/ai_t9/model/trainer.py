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
The original bottleneck was that negative sampling happened in a Python loop
on the CPU, which kept the GPU mostly idle waiting for data.  The current
design removes that bottleneck with three changes:

1. On-GPU negative sampling — torch.randint generates all negative IDs
   directly on the GPU; only (ctx_ids, pos_ids) are transferred from CPU,
   roughly 20× less data per batch than before.

2. Background prefetch thread — a daemon thread runs pair generation and
   numpy array construction while the GPU processes the previous batch,
   overlapping CPU and GPU work.

3. Pinned memory + non-blocking transfers (CUDA only) — page-locked CPU
   buffers allow the DMA engine to copy data asynchronously, letting the
   CPU continue to the next batch immediately after issuing the transfer.
"""

from __future__ import annotations

import math
import queue
import random
import threading
import time
from pathlib import Path
from typing import Iterator

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


def _generate_training_pairs(
    sentences: list[list[int]],
    context_window: int = 3,
) -> Iterator[tuple[list[int], int]]:
    """Yield (context_word_ids, target_word_id) pairs from a sentence corpus."""
    for sent in sentences:
        for t in range(1, len(sent)):
            ctx_start = max(0, t - context_window)
            yield sent[ctx_start:t], sent[t]


# ---------------------------------------------------------------------------
# Batch construction (CPU side — runs in prefetch thread)
# ---------------------------------------------------------------------------

def _build_arrays(
    batch: list[tuple[list[int], int]],
    context_window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a batch of (ctx_ids, pos_id) pairs to numpy arrays.

    Returns:
        ctx_arr: int64 array (batch, context_window) — zero-padded on the left
        pos_arr: int64 array (batch,)
    """
    n = len(batch)
    ctx_arr = np.zeros((n, context_window), dtype=np.int64)
    pos_arr = np.empty(n, dtype=np.int64)
    for i, (ctx_ids, pos_id) in enumerate(batch):
        src = ctx_ids[-context_window:]          # at most context_window entries
        ctx_arr[i, context_window - len(src):] = src  # right-align; left stays 0 (UNK)
        pos_arr[i] = pos_id
    return ctx_arr, pos_arr


def _prefetch_worker(
    sentences: list[list[int]],
    context_window: int,
    batch_size: int,
    out_q: "queue.Queue[tuple[np.ndarray, np.ndarray] | None]",
) -> None:
    """Background thread: generates training pairs and pushes numpy arrays.

    Runs entirely on the CPU.  Pushes (ctx_arr, pos_arr) tuples into out_q,
    then pushes None as a sentinel when all pairs for this epoch are exhausted.
    The queue's maxsize limits how far ahead this thread can run, capping the
    memory used by pre-built batches.
    """
    batch: list[tuple[list[int], int]] = []
    for ctx_ids, target_id in _generate_training_pairs(sentences, context_window):
        batch.append((ctx_ids, target_id))
        if len(batch) >= batch_size:
            out_q.put(_build_arrays(batch, context_window))
            batch.clear()
    if batch:
        out_q.put(_build_arrays(batch, context_window))
    out_q.put(None)  # sentinel — epoch is done


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

        # ---- Phase 4: pair count (single pass, no allocation) ---------------
        n_pairs = sum(max(0, len(s) - 1) for s in sentences)
        self._phase(f"pair count done ({n_pairs:,} pairs)", t_ref)

        # Pinned memory enables async DMA transfers on CUDA; a no-op elsewhere.
        use_pin = str(device).startswith("cuda")

        if verbose:
            print(
                f"Device: {device}  |  training pairs: {n_pairs:,}  |  "
                f"vocab: {vocab_size}  |  embed_dim: {self._embed_dim}  |  "
                f"neg_samples: {self._neg_samples} (on GPU)  |  "
                f"batch: {self._batch_size}  |  prefetch: {self._prefetch_batches}"
            )

        # ---- Training epochs ------------------------------------------------
        for epoch in range(1, epochs + 1):
            random.shuffle(sentences)
            t_epoch = time.monotonic()
            total_loss = 0.0
            n_batches = 0

            # Start prefetch thread for this epoch.
            # maxsize caps how many batches the thread can build ahead, bounding
            # the extra RAM used (each batch is batch_size * context_window * 8 bytes).
            prefetch_q: queue.Queue = queue.Queue(maxsize=self._prefetch_batches)
            prefetch_t = threading.Thread(
                target=_prefetch_worker,
                args=(sentences, self._context_window, self._batch_size, prefetch_q),
                daemon=True,
            )
            prefetch_t.start()

            while True:
                item = prefetch_q.get()
                if item is None:        # sentinel — epoch exhausted
                    break

                ctx_arr, pos_arr = item

                # torch.from_numpy is near zero-copy (shares memory with the array).
                # pin_memory() pages the buffer so DMA can copy without a staging copy.
                # non_blocking=True lets the CPU return immediately after issuing the
                # transfer; PyTorch syncs automatically before the tensor is used.
                ctx_t = torch.from_numpy(ctx_arr)
                pos_t = torch.from_numpy(pos_arr)
                if use_pin:
                    ctx_t = ctx_t.pin_memory()
                    pos_t = pos_t.pin_memory()
                ctx_t = ctx_t.to(device, non_blocking=True)
                pos_t = pos_t.to(device, non_blocking=True)

                # ---- debug: time first batch in detail ----------------------
                if self._debug and n_batches == 0 and epoch == 1:
                    self._phase(f"  first batch on device ({len(ctx_arr)} pairs)", t_ref)
                    optimizer.zero_grad()
                    logits, labels = model.score_with_negatives(ctx_t, pos_t, torch)
                    self._phase("  first forward pass done", t_ref)
                    loss = bce(logits, labels)
                    loss.backward()
                    self._phase("  first backward pass done", t_ref)
                    optimizer.step()
                    self._phase("  first optimizer step done", t_ref)
                    total_loss += loss.item()
                    n_batches += 1
                    continue
                # -------------------------------------------------------------

                optimizer.zero_grad()
                logits, labels = model.score_with_negatives(ctx_t, pos_t, torch)
                loss = bce(logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            prefetch_t.join()

            elapsed = time.monotonic() - t_epoch
            if verbose:
                avg_loss = total_loss / max(n_batches, 1)
                pairs_per_sec = n_pairs / elapsed if elapsed > 0 else 0
                print(
                    f"  Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  "
                    f"time={elapsed:.1f}s  ({pairs_per_sec:,.0f} pairs/s)"
                )
