"""PyTorch training for the DualEncoder model.

This module is intentionally isolated from the inference path — PyTorch is an
optional dependency (pip install ai-t9[train]).  The output is a .npz file
that the pure-NumPy DualEncoder can load without any ML framework.

Training objective: candidate-only negative sampling
  For each (context_words, target_word) pair drawn from the corpus:
    - Sample `neg_samples` random negatives from the vocabulary
    - Score all (1 + neg_samples) candidates with the dual encoder
    - Apply binary cross-entropy (positive=1, negatives=0)

This is ~1000× faster than full softmax over the whole vocabulary and produces
comparable or better embeddings due to harder negatives.
"""

from __future__ import annotations

import math
import random
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


# ---------------------------------------------------------------------------
# PyTorch model (training-time only)
# ---------------------------------------------------------------------------

class _TorchDualEncoder:
    """Thin wrapper around two nn.Embedding tables, kept in a nested class so
    the import of torch is deferred until training actually starts."""

    def __init__(self, vocab_size: int, embed_dim: int, torch):
        nn = torch.nn
        self.ctx_embed = nn.Embedding(vocab_size, embed_dim)
        self.wrd_embed = nn.Embedding(vocab_size, embed_dim)
        # Xavier init
        nn.init.xavier_uniform_(self.ctx_embed.weight)
        nn.init.xavier_uniform_(self.wrd_embed.weight)

    def parameters(self):
        return list(self.ctx_embed.parameters()) + list(self.wrd_embed.parameters())

    def encode_context(self, ctx_ids, torch):
        """ctx_ids: LongTensor (batch, ctx_len) → (batch, dim)"""
        vecs = self.ctx_embed(ctx_ids)      # (batch, ctx_len, dim)
        return vecs.mean(dim=1)             # (batch, dim)

    def score(self, ctx_ids, cand_ids, torch):
        """
        ctx_ids:  LongTensor (batch, ctx_len)
        cand_ids: LongTensor (batch,)          — one positive or negative per row
        returns:  scalar logit per row (batch,)
        """
        ctx_vec = self.encode_context(ctx_ids, torch)               # (batch, dim)
        cand_vec = self.wrd_embed(cand_ids)                          # (batch, dim)
        # Normalise for stable training
        ctx_norm = torch.nn.functional.normalize(ctx_vec, dim=-1)
        cand_norm = torch.nn.functional.normalize(cand_vec, dim=-1)
        return (ctx_norm * cand_norm).sum(dim=-1)                    # (batch,)


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
    """Yield (context_word_ids, target_word_id) pairs from a sentence corpus.

    For each word at position t we use words [t-context_window … t-1] as
    context and word t as target.
    """
    for sent in sentences:
        for t in range(1, len(sent)):
            ctx_start = max(0, t - context_window)
            ctx_ids = sent[ctx_start:t]
            target_id = sent[t]
            yield ctx_ids, target_id


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
    """

    def __init__(
        self,
        vocab: Vocabulary,
        embed_dim: int = 64,
        context_window: int = 3,
        neg_samples: int = 20,
        lr: float = 0.005,
        batch_size: int = 512,
        seed: int = 42,
    ) -> None:
        self._vocab = vocab
        self._embed_dim = embed_dim
        self._context_window = context_window
        self._neg_samples = neg_samples
        self._lr = lr
        self._batch_size = batch_size
        self._seed = seed
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
        if verbose:
            print("Loading Brown corpus…")
        sentences = _brown_sentence_ids(self._vocab)
        self._train(sentences, epochs=epochs, torch=torch, verbose=verbose)

    def train_from_file(
        self,
        corpus_path: str | Path,
        epochs: int = 3,
        verbose: bool = True,
    ) -> None:
        """Train on a plain-text file (one sentence per line)."""
        torch = _require_torch()
        if verbose:
            print(f"Loading corpus from {corpus_path}…")
        sentences = _corpus_file_sentence_ids(Path(corpus_path), self._vocab)
        if verbose:
            print(f"  {len(sentences)} sentences loaded")
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
    # Internal training loop
    # ------------------------------------------------------------------

    def _train(
        self,
        sentences: list[list[int]],
        epochs: int,
        torch,
        verbose: bool,
    ) -> None:
        torch.manual_seed(self._seed)
        random.seed(self._seed)

        vocab_size = self._vocab.size
        model = _TorchDualEncoder(vocab_size, self._embed_dim, torch)
        self._model = model
        self._torch = torch

        optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)
        bce = torch.nn.BCEWithLogitsLoss()

        # Pre-build all training pairs (fits in RAM for Brown corpus)
        pairs = list(_generate_training_pairs(sentences, self._context_window))
        if verbose:
            print(
                f"Training pairs: {len(pairs):,}  |  vocab: {vocab_size}  |  "
                f"embed_dim: {self._embed_dim}  |  neg_samples: {self._neg_samples}"
            )

        for epoch in range(1, epochs + 1):
            random.shuffle(pairs)
            t0 = time.time()
            total_loss = 0.0
            n_batches = 0

            # Stream pairs in batches
            for batch_start in range(0, len(pairs), self._batch_size):
                batch = pairs[batch_start : batch_start + self._batch_size]
                if not batch:
                    continue

                # Build tensors for this batch (context is padded to context_window)
                ctx_tensor, cand_tensor, label_tensor = self._build_batch(
                    batch, vocab_size, torch
                )

                optimizer.zero_grad()
                logits = model.score(ctx_tensor, cand_tensor, torch)
                loss = bce(logits, label_tensor)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            elapsed = time.time() - t0
            if verbose:
                avg_loss = total_loss / max(n_batches, 1)
                print(
                    f"  Epoch {epoch}/{epochs}  loss={avg_loss:.4f}  "
                    f"time={elapsed:.1f}s"
                )

    def _build_batch(
        self,
        batch: list[tuple[list[int], int]],
        vocab_size: int,
        torch,
    ):
        """Build padded tensors for a batch of (context_ids, target_id) pairs.

        Each positive example is paired with `neg_samples` negatives, so the
        effective batch size is len(batch) * (1 + neg_samples).
        """
        ctx_rows: list[list[int]] = []
        cand_rows: list[int] = []
        label_rows: list[float] = []

        for ctx_ids, pos_id in batch:
            # Pad context to context_window (pad with UNK=0)
            padded_ctx = ctx_ids[-self._context_window:]
            while len(padded_ctx) < self._context_window:
                padded_ctx = [0] + padded_ctx

            # Positive example
            ctx_rows.append(padded_ctx)
            cand_rows.append(pos_id)
            label_rows.append(1.0)

            # Negative examples
            for _ in range(self._neg_samples):
                neg_id = random.randint(1, vocab_size - 1)
                ctx_rows.append(padded_ctx)
                cand_rows.append(neg_id)
                label_rows.append(0.0)

        ctx_tensor = torch.tensor(ctx_rows, dtype=torch.long)
        cand_tensor = torch.tensor(cand_rows, dtype=torch.long)
        label_tensor = torch.tensor(label_rows, dtype=torch.float32)
        return ctx_tensor, cand_tensor, label_tensor
