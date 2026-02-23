"""DualEncoder: pure-NumPy inference engine for context-aware word scoring.

The model stores two embedding matrices:
  - context_embeds[word_id]  : how word_id looks when it appears *before* the target
  - word_embeds[word_id]     : how word_id looks when it is the *target*

At inference:
  ctx_vec  = mean(context_embeds[context_word_ids])   # 64-dim
  scores   = word_embeds[candidate_ids] @ ctx_vec      # dot product per candidate

Weights are stored as a .npz file — two float32 arrays, typically ~6 MB for
50k vocab × 64 dims.  Loading takes < 50 ms even on slow flash storage.
"""

from __future__ import annotations

import io
import math
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np

from .vocab import Vocabulary


class DualEncoder:
    """Context-aware word scorer using mean-pooled embedding dot products.

    Pure NumPy — no ML framework required at inference time.
    """

    def __init__(
        self,
        context_embeds: np.ndarray,
        word_embeds: np.ndarray,
        vocab: Vocabulary,
    ) -> None:
        """
        Args:
            context_embeds: float32 array of shape (vocab_size, embed_dim)
            word_embeds:    float32 array of shape (vocab_size, embed_dim)
            vocab:          matching Vocabulary instance
        """
        assert context_embeds.shape == word_embeds.shape, (
            "context_embeds and word_embeds must have the same shape"
        )
        assert context_embeds.shape[0] == vocab.size, (
            f"Embedding vocab size {context_embeds.shape[0]} != vocab size {vocab.size}"
        )
        self._ctx = context_embeds.astype(np.float32)
        self._wrd = word_embeds.astype(np.float32)
        self._vocab = vocab
        self._dim = context_embeds.shape[1]

        # 1-slot context cache — encode_context() is called on every keypress
        # with the same context; caching eliminates redundant computation.
        self._ctx_cache_key: tuple[int, ...] | None = None
        self._ctx_cache_vec: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def encode_context(self, context_ids: Sequence[int]) -> np.ndarray:
        """Produce a context vector by mean-pooling context word embeddings.

        Results are cached by context key — repeated calls with the same context
        (common during a typing burst) return the cached vector immediately.

        If context_ids is empty, returns the zero vector.
        """
        key = tuple(context_ids)
        if key == self._ctx_cache_key and self._ctx_cache_vec is not None:
            return self._ctx_cache_vec

        vec = self._encode_context_impl(context_ids)
        self._ctx_cache_key = key
        self._ctx_cache_vec = vec
        return vec

    def _encode_context_impl(self, context_ids: Sequence[int]) -> np.ndarray:
        if not context_ids:
            return np.zeros(self._dim, dtype=np.float32)
        vecs = self._ctx[np.asarray(context_ids, dtype=np.intp)]  # (n, dim)
        ctx_vec = vecs.mean(axis=0)                                 # (dim,)
        norm = np.linalg.norm(ctx_vec)
        if norm > 0:
            ctx_vec = ctx_vec / norm
        return ctx_vec.astype(np.float32)

    def score_candidates(
        self,
        context_ids: Sequence[int],
        candidate_ids: Sequence[int],
    ) -> np.ndarray:
        """Score each candidate word given the context.

        Returns a float32 array of shape (len(candidate_ids),).
        Higher = more likely given context.
        """
        if not candidate_ids:
            return np.array([], dtype=np.float32)
        ctx_vec = self.encode_context(context_ids)
        cand_vecs = self._wrd[list(candidate_ids)]   # (n_cands, dim)
        return cand_vecs @ ctx_vec                    # (n_cands,)

    @property
    def embed_dim(self) -> int:
        return self._dim

    @property
    def vocab(self) -> Vocabulary:
        return self._vocab

    # ------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------

    def quantize_int8(self) -> "DualEncoder":
        """Return a new DualEncoder with int8-quantized embeddings (~4× smaller).

        Scores remain comparable (scaling is absorbed into dot product norms).
        """
        def _quant(arr: np.ndarray) -> np.ndarray:
            scale = np.abs(arr).max(axis=1, keepdims=True).clip(min=1e-8)
            q = np.round(arr / scale * 127).clip(-127, 127).astype(np.int8)
            # Dequantize back so the interface stays the same (float32 output)
            return (q.astype(np.float32) / 127.0) * scale

        return DualEncoder(
            _quant(self._ctx),
            _quant(self._wrd),
            self._vocab,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save embeddings to a .npz file.

        Writes via an in-memory buffer so the output is always a single
        sequential write, compatible with S3 CloudBucketMounts (Mountpoint
        does not support the random seeks that np.savez makes when given
        a file-path directly).
        """
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            context_embeds=self._ctx,
            word_embeds=self._wrd,
        )
        buf.seek(0)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            shutil.copyfileobj(buf, f)

    @classmethod
    def load(cls, path: str | Path, vocab: Vocabulary) -> "DualEncoder":
        """Load embeddings from a .npz file."""
        path = Path(path)
        arrays = np.load(str(path))
        return cls(
            context_embeds=arrays["context_embeds"],
            word_embeds=arrays["word_embeds"],
            vocab=vocab,
        )

    # ------------------------------------------------------------------
    # Random initialisation (useful for testing without training)
    # ------------------------------------------------------------------

    @classmethod
    def random_init(
        cls,
        vocab: Vocabulary,
        embed_dim: int = 64,
        seed: int = 0,
    ) -> "DualEncoder":
        rng = np.random.default_rng(seed)
        scale = 1.0 / math.sqrt(embed_dim)
        ctx = rng.normal(0, scale, (vocab.size, embed_dim)).astype(np.float32)
        wrd = rng.normal(0, scale, (vocab.size, embed_dim)).astype(np.float32)
        return cls(ctx, wrd, vocab)
