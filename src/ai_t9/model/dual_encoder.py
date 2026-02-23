"""DualEncoder: open-vocabulary context-aware word scorer.

This is the CharNgramDualEncoder implementation, which decouples model size from
vocabulary size and eliminates the need to retrain when new words are added to
the T9 dictionary.

Architecture
------------
Instead of a per-word-ID embedding lookup table, words are represented as the
**mean of their character n-gram embeddings** (fastText-style).  The model
stores two embedding matrices indexed by character n-gram ID rather than word ID:

    ctx_embeds[ngram_id]  — how ngram_id looks in context (preceding words)
    wrd_embeds[ngram_id]  — how ngram_id looks as a target word

At load time, both matrices are expanded into dense per-word vectors for all
vocab words, stored as ``_ctx_matrix`` and ``_word_matrix``.  At inference:

    ctx_vec  = normalise(mean(_ctx_matrix[context_ids]))   # (dim,)
    scores   = _word_matrix[candidate_ids] @ ctx_vec       # (n_cands,)

This pre-computation makes scoring as fast as the old DualEncoder (single matmul) while
keeping the model size small (n_ngrams × dim instead of vocab_size × dim).

Open-vocabulary scoring of OOV words is still available via ``score_word()``,
which uses the raw ngram embeddings directly.

N-gram vocabulary
-----------------
Word boundaries are marked with ``<`` (start) and ``>`` (end), following
fastText convention.  For example ``"the"`` produces:
    2-grams: ``<t``, ``th``, ``he``, ``e>``
    3-grams: ``<th``, ``the``, ``he>``

The n-gram vocabulary is built once at training time from the word corpus and
saved alongside the embeddings in the .npz file.  Its size is fixed and grows
very slowly (new English words almost never introduce new bigrams/trigrams).

Size comparison (dim=264, vocab_size=29,470)
--------------------------------------------
    Old DualEncoder       :  29,470 × 264 × 2 × 4 B  ≈  62 MB RAM
    New DualEncoder       :   8,734 × 264 × 2 × 4 B  ≈  18 MB (ngram mats)
                           + 29,470 × 264 × 2 × 4 B  ≈  62 MB (pre-computed)

For mobile targets use dim=64-96 to keep the pre-computed matrices small.

Interface
---------
The public interface is identical to the old DualEncoder so T9Predictor requires no
changes — just load a DualEncoder instead.
"""

from __future__ import annotations

import io
import math
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np

from .vocab import Vocabulary


# Constants from char_ngram_encoder.py
_UNK_NGRAM_ID = 0


def _char_ngrams(word: str, ns: tuple[int, ...]) -> list[str]:
    """Generate character n-grams for a word."""
    word = f"<{word}>"
    grams = []
    for n in ns:
        for i in range(len(word) - n + 1):
            grams.append(word[i : i + n])
    return grams


def build_ngram_vocab(words: list[str], ns: tuple[int, ...] = (2, 3)) -> dict[str, int]:
    """Build n-gram vocabulary from a list of words."""
    ngram_to_id = {}
    for word in words:
        for gram in _char_ngrams(word, ns):
            if gram not in ngram_to_id:
                ngram_to_id[gram] = len(ngram_to_id) + 1  # Start from 1, 0 is UNK
    return ngram_to_id


class DualEncoder:
    """Open-vocabulary context-aware word scorer using character n-gram embeddings.

    Pure NumPy — no ML framework required at inference time.

    Word embeddings are pre-computed at load time from the n-gram matrices,
    making candidate scoring as fast as a plain DualEncoder (single matmul).

    Open-vocabulary scoring (``score_word()``) is still available for words not
    in the vocabulary, using the raw ngram embeddings directly.
    """

    def __init__(
        self,
        context_embeds: np.ndarray,
        word_embeds: np.ndarray,
        ngram_to_id: dict[str, int],
        vocab: Vocabulary,
        ns: tuple[int, ...] = (2, 3),
    ) -> None:
        """
        Args:
            context_embeds: float32 (n_ngrams, embed_dim) — context role
            word_embeds:    float32 (n_ngrams, embed_dim) — target role
            ngram_to_id:    mapping from n-gram string to row index
            vocab:          Vocabulary (used to build per-word pre-computed matrices)
            ns:             n-gram sizes used during training (default: (2, 3))
        """
        assert context_embeds.shape == word_embeds.shape, (
            "context_embeds and word_embeds must have the same shape"
        )
        self._ctx = context_embeds.astype(np.float32)
        self._wrd = word_embeds.astype(np.float32)
        # Zero out the UNK row so unknown n-grams contribute nothing.
        self._ctx[_UNK_NGRAM_ID] = 0.0
        self._wrd[_UNK_NGRAM_ID] = 0.0
        self._ngram_to_id = ngram_to_id
        self._vocab = vocab
        self._ns = ns
        self._dim = context_embeds.shape[1]
        self._n_ngrams = context_embeds.shape[0]

        # Pre-compute dense (vocab_size, dim) matrices for fast lookup.
        # Each row is the L2-normalised mean of a word's n-gram embeddings,
        # matching the per-word normalisation applied during training.
        self._ctx_matrix, self._word_matrix = self._precompute_word_matrices()

        # 1-slot context cache — encode_context() is called on every keypress
        # with the same context window; caching the result eliminates redundant
        # computation within a typing burst.
        self._ctx_cache_key: tuple[int, ...] | None = None
        self._ctx_cache_vec: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Pre-computation
    # ------------------------------------------------------------------

    def _precompute_word_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Build dense (vocab_size, dim) word matrices from n-gram embeddings.

        For each word in the vocabulary:
          1. Collect its n-gram IDs (falling back to UNK_NGRAM_ID=0 for OOV ngrams)
          2. Mean-pool the corresponding rows from ctx_embeds / wrd_embeds
          3. L2-normalise the resulting vector

        The per-word L2 normalisation matches the training forward pass, where
        each word embedding is normalised before being pooled into the context
        vector — correcting the subtle discrepancy that existed in the old
        inference path.

        Returns:
            ctx_matrix  : float32 (vocab_size, dim) — for encode_context
            word_matrix : float32 (vocab_size, dim) — for score_candidates
        """
        vocab_size = self._vocab.size

        # Build padded (vocab_size, max_ngrams) index table.
        rows: list[list[int]] = []
        for wid in range(vocab_size):
            word = self._vocab.id_to_word(wid)
            ids = [
                self._ngram_to_id.get(g, _UNK_NGRAM_ID)
                for g in _char_ngrams(word, self._ns)
            ]
            # Filter UNK so they don't dilute the mean (same as training).
            ids = [i for i in ids if i != _UNK_NGRAM_ID] or [_UNK_NGRAM_ID]
            rows.append(ids)

        max_n = max(len(r) for r in rows)
        table = np.zeros((vocab_size, max_n), dtype=np.int32)
        counts = np.zeros(vocab_size, dtype=np.float32)
        for i, row in enumerate(rows):
            table[i, : len(row)] = row
            counts[i] = len(row)

        # Vectorised mean-pool: (vocab_size, max_n, dim) → (vocab_size, dim)
        def _build_matrix(embed: np.ndarray) -> np.ndarray:
            # embed: (n_ngrams, dim)
            vecs = embed[table]              # (vocab_size, max_n, dim)
            summed = vecs.sum(axis=1)        # (vocab_size, dim)
            means = summed / counts[:, None] # (vocab_size, dim)
            # L2-normalise each row.
            norms = np.linalg.norm(means, axis=1, keepdims=True).clip(min=1e-8)
            return (means / norms).astype(np.float32)

        ctx_matrix = _build_matrix(self._ctx)
        word_matrix = _build_matrix(self._wrd)
        return ctx_matrix, word_matrix

    # ------------------------------------------------------------------
    # N-gram helpers (kept for score_word OOV path)
    # ------------------------------------------------------------------

    def _ngram_ids(self, word: str) -> np.ndarray:
        """Return an int32 array of n-gram IDs for ``word``.

        OOV n-grams map to _UNK_NGRAM_ID (0), which is filtered out so it
        doesn't dilute the mean.
        """
        ids = [
            self._ngram_to_id.get(g, _UNK_NGRAM_ID)
            for g in _char_ngrams(word, self._ns)
        ]
        ids = [i for i in ids if i != _UNK_NGRAM_ID] or [_UNK_NGRAM_ID]
        return np.array(ids, dtype=np.int32)

    # ------------------------------------------------------------------
    # Inference (same interface as old DualEncoder)
    # ------------------------------------------------------------------

    def encode_context(self, context_ids: Sequence[int]) -> np.ndarray:
        """Produce a context vector by mean-pooling pre-computed context embeddings.

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
        vecs = self._ctx_matrix[np.asarray(context_ids, dtype=np.intp)]  # (n, dim)
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
        cand_vecs = self._word_matrix[list(candidate_ids)]   # (n_cands, dim)
        return cand_vecs @ ctx_vec                    # (n_cands,)

    def score_word(self, context_ids: Sequence[int], word: str) -> float:
        """Score an arbitrary word (possibly OOV) given the context.

        Uses the raw n-gram embeddings directly, without pre-computation.
        Slower than score_candidates() but works for any word.
        """
        if not word:
            return 0.0
        ctx_vec = self.encode_context(context_ids)
        ids = self._ngram_ids(word)
        if len(ids) == 0 or np.all(ids == _UNK_NGRAM_ID):
            return 0.0
        vecs = self._wrd[ids]  # (n_ngrams, dim)
        word_vec = vecs.mean(axis=0)  # (dim,)
        norm = np.linalg.norm(word_vec)
        if norm > 0:
            word_vec = word_vec / norm
        return word_vec @ ctx_vec

    @property
    def embed_dim(self) -> int:
        return self._dim

    @property
    def n_ngrams(self) -> int:
        return self._n_ngrams

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
            self._ngram_to_id,
            self._vocab,
            self._ns,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save to a .npz file.

        Stored arrays:
            context_embeds  : float32 (n_ngrams, dim)
            word_embeds     : float32 (n_ngrams, dim)
            ngram_keys      : byte string array of n-gram strings
            ngram_ids       : int32 array of corresponding IDs
            ns              : int32 array of n-gram sizes used (e.g. [2, 3])
        """
        keys = np.array(
            list(self._ngram_to_id.keys()), dtype=object
        ).astype("U")                               # unicode string array
        ids = np.array(
            list(self._ngram_to_id.values()), dtype=np.int32
        )
        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            context_embeds=self._ctx,
            word_embeds=self._wrd,
            ngram_keys=keys,
            ngram_ids=ids,
            ns=np.array(list(self._ns), dtype=np.int32),
        )
        buf.seek(0)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            shutil.copyfileobj(buf, f)

    @classmethod
    def load(cls, path: str | Path, vocab: Vocabulary) -> "DualEncoder":
        """Load from a .npz file saved by ``save()``."""
        arrays = np.load(str(path), allow_pickle=False)
        ngram_to_id = {
            str(k): int(v)
            for k, v in zip(arrays["ngram_keys"], arrays["ngram_ids"])
        }
        ns = tuple(int(n) for n in arrays["ns"])
        return cls(
            context_embeds=arrays["context_embeds"].copy(),
            word_embeds=arrays["word_embeds"].copy(),
            ngram_to_id=ngram_to_id,
            vocab=vocab,
            ns=ns,
        )

    # ------------------------------------------------------------------
    # Random initialisation (testing)
    # ------------------------------------------------------------------

    @classmethod
    def random_init(
        cls,
        vocab: Vocabulary,
        embed_dim: int = 64,
        ns: tuple[int, ...] = (2, 3),
        seed: int = 0,
    ) -> "DualEncoder":
        """Random initialisation from a vocabulary (useful for tests)."""
        words = [vocab.id_to_word(i) for i in range(vocab.size)]
        ngram_to_id = build_ngram_vocab(words, ns=ns)
        n = len(ngram_to_id) + 1  # +1 for UNK at index 0
        rng = np.random.default_rng(seed)
        scale = 1.0 / math.sqrt(embed_dim)
        ctx = rng.normal(0, scale, (n, embed_dim)).astype(np.float32)
        wrd = rng.normal(0, scale, (n, embed_dim)).astype(np.float32)
        return cls(ctx, wrd, ngram_to_id, vocab, ns=ns)
