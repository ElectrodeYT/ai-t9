"""DualEncoder: open-vocabulary context-aware word scorer.

Architecture
------------
Words are represented as the **mean of their character n-gram embeddings**
(fastText-style), decoupling model size from vocabulary size.  The model
stores two embedding matrices indexed by character n-gram ID:

    ctx_embeds[ngram_id]  — how ngram_id looks in context (preceding words)
    wrd_embeds[ngram_id]  — how ngram_id looks as a target word

Context encoding — GRU with positional embeddings
--------------------------------------------------
The context window is encoded by a small GRU rather than plain mean pooling.
This captures word order (``"going to ___"`` differs from ``"to going ___"``)
and sequential dependencies that mean pooling discards.

Before the GRU, each context word embedding has a learned **positional
embedding** added to it, giving the GRU an explicit signal about each word's
position in the window (oldest → most-recent).  Padding positions (word_id=0)
are masked out before being fed to the GRU.

The GRU is a standard single-layer GRU (PyTorch convention) with
hidden_size = embed_dim.  Its weights are stored alongside the n-gram
embeddings in the .npz file and executed in pure NumPy at inference:

    σ(x)     = 1 / (1 + exp(-x))
    r_t = σ(W_ir x_t + b_ir + W_hr h_{t−1} + b_hr)
    z_t = σ(W_iz x_t + b_iz + W_hz h_{t−1} + b_hz)
    n_t = tanh(W_in x_t + b_in + r_t * (W_hn h_{t−1} + b_hn))
    h_t = (1 − z_t) * n_t + z_t * h_{t−1}

At load time, n-gram matrices are expanded into dense (vocab_size, dim)
per-word vectors for fast candidate scoring (single matmul at inference).

N-gram vocabulary
-----------------
Word boundaries are marked with ``<`` (start) and ``>`` (end), following
fastText convention.  For example ``"the"`` produces:
    2-grams: ``<t``, ``th``, ``he``, ``e>``
    3-grams: ``<th``, ``the``, ``he>``

Size comparison (dim=128, vocab_size=29,470)
--------------------------------------------
    n-gram mats  :   8,734 × 128 × 2 × 4 B  ≈   9 MB
    pre-computed :  29,470 × 128 × 2 × 4 B  ≈  30 MB
    GRU weights  :   4 × (3×128×128) × 4 B  ≈   0.8 MB
    pos_embed    :        window × 128 × 4 B  ≈  negligible
"""

from __future__ import annotations

import io
import math
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np

from .vocab import Vocabulary


_UNK_NGRAM_ID = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x.astype(np.float32)))


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
    ngram_to_id: dict[str, int] = {}
    for word in words:
        for gram in _char_ngrams(word, ns):
            if gram not in ngram_to_id:
                ngram_to_id[gram] = len(ngram_to_id) + 1  # 0 is reserved for UNK
    return ngram_to_id


# ---------------------------------------------------------------------------
# DualEncoder
# ---------------------------------------------------------------------------

class DualEncoder:
    """Open-vocabulary context-aware word scorer.

    Context words are encoded by a GRU (with positional embeddings on the
    inputs) rather than simple mean pooling.  Word embeddings use fastText-
    style character n-gram mean pooling.

    Pure NumPy — no ML framework required at inference time.
    """

    def __init__(
        self,
        context_embeds: np.ndarray,
        word_embeds: np.ndarray,
        ngram_to_id: dict[str, int],
        vocab: Vocabulary,
        ns: tuple[int, ...] = (2, 3),
        gru_weights: "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None" = None,
        pos_embed: "np.ndarray | None" = None,
        context_window: int = 3,
    ) -> None:
        """
        Args:
            context_embeds: float32 (n_ngrams, embed_dim)
            word_embeds:    float32 (n_ngrams, embed_dim)
            ngram_to_id:    mapping from n-gram string to row index
            vocab:          Vocabulary
            ns:             n-gram sizes used during training
            gru_weights:    (W_ih, W_hh, b_ih, b_hh) — PyTorch GRU convention.
                            W_ih / W_hh : (3*dim, dim), b_ih / b_hh : (3*dim,).
                            Gates stacked as [reset, update, new].
                            None falls back to mean-pooling (no GRU).
            pos_embed:      float32 (context_window, embed_dim) positional
                            embeddings added to context words before the GRU.
            context_window: number of preceding words in the context window.
        """
        assert context_embeds.shape == word_embeds.shape
        self._ctx = context_embeds.astype(np.float32)
        self._wrd = word_embeds.astype(np.float32)
        # Zero the UNK row so unknown n-grams contribute nothing.
        self._ctx[_UNK_NGRAM_ID] = 0.0
        self._wrd[_UNK_NGRAM_ID] = 0.0
        self._ngram_to_id = ngram_to_id
        self._vocab = vocab
        self._ns = ns
        self._dim = context_embeds.shape[1]
        self._n_ngrams = context_embeds.shape[0]
        self._context_window = context_window

        # GRU weights (None → fall back to mean-pooling for backward compat)
        if gru_weights is not None:
            w_ih, w_hh, b_ih, b_hh = gru_weights
            self._gru_W_ih = w_ih.astype(np.float32)
            self._gru_W_hh = w_hh.astype(np.float32)
            self._gru_b_ih = b_ih.astype(np.float32)
            self._gru_b_hh = b_hh.astype(np.float32)
        else:
            self._gru_W_ih = self._gru_W_hh = self._gru_b_ih = self._gru_b_hh = None

        # Positional embeddings: (context_window, dim)
        self._pos_embed = pos_embed.astype(np.float32) if pos_embed is not None else None

        # Pre-compute dense (vocab_size, dim) word matrices.
        self._ctx_matrix, self._word_matrix = self._precompute_word_matrices()

        # 1-slot context cache — eliminates redundant computation during a
        # typing burst where the context is unchanged.
        self._ctx_cache_key: tuple[int, ...] | None = None
        self._ctx_cache_vec: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Pre-computation
    # ------------------------------------------------------------------

    def _precompute_word_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Build dense (vocab_size, dim) word matrices from n-gram embeddings."""
        vocab_size = self._vocab.size
        rows: list[list[int]] = []
        for wid in range(vocab_size):
            word = self._vocab.id_to_word(wid)
            ids = [
                self._ngram_to_id.get(g, _UNK_NGRAM_ID)
                for g in _char_ngrams(word, self._ns)
            ]
            ids = [i for i in ids if i != _UNK_NGRAM_ID] or [_UNK_NGRAM_ID]
            rows.append(ids)

        max_n = max(len(r) for r in rows)
        table = np.zeros((vocab_size, max_n), dtype=np.int32)
        counts = np.zeros(vocab_size, dtype=np.float32)
        for i, row in enumerate(rows):
            table[i, : len(row)] = row
            counts[i] = len(row)

        def _build_matrix(embed: np.ndarray) -> np.ndarray:
            vecs = embed[table]              # (vocab_size, max_n, dim)
            summed = vecs.sum(axis=1)        # (vocab_size, dim)
            means = summed / counts[:, None] # (vocab_size, dim)
            norms = np.linalg.norm(means, axis=1, keepdims=True).clip(min=1e-8)
            return (means / norms).astype(np.float32)

        return _build_matrix(self._ctx), _build_matrix(self._wrd)

    # ------------------------------------------------------------------
    # N-gram helpers (OOV path)
    # ------------------------------------------------------------------

    def _ngram_ids(self, word: str) -> np.ndarray:
        ids = [
            self._ngram_to_id.get(g, _UNK_NGRAM_ID)
            for g in _char_ngrams(word, self._ns)
        ]
        ids = [i for i in ids if i != _UNK_NGRAM_ID] or [_UNK_NGRAM_ID]
        return np.array(ids, dtype=np.int32)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def encode_context(self, context_ids: Sequence[int]) -> np.ndarray:
        """Produce a context vector from preceding word IDs.

        Results are cached by context key.  Returns the zero vector if
        context_ids is empty.
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
        if self._gru_W_ih is not None:
            return self._encode_context_gru(context_ids)
        # Fallback: mean pooling (no GRU weights loaded)
        vecs = self._ctx_matrix[np.asarray(context_ids, dtype=np.intp)]
        ctx_vec = vecs.mean(axis=0)
        norm = np.linalg.norm(ctx_vec)
        return (ctx_vec / norm if norm > 1e-8 else ctx_vec).astype(np.float32)

    def _encode_context_gru(self, context_ids: Sequence[int]) -> np.ndarray:
        """Encode context via GRU with positional embeddings.

        Replicates the PyTorch training forward pass exactly:
          1. Build left-padded context window (zeros for missing positions).
          2. Look up L2-normalised word vectors from the pre-computed matrix.
          3. Add positional embeddings.
          4. Zero out padding positions (word_id == 0).
          5. GRU forward pass (left → right).
          6. L2-normalise the final hidden state.
        """
        window = self._context_window
        dim = self._dim

        # Build left-padded window: pad on left, real words on right.
        padded_ids = np.zeros(window, dtype=np.intp)
        n = min(len(context_ids), window)
        padded_ids[window - n :] = np.asarray(context_ids, dtype=np.intp)[-n:]

        # Word vectors (L2-normalised per word from pre-computed matrix).
        word_vecs = self._ctx_matrix[padded_ids].copy()     # (window, dim)

        # Positional embeddings — add before masking.
        if self._pos_embed is not None:
            word_vecs += self._pos_embed                    # (window, dim)

        # Normalise each position's (word + pos) vector to unit norm so the
        # GRU always receives unit-norm inputs regardless of the relative
        # magnitudes of the word and positional embeddings.  Mirrors the
        # F.normalize() applied in the training forward pass.
        norms = np.linalg.norm(word_vecs, axis=1, keepdims=True)
        word_vecs = np.where(norms > 1e-8, word_vecs / norms, word_vecs)

        # Mask out padding positions (zeroes the normalised padding vectors).
        mask = (padded_ids != 0).astype(np.float32)[:, None]  # (window, 1)
        word_vecs *= mask

        # GRU forward (PyTorch gate order: reset, update, new).
        H = dim
        W_ih = self._gru_W_ih   # (3H, H)
        W_hh = self._gru_W_hh   # (3H, H)
        b_ih = self._gru_b_ih   # (3H,)
        b_hh = self._gru_b_hh   # (3H,)
        h = np.zeros(H, dtype=np.float32)

        for x in word_vecs:
            ih = W_ih @ x + b_ih    # (3H,)
            hh = W_hh @ h + b_hh    # (3H,)
            r = _sigmoid(ih[:H] + hh[:H])
            z = _sigmoid(ih[H:2*H] + hh[H:2*H])
            n_gate = np.tanh(ih[2*H:] + r * hh[2*H:])
            h = (1.0 - z) * n_gate + z * h

        norm = np.linalg.norm(h)
        return (h / norm if norm > 1e-8 else h).astype(np.float32)

    def score_candidates(
        self,
        context_ids: Sequence[int],
        candidate_ids: Sequence[int],
    ) -> np.ndarray:
        """Score candidate words given context. Returns float32 (n_cands,)."""
        if not candidate_ids:
            return np.array([], dtype=np.float32)
        ctx_vec = self.encode_context(context_ids)
        cand_vecs = self._word_matrix[list(candidate_ids)]  # (n_cands, dim)
        return cand_vecs @ ctx_vec                          # (n_cands,)

    def score_word(self, context_ids: Sequence[int], word: str) -> float:
        """Score an arbitrary (possibly OOV) word given context."""
        if not word:
            return 0.0
        ctx_vec = self.encode_context(context_ids)
        # Fast path: use the precomputed L2-normalised matrix for in-vocab words.
        wid = self._vocab.word_to_id(word)
        if wid != _UNK_NGRAM_ID:
            return float(self._word_matrix[wid] @ ctx_vec)
        # OOV path: compute from n-gram embeddings on the fly.
        ids = self._ngram_ids(word)
        if len(ids) == 0 or np.all(ids == _UNK_NGRAM_ID):
            return 0.0
        vecs = self._wrd[ids]
        word_vec = vecs.mean(axis=0)
        norm = np.linalg.norm(word_vec)
        if norm > 1e-8:
            word_vec = word_vec / norm
        return float(word_vec @ ctx_vec)

    @property
    def embed_dim(self) -> int:
        return self._dim

    @property
    def n_ngrams(self) -> int:
        return self._n_ngrams

    @property
    def vocab(self) -> Vocabulary:
        return self._vocab

    @property
    def context_window(self) -> int:
        return self._context_window

    # ------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------

    def quantize_int8(self) -> "DualEncoder":
        """Return a new DualEncoder with int8-precision n-gram embeddings.

        The embeddings are quantized to int8 and immediately dequantized back
        to float32.  This does **not** reduce runtime memory usage; the benefit
        is a smaller ``.npz`` file on disk because the rounded float32 values
        compress better than the originals.  GRU weights and positional
        embeddings are kept in full float32.

        For actual runtime memory reduction the embedding matrices would need
        to stay as int8 with on-the-fly dequantization during scoring — that
        is a more invasive change left as a future enhancement.
        """
        def _quant(arr: np.ndarray) -> np.ndarray:
            scale = np.abs(arr).max(axis=1, keepdims=True).clip(min=1e-8)
            q = np.round(arr / scale * 127).clip(-127, 127).astype(np.int8)
            return (q.astype(np.float32) / 127.0) * scale

        gru_weights = None
        if self._gru_W_ih is not None:
            gru_weights = (
                self._gru_W_ih, self._gru_W_hh,
                self._gru_b_ih, self._gru_b_hh,
            )
        return DualEncoder(
            _quant(self._ctx),
            _quant(self._wrd),
            self._ngram_to_id,
            self._vocab,
            self._ns,
            gru_weights=gru_weights,
            pos_embed=self._pos_embed,
            context_window=self._context_window,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save to a .npz file."""
        keys = np.array(list(self._ngram_to_id.keys()), dtype=object).astype("U")
        ids = np.array(list(self._ngram_to_id.values()), dtype=np.int32)

        save_kwargs: dict = dict(
            context_embeds=self._ctx,
            word_embeds=self._wrd,
            ngram_keys=keys,
            ngram_ids=ids,
            ns=np.array(list(self._ns), dtype=np.int32),
            context_window=np.array(self._context_window, dtype=np.int32),
        )
        if self._gru_W_ih is not None:
            save_kwargs["gru_W_ih"] = self._gru_W_ih
            save_kwargs["gru_W_hh"] = self._gru_W_hh
            save_kwargs["gru_b_ih"] = self._gru_b_ih
            save_kwargs["gru_b_hh"] = self._gru_b_hh
        if self._pos_embed is not None:
            save_kwargs["pos_embed"] = self._pos_embed

        buf = io.BytesIO()
        np.savez_compressed(buf, **save_kwargs)
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
        context_window = int(arrays["context_window"]) if "context_window" in arrays else 3

        gru_weights = None
        if "gru_W_ih" in arrays:
            gru_weights = (
                arrays["gru_W_ih"].copy(),
                arrays["gru_W_hh"].copy(),
                arrays["gru_b_ih"].copy(),
                arrays["gru_b_hh"].copy(),
            )
        pos_embed = arrays["pos_embed"].copy() if "pos_embed" in arrays else None

        return cls(
            context_embeds=arrays["context_embeds"].copy(),
            word_embeds=arrays["word_embeds"].copy(),
            ngram_to_id=ngram_to_id,
            vocab=vocab,
            ns=ns,
            gru_weights=gru_weights,
            pos_embed=pos_embed,
            context_window=context_window,
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
        context_window: int = 3,
    ) -> "DualEncoder":
        """Random initialisation from a vocabulary (useful for tests)."""
        words = [vocab.id_to_word(i) for i in range(vocab.size)]
        ngram_to_id = build_ngram_vocab(words, ns=ns)
        n = len(ngram_to_id) + 1  # +1 for UNK at index 0
        rng = np.random.default_rng(seed)
        scale = 1.0 / math.sqrt(embed_dim)
        ctx = rng.normal(0, scale, (n, embed_dim)).astype(np.float32)
        wrd = rng.normal(0, scale, (n, embed_dim)).astype(np.float32)
        gru_scale = 1.0 / math.sqrt(embed_dim)
        gru_W_ih = rng.normal(0, gru_scale, (3 * embed_dim, embed_dim)).astype(np.float32)
        gru_W_hh = rng.normal(0, gru_scale, (3 * embed_dim, embed_dim)).astype(np.float32)
        gru_b_ih = np.zeros(3 * embed_dim, dtype=np.float32)
        gru_b_hh = np.zeros(3 * embed_dim, dtype=np.float32)
        pos_embed = rng.normal(0, scale, (context_window, embed_dim)).astype(np.float32)
        return cls(
            ctx, wrd, ngram_to_id, vocab, ns=ns,
            gru_weights=(gru_W_ih, gru_W_hh, gru_b_ih, gru_b_hh),
            pos_embed=pos_embed,
            context_window=context_window,
        )
