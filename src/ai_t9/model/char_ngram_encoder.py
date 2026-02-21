"""CharNgramDualEncoder: open-vocabulary context-aware word scorer.

This is a drop-in replacement for DualEncoder that decouples model size from
vocabulary size and eliminates the need to retrain when new words are added to
the T9 dictionary.

Architecture
------------
Instead of a per-word-ID embedding lookup table, words are represented as the
**mean of their character n-gram embeddings** (fastText-style).  The model
stores two embedding matrices indexed by character n-gram ID rather than word ID:

    ctx_embeds[ngram_id]  — how ngram_id looks in context (preceding words)
    wrd_embeds[ngram_id]  — how ngram_id looks as a target word

At inference, for a word string ``w``:

    ngrams  = char_ngrams(w, ns=(2, 3))         # e.g. ["<th","the","he>","<t","th","he","e>"]
    ids     = [ngram_to_id[g] for g in ngrams]  # OOV n-grams → UNK_NGRAM_ID (0), ignored
    embed_w = mean(wrd_embeds[ids])              # (dim,)

Because every English word decomposes into bigrams and trigrams that are
virtually certain to have been seen during training, **new words added to the
T9 dictionary receive meaningful embeddings with zero retraining**.

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
    DualEncoder       :  29,470 × 264 × 2 × 4 B  ≈  62 MB RAM
    CharNgramDualEncoder:  8,734 × 264 × 2 × 4 B  ≈  18 MB RAM

Interface
---------
The public interface is identical to DualEncoder so T9Predictor requires no
changes — just load a CharNgramDualEncoder instead of a DualEncoder.

    encoder = CharNgramDualEncoder.load("model_ngram.npz", vocab)
    scores  = encoder.score_candidates(context_ids, candidate_ids)
"""

from __future__ import annotations

import io
import math
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np

from .vocab import Vocabulary


# Sentinel n-gram ID for n-grams not seen during training.  Embeddings at ID 0
# are zeroed out so that unknown n-grams contribute nothing to the mean.
_UNK_NGRAM_ID = 0


def _char_ngrams(word: str, ns: tuple[int, ...] = (2, 3)) -> list[str]:
    """Return all character n-grams for ``word`` with fastText-style boundaries.

    The word is wrapped in ``<`` and ``>`` to distinguish prefixes/suffixes
    from mid-word substrings.  For example ``"the"`` with ``ns=(2, 3)`` gives:
        ``["<t", "th", "he", "e>", "<th", "the", "he>"]``
    """
    marked = "<" + word.lower() + ">"
    ngrams: list[str] = []
    for n in ns:
        for i in range(len(marked) - n + 1):
            ngrams.append(marked[i : i + n])
    return ngrams


def build_ngram_vocab(
    words: Sequence[str],
    ns: tuple[int, ...] = (2, 3),
) -> dict[str, int]:
    """Build a {ngram: id} mapping from a collection of words.

    ID 0 is reserved for the UNK sentinel (unseen n-grams contribute zero).
    IDs are assigned in order of first appearance, deterministically sorted.
    """
    seen: set[str] = set()
    for w in words:
        seen.update(_char_ngrams(w, ns))
    # Sort for determinism; reserve 0 for UNK.
    ngram_to_id: dict[str, int] = {g: i + 1 for i, g in enumerate(sorted(seen))}
    return ngram_to_id


class CharNgramDualEncoder:
    """Open-vocabulary context-aware word scorer using character n-gram embeddings.

    Pure NumPy — no ML framework required at inference time.

    This is a drop-in replacement for DualEncoder with the same public interface.
    The key difference is that embeddings are indexed by character n-gram rather
    than word ID, so new words added to the T9 dictionary get embeddings
    automatically without any retraining.
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
            vocab:          Vocabulary (used only for word ID → string lookups)
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

        # Cache: word_id → pre-computed int32 n-gram ID array.
        # Filled lazily on first access; most queries touch a small candidate
        # set repeatedly across a typing session, so this pays off quickly.
        self._ngram_id_cache: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # N-gram helpers
    # ------------------------------------------------------------------

    def _ngram_ids(self, word: str) -> np.ndarray:
        """Return an int32 array of n-gram IDs for ``word``.

        OOV n-grams map to _UNK_NGRAM_ID (0), whose embedding row is zeroed.
        """
        ids = [
            self._ngram_to_id.get(g, _UNK_NGRAM_ID)
            for g in _char_ngrams(word, self._ns)
        ]
        # Filter UNKs so they don't dilute the mean — same effect as zeroing,
        # but avoids storing lots of zero rows in the mean computation.
        ids = [i for i in ids if i != _UNK_NGRAM_ID] or [_UNK_NGRAM_ID]
        return np.array(ids, dtype=np.int32)

    def _wid_ngram_ids(self, wid: int) -> np.ndarray:
        """Return cached n-gram ID array for vocab word ID ``wid``."""
        if wid not in self._ngram_id_cache:
            word = self._vocab.id_to_word(wid)
            self._ngram_id_cache[wid] = self._ngram_ids(word)
        return self._ngram_id_cache[wid]

    def _embed_word_ids(self, matrix: np.ndarray, wid: int) -> np.ndarray:
        """Mean-pool n-gram embeddings from ``matrix`` for word ID ``wid``."""
        ids = self._wid_ngram_ids(wid)
        return matrix[ids].mean(axis=0)

    # ------------------------------------------------------------------
    # Inference (same interface as DualEncoder)
    # ------------------------------------------------------------------

    def encode_context(self, context_ids: Sequence[int]) -> np.ndarray:
        """Produce a context vector by mean-pooling n-gram context embeddings.

        Each context word is first embedded via its n-grams; those per-word
        vectors are then mean-pooled and L2-normalised.
        If context_ids is empty, returns the zero vector.
        """
        if not context_ids:
            return np.zeros(self._dim, dtype=np.float32)
        word_vecs = np.stack(
            [self._embed_word_ids(self._ctx, wid) for wid in context_ids]
        )                                       # (n_ctx, dim)
        ctx_vec = word_vecs.mean(axis=0)        # (dim,)
        norm = np.linalg.norm(ctx_vec)
        if norm > 0:
            ctx_vec /= norm
        return ctx_vec.astype(np.float32)

    def score_candidates(
        self,
        context_ids: Sequence[int],
        candidate_ids: Sequence[int],
    ) -> np.ndarray:
        """Score each candidate word given the context.

        Returns a float32 array of shape (len(candidate_ids),).
        Higher = more likely given context.

        Interface is identical to DualEncoder.score_candidates so T9Predictor
        needs no changes.
        """
        if not candidate_ids:
            return np.array([], dtype=np.float32)
        ctx_vec = self.encode_context(context_ids)          # (dim,)
        cand_vecs = np.stack(
            [self._embed_word_ids(self._wrd, wid) for wid in candidate_ids]
        )                                                    # (n_cands, dim)
        return (cand_vecs @ ctx_vec).astype(np.float32)     # (n_cands,)

    def score_word(self, context_ids: Sequence[int], word: str) -> float:
        """Score an arbitrary word string (not necessarily in the vocab).

        This is the key open-vocabulary method: any word, even one never seen
        during training, can be scored against context.
        """
        ctx_vec = self.encode_context(context_ids)
        ids = self._ngram_ids(word)
        wrd_vec = self._wrd[ids].mean(axis=0)
        return float(wrd_vec @ ctx_vec)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

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
    def load(cls, path: str | Path, vocab: Vocabulary) -> "CharNgramDualEncoder":
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
    ) -> "CharNgramDualEncoder":
        """Random initialisation from a vocabulary (useful for tests)."""
        words = [vocab.id_to_word(i) for i in range(vocab.size)]
        ngram_to_id = build_ngram_vocab(words, ns=ns)
        n = len(ngram_to_id) + 1  # +1 for UNK at index 0
        rng = np.random.default_rng(seed)
        scale = 1.0 / math.sqrt(embed_dim)
        ctx = rng.normal(0, scale, (n, embed_dim)).astype(np.float32)
        wrd = rng.normal(0, scale, (n, embed_dim)).astype(np.float32)
        return cls(ctx, wrd, ngram_to_id, vocab, ns=ns)
