"""Bigram language model with Laplace (add-k) smoothing.

Serves as a fast, always-available scoring fallback and can be blended into
the final ranking alongside the dual-encoder model.  No ML framework required.

Storage format
--------------
Bigram counts are stored as a **Compressed Sparse Row (CSR)** NumPy structure
rather than nested Python dicts.  For a 29 k vocab with ~2.8 M observed pairs
this cuts runtime RAM from ~310 MB (Python dict overhead) down to ~22 MB:

    row_ptr  : int32 (vocab_size + 1,)  — row slice boundaries
    col_idx  : int32 (n_bigrams,)       — next-word IDs, sorted within each row
    counts   : int32 (n_bigrams,)       — co-occurrence counts
    unigrams : int32 (vocab_size,)      — per-word totals

Serialisation uses a .npz file (~11 MB compressed).  The legacy JSON format is
still accepted by load() for backward compatibility (with a deprecation warning)
and the CSR structure is rebuilt automatically on load.
"""

from __future__ import annotations

import io
import json
import math
import shutil
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from .model.vocab import Vocabulary


class BigramScorer:
    """Bigram language model: P(word | prev_word) with add-k smoothing.

    Trained from a sequence of word IDs.  Stores bigram counts as a NumPy CSR
    sparse matrix so it stays small at runtime; probabilities are computed on
    the fly with smoothing applied.
    """

    def __init__(self, vocab: Vocabulary, k: float = 0.5) -> None:
        self._vocab = vocab
        self._k = k
        # Raw counters used during training (freed after CSR build).
        self._raw_bigrams: dict[int, dict[int, int]] | None = defaultdict(
            lambda: defaultdict(int)
        )
        self._raw_unigrams: dict[int, int] | None = defaultdict(int)
        self._built = False
        # CSR arrays — set by _build_csr(), None until then.
        self._row_ptr: np.ndarray | None = None   # int32 (vocab_size+1,)
        self._col_idx: np.ndarray | None = None   # int32 (n_bigrams,)
        self._counts:  np.ndarray | None = None   # int32 (n_bigrams,)
        self._unigrams: np.ndarray | None = None  # int32 (vocab_size,)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_on_ids(self, word_ids: list[int]) -> None:
        """Accumulate bigram counts from a flat sequence of word IDs."""
        if self._built:
            raise RuntimeError(
                "Cannot call train_on_ids() after the model has been built/saved/used for scoring."
            )
        for i, wid in enumerate(word_ids):
            self._raw_unigrams[wid] += 1  # type: ignore[index]
            if i > 0:
                self._raw_bigrams[word_ids[i - 1]][wid] += 1  # type: ignore[index]

    def _build_csr(self) -> None:
        """Convert raw accumulator dicts into compact CSR NumPy arrays."""
        vs = self._vocab.size
        raw_bg = self._raw_bigrams or {}
        raw_uni = self._raw_unigrams or {}

        # Unigrams
        uni = np.zeros(vs, dtype=np.int32)
        for wid, cnt in raw_uni.items():
            if 0 <= wid < vs:
                uni[wid] = cnt

        # Build row_ptr
        row_ptr = np.zeros(vs + 1, dtype=np.int32)
        for prev in raw_bg:
            if 0 <= prev < vs:
                row_ptr[prev + 1] = len(raw_bg[prev])
        np.cumsum(row_ptr, out=row_ptr)

        n = int(row_ptr[-1])
        col_idx = np.empty(n, dtype=np.int32)
        counts  = np.empty(n, dtype=np.int32)

        for prev in range(vs):
            entries = raw_bg.get(prev)
            if not entries:
                continue
            start = int(row_ptr[prev])
            sorted_pairs = sorted(entries.items())  # sorted by next_id
            for i, (nxt, cnt) in enumerate(sorted_pairs):
                col_idx[start + i] = nxt
                counts[start + i]  = cnt

        self._row_ptr  = row_ptr
        self._col_idx  = col_idx
        self._counts   = counts
        self._unigrams = uni
        self._built    = True
        # Free the training dicts — this is the primary RAM saving.
        self._raw_bigrams  = None
        self._raw_unigrams = None

    @property
    def n_unique_contexts(self) -> int:
        """Number of distinct first-words (rows) that have at least one bigram entry."""
        self._ensure_built()
        import numpy as _np
        return int(_np.count_nonzero(_np.diff(self._row_ptr)))  # type: ignore[arg-type]

    @property
    def n_bigram_pairs(self) -> int:
        """Total number of stored (prev, next) bigram pairs."""
        self._ensure_built()
        return int(len(self._col_idx))  # type: ignore[arg-type]

    def _ensure_built(self) -> None:
        if not self._built:
            self._build_csr()

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _lookup_count(self, prev_id: int, next_id: int) -> int:
        """Return the raw bigram count for (prev_id, next_id), or 0."""
        vs = self._vocab.size
        if not (0 <= prev_id < vs):
            return 0
        start = int(self._row_ptr[prev_id])       # type: ignore[index]
        end   = int(self._row_ptr[prev_id + 1])   # type: ignore[index]
        if start == end:
            return 0
        row = self._col_idx[start:end]             # type: ignore[index]
        pos = int(np.searchsorted(row, next_id))
        if pos < len(row) and row[pos] == next_id:
            return int(self._counts[start + pos])  # type: ignore[index]
        return 0

    def log_prob(self, prev_word_id: int, candidate_id: int) -> float:
        """Log P(candidate | prev_word) with add-k smoothing."""
        self._ensure_built()
        v  = self._vocab.size
        k  = self._k
        count_prev   = int(self._unigrams[prev_word_id]) if 0 <= prev_word_id < v else 0  # type: ignore[index]
        count_bigram = self._lookup_count(prev_word_id, candidate_id)
        return math.log((count_bigram + k) / (count_prev + k * v))

    def score_candidates(
        self,
        prev_word_id: int,
        candidate_ids: list[int],
    ) -> list[float]:
        """Return log P(candidate | prev_word) for each candidate.

        Extracts the CSR row for prev_word once and binary-searches for each
        candidate, avoiding repeated row look-ups.
        """
        self._ensure_built()
        if not candidate_ids:
            return []
        vs = self._vocab.size
        k  = self._k

        count_prev = int(self._unigrams[prev_word_id]) if 0 <= prev_word_id < vs else 0  # type: ignore[index]
        denom = count_prev + k * vs

        # Extract the CSR row for prev_word once.
        if 0 <= prev_word_id < vs:
            start    = int(self._row_ptr[prev_word_id])      # type: ignore[index]
            end      = int(self._row_ptr[prev_word_id + 1])  # type: ignore[index]
            row_cols = self._col_idx[start:end]               # type: ignore[index]
            row_data = self._counts[start:end]                # type: ignore[index]
        else:
            row_cols = np.empty(0, dtype=np.int32)
            row_data = np.empty(0, dtype=np.int32)

        results: list[float] = []
        for cid in candidate_ids:
            pos = int(np.searchsorted(row_cols, cid))
            cnt = float(row_data[pos]) if (pos < len(row_cols) and row_cols[pos] == cid) else 0.0
            results.append(math.log((cnt + k) / denom))
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Save the model to a .npz file.

        The path extension is rewritten to .npz if .json was given (legacy).
        Returns the actual path written.
        """
        self._ensure_built()
        path = Path(path)
        if path.suffix == ".json":
            path = path.with_suffix(".npz")

        buf = io.BytesIO()
        np.savez_compressed(
            buf,
            row_ptr  = self._row_ptr,
            col_idx  = self._col_idx,
            counts   = self._counts,
            unigrams = self._unigrams,
            k        = np.array(self._k, dtype=np.float64),
        )
        buf.seek(0)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            shutil.copyfileobj(buf, f)
        return path

    @classmethod
    def load(cls, path: str | Path, vocab: Vocabulary) -> "BigramScorer":
        """Load a bigram model.

        Accepts .npz (compact, current format) or the legacy .json format.
        When given a .json path (or a stem path with no extension), also tries
        the .npz sibling so callers that still reference 'bigram.json' pick up
        a re-saved compact file automatically.
        """
        path = Path(path)

        # Resolution order: exact path → .npz sibling → .json sibling
        candidates = [path]
        if path.suffix != ".npz":
            candidates.append(path.with_suffix(".npz"))
        if path.suffix != ".json":
            candidates.append(path.with_suffix(".json"))

        for p in candidates:
            if not p.exists():
                continue
            if p.suffix == ".json":
                return cls._load_json(p, vocab)
            return cls._load_npz(p, vocab)

        raise FileNotFoundError(
            f"BigramScorer: could not find model at {path} "
            "(tried .npz and .json variants)"
        )

    @classmethod
    def _load_npz(cls, path: Path, vocab: Vocabulary) -> "BigramScorer":
        data = np.load(str(path))
        scorer = cls(vocab, k=float(data["k"]))
        scorer._row_ptr  = data["row_ptr"]
        scorer._col_idx  = data["col_idx"]
        scorer._counts   = data["counts"]
        scorer._unigrams = data["unigrams"]
        scorer._built    = True
        scorer._raw_bigrams  = None
        scorer._raw_unigrams = None
        return scorer

    @classmethod
    def _load_json(cls, path: Path, vocab: Vocabulary) -> "BigramScorer":
        """Load legacy JSON format and immediately build CSR structure."""
        warnings.warn(
            f"BigramScorer: loading legacy JSON from {path}. "
            "Re-save with scorer.save() to use the compact .npz format "
            "(reduces runtime RAM from ~300 MB to ~22 MB).",
            DeprecationWarning,
            stacklevel=3,
        )
        raw = json.loads(path.read_text(encoding="utf-8"))
        vs  = vocab.size
        scorer = cls(vocab, k=raw["k"])

        # Build unigrams array directly.
        uni = np.zeros(vs, dtype=np.int32)
        for k_str, cnt in raw["unigrams"].items():
            wid = int(k_str)
            if 0 <= wid < vs:
                uni[wid] = cnt

        # Build CSR directly from the JSON structure (grouped by prev_id).
        rows: dict[int, list[tuple[int, int]]] = {}
        for prev_str, nexts in raw["bigrams"].items():
            prev = int(prev_str)
            if not (0 <= prev < vs):
                continue
            entries = [(int(nxt), cnt) for nxt, cnt in nexts.items() if 0 <= int(nxt) < vs]
            entries.sort()
            rows[prev] = entries

        row_ptr = np.zeros(vs + 1, dtype=np.int32)
        for prev in range(vs):
            row_ptr[prev + 1] = row_ptr[prev] + len(rows.get(prev, []))

        n = int(row_ptr[-1])
        col_idx = np.empty(n, dtype=np.int32)
        counts  = np.empty(n, dtype=np.int32)
        for prev in range(vs):
            start   = int(row_ptr[prev])
            entries = rows.get(prev, [])
            for i, (nxt, cnt) in enumerate(entries):
                col_idx[start + i] = nxt
                counts[start + i]  = cnt

        scorer._row_ptr  = row_ptr
        scorer._col_idx  = col_idx
        scorer._counts   = counts
        scorer._unigrams = uni
        scorer._built    = True
        scorer._raw_bigrams  = None
        scorer._raw_unigrams = None
        return scorer
