"""T9Predictor: the main public API combining dictionary, n-gram, and dual-encoder."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .dictionary import T9Dictionary
from .model.vocab import Vocabulary
from .model.dual_encoder import DualEncoder
from .ngram import BigramScorer
from .t9_map import is_valid_digit_sequence


class RankedCandidate:
    """A single ranked prediction with score breakdown for introspection."""

    __slots__ = ("word", "word_id", "freq_score", "model_score", "ngram_score", "final_score")

    def __init__(
        self,
        word: str,
        word_id: int,
        freq_score: float,
        model_score: float,
        ngram_score: float,
        final_score: float,
    ) -> None:
        self.word = word
        self.word_id = word_id
        self.freq_score = freq_score
        self.model_score = model_score
        self.ngram_score = ngram_score
        self.final_score = final_score

    def __repr__(self) -> str:
        return (
            f"RankedCandidate({self.word!r}, final={self.final_score:.3f}, "
            f"freq={self.freq_score:.3f}, model={self.model_score:.3f}, "
            f"ngram={self.ngram_score:.3f})"
        )


class T9Predictor:
    """Context-aware T9 word predictor.

    Ranks candidate words for a given digit sequence using up to three signals:

    1. **freq** (always on): log-frequency of the word in the training corpus.
    2. **model** (if loaded): dual-encoder cosine similarity between a mean-
       pooled context embedding and the candidate word embedding.
    3. **ngram** (if loaded): bigram log-probability P(candidate | prev_word).

    Final score::

        score = w_freq * freq_score + w_model * model_score + w_ngram * ngram_score

    All three weights are configurable at construction time.  The weights are
    normalised automatically (they don't need to sum to 1.0).

    Signals that are not loaded contribute a weight of 0.
    """

    def __init__(
        self,
        dictionary: T9Dictionary,
        model: DualEncoder | None = None,
        ngram: BigramScorer | None = None,
        w_freq: float = 0.35,
        w_model: float = 0.50,
        w_ngram: float = 0.15,
    ) -> None:
        self._dict = dictionary
        self._vocab = dictionary.vocab
        self._model = model
        self._ngram = ngram

        # Effective weights (zero out missing signals and renormalise)
        wf = w_freq
        wm = w_model if model is not None else 0.0
        wn = w_ngram if ngram is not None else 0.0
        total = wf + wm + wn
        if total == 0:
            raise ValueError("At least one signal weight must be > 0")
        self._wf = wf / total
        self._wm = wm / total
        self._wn = wn / total

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        digit_seq: str,
        context: Sequence[str] = (),
        top_k: int = 5,
        completions: bool = False,
        max_extra_digits: int | None = None,
        w_length: float = 0.30,
        min_model_score: float = 0.0,
        return_details: bool = False,
        trace: bool = False,
    ) -> "list[str] | list[RankedCandidate] | tuple[list[RankedCandidate], dict]":
        """Return the top-k predicted words for the given digit sequence.

        Args:
            digit_seq:        T9 digit string, e.g. ``"4663"``
            context:          Previously typed words (most recent last),
                              e.g. ``["i", "am", "going"]``
            top_k:            Number of results to return (default 5)
            completions:      If True, predict completions extending the digit prefix
                              instead of exact matches.
            max_extra_digits: Maximum extra digits for completions.  ``None`` (default)
                              activates adaptive scaling based on prefix length.
                              Ignored when completions=False.
            w_length:         Weight for length bonus in completions (ignored if completions=False)
            min_model_score:  When > 0 and a model is loaded, completion candidates whose
                              normalised model score falls below this threshold are filtered out.
                              Ignored for exact-match prediction and when no model is loaded.
            return_details:   If True, return RankedCandidate objects with score
                              breakdown instead of plain strings.
            trace:            If True, return a tuple of (ranked_candidates, trace_dict)
                              instead of just the results. Forces return_details=True.

        Returns:
            List of word strings (default) or RankedCandidate objects, or tuple with trace.
        """
        return self._predict_core(
            digit_seq=digit_seq,
            context=context,
            top_k=top_k,
            is_completions=completions,
            max_extra_digits=max_extra_digits,
            w_length=w_length,
            min_model_score=min_model_score,
            return_details=return_details,
            trace=trace,
        )

    def predict_completions(
        self,
        digit_prefix: str,
        context: Sequence[str] = (),
        top_k: int = 5,
        max_extra_digits: int | None = None,
        w_length: float = 0.30,
        min_model_score: float = 0.0,
        return_details: bool = False,
    ) -> "list[str] | list[RankedCandidate]":
        """Return top-k word completions extending the given digit prefix.

        Like :meth:`predict` but searches for words *longer* than the typed
        digits.  A **length bonus** signal favours shorter completions (fewer
        remaining keypresses).

        Completions are scaled adaptively by default (``max_extra_digits=None``):
        short prefixes are conservative (huge fan-out = noisy), long prefixes
        are expansive (few candidates = reliable).  Pass an explicit integer to
        override this behaviour.

        Scoring::

            base_blend = wf * freq + wm * model + wn * ngram   # existing signals
            final      = (1 - w_length) * base_blend + w_length * length_bonus

        ``length_bonus`` is rank-normalised — the shortest completion(s) among
        the candidates receive the highest score.  When all candidates have the
        same extra length, the signal is zero and only the base signals decide.

        Args:
            digit_prefix:     Partial T9 digit string, e.g. ``"466"``
            context:          Previously typed words (most recent last)
            top_k:            Number of results to return (adaptive mode may reduce this)
            max_extra_digits: Maximum extra digits beyond the prefix.  ``None`` → adaptive.
            w_length:         Weight for the completion-length bonus ∈ [0, 1).
            min_model_score:  Filter completions below this normalised model score.
            return_details:   If True, return :class:`RankedCandidate` objects.

        Returns:
            List of word strings (default) or RankedCandidate objects.
        """
        return self.predict(
            digit_seq=digit_prefix,
            context=context,
            top_k=top_k,
            completions=True,
            max_extra_digits=max_extra_digits,
            w_length=w_length,
            min_model_score=min_model_score,
            return_details=return_details,
            trace=False,
        )

    def predict_with_trace(
        self,
        digit_seq: str,
        context: Sequence[str] = (),
        top_k: int = 5,
    ) -> "tuple[list[RankedCandidate], dict]":
        """Like predict(return_details=True) but also returns a trace dict.

        The trace dict exposes every intermediate array so that debug tooling
        can render per-stage score breakdowns without reimplementing scoring.

        Returns:
            (ranked_candidates, trace)

        Trace keys:
            digit_seq       – the input sequence
            context         – the context list used
            dict_hits       – total candidates from the dictionary (before top_k)
            candidates_raw  – list of (word, wid) in original dict order
            freq_raw        – raw log-freq array (same order as dict_hits)
            freq_norm       – min-max normalised freq_raw
            model_raw       – raw model cosine-sim array, or None
            model_norm      – normalised model_raw, or None
            ngram_raw       – raw bigram log-prob array, or None
            ngram_norm      – normalised ngram_raw, or None
            final           – final blended score array
            order           – argsort indices (descending) used for ranking
            weights         – effective weights dict
            timing_ms       – wall-clock ms for each pipeline stage;
                              keys: "dict", "freq", "model", "ngram", "blend", "total"
        """
        result = self.predict(
            digit_seq=digit_seq,
            context=context,
            top_k=top_k,
            completions=False,
            return_details=True,
            trace=True,
        )
        assert isinstance(result, tuple)
        return result

    def _predict_core(
        self,
        digit_seq: str,
        context: Sequence[str],
        top_k: int,
        is_completions: bool,
        max_extra_digits: int | None = None,
        w_length: float = 0.30,
        min_model_score: float = 0.0,
        return_details: bool = False,
        trace: bool = False,
    ) -> "list[str] | list[RankedCandidate] | tuple[list[RankedCandidate], dict]":
        """Core prediction logic shared by all public methods."""
        if not is_valid_digit_sequence(digit_seq):
            raise ValueError(
                f"Invalid digit sequence {digit_seq!r}. "
                "Use digits 2-9 only (T9 keypad)."
            )

        # Resolve adaptive completion parameters.
        if is_completions:
            if max_extra_digits is None:
                max_extra_digits, top_k = _adaptive_completion_params(len(digit_seq), top_k)
            # else: caller supplied explicit values, use them as-is

        _t0 = time.perf_counter_ns() if trace else 0

        # ── Stage 1: dict lookup ──────────────────────────────────────
        _td0 = time.perf_counter_ns() if trace else 0
        if is_completions:
            candidates = self._dict.prefix_lookup(digit_seq, max_extra_digits=max_extra_digits)
            if candidates:
                words, word_ids, full_digits = zip(*candidates)
            else:
                words, word_ids, full_digits = (), (), ()
        else:
            candidates = self._dict.lookup(digit_seq)
            if candidates:
                words, word_ids = zip(*candidates)
                full_digits = None
            else:
                words, word_ids, full_digits = (), (), None
        _td1 = time.perf_counter_ns() if trace else 0

        if not candidates:
            if trace:
                empty_trace: dict = {
                    "digit_seq": digit_seq,
                    "context": list(context),
                    "dict_hits": 0,
                    "candidates_raw": [],
                    "freq_raw": np.array([], dtype=np.float32),
                    "freq_norm": np.array([], dtype=np.float32),
                    "model_raw": None,
                    "model_norm": None,
                    "ngram_raw": None,
                    "ngram_norm": None,
                    "final": np.array([], dtype=np.float32),
                    "order": np.array([], dtype=np.intp),
                    "weights": self.weights,
                    "timing_ms": {
                        "dict": (_td1 - _td0) / 1e6 if trace else 0.0,
                        "freq": 0.0, "model": 0.0, "ngram": 0.0, "blend": 0.0,
                        "total": (time.perf_counter_ns() - _t0) / 1e6 if trace else 0.0,
                    },
                }
                return [], empty_trace
            else:
                return []

        word_ids_list = list(word_ids)

        # ── Stage 2: frequency scoring ────────────────────────────────
        _tf0 = time.perf_counter_ns() if trace else 0
        freq_raw = np.array(
            [self._vocab.logfreq(wid) for wid in word_ids_list], dtype=np.float32
        )
        freq_norm = _normalise(freq_raw)
        _tf1 = time.perf_counter_ns() if trace else 0

        # ── Stage 3: model scoring ────────────────────────────────────
        _tm0 = time.perf_counter_ns() if trace else 0
        if self._model is not None and self._wm > 0:
            ctx_ids = self._vocab.words_to_ids(list(context))
            model_raw = self._model.score_candidates(ctx_ids, word_ids_list)
            model_norm = _normalise(model_raw)
        else:
            model_raw = None
            model_norm = np.zeros(len(word_ids_list), dtype=np.float32)
        _tm1 = time.perf_counter_ns() if trace else 0

        # ── Stage 4: ngram scoring ────────────────────────────────────
        _tn0 = time.perf_counter_ns() if trace else 0
        if self._ngram is not None and self._wn > 0 and context:
            prev_id = self._vocab.word_to_id(context[-1].lower())
            ngram_raw = np.array(
                self._ngram.score_candidates(prev_id, word_ids_list), dtype=np.float32
            )
            ngram_norm = _normalise(ngram_raw)
        else:
            ngram_raw = None
            ngram_norm = np.zeros(len(word_ids_list), dtype=np.float32)
        _tn1 = time.perf_counter_ns() if trace else 0

        # ── Stage 4b: model-score gate (completions only) ────────────
        if is_completions and min_model_score > 0.0 and model_norm is not None:
            keep = model_norm >= min_model_score
            if keep.any() and not keep.all():
                words = tuple(w for w, k in zip(words, keep) if k)
                word_ids_list = [wid for wid, k in zip(word_ids_list, keep) if k]
                full_digits = tuple(fd for fd, k in zip(full_digits, keep) if k)
                freq_raw = freq_raw[keep]
                freq_norm = freq_norm[keep]
                model_raw = model_raw[keep] if model_raw is not None else None
                model_norm = model_norm[keep]
                ngram_raw = ngram_raw[keep] if ngram_raw is not None else None
                ngram_norm = ngram_norm[keep]

        # ── Stage 5: length bonus (completions only) ──────────────────
        if is_completions:
            prefix_len = len(digit_seq)
            extra = np.array(
                [len(fd) - prefix_len for fd in full_digits], dtype=np.float32
            )
            length_norm = _normalise(-extra)
        else:
            length_norm = None

        # ── Stage 6: blend + sort ─────────────────────────────────────
        _tb0 = time.perf_counter_ns() if trace else 0
        base_blend = (
            self._wf * freq_norm
            + self._wm * model_norm
            + self._wn * ngram_norm
        )
        if is_completions:
            w_len = max(0.0, min(float(w_length), 0.99))
            final = (1.0 - w_len) * base_blend + w_len * length_norm
        else:
            final = base_blend

        order = np.argsort(-final)
        top_indices = order[:top_k]

        ranked = [
            RankedCandidate(
                word=words[i],
                word_id=word_ids_list[i],
                freq_score=float(freq_norm[i]),
                model_score=float(model_norm[i]),
                ngram_score=float(ngram_norm[i]),
                final_score=float(final[i]),
            )
            for i in top_indices
        ]
        _tb1 = time.perf_counter_ns() if trace else 0
        _t1 = time.perf_counter_ns() if trace else 0

        if trace:
            trace_dict: dict = {
                "digit_seq": digit_seq,
                "context": list(context),
                "dict_hits": len(candidates),
                "candidates_raw": list(zip(words, word_ids_list)),
                "freq_raw": freq_raw,
                "freq_norm": freq_norm,
                "model_raw": model_raw,
                "model_norm": model_norm if model_raw is not None else None,
                "ngram_raw": ngram_raw,
                "ngram_norm": ngram_norm if ngram_raw is not None else None,
                "final": final,
                "order": order,
                "weights": self.weights,
                "timing_ms": {
                    "dict": (_td1 - _td0) / 1e6,
                    "freq": (_tf1 - _tf0) / 1e6,
                    "model": (_tm1 - _tm0) / 1e6,
                    "ngram": (_tn1 - _tn0) / 1e6,
                    "blend": (_tb1 - _tb0) / 1e6,
                    "total": (_t1 - _t0) / 1e6,
                },
            }
            return ranked, trace_dict

        if return_details:
            return ranked
        else:
            return [rc.word for rc in ranked]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def has_model(self) -> bool:
        return self._model is not None

    @property
    def has_ngram(self) -> bool:
        return self._ngram is not None

    @property
    def weights(self) -> dict[str, float]:
        return {"freq": self._wf, "model": self._wm, "ngram": self._wn}

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_files(
        cls,
        vocab_path: str | Path,
        dict_path: str | Path,
        model_path: str | Path | None = None,
        ngram_path: str | Path | None = None,
        **kwargs,
    ) -> "T9Predictor":
        """Load a predictor from saved files.

        At minimum, vocab_path and dict_path are required.
        Pass model_path and/or ngram_path to enable those signals.

        The model type (DualEncoder vs CharNgramDualEncoder) is detected
        automatically by inspecting the .npz file contents — callers do not
        need to specify which type was trained.
        """
        vocab = Vocabulary.load(vocab_path)
        dictionary = T9Dictionary.load(dict_path, vocab)
        model = _load_model_auto(model_path, vocab) if model_path else None
        ngram = BigramScorer.load(ngram_path, vocab) if ngram_path else None
        return cls(dictionary, model=model, ngram=ngram, **kwargs)

    @classmethod
    def build_default(
        cls,
        max_words: int = 50_000,
        verbose: bool = True,
        with_ngram: bool = True,
    ) -> "T9Predictor":
        """Build a predictor from NLTK data with no pre-trained model.

        This downloads the NLTK Brown corpus (~15 MB on first use) and builds
        the vocabulary, dictionary, and optionally the bigram scorer in-memory.
        No trained dual-encoder model is loaded; ranking uses frequency only
        (+ bigram if with_ngram=True).

        To add the neural model, train with DualEncoderTrainer and then
        load with T9Predictor.from_files().
        """
        vocab = Vocabulary.build_from_nltk(max_words=max_words, verbose=verbose)
        dictionary = T9Dictionary.build(vocab, verbose=verbose)

        ngram = None
        if with_ngram:
            ngram = BigramScorer.build_from_nltk(vocab, verbose=verbose)

        # With no model, only freq + ngram are active; adjust weights
        return cls(
            dictionary,
            model=None,
            ngram=ngram,
            w_freq=0.5,
            w_model=0.0,
            w_ngram=0.5,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model_auto(path: "str | Path", vocab: Vocabulary) -> "DualEncoder":
    """Load a DualEncoder or CharNgramDualEncoder by inspecting the .npz file."""
    import numpy as np
    arrays = np.load(str(path))
    if "ngram_keys" in arrays:
        from .model.char_ngram_encoder import CharNgramDualEncoder
        return CharNgramDualEncoder.load(path, vocab)
    return DualEncoder.load(path, vocab)


def _adaptive_completion_params(prefix_len: int, top_k: int) -> tuple[int, int]:
    """Return (max_extra_digits, effective_top_k) scaled to prefix length.

    Short prefixes → conservative (huge fan-out, completions are noisy).
    Long prefixes  → expansive (few candidates, completions are reliable).

    prefix_len | max_extra_digits | effective_top_k
        1-2    |       1          |   1
         3     |       2          |   2
         4     |       3          |   max(2, top_k // 2)
         5     |       4          |   max(3, top_k * 2 // 3)
        6+     |       5          |   top_k
    """
    if prefix_len <= 2:
        return 1, 1
    if prefix_len == 3:
        return 2, 2
    if prefix_len == 4:
        return 3, max(2, top_k // 2)
    if prefix_len == 5:
        return 4, max(3, top_k * 2 // 3)
    return 5, top_k


def _normalise(scores: np.ndarray) -> np.ndarray:
    """Rank-based normalisation to [0, 1].

    Each score is replaced by its fractional rank: ``rank / (n - 1)`` for
    *n* candidates, producing values in [0, 1] where 1.0 is the highest-
    scoring candidate and 0.0 is the lowest.  Ties receive the same rank
    (mean of the ranks they would span).

    Compared to min-max normalisation, this is:
      - **invariant** to the raw score distribution (log-probs, cosine sims,
        whatever) — only the relative ordering matters;
      - **robust** to outliers — a single extreme score cannot dominate;
      - **stable** across queries — the same relative rank always maps to
        the same normalised value, regardless of what other candidates happen
        to be in the set.

    With a single candidate (or all-identical scores), returns all zeros so
    the signal contributes nothing — identical behaviour to the old min-max.
    """
    n = len(scores)
    if n <= 1:
        return np.zeros_like(scores)
    # argsort twice gives rank (0-based, ascending)
    order = scores.argsort()
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.arange(n, dtype=np.float32)
    # Handle ties: average the ranks of tied values
    sorted_scores = scores[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] - sorted_scores[i] < 1e-12:
            j += 1
        if j > i + 1:
            avg_rank = np.mean(np.arange(i, j, dtype=np.float32))
            for idx in range(i, j):
                ranks[order[idx]] = avg_rank
        i = j
    # All-identical scores → all ranks equal → return zeros (no signal)
    denom = n - 1
    result = ranks / denom
    if result.max() - result.min() < 1e-9:
        return np.zeros(n, dtype=np.float32)
    return result
