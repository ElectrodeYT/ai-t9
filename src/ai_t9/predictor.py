"""T9Predictor: the main public API combining dictionary, n-gram, and dual-encoder."""

from __future__ import annotations

import math
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
        return_details: bool = False,
    ) -> list[str] | list[RankedCandidate]:
        """Return the top-k predicted words for the given digit sequence.

        Args:
            digit_seq:      T9 digit string, e.g. ``"4663"``
            context:        Previously typed words (most recent last),
                            e.g. ``["i", "am", "going"]``
            top_k:          Number of results to return (default 5)
            return_details: If True, return RankedCandidate objects with score
                            breakdown instead of plain strings.

        Returns:
            List of word strings (default) or RankedCandidate objects.
        """
        if not is_valid_digit_sequence(digit_seq):
            raise ValueError(
                f"Invalid digit sequence {digit_seq!r}. "
                "Use digits 2-9 only (T9 keypad)."
            )

        candidates = self._dict.lookup(digit_seq)
        if not candidates:
            return []

        words, word_ids = zip(*candidates)
        word_ids_list = list(word_ids)

        # ---- freq score ------------------------------------------------
        freq_scores = np.array(
            [self._vocab.logfreq(wid) for wid in word_ids_list], dtype=np.float32
        )
        freq_scores = _normalise(freq_scores)

        # ---- model score -----------------------------------------------
        if self._model is not None and self._wm > 0:
            ctx_ids = self._vocab.words_to_ids(list(context))
            raw_model = self._model.score_candidates(ctx_ids, word_ids_list)
            model_scores = _normalise(raw_model)
        else:
            model_scores = np.zeros(len(word_ids_list), dtype=np.float32)

        # ---- ngram score -----------------------------------------------
        if self._ngram is not None and self._wn > 0 and context:
            prev_id = self._vocab.word_to_id(context[-1].lower())
            raw_ngram = np.array(
                self._ngram.score_candidates(prev_id, word_ids_list), dtype=np.float32
            )
            ngram_scores = _normalise(raw_ngram)
        else:
            ngram_scores = np.zeros(len(word_ids_list), dtype=np.float32)

        # ---- combine ---------------------------------------------------
        final = (
            self._wf * freq_scores
            + self._wm * model_scores
            + self._wn * ngram_scores
        )

        # Sort descending
        order = np.argsort(-final)
        top_indices = order[:top_k]

        if return_details:
            return [
                RankedCandidate(
                    word=words[i],
                    word_id=word_ids_list[i],
                    freq_score=float(freq_scores[i]),
                    model_score=float(model_scores[i]),
                    ngram_score=float(ngram_scores[i]),
                    final_score=float(final[i]),
                )
                for i in top_indices
            ]
        else:
            return [words[i] for i in top_indices]

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
        """
        vocab = Vocabulary.load(vocab_path)
        dictionary = T9Dictionary.load(dict_path, vocab)
        model = DualEncoder.load(model_path, vocab) if model_path else None
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

def _normalise(scores: np.ndarray) -> np.ndarray:
    """Min-max normalise to [0, 1]; return zeros if all scores are identical."""
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-9:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)
