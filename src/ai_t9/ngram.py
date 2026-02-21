"""Bigram language model with Laplace (add-k) smoothing.

Serves as a fast, always-available scoring fallback and can be blended into
the final ranking alongside the dual-encoder model.  No ML framework required.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

from .model.vocab import Vocabulary


class BigramScorer:
    """Bigram language model: P(word | prev_word) with add-k smoothing.

    Trained from a sequence of word IDs.  Stores only bigram counts (not full
    probability tables) so it stays small; probabilities are computed on the
    fly with smoothing applied.
    """

    def __init__(self, vocab: Vocabulary, k: float = 0.5) -> None:
        self._vocab = vocab
        self._k = k
        # counts[prev_id][next_id] = count
        self._bigrams: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._unigrams: dict[int, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_on_ids(self, word_ids: list[int]) -> None:
        """Accumulate bigram counts from a flat sequence of word IDs."""
        for i, wid in enumerate(word_ids):
            self._unigrams[wid] += 1
            if i > 0:
                self._bigrams[word_ids[i - 1]][wid] += 1

    @classmethod
    def build_from_nltk(
        cls,
        vocab: Vocabulary,
        k: float = 0.5,
        verbose: bool = True,
    ) -> "BigramScorer":
        """Build a bigram model from the NLTK Brown corpus."""
        try:
            import nltk
        except ImportError:
            raise ImportError("nltk is required: pip install nltk")

        try:
            nltk.data.find("corpora/brown")
        except LookupError:
            if verbose:
                print("Downloading NLTK 'brown' corpus…")
            nltk.download("brown", quiet=not verbose)

        from nltk.corpus import brown

        if verbose:
            print("Training bigram model on Brown corpus…")

        scorer = cls(vocab, k=k)
        # Train sentence by sentence to respect sentence boundaries
        for sent in brown.sents():
            ids = [vocab.word_to_id(w.lower()) for w in sent if w.isalpha()]
            if len(ids) > 1:
                scorer.train_on_ids(ids)

        if verbose:
            print(
                f"Bigram model trained: {len(scorer._bigrams)} context words, "
                f"{sum(len(v) for v in scorer._bigrams.values())} bigram types"
            )
        return scorer

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def log_prob(self, prev_word_id: int, candidate_id: int) -> float:
        """Log P(candidate | prev_word) with add-k smoothing."""
        v = self._vocab.size
        k = self._k
        count_prev = self._unigrams.get(prev_word_id, 0)
        count_bigram = 0
        if prev_word_id in self._bigrams:
            count_bigram = self._bigrams[prev_word_id].get(candidate_id, 0)
        # P(w|prev) = (count(prev, w) + k) / (count(prev) + k * V)
        prob = (count_bigram + k) / (count_prev + k * v)
        return math.log(prob)

    def score_candidates(
        self,
        prev_word_id: int,
        candidate_ids: list[int],
    ) -> list[float]:
        """Return a list of log-probabilities for each candidate given prev_word."""
        return [self.log_prob(prev_word_id, cid) for cid in candidate_ids]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = {
            "k": self._k,
            "unigrams": {str(k): v for k, v in self._unigrams.items()},
            "bigrams": {
                str(prev): {str(nxt): cnt for nxt, cnt in nexts.items()}
                for prev, nexts in self._bigrams.items()
            },
        }
        path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, vocab: Vocabulary) -> "BigramScorer":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        scorer = cls(vocab, k=data["k"])
        scorer._unigrams = defaultdict(int, {int(k): v for k, v in data["unigrams"].items()})
        scorer._bigrams = defaultdict(
            lambda: defaultdict(int),
            {
                int(prev): defaultdict(int, {int(nxt): cnt for nxt, cnt in nexts.items()})
                for prev, nexts in data["bigrams"].items()
            },
        )
        return scorer
