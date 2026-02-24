"""Vocabulary: word ↔ id mapping with log-frequency weights."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path


_UNKNOWN = "<unk>"
_UNK_ID = 0


class Vocabulary:
    """Mapping between words and integer IDs with associated log-frequencies.

    The vocabulary is fixed at construction time. Unknown words map to
    UNK_ID (0) and receive a very low log-frequency score.
    """

    UNK = _UNKNOWN
    UNK_ID = _UNK_ID

    def __init__(
        self,
        words: list[str],
        counts: list[int],
    ) -> None:
        """Build vocabulary from a list of words and their corpus counts.

        words[0] is reserved for <unk>; if not already present it is inserted.
        words and counts must be parallel lists, same length, sorted descending
        by count (most frequent first).
        """
        # Ensure <unk> is at index 0
        if words and words[0] != _UNKNOWN:
            words = [_UNKNOWN] + list(words)
            counts = [0] + list(counts)

        self._words: list[str] = list(words)
        self._counts: list[int] = list(counts)
        self._word2id: dict[str, int] = {w: i for i, w in enumerate(self._words)}

        total = max(sum(self._counts), 1)
        # Log-frequency score: log(count / total), floored at log(1/total)
        min_logfreq = math.log(1.0 / total)
        self._logfreq: list[float] = [
            math.log(max(c, 1) / total) if c > 0 else min_logfreq
            for c in self._counts
        ]
        # UNK gets a score strictly below the minimum word score (log(0.5/total))
        # so that even floor-count words (count=1, from wordlist merges) rank
        # above UNK in frequency-based scoring.
        self._logfreq[_UNK_ID] = math.log(0.5 / total)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._words)

    def __contains__(self, word: str) -> bool:
        return word in self._word2id

    def word_to_id(self, word: str) -> int:
        return self._word2id.get(word, _UNK_ID)

    def id_to_word(self, wid: int) -> str:
        return self._words[wid]

    def words_to_ids(self, words: list[str]) -> list[int]:
        return [self._word2id.get(w, _UNK_ID) for w in words]

    def logfreq(self, word_id: int) -> float:
        return self._logfreq[word_id]

    def logfreq_array(self) -> "list[float]":
        return self._logfreq

    @property
    def size(self) -> int:
        return len(self._words)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = {"words": self._words, "counts": self._counts}
        path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["words"], data["counts"])

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def build_from_counts(
        cls,
        counter: Counter,
        max_words: int = 50_000,
        min_count: int = 2,
    ) -> "Vocabulary":
        """Build from a Counter of {word: count}, keeping the most frequent words."""
        filtered = [
            (w, c)
            for w, c in counter.most_common()
            if c >= min_count and w.isalpha() and w.islower()
        ][:max_words]
        if not filtered:
            raise ValueError("No words passed the filter — check your corpus.")
        words, counts = zip(*filtered)
        return cls(list(words), list(counts))

    def merge_wordlist(self, wordlist: set[str]) -> "Vocabulary":
        """Return a new Vocabulary with wordlist words added at floor frequency.

        Words already in the vocabulary keep their existing counts.  Words in
        the wordlist that are not yet in the vocabulary are appended with a
        count of 1, giving them the lowest log-frequency above the UNK floor.

        This is intended for incorporating a verified dictionary: all wordlist
        words will have real vocab IDs (not UNK), so the model can assign them
        meaningful embeddings.
        """
        new_words = list(self._words)
        new_counts = list(self._counts)
        existing = set(self._word2id.keys())
        added = 0
        for w in sorted(wordlist):
            wl = w.lower()
            if wl.isalpha() and wl not in existing:
                new_words.append(wl)
                new_counts.append(1)
                existing.add(wl)
                added += 1
        if added == 0:
            return self
        return Vocabulary(new_words, new_counts)

