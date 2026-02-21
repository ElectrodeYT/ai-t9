"""T9Dictionary: maps digit sequences to candidate words with their vocab IDs."""

from __future__ import annotations

import json
from pathlib import Path

from .t9_map import word_to_digits
from .model.vocab import Vocabulary


class T9Dictionary:
    """Pre-computed index from T9 digit sequences to matching vocabulary words.

    Built once from a Vocabulary, then used at inference time for O(1) lookups.
    Each entry stores (word, word_id) pairs sorted by descending log-frequency
    so that pure-frequency ranking is free.
    """

    def __init__(self, vocab: Vocabulary) -> None:
        self._vocab = vocab
        self._index: dict[str, list[tuple[str, int]]] = {}
        self._build(vocab)

    def _build(self, vocab: Vocabulary) -> None:
        index: dict[str, list[tuple[str, int]]] = {}
        for wid in range(vocab.size):
            word = vocab.id_to_word(wid)
            if word == vocab.UNK:
                continue
            digits = word_to_digits(word)
            if digits is None:
                continue
            index.setdefault(digits, []).append((word, wid))
        # Sort each bucket by descending log-frequency (already in vocab order)
        # vocab words are stored most-frequent-first so wid order is freq order
        for digits in index:
            index[digits].sort(key=lambda t: vocab.logfreq(t[1]), reverse=True)
        self._index = index

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(self, digit_seq: str) -> list[tuple[str, int]]:
        """Return (word, word_id) pairs for the given digit sequence.

        Returns an empty list if no words match.
        Results are pre-sorted by descending frequency.
        """
        return self._index.get(digit_seq, [])

    def digit_sequences(self) -> list[str]:
        """All digit sequences that have at least one matching word."""
        return list(self._index.keys())

    @property
    def vocab(self) -> Vocabulary:
        return self._vocab

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save index to a compact JSON file."""
        path = Path(path)
        # Store as {digits: [[word, wid], ...]} — compact but human-readable
        serialisable = {
            digits: [[w, wid] for w, wid in entries]
            for digits, entries in self._index.items()
        }
        path.write_text(
            json.dumps(serialisable, separators=(",", ":")), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path, vocab: Vocabulary) -> "T9Dictionary":
        """Load a previously saved index (the vocab must be the same one used to build it)."""
        path = Path(path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        obj = cls.__new__(cls)
        obj._vocab = vocab
        obj._index = {
            digits: [(entry[0], entry[1]) for entry in entries]
            for digits, entries in raw.items()
        }
        return obj

    @classmethod
    def build(cls, vocab: Vocabulary, verbose: bool = True) -> "T9Dictionary":
        """Convenience constructor that prints progress."""
        if verbose:
            print(f"Building T9 index for {vocab.size} words…")
        d = cls(vocab)
        if verbose:
            print(f"T9 index: {len(d._index)} unique digit sequences")
        return d
