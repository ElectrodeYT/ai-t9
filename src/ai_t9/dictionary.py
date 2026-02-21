"""T9Dictionary: maps digit sequences to candidate words with their vocab IDs."""

from __future__ import annotations

import json
from pathlib import Path

from .t9_map import word_to_digits
from .model.vocab import Vocabulary


def load_wordlist(path: str | Path) -> set[str]:
    """Load a plain-text wordlist (one lowercase word per line).

    Lines are stripped, lowercased, and filtered to pure-alpha tokens.
    Blank lines and lines starting with ``#`` are skipped.
    """
    path = Path(path)
    words: set[str] = set()
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip().lower()
            if w and not w.startswith("#") and w.isalpha():
                words.add(w)
    return words


class T9Dictionary:
    """Pre-computed index from T9 digit sequences to matching vocabulary words.

    Built once from a Vocabulary, then used at inference time for O(1) lookups.
    Each entry stores (word, word_id) pairs sorted by descending log-frequency
    so that pure-frequency ranking is free.

    An optional *wordlist* restricts which words are indexed.  When provided,
    only words present in the wordlist are included — this allows the dictionary
    to be sourced from a verified, typo-free word list (e.g. an actual English
    dictionary) while frequencies and embeddings still come from the corpus-
    derived Vocabulary.

    Words in the wordlist that are not in the Vocabulary are still indexed
    (with the UNK word ID) so they can appear as candidates, albeit with the
    lowest possible frequency score.
    """

    def __init__(
        self,
        vocab: Vocabulary,
        wordlist: set[str] | None = None,
    ) -> None:
        self._vocab = vocab
        self._index: dict[str, list[tuple[str, int]]] = {}
        self._build(vocab, wordlist)

    def _build(self, vocab: Vocabulary, wordlist: set[str] | None) -> None:
        index: dict[str, list[tuple[str, int]]] = {}

        if wordlist is not None:
            # Restricted mode: only index words from the verified wordlist.
            for word in wordlist:
                digits = word_to_digits(word)
                if digits is None:
                    continue
                wid = vocab.word_to_id(word)
                index.setdefault(digits, []).append((word, wid))
        else:
            # Unrestricted mode: index every vocab word (original behaviour).
            for wid in range(vocab.size):
                word = vocab.id_to_word(wid)
                if word == vocab.UNK:
                    continue
                digits = word_to_digits(word)
                if digits is None:
                    continue
                index.setdefault(digits, []).append((word, wid))

        # Sort each bucket by descending log-frequency.
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
    def build(
        cls,
        vocab: Vocabulary,
        wordlist: set[str] | None = None,
        verbose: bool = True,
    ) -> "T9Dictionary":
        """Convenience constructor that prints progress.

        If *wordlist* is given, only words in that set are indexed.
        """
        if verbose:
            msg = f"Building T9 index for {vocab.size} words"
            if wordlist is not None:
                msg += f" (restricted to {len(wordlist):,}-word wordlist)"
            print(f"{msg}…")
        d = cls(vocab, wordlist=wordlist)
        if verbose:
            print(f"T9 index: {len(d._index)} unique digit sequences")
        return d
