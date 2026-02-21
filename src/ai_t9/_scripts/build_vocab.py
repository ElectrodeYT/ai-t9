"""CLI: Build vocabulary and T9 dictionary from a text corpus.

Usage::

    # From NLTK Brown corpus (default)
    ai-t9-build-vocab --output data/

    # From a single corpus file
    ai-t9-build-vocab --corpus corpuses/mytext.txt --output data/

    # From a folder of corpus files (all *.txt files are combined)
    ai-t9-build-vocab --corpus corpuses/ --output data/

    # Restrict the T9 dictionary to a verified wordlist (typo-free)
    ai-t9-build-vocab --corpus corpuses/ --dictionary wordlist.txt --output data/
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path


def _count_words_from_file(path: Path, counter: Counter, verbose: bool) -> int:
    """Accumulate word counts from a single text file into counter.

    Returns the number of lines processed.
    """
    lines = 0
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            for word in line.strip().lower().split():
                if word.isalpha():
                    counter[word] += 1
            lines += 1
    return lines


def _resolve_corpus_files(corpus_path: Path) -> list[Path]:
    """Return a list of text files to read from a file or directory path."""
    if corpus_path.is_dir():
        files = sorted(corpus_path.glob("*.txt"))
        if not files:
            raise FileNotFoundError(f"No *.txt files found in {corpus_path}")
        return files
    if corpus_path.is_file():
        return [corpus_path]
    raise FileNotFoundError(f"Corpus path not found: {corpus_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build ai-t9 vocabulary and T9 dictionary index"
    )
    parser.add_argument(
        "--corpus",
        metavar="FILE_OR_DIR",
        default=None,
        help="Plain-text corpus file, or a directory of *.txt files whose "
             "word counts are combined. Defaults to NLTK Brown corpus.",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="DIR",
        default="data",
        help="Output directory for vocab.json and dict.json (default: data/)",
    )
    parser.add_argument(
        "--max-words",
        type=int,
        default=50_000,
        metavar="N",
        help="Maximum vocabulary size (default: 50000)",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=2,
        metavar="N",
        help="Minimum word frequency to include (default: 2)",
    )
    parser.add_argument(
        "--dictionary",
        metavar="FILE",
        default=None,
        help="Plain-text wordlist file (one word per line) to restrict the "
             "T9 dictionary.  Only words present in this file will appear as "
             "prediction candidates.  The vocabulary and model still use the "
             "full corpus — this only filters the dictionary output.  Useful "
             "for ensuring a typo-free candidate set.",
    )
    args = parser.parse_args(argv)

    from ai_t9.model.vocab import Vocabulary
    from ai_t9.dictionary import T9Dictionary, load_wordlist

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build vocabulary -------------------------------------------------
    if args.corpus:
        corpus_path = Path(args.corpus)
        try:
            files = _resolve_corpus_files(corpus_path)
        except FileNotFoundError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

        counter: Counter = Counter()
        for path in files:
            lines = _count_words_from_file(path, counter, verbose=True)
            print(f"  {path.name}: {lines:,} lines  ({sum(counter.values()):,} words so far)")

        vocab = Vocabulary.build_from_counts(
            counter, max_words=args.max_words, min_count=args.min_count
        )
    else:
        vocab = Vocabulary.build_from_nltk(
            max_words=args.max_words, min_count=args.min_count, verbose=True
        )

    vocab_path = out_dir / "vocab.json"
    vocab.save(vocab_path)
    print(f"Saved vocabulary ({vocab.size} words) → {vocab_path}")

    # ---- Load optional verified wordlist -----------------------------------
    wordlist: set[str] | None = None
    if args.dictionary:
        dict_file = Path(args.dictionary)
        if not dict_file.exists():
            print(f"ERROR: dictionary file not found: {dict_file}", file=sys.stderr)
            return 1
        wordlist = load_wordlist(dict_file)
        print(f"Loaded wordlist: {len(wordlist):,} verified words from {dict_file.name}")

    # ---- Build T9 dictionary index ----------------------------------------
    dictionary = T9Dictionary.build(vocab, wordlist=wordlist, verbose=True)
    dict_path = out_dir / "dict.json"
    dictionary.save(dict_path)
    print(f"Saved T9 dictionary index → {dict_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
