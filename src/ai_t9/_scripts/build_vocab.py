"""CLI: Build vocabulary and T9 dictionary from a text corpus.

Usage::

    # From NLTK Brown corpus (default)
    ai-t9-build-vocab --output data/

    # From your own corpus file
    ai-t9-build-vocab --corpus mytext.txt --output data/ --max-words 30000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build ai-t9 vocabulary and T9 dictionary index"
    )
    parser.add_argument(
        "--corpus",
        metavar="FILE",
        default=None,
        help="Plain-text corpus file (one sentence per line). "
             "Defaults to NLTK Brown corpus.",
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
    args = parser.parse_args(argv)

    from collections import Counter
    from ai_t9.model.vocab import Vocabulary
    from ai_t9.dictionary import T9Dictionary

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build vocabulary -------------------------------------------------
    if args.corpus:
        corpus_path = Path(args.corpus)
        if not corpus_path.exists():
            print(f"ERROR: corpus file not found: {corpus_path}", file=sys.stderr)
            return 1
        print(f"Reading corpus from {corpus_path}…")
        counter: Counter = Counter()
        with corpus_path.open(encoding="utf-8", errors="ignore") as f:
            for line in f:
                for word in line.strip().lower().split():
                    if word.isalpha():
                        counter[word] += 1
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

    # ---- Build T9 dictionary index ----------------------------------------
    dictionary = T9Dictionary.build(vocab, verbose=True)
    dict_path = out_dir / "dict.json"
    dictionary.save(dict_path)
    print(f"Saved T9 dictionary index → {dict_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
