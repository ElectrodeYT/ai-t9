"""Estimate ai-t9 model file size and runtime RAM for a given embedding dim.

Usage::

    # Show table of common dimensions around a target
    python scripts/model_size.py

    # Inspect a specific dimension
    python scripts/model_size.py --dim 128

    # Find the largest dim that fits within a RAM budget
    python scripts/model_size.py --budget 100

    # Use a different vocab size (defaults to reading data/vocab.json)
    python scripts/model_size.py --vocab-size 50000 --budget 80
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Core formula
# ---------------------------------------------------------------------------

def model_ram_mb(vocab_size: int, embed_dim: int) -> float:
    """Uncompressed RAM for the DualEncoder embedding matrices (float32).

    Two matrices (context_embeds + word_embeds), each vocab_size × embed_dim,
    stored as float32 (4 bytes each).
    """
    return 2 * vocab_size * embed_dim * 4 / 1e6


def ngram_vocab_size(word_vocab_size: int, ns: tuple[int, ...] = (2, 3)) -> int:
    """Estimate n-gram vocabulary size for an English word vocab.

    Uses the empirically measured ratio from the project's 29,470-word vocab:
        2-grams:  939 unique
        3-grams:  7,795 unique
    These counts are stable for English and scale negligibly with corpus size.
    The returned value is the total unique n-grams across all `ns` sizes.
    """
    # Empirical counts per n-gram size (measured from 29,470 English words)
    _EMPIRICAL = {2: 939, 3: 7_795, 4: 29_944}
    # Scale linearly with vocab size relative to baseline
    scale = word_vocab_size / 29_470
    total = 0
    for n in ns:
        base = _EMPIRICAL.get(n, 30_000)
        total += int(base * min(scale, 1.0))  # n-gram counts saturate quickly
    return total


def model_file_mb(vocab_size: int, embed_dim: int, compression_ratio: float = 0.90) -> float:
    """Estimated .npz compressed file size.

    Default compression_ratio=0.90 was measured empirically on the Brown
    corpus embeddings (13.58 MB file / 15.09 MB uncompressed ≈ 0.90).
    """
    return model_ram_mb(vocab_size, embed_dim) * compression_ratio


def total_ram_mb(vocab_size: int, embed_dim: int, fixed_mb: float) -> float:
    """Total predicted runtime RSS = fixed components + model matrices."""
    return fixed_mb + model_ram_mb(vocab_size, embed_dim)


def optimal_dim(vocab_size: int, fixed_mb: float, budget_mb: float, align: int = 8) -> int:
    """Return the largest embed_dim (multiple of `align`) that stays under budget_mb."""
    model_budget = budget_mb - fixed_mb
    if model_budget <= 0:
        return 0
    raw = model_budget * 1e6 / (2 * vocab_size * 4)
    return int(raw // align) * align


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_vocab_size(vocab_json: Path) -> int:
    import json
    data = json.loads(vocab_json.read_text())
    return len(data["words"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Estimate ai-t9 model size and RAM for a given embed-dim."
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=None,
        help="Embedding dimension to inspect (prints a single row).",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=100.0,
        help="RAM budget in MB (default: 100). Used to find the optimal dim.",
    )
    parser.add_argument(
        "--fixed",
        type=float,
        default=None,
        help="Fixed RAM for vocab+dict+bigram in MB. "
             "Defaults to 37.0 MB (measured with data/bigram.npz).",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Vocabulary size. Defaults to reading data/vocab.json.",
    )
    parser.add_argument(
        "--vocab",
        metavar="FILE",
        default="data/vocab.json",
        help="Path to vocab.json (used to read vocab size if --vocab-size not given).",
    )
    parser.add_argument(
        "--compression",
        type=float,
        default=0.90,
        help="Compression ratio for .npz file size estimate (default: 0.90).",
    )
    args = parser.parse_args(argv)

    # Resolve vocab size
    if args.vocab_size is not None:
        vocab_size = args.vocab_size
    else:
        vocab_path = Path(args.vocab)
        if vocab_path.exists():
            vocab_size = _load_vocab_size(vocab_path)
        else:
            print(
                f"WARNING: vocab file not found at {vocab_path}; "
                "using default vocab_size=29470.",
                file=sys.stderr,
            )
            vocab_size = 29_470

    fixed_mb = args.fixed if args.fixed is not None else 37.0
    budget = args.budget

    # ---- Single-dim mode ------------------------------------------------
    if args.dim is not None:
        d = args.dim
        ng_vocab = ngram_vocab_size(vocab_size)
        ram_word  = total_ram_mb(vocab_size, d, fixed_mb)
        ram_ngram = total_ram_mb(ng_vocab,   d, fixed_mb)
        file_word  = model_file_mb(vocab_size, d, args.compression)
        file_ngram = model_file_mb(ng_vocab,   d, args.compression)
        model_word  = model_ram_mb(vocab_size, d)
        model_ngram = model_ram_mb(ng_vocab,   d)
        print(f"vocab_size:         {vocab_size:,}  words")
        print(f"ngram_vocab_size:   {ng_vocab:,}  (2+3-grams, empirical)")
        print(f"embed_dim:          {d}")
        print()
        print(f"{'':30s}  {'DualEncoder':>18}  {'CharNgramEncoder':>18}")
        print(f"  {'model RAM':<28}  {model_word:>15.2f} MB  {model_ngram:>15.2f} MB")
        print(f"  {'total RAM':<28}  {ram_word:>15.2f} MB  {ram_ngram:>15.2f} MB")
        print(f"  {'file size (compressed)':<28}  {file_word:>14.1f} MB  {file_ngram:>14.1f} MB")
        print(f"  {'budget use ({:.0f} MB)':<28}  {ram_word/budget*100:>14.1f} %  {ram_ngram/budget*100:>14.1f} %".format(budget))
        return 0

    # ---- Table mode ------------------------------------------------------
    best_word  = optimal_dim(vocab_size,                   fixed_mb, budget)
    best_ngram = optimal_dim(ngram_vocab_size(vocab_size), fixed_mb, budget)
    ng_vocab   = ngram_vocab_size(vocab_size)

    # Build candidate list: powers of 2 plus ±32 neighbourhood of each optimum
    candidates: set[int] = set()
    for exp in range(4, 12):       # 16 … 2048
        candidates.add(2 ** exp)
    for best in (best_word, best_ngram):
        for d in range(max(8, best - 32), best + 40, 8):
            candidates.add(d)

    dims = sorted(candidates)

    # marker column: "W>" = DualEncoder optimum, "N>" = CharNgram optimum, "  " = neither
    hdr = "{:2}  {:>6}  {:>12}  {:>9}  {:>12}  {:>9}"
    sep = "-" * 60
    print(f"\nvocab_size={vocab_size:,}  ngram_vocab={ng_vocab:,}  fixed={fixed_mb:.1f} MB  budget={budget:.0f} MB\n")
    print(f"{'':10}  {'-- DualEncoder --':^22}  {'-- CharNgramEncoder --':^22}")
    print(hdr.format("", "dim", "total RAM", "budget %", "total RAM", "budget %"))
    print(sep)
    for d in dims:
        r_w = total_ram_mb(vocab_size, d, fixed_mb)
        r_n = total_ram_mb(ng_vocab,   d, fixed_mb)
        p_w = r_w / budget * 100
        p_n = r_n / budget * 100
        if d == best_word and d == best_ngram:
            marker = "WN"
        elif d == best_word:
            marker = "W>"
        elif d == best_ngram:
            marker = "N>"
        else:
            marker = ""
        over_w = "!" if r_w > budget else " "
        over_n = "!" if r_n > budget else " "
        print(hdr.format(
            marker,
            d,
            f"{r_w:.1f} MB {over_w}",
            f"{p_w:.1f}%",
            f"{r_n:.1f} MB {over_n}",
            f"{p_n:.1f}%",
        ))
    print(sep)
    print(f"\n  DualEncoder optimal:        embed_dim={best_word:4d}  →  {total_ram_mb(vocab_size, best_word, fixed_mb):.1f} MB")
    print(f"  CharNgramDualEncoder optimal: embed_dim={best_ngram:4d}  →  {total_ram_mb(ng_vocab,   best_ngram, fixed_mb):.1f} MB")
    print(f"  (largest multiple of 8 within {budget:.0f} MB budget)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
