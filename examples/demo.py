#!/usr/bin/env python3
"""Interactive T9 demo.

Run with::

    # Quickstart (frequency + bigram, no neural model)
    python examples/demo.py

    # With a trained neural model
    python examples/demo.py --model data/model.npz --vocab data/vocab.json --dict data/dict.json

Commands during the demo
-------------------------
  <digits>          Predict words for that T9 sequence
  <digits> -v       Predict with score breakdown
  .<word>           Manually add a word to context (e.g. ".hello")
  reset / clear     Clear the current context
  ctx / context     Show the current context window
  help              Show this help
  quit / exit / q   Exit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_predictor(args) -> "T9Predictor":  # noqa: F821
    from ai_t9 import T9Predictor

    if args.vocab and args.dict:
        print(f"Loading from files: {args.vocab}, {args.dict}")
        predictor = T9Predictor.from_files(
            vocab_path=args.vocab,
            dict_path=args.dict,
            model_path=args.model,
            ngram_path=args.ngram,
        )
    else:
        print(
            "ERROR: --vocab and --dict are required.\n"
            "Run the training pipeline first:\n"
            "  ai-t9-run configs/default.yaml\n"
            "Then:\n"
            "  python examples/demo.py --vocab data/vocab.json --dict data/dict.json",
            file=sys.stderr,
        )
        sys.exit(1)

    print()
    print("Signals active:")
    for signal, weight in predictor.weights.items():
        status = f"{weight:.2f}" if weight > 0 else "disabled"
        print(f"  {signal:8s}: {status}")
    print()
    return predictor


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive ai-t9 demo")
    parser.add_argument("--vocab",  metavar="FILE", default=None, help="vocab.json path")
    parser.add_argument("--dict",   metavar="FILE", default=None, help="dict.json path")
    parser.add_argument("--model",  metavar="FILE", default=None, help="model.npz path (optional)")
    parser.add_argument("--ngram",  metavar="FILE", default=None, help="bigram.npz path (optional)")
    parser.add_argument("--top-k",  type=int, default=5, help="Candidates to show (default 5)")
    args = parser.parse_args()

    from ai_t9 import T9Session
    from ai_t9.t9_map import VALID_DIGITS

    predictor = build_predictor(args)
    session = T9Session(predictor, context_window=5)

    print("=" * 55)
    print("  ai-t9 interactive demo  (type 'help' for commands)")
    print("=" * 55)
    print()

    while True:
        try:
            raw = input("t9> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        lower = raw.lower()

        if lower in ("quit", "exit", "q"):
            break

        if lower == "help":
            print(__doc__)
            continue

        if lower in ("reset", "clear"):
            session.reset()
            print("  Context cleared.")
            continue

        if lower in ("ctx", "context"):
            ctx = session.context
            if ctx:
                print("  Context:", " ".join(ctx))
            else:
                print("  Context: (empty)")
            continue

        if lower.startswith("."):
            # Manual context word
            word = lower[1:].strip()
            if word:
                session.confirm(word)
                print(f"  Added {word!r} to context. Context: {session.context}")
            continue

        # Check for -v flag
        verbose = raw.endswith(" -v")
        digit_seq = raw.removesuffix(" -v").strip()

        if not all(d in VALID_DIGITS for d in digit_seq):
            print(f"  Unknown command or invalid digits: {raw!r}")
            print("  Use digits 2-9, or type 'help'.")
            continue

        # Predict
        if verbose:
            results = session.dial(digit_seq, top_k=args.top_k, return_details=True)
            if not results:
                print("  No matches found.")
            else:
                print(f"  Candidates for {digit_seq!r} "
                      f"(context: {session.context or 'none'}):")
                for i, r in enumerate(results, 1):
                    print(
                        f"  {i}. {r.word:<15s}  final={r.final_score:.3f}  "
                        f"freq={r.freq_score:.3f}  model={r.model_score:.3f}  "
                        f"ngram={r.ngram_score:.3f}"
                    )
        else:
            results = session.dial(digit_seq, top_k=args.top_k)
            if not results:
                print("  No matches found.")
            else:
                print(f"  {digit_seq!r} → {', '.join(results)}")
                print(f"  (context: {session.context or 'none'})")

        # Let user confirm a word by number
        if results:
            try:
                choice_raw = input("  Confirm [1-5 or Enter to skip]: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if choice_raw.isdigit():
                idx = int(choice_raw) - 1
                word_list = [r.word if hasattr(r, "word") else r for r in results]
                if 0 <= idx < len(word_list):
                    chosen = word_list[idx]
                    session.confirm(chosen)
                    print(f"  Confirmed {chosen!r}. Context: {session.context}")

        print()

    print("Bye!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
