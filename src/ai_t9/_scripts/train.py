"""CLI: Train the DualEncoder model and save it as a .npz file.

Usage::

    # Train on NLTK Brown corpus (default)
    ai-t9-train --vocab data/vocab.json --output data/model.npz

    # Train on your own corpus
    ai-t9-train --vocab data/vocab.json --corpus mytext.txt --output data/model.npz \\
                --epochs 5 --embed-dim 64 --neg-samples 20

    # Also export a bigram model for the ngram signal
    ai-t9-train --vocab data/vocab.json --output data/model.npz --save-ngram data/bigram.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train ai-t9 DualEncoder model"
    )
    parser.add_argument(
        "--vocab",
        metavar="FILE",
        required=True,
        help="Path to vocab.json (built by ai-t9-build-vocab)",
    )
    parser.add_argument(
        "--corpus",
        metavar="FILE",
        default=None,
        help="Plain-text corpus file. Defaults to NLTK Brown corpus.",
    )
    parser.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        default="model.npz",
        help="Output path for model.npz (default: model.npz)",
    )
    parser.add_argument(
        "--save-ngram",
        metavar="FILE",
        default=None,
        help="If given, also save a trained bigram model to this path.",
    )
    parser.add_argument("--epochs",      type=int,   default=3,    help="Training epochs (default: 3)")
    parser.add_argument("--embed-dim",   type=int,   default=64,   help="Embedding dimension (default: 64)")
    parser.add_argument("--context-window", type=int, default=3,   help="Context words to use (default: 3)")
    parser.add_argument("--neg-samples", type=int,   default=20,   help="Negatives per positive (default: 20)")
    parser.add_argument("--lr",          type=float, default=0.005,help="Learning rate (default: 0.005)")
    parser.add_argument("--batch-size",  type=int,   default=512,  help="Batch size (default: 512)")
    parser.add_argument("--seed",        type=int,   default=42,   help="Random seed (default: 42)")
    args = parser.parse_args(argv)

    from ai_t9.model.vocab import Vocabulary
    from ai_t9.model.trainer import DualEncoderTrainer

    # ---- Load vocabulary -------------------------------------------------
    vocab_path = Path(args.vocab)
    if not vocab_path.exists():
        print(f"ERROR: vocab file not found: {vocab_path}", file=sys.stderr)
        print("Run 'ai-t9-build-vocab' first to create vocab.json", file=sys.stderr)
        return 1

    print(f"Loading vocabulary from {vocab_path}…")
    vocab = Vocabulary.load(vocab_path)
    print(f"  {vocab.size} words loaded")

    # ---- Train dual encoder ---------------------------------------------
    trainer = DualEncoderTrainer(
        vocab=vocab,
        embed_dim=args.embed_dim,
        context_window=args.context_window,
        neg_samples=args.neg_samples,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    if args.corpus:
        trainer.train_from_file(args.corpus, epochs=args.epochs, verbose=True)
    else:
        trainer.train_from_nltk(epochs=args.epochs, verbose=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_numpy(out_path)

    # ---- Optionally train bigram model -----------------------------------
    if args.save_ngram:
        from ai_t9.ngram import BigramScorer
        print("Training bigram model…")
        if args.corpus:
            # Re-read sentences for bigram training
            from ai_t9.model.trainer import _corpus_file_sentence_ids
            sentences = _corpus_file_sentence_ids(Path(args.corpus), vocab)
            scorer = BigramScorer(vocab)
            for sent in sentences:
                scorer.train_on_ids(sent)
        else:
            scorer = BigramScorer.build_from_nltk(vocab, verbose=True)
        ngram_path = Path(args.save_ngram)
        ngram_path.parent.mkdir(parents=True, exist_ok=True)
        scorer.save(ngram_path)
        print(f"Saved bigram model → {ngram_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
