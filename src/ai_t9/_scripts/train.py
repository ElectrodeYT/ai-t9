"""CLI: Train the DualEncoder model and save it as a .npz file.

Usage::

    # Train on NLTK Brown corpus (default)
    ai-t9-train --vocab data/vocab.json --output data/model.npz

    # Train on a single corpus file
    ai-t9-train --vocab data/vocab.json --corpus mytext.txt --output data/model.npz \\
                --epochs 5 --embed-dim 64 --neg-samples 20

    # Train on a folder of corpus files (all *.txt files are combined)
    ai-t9-train --vocab data/vocab.json --corpus corpuses/ --output data/model.npz

    # Also export a bigram model for the ngram signal
    ai-t9-train --vocab data/vocab.json --output data/model.npz --save-ngram data/bigram.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _resolve_corpus_files(corpus_path: Path) -> list[Path] | None:
    """Return sorted list of *.txt files from a file or directory path.

    Prints an error and returns None if the path doesn't exist or a directory
    contains no .txt files (so callers can return 1 immediately).
    """
    if corpus_path.is_dir():
        files = sorted(corpus_path.glob("*.txt"))
        if not files:
            print(f"ERROR: no *.txt files found in {corpus_path}", file=sys.stderr)
            return None
        return files
    if corpus_path.is_file():
        return [corpus_path]
    print(f"ERROR: corpus path not found: {corpus_path}", file=sys.stderr)
    return None


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
        metavar="FILE_OR_DIR",
        default=None,
        help="Plain-text corpus file, or a directory of *.txt files whose "
             "sentences are combined. Defaults to NLTK Brown corpus.",
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
    parser.add_argument("--epochs",           type=int,   default=3,    help="Training epochs (default: 3)")
    parser.add_argument("--embed-dim",        type=int,   default=64,   help="Embedding dimension (default: 64)")
    parser.add_argument("--context-window",   type=int,   default=3,    help="Context words to use (default: 3)")
    parser.add_argument("--neg-samples",      type=int,   default=20,   help="Negatives per positive, sampled on GPU (default: 20)")
    parser.add_argument("--lr",               type=float, default=0.005,help="Learning rate (default: 0.005)")
    parser.add_argument("--batch-size",       type=int,   default=2048, help="Pairs per batch (default: 2048; try 4096–8192 on large GPUs)")
    parser.add_argument("--seed",             type=int,   default=42,   help="Random seed (default: 42)")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device (default: auto — picks CUDA > MPS > CPU)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print timestamped phase breakdown to diagnose startup bottlenecks",
    )
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
        device=args.device,
        debug=args.debug,
    )

    if args.corpus:
        corpus_files = _resolve_corpus_files(Path(args.corpus))
        if corpus_files is None:
            return 1
        trainer.train_from_files(corpus_files, epochs=args.epochs, verbose=True)
    else:
        trainer.train_from_nltk(epochs=args.epochs, verbose=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_numpy(out_path)

    # ---- Optionally train bigram model -----------------------------------
    if args.save_ngram:
        from ai_t9.ngram import BigramScorer
        from ai_t9.model.trainer import _corpus_file_sentence_ids
        print("Training bigram model…")
        if args.corpus:
            scorer = BigramScorer(vocab)
            for path in corpus_files:
                sents = _corpus_file_sentence_ids(path, vocab)
                for sent in sents:
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
