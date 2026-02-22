"""CLI: Train the DualEncoder model and save it as a .npz file.

Usage::

    # Train on NLTK Brown corpus (default)
    ai-t9-train --vocab data/vocab.json --output data/model.npz

    # Train on a single corpus file
    ai-t9-train --vocab data/vocab.json --corpus mytext.txt --output data/model.npz \\
                --epochs 5 --embed-dim 64 --temperature 0.07

    # Train on a folder of corpus files (all *.txt files are combined)
    ai-t9-train --vocab data/vocab.json --corpus corpuses/ --output data/model.npz

    # Also export a bigram model for the ngram signal
    ai-t9-train --vocab data/vocab.json --output data/model.npz --save-ngram data/bigram.npz

    # Precompute pairs once (CPU job) and save for later GPU training runs:
    ai-t9-train --vocab data/vocab.json --corpus corpuses/ \\
                --save-pairs data/pairs.npz --pairs-only

    # Precompute into shards (large corpora that don't fit in RAM):
    ai-t9-train --vocab data/vocab.json --corpus corpuses/ \\
                --save-pairs data/pairs/pairs.npz --shard-size 10000000 --pairs-only

    # GPU training job that reuses saved pairs (no corpus loading):
    ai-t9-train --vocab data/vocab.json --load-pairs data/pairs.npz \\
                --output data/model.npz --epochs 10

    # GPU training from a directory of sharded pairs:
    ai-t9-train --vocab data/vocab.json --pairs-dir data/pairs/ \\
                --output data/model.npz --epochs 10 --accumulate-grad-batches 4
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

    # ---- Input / output -------------------------------------------------
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

    # ---- Pairs precomputation / loading ----------------------------------
    parser.add_argument(
        "--save-pairs",
        metavar="FILE",
        default=None,
        help="After loading the corpus, precompute training pairs and save them "
             "to this .npz file (or shard prefix when --shard-size is set). "
             "Use with --pairs-only to skip training entirely.",
    )
    parser.add_argument(
        "--load-pairs",
        metavar="FILE",
        default=None,
        help="Load precomputed pairs from this .npz file instead of reading a corpus. "
             "Skips all corpus I/O and pair computation — ideal for GPU cloud jobs.",
    )
    parser.add_argument(
        "--pairs-dir",
        metavar="DIR",
        default=None,
        help="Directory of sharded pairs files (pairs_*.npz) to train from. "
             "Alternative to --load-pairs for large corpora. "
             "Shards are shuffled each epoch for gradient diversity.",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=None,
        metavar="N",
        help="Maximum pairs per shard when using --save-pairs. "
             "When set, writes pairs_000.npz, pairs_001.npz, … "
             "(default: None — single file).",
    )
    parser.add_argument(
        "--pairs-only",
        action="store_true",
        help="Only precompute and save pairs (requires --save-pairs); exit without training.",
    )

    # ---- Model architecture ---------------------------------------------
    parser.add_argument(
        "--model-type",
        choices=["dual-encoder", "char-ngram"],
        default="char-ngram",
        help="Model architecture to train (default: char-ngram, which is more accurate "
             "but slightly slower than dual-encoder)",
    )
    parser.add_argument("--embed-dim",      type=int,   default=64,    help="Embedding dimension (default: 64)")
    parser.add_argument("--context-window", type=int,   default=3,     help="Context words to use (default: 3)")

    # ---- Optimiser / schedule -------------------------------------------
    parser.add_argument("--epochs",         type=int,   default=3,     help="Training epochs (default: 3)")
    parser.add_argument("--lr",             type=float, default=0.001, help="Peak learning rate (default: 0.001)")
    parser.add_argument("--weight-decay",   type=float, default=1e-4,  help="AdamW weight decay (default: 1e-4)")
    parser.add_argument("--warmup-frac",    type=float, default=0.05,  help="Fraction of steps used for linear LR warmup (default: 0.05)")
    parser.add_argument("--min-lr-frac",    type=float, default=0.01,  help="Cosine decay floor as fraction of peak LR (default: 0.01)")
    parser.add_argument("--temperature",    type=float, default=0.07,  help="In-batch negative softmax temperature (default: 0.07)")

    # ---- Batch / accumulation -------------------------------------------
    parser.add_argument("--batch-size",             type=int,   default=0,    help="Pairs per micro-batch. 0 (default) auto-selects based on GPU VRAM; typical auto values are 16384\u2013131072.")
    parser.add_argument("--accumulate-grad-batches", type=int,  default=1,    help="Gradient accumulation steps (default: 1; effective batch = batch-size × this)")
    parser.add_argument("--clip-grad-norm",          type=float, default=1.0, help="Max gradient norm for clipping, 0 to disable (default: 1.0)")

    # ---- Misc -----------------------------------------------------------
    parser.add_argument("--seed",  type=int, default=42,   help="Random seed (default: 42)")
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
    from ai_t9.model.trainer import DualEncoderTrainer, CharNgramDualEncoderTrainer

    match args.model_type:
        case "dual-encoder":
            print("Selected model type: DualEncoder (word-level)")
            TrainerCls = DualEncoderTrainer
        case "char-ngram":
            print("Selected model type: CharNgramDualEncoder (character n-gram)")
            TrainerCls = CharNgramDualEncoderTrainer
        case _:
            print(f"ERROR: unknown model type: {args.model_type}", file=sys.stderr)
            return 1

    # ---- Validate flag combinations ------------------------------------
    if args.pairs_only and not args.save_pairs:
        print("ERROR: --pairs-only requires --save-pairs", file=sys.stderr)
        return 1

    input_sources = sum([
        bool(args.load_pairs),
        bool(args.pairs_dir),
        bool(args.corpus),
    ])
    if input_sources > 1:
        print("ERROR: --load-pairs, --pairs-dir, and --corpus are mutually exclusive", file=sys.stderr)
        return 1
    if args.load_pairs and args.pairs_only:
        print("ERROR: --load-pairs and --pairs-only are mutually exclusive", file=sys.stderr)
        return 1
    if args.pairs_dir and args.pairs_only:
        print("ERROR: --pairs-dir and --pairs-only are mutually exclusive", file=sys.stderr)
        return 1
    if args.pairs_dir and args.save_pairs:
        print("ERROR: --pairs-dir and --save-pairs are mutually exclusive", file=sys.stderr)
        return 1

    # ---- Load vocabulary -------------------------------------------------
    vocab_path = Path(args.vocab)
    if not vocab_path.exists():
        print(f"ERROR: vocab file not found: {vocab_path}", file=sys.stderr)
        print("Run 'ai-t9-build-vocab' first to create vocab.json", file=sys.stderr)
        return 1

    print(f"Loading vocabulary from {vocab_path}…")
    vocab = Vocabulary.load(vocab_path)
    print(f"  {vocab.size} words loaded")

    # ---- Build trainer --------------------------------------------------
    trainer = TrainerCls(
        vocab=vocab,
        embed_dim=args.embed_dim,
        context_window=args.context_window,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_frac=args.warmup_frac,
        min_lr_frac=args.min_lr_frac,
        temperature=args.temperature,
        batch_size=args.batch_size,
        accumulate_grad_batches=args.accumulate_grad_batches,
        clip_grad_norm=args.clip_grad_norm,
        seed=args.seed,
        device=args.device,
        debug=args.debug,
    )

    # ---- Train ----------------------------------------------------------
    # corpus_files may be set below; initialize here so the bigram section can
    # always reference it regardless of which training path was taken.
    corpus_files = None

    if args.pairs_dir:
        # Sharded-directory path: shuffle shards each epoch, optional prefetch.
        pairs_dir = Path(args.pairs_dir)
        if not pairs_dir.is_dir():
            print(f"ERROR: pairs directory not found: {pairs_dir}", file=sys.stderr)
            return 1
        trainer.train_from_pairs_dir(pairs_dir, epochs=args.epochs, verbose=True)

    elif args.load_pairs:
        # Single-file path: skip corpus loading, use precomputed pairs directly.
        pairs_path = Path(args.load_pairs)
        if not pairs_path.exists():
            print(f"ERROR: pairs file not found: {pairs_path}", file=sys.stderr)
            return 1
        trainer.train_from_pairs_file(pairs_path, epochs=args.epochs, verbose=True)

    else:
        # Corpus path: load sentences, optionally save pairs, optionally train.
        if args.corpus:
            corpus_files = _resolve_corpus_files(Path(args.corpus))
            if corpus_files is None:
                return 1

        if args.save_pairs:
            from ai_t9.model.trainer import (
                _corpus_file_sentence_ids,
                _brown_sentence_ids,
                save_pairs,
            )
            print("Loading corpus for pair precomputation…")
            if corpus_files:
                sentences: list[list[int]] = []
                for p in corpus_files:
                    sentences.extend(_corpus_file_sentence_ids(p, vocab))
            else:
                sentences = _brown_sentence_ids(vocab)
            pairs_out = Path(args.save_pairs)
            n = save_pairs(
                sentences,
                context_window=args.context_window,
                vocab_size=vocab.size,
                path=pairs_out,
                verbose=True,
                max_shard_pairs=args.shard_size,
            )
            print(f"Precomputed {n:,} pairs → {pairs_out}")
            if args.pairs_only:
                return 0
            # Train from the just-written file (validates the roundtrip too).
            resolved = pairs_out if str(pairs_out).endswith(".npz") else Path(str(pairs_out) + ".npz")
            trainer.train_from_pairs_file(resolved, epochs=args.epochs, verbose=True)

        elif corpus_files:
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
        if corpus_files:
            scorer = BigramScorer(vocab)
            for path in corpus_files:
                sents = _corpus_file_sentence_ids(path, vocab)
                for sent in sents:
                    scorer.train_on_ids(sent)
        else:
            scorer = BigramScorer.build_from_nltk(vocab, verbose=True)
        ngram_path = Path(args.save_ngram)
        ngram_path.parent.mkdir(parents=True, exist_ok=True)
        actual_ngram_path = scorer.save(ngram_path)
        print(f"Saved bigram model → {actual_ngram_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
