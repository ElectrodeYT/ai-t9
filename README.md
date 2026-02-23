# ai-t9

Advanced predictive T9 text input powered by a trainable dual-encoder neural model.

`ai-t9` maps digit sequences (2–9) to ranked word candidates using up to three
blended signals:

| Signal | Source | Always on? |
|--------|--------|------------|
| **freq** | Log-frequency from the training corpus | Yes |
| **model** | Dual-encoder cosine similarity (context → word) | If `model.npz` loaded |
| **ngram** | Bigram log-probability P(word \| prev_word) | If `bigram.json` loaded |

Scores are rank-normalized and weighted, so predictions adapt to conversational
context in real time.

## Quick start

### Install

```bash
pip install -e .
```

For training support (requires PyTorch):

```bash
pip install -e ".[train]"
```

For the GUI demo (requires PyQt6):

```bash
pip install -e ".[gui]"
```

### Build vocabulary and dictionary from a corpus

The build step creates two artifacts — `vocab.json` (word↔id mapping with
frequency counts) and `dict.json` (digit-sequence → candidate index).

**From the NLTK Brown corpus (default — downloads ~15 MB on first run):**

```bash
ai-t9-build-vocab --output data/
```

**From your own plain-text corpus (one utterance per line, UTF-8):**

```bash
# Single file
ai-t9-build-vocab --corpus corpuses/mytext.txt --output data/

# Directory of *.txt files (counts are combined)
ai-t9-build-vocab --corpus corpuses/ --output data/
```

**Restricting the dictionary to a verified wordlist:**

If your corpus contains typos or domain jargon you don't want surfaced as
predictions, pass a clean wordlist. The full corpus is still used for frequency
counts and model training — only the dictionary output is filtered:

```bash
ai-t9-build-vocab --corpus corpuses/ --dictionary tt9-dictionary/en-utf8.txt --output data/
```

Words in the wordlist that aren't in the corpus are automatically merged into
the vocabulary at floor frequency so the model and n-gram scorer can still
assign them meaningful scores.

Additional options:

| Flag | Default | Description |
|------|---------|-------------|
| `--max-words N` | 50 000 | Maximum vocabulary size |
| `--min-count N` | 2 | Minimum word frequency to include |

### Train the neural model (optional)

Training produces `model.npz` (DualEncoder embeddings) and optionally
`bigram.json` (smoothed bigram counts). Requires the `[train]` extra.

```bash
# Train on NLTK Brown corpus
ai-t9-train --vocab data/vocab.json --output data/model.npz --save-ngram data/bigram.json

# Train on a custom corpus
ai-t9-train --vocab data/vocab.json --corpus corpuses/ --output data/model.npz \
            --save-ngram data/bigram.json --epochs 5 --embed-dim 64
```

Training options:

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs N` | 3 | Training epochs |
| `--embed-dim N` | 64 | Embedding dimension |
| `--context-window N` | 3 | Context words for the encoder |
| `--neg-samples N` | 20 | Negatives per positive (frequency-weighted) |
| `--lr F` | 0.005 | Learning rate |
| `--batch-size N` | 2048 | Pairs per batch |
| `--device` | auto | `cuda`, `mps`, or `cpu` |

### Precompute training pairs (optional)

For large datasets or cloud training, you can split data preparation (CPU)
from model training (GPU). Precomputed pairs are portable `.npz` files:

```bash
# Precompute pairs from a corpus (CPU-heavy, run once)
ai-t9-train --vocab data/vocab.json --corpus corpuses/ \
            --save-pairs data/pairs.npz --pairs-only

# Later: train from precomputed pairs (GPU-heavy, fast startup)
ai-t9-train --vocab data/vocab.json --load-pairs data/pairs.npz \
            --output data/model.npz --epochs 10
```

### Cloud training with Modal (optional)

Train on cloud GPUs using [Modal](https://modal.com). The Modal app wraps
the same CLIs used locally — no code duplication.

**Setup:**

```bash
pip install modal
modal setup
modal volume create ai-t9-data    # persistent storage for artifacts
```

**Upload corpus files to the Volume:**

```bash
# Upload local corpus files
modal volume put ai-t9-data corpuses/ corpuses/

# Or ingest a HuggingFace dataset directly on Modal (no local download)
modal run modal_app.py --ingest wikitext \
    --ingest-config wikitext-103-raw-v1 --ingest-filename wikitext.txt
```

**Run the full pipeline (prep → train → download):**

```bash
# Full pipeline: build vocab + pairs on CPU, train on GPU, download results
modal run modal_app.py --use-volume-corpus --gpu L4

# Override hyperparameters
modal run modal_app.py --use-volume-corpus --gpu A100 \
    --epochs 10 --embed-dim 128 --batch-size 8192

# Skip prep if pairs are already on the Volume
modal run modal_app.py --skip-prep --gpu H100 --epochs 20

# Prep only (no training)
modal run modal_app.py --prep-only --use-volume-corpus

# Download artifacts from a previous run
modal run modal_app.py --download-only

# Long-running job (survives terminal disconnect)
modal run --detach modal_app.py --use-volume-corpus --gpu A100 --epochs 20

# List what's on the Volume
modal run modal_app.py --list-files
```

Available GPU tiers:

| Flag | VRAM | Cost | Notes |
|------|------|------|-------|
| `--gpu L4` | 24 GB | ~$0.60/hr | Good for iteration |
| `--gpu A100` | 40 GB | ~$2.80/hr | Fast training |
| `--gpu A100-80GB` | 80 GB | ~$3.70/hr | Large vocab/embed_dim |
| `--gpu H100` | 80 GB | ~$4.50/hr | Fastest |

### Run the interactive demo

```bash
# With pre-built data files (instant startup)
python examples/demo.py \
    --vocab data/vocab.json --dict data/dict.json \
    --model data/model.npz --ngram data/bigram.json

# Fall back to NLTK (no data files needed)
python examples/demo.py
```

Type T9 digit sequences at the `t9>` prompt. Append `-v` for a score
breakdown. Type `help` for all commands.

### Run the GUI demo

A phone-style T9 interface built with PyQt6:

```bash
python examples/gui_demo.py \
    --vocab data/vocab.json --dict data/dict.json \
    --model data/model.npz --ngram data/bigram.json
```

## Using as a library

```python
from ai_t9 import T9Predictor, T9Session

# Load from pre-built files
predictor = T9Predictor.from_files(
    vocab_path="data/vocab.json",
    dict_path="data/dict.json",
    model_path="data/model.npz",     # optional
    ngram_path="data/bigram.json",    # optional
)

# One-shot prediction
print(predictor.predict("4663"))  # ["home", "good", "gone", ...]

# Stateful session with rolling context
session = T9Session(predictor)
session.add_context("i", "am", "going")
print(session.dial("4663"))   # context-aware ranking
session.confirm("home")
print(session.dial("269"))    # next word, informed by "home"
```

## Preparing a custom corpus

Corpus files should be UTF-8 plain text with one utterance per line. Only
alphabetic tokens are counted (punctuation and numbers are ignored).

## Data artifacts

A complete setup uses four files in the `data/` directory:

| File | Built by | Description |
|------|----------|-------------|
| `vocab.json` | `ai-t9-build-vocab` | Word↔ID mapping with frequency counts |
| `dict.json` | `ai-t9-build-vocab` | Digit-sequence → candidate word index |
| `model.npz` | `ai-t9-train` | DualEncoder context/word embeddings (NumPy) |
| `bigram.json` | `ai-t9-train --save-ngram` | Smoothed bigram transition counts |
| `pairs.npz` | `ai-t9-train --save-pairs` | Precomputed (context, target) training pairs |

All artifacts must be built from the same vocabulary. Do not mix files from
different builds.

## S3/R2 bucket management

The `ai-t9-data` CLI manages training data in any S3-compatible bucket
(Cloudflare R2 recommended for zero egress fees):

```bash
pip install -e ".[data]"

# Configure (set once in your shell profile)
export AI_T9_S3_ENDPOINT="https://<account>.r2.cloudflarestorage.com"
export AI_T9_S3_BUCKET="my-ai-t9-bucket"
export AI_T9_S3_ACCESS_KEY="..."
export AI_T9_S3_SECRET_KEY="..."

# List bucket contents
ai-t9-data ls
ai-t9-data ls corpuses/

# Upload / download artifacts
ai-t9-data upload data/vocab.json vocab/vocab.json
ai-t9-data download vocab/vocab.json data/vocab.json

# Stream a HuggingFace dataset directly to the bucket
ai-t9-data fetch-hf wikitext wikitext-103-raw-v1 train corpuses/wiki.txt

# Stream a HuggingFace dataset to a local/mounted path (for Modal)
ai-t9-data fetch-hf-local wikitext wikitext-103-raw-v1 train /data/corpuses/wiki.txt
```

## Architecture

- **Inference** is NumPy-only — no PyTorch dependency at runtime.
- **Training** uses PyTorch with AMP, TF32, and optional CUDA graphs for fast
  GPU training.
- The **DualEncoder** model learns separate context and word embedding tables.
  At inference time, context word embeddings are mean-pooled and compared to
  candidate word embeddings via cosine similarity.
- **Negative sampling** during training is frequency-weighted (Word2Vec-style
  $f^{0.75}$ distribution) for better rare-word representations.
- The `<unk>` token (ID 0) is excluded from training pairs and negative
  samples, preventing corpus noise from leaking into embeddings.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

Tests are fast, offline, and do not require NLTK data downloads.

### Optional extras

| Extra | Install | Provides |
|-------|---------|----------|
| `train` | `pip install -e ".[train]"` | PyTorch + tqdm for model training |
| `data` | `pip install -e ".[data]"` | boto3 for S3/R2 bucket management |
| `gui` | `pip install -e ".[gui]"` | PyQt6 for the phone-style GUI demo |
| `dev` | `pip install -e ".[dev]"` | pytest + PyTorch for developing/testing |

## T9 keypad reference

```
┌───────┬───────┬───────┐
│ 2     │ 3     │ 4     │
│ a b c │ d e f │ g h i │
├───────┼───────┼───────┤
│ 5     │ 6     │ 7     │
│ j k l │ m n o │ p q r s│
├───────┼───────┼───────┤
│ 8     │ 9     │
│ t u v │ w x y z│
└───────┴───────┘
```

## License

See [pyproject.toml](pyproject.toml) for package metadata.
