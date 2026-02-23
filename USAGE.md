# ai-t9 Usage Guide

- [Training pipeline](#training-pipeline)
  - [Full pipeline with `ai-t9-run`](#full-pipeline-with-ai-t9-run)
  - [Preparing a corpus](#preparing-a-corpus)
  - [Config file reference](#config-file-reference)
  - [Running individual steps](#running-individual-steps)
- [Individual CLI tools](#individual-cli-tools)
  - [Build vocabulary and dictionary](#build-vocabulary-and-dictionary-ai-t9-build-vocab)
  - [Train the neural model](#train-the-neural-model-ai-t9-train)
  - [Precompute training pairs](#precompute-training-pairs)
- [Data artifacts](#data-artifacts)
- [Cloud training](#cloud-training)
  - [Vast.ai](#vastai)
  - [Modal](#modal)
- [S3/R2 bucket management](#s3r2-bucket-management)
- [Demos](#demos)
  - [Interactive CLI demo](#interactive-cli-demo)
  - [GUI demo](#gui-demo)

---

## Training pipeline

### Full pipeline with `ai-t9-run`

The recommended way to train is with `ai-t9-run` and a YAML config file. It
runs all steps end-to-end and handles S3 upload/download between steps when
configured.

```bash
# Install training dependencies
pip install -e ".[train,data]"

# Run the full pipeline
ai-t9-run configs/default.yaml
```

Pipeline steps (executed in order):

| Step | What it does |
|------|-------------|
| `corpus` | Combines all configured dataset sources into a single `corpus.txt` |
| `vocab` | Counts words and builds `vocab.json` + `dict.json` |
| `pairs` | Precomputes (context, target) training pairs into `pairs/` shards |
| `train` | Trains the DualEncoder model, writes `model.npz` |
| `ngram` | Trains a bigram language model, writes `bigram.npz` |

### Preparing a corpus

Corpus files must be UTF-8 plain text with one utterance per line. Only
alphabetic tokens are counted; punctuation, numbers, and mixed-case forms are
normalised to lowercase and stripped.

```
corpuses/
  mytext.txt      ← one sentence / utterance per line
  mortext.txt
```

The `corpus` step in the pipeline supports three source types in the config:

```yaml
datasets:
  # Local files or directories of *.txt files
  - type: local
    path: "corpuses/"

  # Stream a HuggingFace dataset (uses streaming=True, no full download)
  - type: huggingface
    name: "wikitext"
    config: "wikitext-103-raw-v1"
    split: "train"
    column: "text"

  # Download a file from your S3/R2 bucket
  - type: s3
    key: "corpuses/extra-corpus.txt"
```

### Config file reference

See `configs/default.yaml` for a fully commented example. Key fields:

```yaml
name: "my-run"

datasets: [...]               # corpus sources (see above)

dictionary: "path/to/wordlist.txt"  # optional: restrict T9 candidates

vocab:
  max_words: 500_000
  min_count: 20

model:
  embed_dim: 64
  context_window: 3

training:
  epochs: 5
  lr: 0.001
  objective: "sgns"           # "sgns" (default) or "clip"
  n_negatives: 15             # SGNS only
  batch_size: 0               # 0 = auto-detect from GPU VRAM
  accumulate_grad_batches: 1
  device: "auto"

pairs:
  shard_size: 10_000_000      # pairs per shard file; null for single file

ngram: true                   # set false to skip bigram model

output_dir: "data"

steps:
  - corpus
  - vocab
  - pairs
  - train
  - ngram

s3:                           # optional — omit if not using S3/R2
  endpoint:   "${AI_T9_S3_ENDPOINT}"
  bucket:     "${AI_T9_S3_BUCKET}"
  access_key: "${AI_T9_S3_ACCESS_KEY}"
  secret_key: "${AI_T9_S3_SECRET_KEY}"
  region:     "auto"
  upload: true
  paths:
    vocab:  "vocab/vocab.json"
    dict:   "vocab/dict.json"
    pairs:  "pairs/"
    model:  "models/model.npz"
    ngram:  "ngrams/bigram.npz"
    corpus: "corpuses/"
```

### Running individual steps

```bash
# Run only specific steps (skipping the rest)
ai-t9-run configs/default.yaml --step train --step ngram

# Skip specific steps (run everything else)
ai-t9-run configs/default.yaml --skip corpus --skip vocab
```

When a required input file is missing locally and S3 is configured, the runner
downloads it automatically, so steps can be split across machines.

---

## Individual CLI tools

These tools offer finer control than `ai-t9-run`. They are primarily for one-off
operations or when integrating into an existing pipeline.

### Build vocabulary and dictionary (`ai-t9-build-vocab`)

```bash
# From a single file
ai-t9-build-vocab --corpus corpuses/mytext.txt --output data/

# From a directory of *.txt files (counts are combined)
ai-t9-build-vocab --corpus corpuses/ --output data/

# Restrict T9 dictionary to a verified wordlist
ai-t9-build-vocab --corpus corpuses/ --dictionary tt9-dictionary/en-utf8.txt --output data/
```

Words in the wordlist that are absent from the corpus are merged into the
vocabulary at floor frequency so the model and n-gram scorer can still assign
them meaningful scores.

| Flag | Default | Description |
|------|---------|-------------|
| `--corpus` | required | Corpus file or directory of `*.txt` files |
| `--dictionary` | none | Wordlist to filter T9 candidates |
| `--max-words N` | 50 000 | Maximum vocabulary size |
| `--min-count N` | 2 | Minimum corpus frequency to include a word |
| `--output DIR` | required | Directory to write `vocab.json` and `dict.json` |

### Train the neural model (`ai-t9-train`)

```bash
# Train on a corpus file
ai-t9-train --vocab data/vocab.json --corpus corpuses/ \
            --output data/model.npz --save-ngram data/bigram.npz \
            --epochs 5 --embed-dim 64

# CLIP-style objective instead of SGNS
ai-t9-train --vocab data/vocab.json --corpus corpuses/ \
            --output data/model.npz --objective clip --temperature 0.07
```

| Flag | Default | Description |
|------|---------|-------------|
| `--vocab` | required | `vocab.json` path |
| `--corpus` | — | Corpus file or directory (`*.txt`) |
| `--output` | required | Output `model.npz` path |
| `--save-ngram FILE` | none | Also train and save a bigram model |
| `--epochs N` | 3 | Training epochs |
| `--embed-dim N` | 64 | Embedding dimension |
| `--context-window N` | 3 | Context words fed to the encoder |
| `--objective` | `sgns` | `sgns` (O(B×k)) or `clip` (O(B²)) |
| `--n-negatives N` | 15 | SGNS negatives per positive |
| `--temperature F` | 0.07 | CLIP softmax temperature |
| `--lr F` | 0.001 | Learning rate |
| `--batch-size N` | auto | Pairs per batch |
| `--device` | auto | `cuda`, `mps`, or `cpu` |

### Precompute training pairs

For large datasets or cloud GPU workflows, separate the CPU-heavy pair
precomputation from the GPU training step:

```bash
# Precompute pairs into shards (CPU job, run once)
ai-t9-train --vocab data/vocab.json --corpus corpuses/ \
            --save-pairs data/pairs/pairs.npz --shard-size 10000000 --pairs-only

# Train from precomputed shards (GPU job, fast startup)
ai-t9-train --vocab data/vocab.json --pairs-dir data/pairs/ \
            --output data/model.npz --epochs 10 --accumulate-grad-batches 4
```

---

## Data artifacts

All artifacts must be built from the same vocabulary. Do not mix files from
different builds.

| File | Built by | Description |
|------|----------|-------------|
| `vocab.json` | `ai-t9-build-vocab` / `ai-t9-run` | Word↔ID mapping with frequency counts |
| `dict.json` | `ai-t9-build-vocab` / `ai-t9-run` | Digit-sequence → candidate word index |
| `model.npz` | `ai-t9-train` / `ai-t9-run` | DualEncoder embeddings (NumPy) |
| `bigram.npz` | `ai-t9-train --save-ngram` / `ai-t9-run` | Smoothed bigram transition counts |
| `pairs/pairs_*.npz` | `ai-t9-train --save-pairs` / `ai-t9-run` | Precomputed (context, target) training pairs |

---

## Cloud training

### Vast.ai

`scripts/vast_orchestrate.py` provisions a GPU instance, uploads the current
local package as a wheel, runs `ai-t9-run`, downloads the artifacts, and tears
down the instance.

**Prerequisites:**

```bash
pip install -e ".[vast]"
pip install vastai
vastai apikey set <your-key>
```

**Run the full pipeline on a Vast.ai GPU:**

```bash
python scripts/vast_orchestrate.py configs/vast-large.yaml
```

**Common options:**

```bash
# Preview the cheapest matching offer without spending anything
python scripts/vast_orchestrate.py configs/vast-large.yaml --dry-run

# Override GPU type and price cap
python scripts/vast_orchestrate.py configs/vast-large.yaml \
    --gpu H100 --max-price 3.0

# Skip specific steps (e.g. corpus and vocab already in S3)
python scripts/vast_orchestrate.py configs/vast-large.yaml \
    --skip corpus --skip vocab

# Reuse an already-running instance
python scripts/vast_orchestrate.py configs/vast-large.yaml \
    --instance-id 12345678

# Keep the instance alive after training (for inspection)
python scripts/vast_orchestrate.py configs/vast-large.yaml --no-destroy
```

| Flag | Default | Description |
|------|---------|-------------|
| `--gpu NAME` | `RTX_3090` | GPU name filter for offer search |
| `--min-vram N` | 16 | Minimum VRAM in GB |
| `--max-price F` | 1.0 | Max $/hour |
| `--cuda-version V` | `12.0` | Minimum CUDA version |
| `--disk N` | 30 | Instance disk size in GB |
| `--image IMAGE` | PyTorch 2.5 / CUDA 12.4 | Docker image |
| `--install` | `wheel` | `wheel` = build + upload local wheel; `skip` = image has it |
| `--ssh-retries N` | 8 | SSH connection attempts before giving up |
| `--stabilize-wait N` | 30 | Seconds to wait after "running" before SSH |
| `--instance-id N` | — | Reuse an existing instance |
| `--no-destroy` | off | Keep the instance after training |
| `--dry-run` | off | Print best offer and exit |

**S3 credentials** (`AI_T9_S3_ENDPOINT`, `AI_T9_S3_BUCKET`,
`AI_T9_S3_ACCESS_KEY`, `AI_T9_S3_SECRET_KEY`) and HuggingFace tokens
(`HF_TOKEN`) are automatically forwarded from your local environment to the
remote instance.

**Custom Docker image** (optional — avoids the PyTorch download on every run):

```bash
# Build the image locally
bash scripts/build_image.sh               # tags as ai-t9-trainer:latest

# Push to a registry so Vast.ai can pull it
docker tag ai-t9-trainer:latest myuser/ai-t9-trainer:latest
docker push myuser/ai-t9-trainer:latest

# Use it — skip the wheel install since the package is baked in
python scripts/vast_orchestrate.py configs/vast-large.yaml \
    --image myuser/ai-t9-trainer:latest --install skip
```

### Modal

Train on serverless cloud GPUs using [Modal](https://modal.com). The Modal app
wraps the same CLIs used locally.

**Setup:**

```bash
pip install modal
modal setup
modal volume create ai-t9-data    # persistent storage for artifacts
```

**Upload corpus files to the Volume:**

```bash
modal volume put ai-t9-data corpuses/ corpuses/

# Or ingest a HuggingFace dataset directly on Modal (no local download)
modal run modal_app.py --ingest wikitext \
    --ingest-config wikitext-103-raw-v1 --ingest-filename wikitext.txt
```

**Run the pipeline:**

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

| Flag | VRAM | Notes |
|------|------|-------|
| `--gpu L4` | 24 GB | Good for iteration |
| `--gpu A100` | 40 GB | Fast training |
| `--gpu A100-80GB` | 80 GB | Large vocab / embed_dim |
| `--gpu H100` | 80 GB | Fastest |

---

## S3/R2 bucket management

The `ai-t9-data` CLI manages training data in any S3-compatible bucket.
Cloudflare R2 is recommended for zero egress fees.

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

---

## Demos

### Interactive CLI demo

```bash
python examples/demo.py \
    --vocab data/vocab.json \
    --dict  data/dict.json \
    --model data/model.npz \
    --ngram data/bigram.npz
```

Type T9 digit sequences at the `t9>` prompt. Append `-v` to a sequence for a
per-signal score breakdown. Other commands:

| Command | Action |
|---------|--------|
| `<digits>` | Predict words |
| `<digits> -v` | Predict with score breakdown |
| `.<word>` | Manually add a word to context |
| `reset` / `clear` | Clear current context |
| `ctx` / `context` | Show current context window |
| `help` | Show command reference |
| `quit` / `q` | Exit |

### GUI demo

A phone-style T9 interface built with PyQt6:

```bash
pip install -e ".[gui]"

python examples/gui_demo.py \
    --vocab data/vocab.json \
    --dict  data/dict.json \
    --model data/model.npz \
    --ngram data/bigram.npz
```

Keyboard shortcuts:

| Key | Action |
|-----|--------|
| `2`–`9` | T9 digit input |
| `0` / Space | Confirm word + space |
| Backspace | Delete last digit (or undo last word if buffer empty) |
| Tab / `#` | Cycle through candidates |
| `1` / `.` | Punctuation cycling |
| `M` | Toggle T9 ↔ ABC (multi-tap) mode |
| Escape | Clear all |
| Enter | Confirm word |
| Ctrl+C | Copy composed text to clipboard |
