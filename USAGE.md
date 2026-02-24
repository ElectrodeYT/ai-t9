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
  - [Vast.ai (blocking)](#vastai-blocking)
  - [Vast.ai autonomous manager](#vastai-autonomous-manager)
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

### Vast.ai (blocking)

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

# Multi-GPU instance (2× RTX 3090)
python scripts/vast_orchestrate.py configs/vast-large.yaml \
    --num-gpus 2 --max-price 2.0

# Interruptable (spot-like) instance — cheaper, may be preempted
python scripts/vast_orchestrate.py configs/vast-large.yaml --interruptable

# Interruptable with automatic respawn on interruption (requires S3 for state)
python scripts/vast_orchestrate.py configs/vast-large.yaml \
    --interruptable --retries 3

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
| `--num-gpus N` | 1 | Number of GPUs to search for |
| `--min-vram N` | 16 | Minimum VRAM in GB per GPU |
| `--max-price F` | 1.0 | Max $/hour |
| `--cuda-version V` | `12.0` | Minimum CUDA version |
| `--disk N` | 30 | Instance disk size in GB |
| `--image IMAGE` | PyTorch 2.5 / CUDA 12.4 | Docker image |
| `--interruptable` | off | Rent as spot-like instance (cheaper, may be preempted) |
| `--retries N` | 0 | Re-provision and retry on SSH disconnection (useful with `--interruptable` + S3) |
| `--install` | `wheel` | `wheel` = build + upload local wheel; `skip` = image has it |
| `--ssh-retries N` | 8 | SSH connection attempts before giving up |
| `--stabilize-wait N` | 30 | Seconds to wait after "running" before SSH |
| `--instance-id N` | — | Reuse an existing instance |
| `--no-destroy` | off | Keep the instance after training |
| `--dry-run` | off | Print best offer and exit |
| `--detach` | off | Upload artifacts to S3, start supervisor in tmux, exit immediately (see [Autonomous manager](#vastai-autonomous-manager)) |

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

### Vast.ai autonomous manager

The **autonomous manager** (`scripts/vast_manager.py`) solves the single biggest
problem with interruptable (spot) instances: if your laptop goes to sleep or
loses internet while training, the blocking SSH session dies and training never
resumes.

The manager runs on a cheap always-on VPS. It polls S3 for heartbeats written by
a lightweight supervisor process on the GPU instance. When the instance is
preempted (or the heartbeat goes stale), the manager automatically provisions a
new instance and picks up where training left off — no human intervention needed.

```
[Browser] ──HTTP:7860──► [VPS: vast_manager.py]
                               │ polls S3 every 2 min
                               ▼
                          [S3: jobs/ prefix]  ◄── [GPU: supervisor.py writes heartbeat]
                               │                        │
                               └──── provision ────────►└── runs ai-t9-run
```

#### Architecture

| Component | Where it runs | What it does |
|-----------|---------------|--------------|
| `vast_manager.py` | VPS (always-on) | Poll loop, HTTP API, web UI |
| `vast_supervisor.py` | GPU instance (tmux) | Wraps `ai-t9-run`, heartbeats S3 every 60 s |
| `jobs/` S3 prefix | S3 bucket | Single source of truth for all job state |

The manager is **stateless** — all job state lives in S3. You can restart the
manager at any time without affecting running jobs.

#### VPS setup

```bash
# Install deps
pip install boto3 paramiko vastai pyyaml

# Set credentials (add to /etc/environment or a .env file)
export AI_T9_S3_ENDPOINT="https://<account>.r2.cloudflarestorage.com"
export AI_T9_S3_BUCKET="my-ai-t9-bucket"
export AI_T9_S3_ACCESS_KEY="..."
export AI_T9_S3_SECRET_KEY="..."
export AI_T9_S3_REGION="auto"
export AI_T9_JOBS_PREFIX="jobs/"          # optional, default: jobs/

# Authenticate vastai CLI
vastai apikey set <your-api-key>

# Start the manager
python scripts/vast_manager.py --port 7860
```

Open `http://<vps-ip>:7860` in your browser. The dashboard shows all jobs with
live status and a log of events.

**Tip:** expose behind nginx with HTTP Basic Auth if the port is publicly
accessible.

#### Systemd service (recommended)

Generate a ready-to-use unit file:

```bash
python scripts/vast_manager.py --print-systemd-unit \
    | sudo tee /etc/systemd/system/ai-t9-manager.service
sudo systemctl enable --now ai-t9-manager
sudo journalctl -u ai-t9-manager -f
```

The generated unit file uses `EnvironmentFile=` for credentials and
`Restart=on-failure` so the manager comes back automatically after a VPS reboot
or crash.

#### Creating a job via the web UI

1. Open `http://<vps-ip>:7860`
2. Click **+ New Job**
3. Paste your YAML config, set GPU/price/options, click **Create Job**

The manager uploads the config (and wheel if install=wheel) to S3, provisions
the instance, deploys the supervisor into a tmux session, and returns the job
card. Creation takes ~2 minutes; the modal shows a spinner while it waits.

#### Creating a job via the API

```bash
curl -s -X POST http://<vps-ip>:7860/api/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "config_yaml": "output_dir: data\n...",
    "gpu": "RTX_3090",
    "num_gpus": 1,
    "max_price": 0.5,
    "min_vram": 16,
    "interruptable": true,
    "install": "wheel",
    "skip_steps": ["corpus", "vocab"]
  }'
```

Response is a job JSON object with `job_id`, `status`, `instance_id`, and an
`events` array.

#### Using `--detach` from the orchestrator

If you prefer to trigger jobs from the command line rather than the web UI:

```bash
# Upload artifacts to S3, provision instance, exit immediately.
# The job is then managed by the autonomous manager running on your VPS.
python scripts/vast_orchestrate.py configs/vast-large.yaml \
    --interruptable --detach
```

This prints the `job_id` and exits. You can monitor the job in the manager UI
or by checking S3 directly:

```bash
aws s3 cp s3://<bucket>/jobs/<job-id>/heartbeat.json - | python -m json.tool
aws s3 cp s3://<bucket>/jobs/<job-id>/manifest.json  - | python -m json.tool
```

#### Stopping and restarting jobs

Via the web UI, use the **Stop** and **Restart** buttons on each job card.

Via the API:

```bash
# Stop a running job (destroys the instance)
curl -s -X POST http://<vps-ip>:7860/api/jobs/<job-id>/stop

# Manually restart a stopped/failed/completed job
curl -s -X POST http://<vps-ip>:7860/api/jobs/<job-id>/restart
```

#### S3 layout

```
jobs/
  train-20240215-a3f2b1/
    manifest.json          job state (status, events, instance info)
    heartbeat.json         written by supervisor every 60 s
    supervisor.py          uploaded at job creation for re-deployment
    train_config.yaml      YAML config used for this run
    wheel/
      ai_t9-*.whl          uploaded at job creation (if install==wheel)

checkpoints/               written by ai-t9-run (unchanged)
  ckpt_0.pt
  ptr.json                 read by manager for epoch/shard progress display
```

#### How preemption recovery works

1. Supervisor writes `heartbeat.json` every 60 s. If the instance is preempted,
   writes stop.
2. Manager's poll loop (every 2 min by default) reads the heartbeat. If it's
   older than 5 min it calls `vastai show instance` to verify the instance is
   still alive.
3. If the instance is gone, the manager provisions a fresh one, re-uploads the
   supervisor and config from S3, starts the supervisor in tmux, and increments
   `restart_count` in the manifest.
4. `ai-t9-run` picks up from the latest checkpoint in S3 automatically (as long
   as your config has S3 upload enabled for checkpoints).

#### Job status reference

| Status | Meaning |
|--------|---------|
| `pending` | Job created, not yet provisioned (or between retries) |
| `running` | Instance alive, supervisor heartbeating |
| `completed` | `ai-t9-run` exited 0; supervisor wrote final status |
| `failed` | `ai-t9-run` exited non-zero, or provisioning failed |
| `stopped` | Manually stopped via UI / API |
| `interrupted` | Preempted; manager is respawning |

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
