# ai-t9

Advanced predictive T9 text input powered by a trainable dual-encoder neural model.

`ai-t9` maps digit sequences (2–9) to ranked word candidates using up to three
blended signals:

| Signal | Source | Always on? |
|--------|--------|------------|
| **freq** | Log-frequency from the training corpus | Yes |
| **model** | Dual-encoder cosine similarity (context → word) | If `model.npz` loaded |
| **ngram** | Bigram log-probability P(word \| prev_word) | If `bigram.npz` loaded |

Scores are rank-normalized and weighted, so predictions adapt to conversational
context in real time.

## Install

```bash
pip install -e ".[train,data]"
```

| Extra | Provides |
|-------|----------|
| `train` | PyTorch, tqdm, NLTK for model training |
| `data` | boto3 for S3/R2 bucket management |
| `gui` | PyQt6 for the phone-style GUI demo |
| `dev` | pytest + PyTorch for development |
| `vast` | paramiko for Vast.ai orchestration |

## Quick start

Place one or more UTF-8 plain-text corpus files (one utterance per line) in
`corpuses/`, then run the full pipeline:

```bash
ai-t9-run configs/default.yaml
```

This runs five steps in sequence — corpus aggregation, vocabulary build, training
pair precomputation, dual-encoder training, and bigram model training — writing
all artifacts to `data/`.

See **[USAGE.md](USAGE.md)** for the complete guide: per-step CLI flags, cloud
training on Vast.ai and Modal, the autonomous training manager for unattended
spot-instance training, S3/R2 data management, and demo instructions.

## Using as a library

```python
from ai_t9 import T9Predictor, T9Session

predictor = T9Predictor.from_files(
    vocab_path="data/vocab.json",
    dict_path="data/dict.json",
    model_path="data/model.npz",   # optional
    ngram_path="data/bigram.npz",  # optional
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

## Architecture

- **Inference** is NumPy-only — no PyTorch dependency at runtime.
- **Training** uses PyTorch with AMP, TF32, and optional CUDA graphs for fast
  GPU training.
- The **DualEncoder** model learns separate context and word embedding tables.
  At inference time, context word embeddings are mean-pooled and compared to
  candidate word embeddings via cosine similarity.
- Training objectives are **pluggable** via `TrainingObjective` subclasses in
  `src/ai_t9/model/objectives.py`. The default is **SGNS** (Skip-Gram Negative
  Sampling), O(B × k) per step. A **CLIP**-style objective (O(B²)) is also
  available.
- **Negative sampling** is frequency-weighted (Word2Vec-style f^0.75) for better
  rare-word representations.
- L2 normalisation happens inside the model; SGNS works directly on cosine
  similarities in [-1, 1].
- The `<unk>` token (ID 0) is excluded from training pairs and negative samples.

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

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
