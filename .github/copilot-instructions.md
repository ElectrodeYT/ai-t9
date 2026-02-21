# AI Coding Instructions for `ai-t9`

## Project shape (read this first)
- `ai_t9` is a hybrid T9 predictor: `freq` + optional neural `model` + optional `ngram` (`src/ai_t9/predictor.py`).
- Inference path is NumPy-only (`src/ai_t9/model/dual_encoder.py`); PyTorch is training-only (`src/ai_t9/model/trainer.py`).
- Core runtime flow: digits → dictionary candidates (`src/ai_t9/dictionary.py`) → score blend in `T9Predictor.predict()`.
- Session behavior is stateful and context-driven via `T9Session` (`src/ai_t9/session.py`), with a rolling lowercased context window.

## Data/artifact contract
- Canonical artifacts are:
  - `vocab.json` (word↔id + counts),
  - `dict.json` (digit sequence → `[word, wid]`),
  - `model.npz` (context/word embeddings),
  - `bigram.json` (smoothed bigram counts).
- `T9Predictor.from_files()` expects matching vocab IDs across all loaded artifacts; do not mix files built from different vocabularies.
- Unknown token semantics are fixed: `<unk>` at ID `0` (`src/ai_t9/model/vocab.py`).

## Scoring conventions to preserve
- Keep digit validation strict: only `2-9` are valid (`src/ai_t9/t9_map.py`).
- Dictionary buckets are pre-sorted by descending log-frequency; treat `lookup()` ordering as meaningful (`src/ai_t9/dictionary.py`).
- Predictor combines min-max-normalized signal arrays and renormalized weights (`src/ai_t9/predictor.py`):
  - disabled signals must contribute zero,
  - available signal weights are auto-renormalized.
- `BigramScorer` uses add-k smoothing (`k=0.5` default) and returns log-probabilities (`src/ai_t9/ngram.py`).

## Developer workflows
- Install base/dev deps: `pip install -e .` or `pip install -e ".[dev]"`.
- Run tests: `pytest -q` (tests are intentionally tiny/offline and avoid NLTK downloads; see `tests/test_t9.py`).
- Build vocab + dictionary:
  - `ai-t9-build-vocab --output data/`
  - `ai-t9-build-vocab --corpus corpuses/ --output data/`
- Train model (PyTorch extra required):
  - `pip install -e ".[train]"`
  - `ai-t9-train --vocab data/vocab.json --output data/model.npz --save-ngram data/bigram.json`
- Manual sanity run: `python examples/demo.py --vocab data/vocab.json --dict data/dict.json --model data/model.npz --ngram data/bigram.json`.
- Always run tests and sanity checks after making changes, especially to core logic or data processing.

## Repo-specific patterns
- Keep inference modules free of torch imports; import torch lazily via `_require_torch()` in training code.
- Preserve lowercase/alpha filtering behavior in corpus ingestion (`isalpha()` gates in vocab/trainer scripts).
- If changing corpus preprocessing, align with `scripts/discord_to_corpus.py` output expectation: one utterance per line, UTF-8 plain text.
- Prefer extending existing CLIs (`src/ai_t9/_scripts/build_vocab.py`, `src/ai_t9/_scripts/train.py`) over adding new entry points.
