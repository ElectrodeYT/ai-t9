"""YAML-based training run configuration for ai-t9.

Provides a declarative way to define complete training runs including:

- Multiple dataset sources (local files, HuggingFace, S3)
- Vocabulary and dictionary settings
- Model architecture and training hyperparameters
- Pair precomputation settings
- S3 artifact management
- Pipeline step control

Environment variables can be referenced in string values via ``${VAR_NAME}``
syntax.  Unset variables are left as literal ``${VAR_NAME}`` strings (the
runner decides whether they're required).

Example YAML::

    name: "my-run"
    datasets:
      - type: local
        path: "corpuses/"
      - type: huggingface
        name: "wikitext"
        config: "wikitext-103-raw-v1"
        split: "train"
        column: "text"
    vocab:
      max_words: 500_000
      min_count: 20
    model:
      embed_dim: 64
      context_window: 3
    training:
      epochs: 5
      lr: 0.001
    steps:
      - corpus
      - vocab
      - pairs
      - train
"""

from __future__ import annotations

import os
import re
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DatasetSource:
    """A single dataset source to combine into the training corpus.

    Supported types:

    ``local``
        A local file or directory of ``*.txt`` files.

    ``huggingface``
        A HuggingFace dataset, streamed row-by-row.

    ``s3``
        A file in an S3-compatible bucket (downloaded first).
    """

    type: str  # "local" | "huggingface" | "s3"

    # -- local --
    path: str | None = None

    # -- huggingface --
    name: str | None = None
    config: str | None = None
    split: str = "train"
    column: str = "text"
    max_lines: int | None = None

    # -- s3 --
    key: str | None = None

    @staticmethod
    def from_dict(d: dict) -> DatasetSource:
        return DatasetSource(
            type=d["type"],
            path=d.get("path"),
            name=d.get("name"),
            config=d.get("config"),
            split=d.get("split", "train"),
            column=d.get("column", "text"),
            max_lines=d.get("max_lines"),
            key=d.get("key"),
        )


@dataclass
class VocabConfig:
    """Vocabulary building settings."""

    max_words: int = 500_000
    min_count: int = 20

    @staticmethod
    def from_dict(d: dict | None) -> VocabConfig:
        if not d:
            return VocabConfig()
        return VocabConfig(
            max_words=d.get("max_words", 500_000),
            min_count=d.get("min_count", 20),
        )


@dataclass
class ModelConfig:
    """Model architecture settings."""

    embed_dim: int = 64
    context_window: int = 3

    @staticmethod
    def from_dict(d: dict | None) -> ModelConfig:
        if not d:
            return ModelConfig()
        return ModelConfig(
            embed_dim=d.get("embed_dim", 64),
            context_window=d.get("context_window", 3),
        )


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    epochs: int = 5
    lr: float = 0.001
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05
    min_lr_frac: float = 0.01
    temperature: float = 0.07
    batch_size: int = 0  # 0 = auto-detect
    accumulate_grad_batches: int = 4
    clip_grad_norm: float = 1.0
    seed: int = random.randint(0, 2**32 - 1)
    device: str = "auto"
    checkpoint: str | None = None
    objective: str = "sgns"
    n_negatives: int = 15
    hard_neg_frac: float = 0.5  # fraction of SGNS negatives that are T9-hard

    @staticmethod
    def from_dict(d: dict | None) -> TrainingConfig:
        if not d:
            return TrainingConfig()
        return TrainingConfig(
            epochs=d.get("epochs", 5),
            lr=d.get("lr", 0.001),
            weight_decay=float(d.get("weight_decay", 1e-4)),
            warmup_frac=d.get("warmup_frac", 0.05),
            min_lr_frac=d.get("min_lr_frac", 0.01),
            temperature=d.get("temperature", 0.07),
            batch_size=d.get("batch_size", 0),
            accumulate_grad_batches=d.get("accumulate_grad_batches", 4),
            clip_grad_norm=d.get("clip_grad_norm", 1.0),
            seed=d.get("seed", random.randint(0, 2**32 - 1)),
            device=d.get("device", "auto"),
            checkpoint=d.get("checkpoint"),
            objective=d.get("objective", "sgns"),
            n_negatives=d.get("n_negatives", 15),
            hard_neg_frac=d.get("hard_neg_frac", 0.5),
        )


@dataclass
class PairsConfig:
    """Pair precomputation settings."""

    shard_size: int | None = 10_000_000

    @staticmethod
    def from_dict(d: dict | None) -> PairsConfig:
        if not d:
            return PairsConfig()
        return PairsConfig(shard_size=d.get("shard_size", 10_000_000))


@dataclass
class S3Paths:
    """Remote key layout for S3 artifact storage."""

    vocab: str = "vocab/vocab.json"
    dict: str = "vocab/dict.json"
    pairs: str = "pairs/"
    model: str = "models/model.npz"
    corpus: str = "corpuses/"
    checkpoint: str = "checkpoints/"
    dictionary: str = "dictionaries/words_alpha.txt"

    @staticmethod
    def from_dict(d: dict | None) -> S3Paths:
        if not d:
            return S3Paths()
        return S3Paths(
            vocab=d.get("vocab", "vocab/vocab.json"),
            dict=d.get("dict", "vocab/dict.json"),
            pairs=d.get("pairs", "pairs/"),
            model=d.get("model", "models/model.npz"),
            corpus=d.get("corpus", "corpuses/"),
            checkpoint=d.get("checkpoint", "checkpoints/"),
            dictionary=d.get("dictionary", "dictionaries/words_alpha.txt"),
        )


@dataclass
class S3Config:
    """S3-compatible storage configuration (e.g. Cloudflare R2, AWS S3).

    When ``upload`` is True, artifacts are pushed to the bucket after each
    pipeline step completes.  When a required local file is missing but S3
    is configured, the runner will attempt to download it automatically.
    """

    endpoint: str | None = None
    bucket: str | None = None
    access_key: str | None = None
    secret_key: str | None = None
    region: str = "auto"
    paths: S3Paths = field(default_factory=S3Paths)
    upload: bool = False

    @property
    def enabled(self) -> bool:
        """True when all required connection fields are set (non-empty)."""
        return bool(
            self.endpoint and self.bucket and self.access_key and self.secret_key
        )

    @staticmethod
    def from_dict(d: dict | None) -> S3Config:
        if not d:
            return S3Config()
        return S3Config(
            endpoint=d.get("endpoint"),
            bucket=d.get("bucket"),
            access_key=d.get("access_key"),
            secret_key=d.get("secret_key"),
            region=d.get("region", "auto"),
            paths=S3Paths.from_dict(d.get("paths")),
            upload=d.get("upload", False),
        )


# Default pipeline step order.
DEFAULT_STEPS = ["corpus", "vocab", "pairs", "train"]
VALID_STEPS = set(DEFAULT_STEPS)


@dataclass
class RunConfig:
    """Complete training run configuration."""

    name: str = "default"
    datasets: list[DatasetSource] = field(default_factory=list)
    dictionary: str | None = None
    vocab: VocabConfig = field(default_factory=VocabConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    pairs: PairsConfig = field(default_factory=PairsConfig)
    output_dir: str = "data"
    steps: list[str] = field(default_factory=lambda: list(DEFAULT_STEPS))
    s3: S3Config = field(default_factory=S3Config)

    def validate(self) -> list[str]:
        """Return a list of validation errors (empty if valid)."""
        errors: list[str] = []
        for step in self.steps:
            if step not in VALID_STEPS:
                errors.append(f"Unknown step: '{step}' (valid: {', '.join(VALID_STEPS)})")
        if "corpus" in self.steps and not self.datasets:
            errors.append("'corpus' step requires at least one entry in 'datasets'")
        for i, ds in enumerate(self.datasets):
            if ds.type not in ("local", "huggingface", "s3"):
                errors.append(f"datasets[{i}]: unknown type '{ds.type}'")
            if ds.type == "local" and not ds.path:
                errors.append(f"datasets[{i}]: 'local' type requires 'path'")
            if ds.type == "huggingface" and not ds.name:
                errors.append(f"datasets[{i}]: 'huggingface' type requires 'name'")
            if ds.type == "s3" and not ds.key:
                errors.append(f"datasets[{i}]: 's3' type requires 'key'")
        return errors

    @staticmethod
    def from_dict(d: dict) -> RunConfig:
        datasets = [DatasetSource.from_dict(ds) for ds in d.get("datasets", [])]
        return RunConfig(
            name=d.get("name", "default"),
            datasets=datasets,
            dictionary=d.get("dictionary"),
            vocab=VocabConfig.from_dict(d.get("vocab")),
            model=ModelConfig.from_dict(d.get("model")),
            training=TrainingConfig.from_dict(d.get("training")),
            pairs=PairsConfig.from_dict(d.get("pairs")),
            output_dir=d.get("output_dir", "data"),
            steps=d.get("steps", list(DEFAULT_STEPS)),
            s3=S3Config.from_dict(d.get("s3")),
        )


# ---------------------------------------------------------------------------
# YAML loading with env-var interpolation
# ---------------------------------------------------------------------------

_ENV_RE = re.compile(r"\$\{(\w+)\}")


def _interpolate_env(value: str) -> str:
    """Replace ``${VAR_NAME}`` placeholders with environment variable values.

    Unresolved variables (not set in the environment) are left as-is so the
    runner can decide whether their absence is fatal.
    """

    def _replace(match: re.Match) -> str:
        var = match.group(1)
        return os.environ.get(var, match.group(0))

    return _ENV_RE.sub(_replace, value)


def _interpolate(obj: Any) -> Any:
    """Recursively interpolate environment variables in a parsed YAML tree."""
    if isinstance(obj, str):
        return _interpolate_env(obj)
    if isinstance(obj, dict):
        return {k: _interpolate(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate(item) for item in obj]
    return obj


def load_config(path: str | Path) -> RunConfig:
    """Load a training run configuration from a YAML file.

    Environment variables referenced as ``${VAR_NAME}`` in string values
    are expanded at load time.  The loaded config is validated; a
    ``ValueError`` is raised if there are structural problems.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for config loading: pip install pyyaml"
        )

    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    raw = _interpolate(raw)
    cfg = RunConfig.from_dict(raw)

    errors = cfg.validate()
    if errors:
        msg = f"Config validation errors in {path}:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        raise ValueError(msg)

    return cfg
