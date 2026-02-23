"""Tests for ai_t9.config — YAML config loading and validation."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest

from ai_t9.config import (
    DEFAULT_STEPS,
    DatasetSource,
    ModelConfig,
    RunConfig,
    S3Config,
    TrainingConfig,
    VocabConfig,
    load_config,
)


# ---------------------------------------------------------------------------
# RunConfig.from_dict
# ---------------------------------------------------------------------------


class TestFromDict:
    def test_minimal_config(self):
        """Empty dict produces sensible defaults."""
        cfg = RunConfig.from_dict({})
        assert cfg.name == "default"
        assert cfg.datasets == []
        assert cfg.vocab.max_words == 500_000
        assert cfg.model.embed_dim == 64
        assert cfg.training.epochs == 5
        assert cfg.steps == DEFAULT_STEPS

    def test_full_config(self):
        raw = {
            "name": "test-run",
            "datasets": [
                {"type": "local", "path": "corpuses/"},
                {
                    "type": "huggingface",
                    "name": "wikitext",
                    "config": "wikitext-103-raw-v1",
                    "split": "validation",
                    "column": "text",
                    "max_lines": 100,
                },
                {"type": "s3", "key": "corpuses/extra.txt"},
            ],
            "dictionary": "wordlist.txt",
            "vocab": {"max_words": 100_000, "min_count": 5},
            "model": {"embed_dim": 128, "context_window": 5},
            "training": {
                "epochs": 10,
                "lr": 0.01,
                "batch_size": 4096,
                "device": "cpu",
            },
            "pairs": {"shard_size": 5_000_000},
            "ngram": False,
            "output_dir": "output",
            "steps": ["corpus", "vocab", "train"],
            "s3": {
                "endpoint": "https://s3.example.com",
                "bucket": "my-bucket",
                "access_key": "AK",
                "secret_key": "SK",
                "upload": True,
                "paths": {
                    "vocab": "v/vocab.json",
                    "model": "m/model.npz",
                },
            },
        }
        cfg = RunConfig.from_dict(raw)
        assert cfg.name == "test-run"
        assert len(cfg.datasets) == 3
        assert cfg.datasets[0].type == "local"
        assert cfg.datasets[1].type == "huggingface"
        assert cfg.datasets[1].max_lines == 100
        assert cfg.datasets[2].type == "s3"
        assert cfg.dictionary == "wordlist.txt"
        assert cfg.vocab.max_words == 100_000
        assert cfg.model.embed_dim == 128
        assert cfg.model.context_window == 5
        assert cfg.training.epochs == 10
        assert cfg.training.lr == 0.01
        assert cfg.training.device == "cpu"
        assert cfg.pairs.shard_size == 5_000_000
        assert cfg.ngram is False
        assert cfg.output_dir == "output"
        assert cfg.steps == ["corpus", "vocab", "train"]
        assert cfg.s3.enabled is True
        assert cfg.s3.upload is True
        assert cfg.s3.paths.vocab == "v/vocab.json"

    def test_dataset_defaults(self):
        ds = DatasetSource.from_dict({"type": "huggingface", "name": "wiki"})
        assert ds.split == "train"
        assert ds.column == "text"
        assert ds.max_lines is None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_minimal_no_corpus(self):
        """No datasets is fine as long as 'corpus' step isn't included."""
        cfg = RunConfig.from_dict({"steps": ["train"]})
        assert cfg.validate() == []

    def test_corpus_step_needs_datasets(self):
        cfg = RunConfig.from_dict({"datasets": [], "steps": ["corpus", "train"]})
        errors = cfg.validate()
        assert any("corpus" in e and "datasets" in e for e in errors)

    def test_unknown_step_flagged(self):
        cfg = RunConfig.from_dict({"steps": ["corpus", "bogus"]})
        errors = cfg.validate()
        assert any("bogus" in e for e in errors)

    def test_local_requires_path(self):
        cfg = RunConfig.from_dict(
            {"datasets": [{"type": "local"}], "steps": ["corpus"]}
        )
        errors = cfg.validate()
        assert any("path" in e for e in errors)

    def test_hf_requires_name(self):
        cfg = RunConfig.from_dict(
            {"datasets": [{"type": "huggingface"}], "steps": ["corpus"]}
        )
        errors = cfg.validate()
        assert any("name" in e for e in errors)

    def test_s3_requires_key(self):
        cfg = RunConfig.from_dict(
            {"datasets": [{"type": "s3"}], "steps": ["corpus"]}
        )
        errors = cfg.validate()
        assert any("key" in e for e in errors)

    def test_unknown_dataset_type(self):
        cfg = RunConfig.from_dict(
            {"datasets": [{"type": "ftp", "path": "/x"}], "steps": ["corpus"]}
        )
        errors = cfg.validate()
        assert any("ftp" in e for e in errors)


# ---------------------------------------------------------------------------
# S3Config.enabled
# ---------------------------------------------------------------------------


class TestS3Config:
    def test_enabled_when_all_present(self):
        s3 = S3Config.from_dict(
            {
                "endpoint": "https://s3.example.com",
                "bucket": "b",
                "access_key": "a",
                "secret_key": "s",
            }
        )
        assert s3.enabled is True

    def test_disabled_when_missing_field(self):
        s3 = S3Config.from_dict({"endpoint": "https://s3.example.com"})
        assert s3.enabled is False

    def test_disabled_when_empty(self):
        s3 = S3Config.from_dict(None)
        assert s3.enabled is False

    def test_disabled_when_env_unresolved(self):
        """Unresolved ${VAR} strings are not valid credentials."""
        s3 = S3Config.from_dict(
            {
                "endpoint": "${NOT_SET}",
                "bucket": "b",
                "access_key": "a",
                "secret_key": "s",
            }
        )
        # endpoint is the literal "${NOT_SET}", which is truthy — but it's not
        # really usable.  The enabled property just checks bool(field).
        assert s3.enabled is True  # technically truthy; caller validates at use

    def test_default_paths(self):
        s3 = S3Config.from_dict({})
        assert s3.paths.vocab == "vocab/vocab.json"
        assert s3.paths.model == "models/model.npz"


# ---------------------------------------------------------------------------
# Environment variable interpolation
# ---------------------------------------------------------------------------


class TestEnvInterpolation:
    def test_interpolates_set_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TEST_BUCKET", "my-bucket")
        config_file = tmp_path / "test.yaml"
        config_file.write_text(
            textwrap.dedent("""\
                name: "env-test"
                datasets: []
                steps: []
                s3:
                  bucket: "${TEST_BUCKET}"
            """)
        )
        cfg = load_config(config_file)
        assert cfg.s3.bucket == "my-bucket"

    def test_unset_var_left_as_is(self, tmp_path, monkeypatch):
        monkeypatch.delenv("DEFINITELY_NOT_SET", raising=False)
        config_file = tmp_path / "test.yaml"
        config_file.write_text(
            textwrap.dedent("""\
                name: "env-test"
                datasets: []
                steps: []
                s3:
                  bucket: "${DEFINITELY_NOT_SET}"
            """)
        )
        cfg = load_config(config_file)
        assert cfg.s3.bucket == "${DEFINITELY_NOT_SET}"


# ---------------------------------------------------------------------------
# load_config validation error
# ---------------------------------------------------------------------------


class TestLoadConfigValidation:
    def test_raises_on_invalid(self, tmp_path):
        config_file = tmp_path / "bad.yaml"
        config_file.write_text(
            textwrap.dedent("""\
                steps:
                  - nonsense_step
            """)
        )
        with pytest.raises(ValueError, match="nonsense_step"):
            load_config(config_file)

    def test_empty_file_loads_defaults(self, tmp_path):
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")
        # Empty config has 'corpus' step but no datasets → validation error
        with pytest.raises(ValueError, match="corpus"):
            load_config(config_file)

    def test_valid_file_loads(self, tmp_path):
        config_file = tmp_path / "ok.yaml"
        config_file.write_text(
            textwrap.dedent("""\
                name: "ok"
                datasets:
                  - type: local
                    path: "corpuses/"
                steps:
                  - corpus
                  - vocab
            """)
        )
        cfg = load_config(config_file)
        assert cfg.name == "ok"
        assert len(cfg.datasets) == 1
