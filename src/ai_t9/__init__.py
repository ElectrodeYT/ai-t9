"""ai_t9 — Advanced predictive T9 text input library.

Quick start::

    from ai_t9 import T9Predictor, T9Session

    # Load a trained predictor from files produced by the training pipeline
    predictor = T9Predictor.from_files(
        "data/vocab.json", "data/dict.json",
        model_path="data/model.npz",
    )

    # One-shot prediction
    print(predictor.predict("4663"))  # e.g. ["home", "good", "gone", ...]

    # Stateful session with rolling context
    session = T9Session(predictor)
    session.add_context("i", "am", "going")
    print(session.dial("4663"))
    session.confirm("home")
    print(session.dial("269"))  # "any" / "bow" / "cox" …

Training a custom model::

    # Run the full pipeline (corpus → vocab → pairs → train):
    #   ai-t9-run configs/default.yaml
    #
    # Then load the results:
    predictor = T9Predictor.from_files(
        "data/vocab.json", "data/dict.json",
        model_path="data/model.npz",
    )
"""

from .predictor import RankedCandidate, T9Predictor
from .session import T9Session
from .t9_map import T9_MAP, word_to_digits, is_valid_digit_sequence
from .model.dual_encoder import DualEncoder

__all__ = [
    "T9Predictor",
    "T9Session",
    "RankedCandidate",
    "T9_MAP",
    "word_to_digits",
    "is_valid_digit_sequence",
    "DualEncoder",
]
