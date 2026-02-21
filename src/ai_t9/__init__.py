"""ai_t9 — Advanced predictive T9 text input library.

Quick start::

    from ai_t9 import T9Predictor, T9Session

    # Build from NLTK data (downloads ~15 MB on first use)
    predictor = T9Predictor.build_default()

    # One-shot prediction
    print(predictor.predict("4663"))  # e.g. ["home", "good", "gone", ...]

    # Stateful session with rolling context
    session = T9Session(predictor)
    session.add_context("i", "am", "going")
    print(session.dial("4663"))
    session.confirm("home")
    print(session.dial("269"))  # "any" / "bow" / "cox" …

Training a custom model (requires torch)::

    from ai_t9.model.trainer import DualEncoderTrainer
    from ai_t9.model.vocab import Vocabulary

    vocab = Vocabulary.build_from_nltk()
    trainer = DualEncoderTrainer(vocab, embed_dim=64)
    trainer.train_from_nltk(epochs=3)
    trainer.save_numpy("model.npz")

    # Then load it:
    predictor = T9Predictor.from_files(
        "vocab.json", "dict.json", model_path="model.npz"
    )
"""

from .predictor import RankedCandidate, T9Predictor
from .session import T9Session
from .t9_map import T9_MAP, word_to_digits, is_valid_digit_sequence

__all__ = [
    "T9Predictor",
    "T9Session",
    "RankedCandidate",
    "T9_MAP",
    "word_to_digits",
    "is_valid_digit_sequence",
]
