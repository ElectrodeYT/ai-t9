"""Model sub-package: vocabulary, dual-encoder inference, and optional training."""

from .dual_encoder import DualEncoder
from .vocab import Vocabulary

__all__ = ["DualEncoder", "Vocabulary"]
