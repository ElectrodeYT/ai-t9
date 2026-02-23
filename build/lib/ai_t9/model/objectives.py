"""Pluggable training objectives for the DualEncoder.

Each objective takes L2-normalised ``(ctx_vecs, pos_vecs)`` embedding pairs
from the model's forward pass and computes a scalar loss.  Objectives that
need additional word embeddings (e.g. SGNS negatives) receive an ``embed_fn``
callback that maps word IDs to normalised target embeddings.

Adding a new objective
----------------------
1. Subclass :class:`TrainingObjective`.
2. Implement :meth:`compute_loss`.
3. (Optional) override :meth:`setup`, :meth:`max_batch_size`.
4. Register the class in the :data:`OBJECTIVES` dict at the bottom.
5. It is then available via ``--objective <name>`` on the CLI.

Built-in objectives
-------------------
``sgns``
    Skip-Gram Negative Sampling (Word2Vec / fastText style).
    Cost: O(B × k) per step — linear in batch size.

``clip``
    CLIP-style in-batch negatives with symmetric cross-entropy.
    Cost: O(B²) per step — quadratic in batch size.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class TrainingObjective(ABC):
    """Base class for pluggable training objectives."""

    @abstractmethod
    def compute_loss(
        self,
        ctx_vecs: "torch.Tensor",
        pos_vecs: "torch.Tensor",
        embed_fn: "Callable[[torch.Tensor], torch.Tensor] | None" = None,
    ) -> "torch.Tensor":
        """Compute scalar loss from context and positive target embeddings.

        Args:
            ctx_vecs:  (B, dim) L2-normalised context vectors.
            pos_vecs:  (B, dim) L2-normalised positive target vectors.
            embed_fn:  Optional callback ``word_ids → (n, dim)`` that embeds
                       arbitrary word IDs through the target embedding table.
                       Required by objectives that sample negatives (e.g. SGNS).
        """
        ...

    def setup(self, device: "torch.device") -> None:
        """Move internal state (e.g. sampling weights) to *device*.

        Called once before the training loop starts.
        """

    def max_batch_size(self, free_vram_bytes: int) -> int | None:
        """Return the maximum batch size this objective supports.

        Given the free VRAM in bytes, return a batch size cap, or ``None``
        if the objective imposes no constraint (the default).
        """
        return None

    @property
    def label(self) -> str:
        """Short description for verbose training output."""
        return type(self).__name__


# ---------------------------------------------------------------------------
# SGNS — Skip-Gram Negative Sampling
# ---------------------------------------------------------------------------

class SGNSObjective(TrainingObjective):
    """Skip-Gram Negative Sampling (Word2Vec / fastText style).

    For each (context, target) pair in the batch::

        L = −log σ(c · w⁺) − (1/k) Σᵢ log σ(−c · wᵢ⁻)

    Negatives are drawn from a frequency-weighted unigram distribution
    (count^power), with UNK (id 0) excluded.

    Because both context and target vectors are L2-normalised (matching the
    inference path), dot products lie in [-1, 1].  The sigmoid has good
    gradient signal in this range — no temperature scaling is needed.

    Cost per step: O(B × k × d) — **linear** in batch size.

    Args:
        counts:  Per-word corpus counts (list or tensor, length = vocab_size).
        k:       Number of negative samples per positive.
        power:   Exponent for the frequency-based sampling distribution.
                 0.75 is the Word2Vec default.
    """

    def __init__(
        self,
        counts: "list[int] | torch.Tensor",
        k: int = 15,
        power: float = 0.75,
    ) -> None:
        import torch

        if not isinstance(counts, torch.Tensor):
            counts = torch.tensor(counts, dtype=torch.float32)
        else:
            counts = counts.float()

        weights = counts.pow(power)
        weights[0] = 0.0  # exclude UNK
        total = weights.sum()
        self._neg_weights = weights / total if total > 0 else weights
        self._k = k

    def setup(self, device: "torch.device") -> None:
        self._neg_weights = self._neg_weights.to(device)

    def compute_loss(self, ctx_vecs, pos_vecs, embed_fn=None):
        import torch
        F = torch.nn.functional

        B, dim = ctx_vecs.shape

        # ── Positive term ─────────────────────────────────────────────
        pos_dots = (ctx_vecs * pos_vecs).sum(-1)                    # (B,)
        pos_loss = -F.logsigmoid(pos_dots).mean()

        # ── Negative sampling ─────────────────────────────────────────
        assert embed_fn is not None, "SGNSObjective requires embed_fn"
        neg_ids = torch.multinomial(
            self._neg_weights, B * self._k, replacement=True,
        )                                                           # (B*k,)
        neg_vecs = embed_fn(neg_ids).reshape(B, self._k, dim)      # (B, k, dim)
        neg_dots = torch.bmm(                                       # (B, k)
            neg_vecs, ctx_vecs.unsqueeze(2),
        ).squeeze(2)
        neg_loss = -F.logsigmoid(-neg_dots).mean()

        return pos_loss + neg_loss

    @property
    def label(self) -> str:
        return f"SGNS(k={self._k})"


# ---------------------------------------------------------------------------
# CLIP — In-Batch Negative Sampling
# ---------------------------------------------------------------------------

class CLIPObjective(TrainingObjective):
    """CLIP-style in-batch negative sampling with symmetric cross-entropy.

    Builds a (B, B) similarity matrix and applies cross-entropy in both
    directions: each context identifies its target *and* each target
    identifies its context.

    Cost per step: O(B² × d) — **quadratic** in batch size.

    Args:
        temperature: Softmax temperature for the similarity logits.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        self._temperature = temperature

    def compute_loss(self, ctx_vecs, pos_vecs, embed_fn=None):
        import torch

        B = ctx_vecs.size(0)
        labels = torch.arange(B, device=ctx_vecs.device)
        logits = ctx_vecs @ pos_vecs.T / self._temperature
        ce = torch.nn.functional.cross_entropy
        return (ce(logits, labels) + ce(logits.T, labels)) / 2

    def max_batch_size(self, free_vram_bytes: int) -> int | None:
        """Cap batch size so the (B, B) logits matrix fits in 25% of VRAM."""
        # 8 bytes per element: BF16 logits(2) + BF16 grad(2) + FP32 softmax(4)
        return int(math.sqrt(free_vram_bytes * 0.25 / 8))

    @property
    def label(self) -> str:
        return f"CLIP(τ={self._temperature})"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

OBJECTIVES: dict[str, type[TrainingObjective]] = {
    "sgns": SGNSObjective,
    "clip": CLIPObjective,
}
