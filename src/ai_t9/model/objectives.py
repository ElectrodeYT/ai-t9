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

    When ``t9_groups`` is supplied, a fraction of negatives are drawn from the
    T9 ambiguity group of the positive target (words that share the same digit
    sequence).  This directly trains the model on the discrimination task it
    performs at inference time: ranking T9-ambiguous siblings by context.

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
        pos_ids: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """Compute scalar loss from context and positive target embeddings.

        Args:
            ctx_vecs:  (B, dim) L2-normalised context vectors.
            pos_vecs:  (B, dim) L2-normalised positive target vectors.
            embed_fn:  Optional callback ``word_ids → (n, dim)`` that embeds
                       arbitrary word IDs through the target embedding table.
                       Required by objectives that sample negatives (e.g. SGNS).
            pos_ids:   (B,) int tensor of positive target word IDs.
                       Required by objectives that use T9 hard negatives.
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

    When ``t9_groups`` is provided, ``hard_neg_frac`` of the k negatives per
    positive are drawn from the T9 ambiguity group of the target word (i.e.
    other words that map to the same digit sequence).  Words with no T9
    siblings fall back to random negatives for those slots.  This directly
    optimises the model for the T9 ranking task rather than generic
    co-occurrence.

    Cost per step: O(B × k × d) — **linear** in batch size.

    Args:
        counts:        Per-word corpus counts (list or tensor, length = vocab_size).
        k:             Number of negative samples per positive.
        power:         Exponent for the frequency-based sampling distribution.
                       0.75 is the Word2Vec default.
        t9_groups:     Mapping ``word_id → [sibling_word_ids]`` where siblings
                       are words sharing the same T9 digit sequence.  Built by
                       the trainer from the vocabulary.
        hard_neg_frac: Fraction of negatives to draw from T9 siblings when
                       ``t9_groups`` is given.  0.5 means half random, half
                       T9-hard.
    """

    def __init__(
        self,
        counts: "list[int] | torch.Tensor",
        k: int = 15,
        power: float = 0.75,
        t9_groups: "dict[int, list[int]] | None" = None,
        hard_neg_frac: float = 0.5,
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

        # T9 hard negative setup
        self._k_hard = max(1, int(k * hard_neg_frac)) if t9_groups else 0
        self._k_rand = k - self._k_hard
        self._sibling_table: "torch.Tensor | None" = None
        self._has_siblings: "torch.Tensor | None" = None
        self._max_sib = 0

        if t9_groups and self._k_hard > 0:
            vocab_size = len(counts)
            max_sib = max((len(v) for v in t9_groups.values()), default=1)
            self._max_sib = max_sib

            # Build (vocab_size, max_sib) sibling table.
            # Rows for words with no siblings stay all-zero (UNK ID) and are
            # replaced at training time by random fallback negatives.
            table = torch.zeros(vocab_size, max_sib, dtype=torch.long)
            has_sib = torch.zeros(vocab_size, dtype=torch.bool)
            for wid, siblings in t9_groups.items():
                if 0 <= wid < vocab_size and siblings:
                    for j in range(max_sib):
                        table[wid, j] = siblings[j % len(siblings)]
                    has_sib[wid] = True

            self._sibling_table = table
            self._has_siblings = has_sib

    def setup(self, device: "torch.device") -> None:
        self._neg_weights = self._neg_weights.to(device)
        if self._sibling_table is not None:
            self._sibling_table = self._sibling_table.to(device)
        if self._has_siblings is not None:
            self._has_siblings = self._has_siblings.to(device)

    def compute_loss(self, ctx_vecs, pos_vecs, embed_fn=None, pos_ids=None):
        import torch
        F = torch.nn.functional

        B, dim = ctx_vecs.shape

        # ── Positive term ─────────────────────────────────────────────
        pos_dots = (ctx_vecs * pos_vecs).sum(-1)                    # (B,)
        pos_loss = -F.logsigmoid(pos_dots).mean()

        # ── Negative sampling ─────────────────────────────────────────
        assert embed_fn is not None, "SGNSObjective requires embed_fn"

        if (
            self._sibling_table is not None
            and self._has_siblings is not None
            and pos_ids is not None
            and self._k_hard > 0
        ):
            # Hard negatives: sample from T9 sibling table
            # Random column indices to pick from each word's sibling pool
            col_idx = torch.randint(
                self._max_sib, (B, self._k_hard), device=pos_ids.device
            )
            hard_neg_ids = self._sibling_table[pos_ids].gather(1, col_idx)  # (B, k_hard)

            # Replace hard negs for words that have no siblings with random draws
            no_sib = ~self._has_siblings[pos_ids]                           # (B,)
            if no_sib.any():
                n_fill = int(no_sib.sum().item())
                fallback = torch.multinomial(
                    self._neg_weights, n_fill * self._k_hard, replacement=True,
                ).reshape(n_fill, self._k_hard)
                hard_neg_ids[no_sib] = fallback

            hard_neg_vecs = embed_fn(hard_neg_ids.reshape(-1)).reshape(
                B, self._k_hard, dim
            )                                                               # (B, k_hard, dim)

            # Random negatives for remaining slots
            rand_neg_ids = torch.multinomial(
                self._neg_weights, B * self._k_rand, replacement=True,
            )                                                               # (B*k_rand,)
            rand_neg_vecs = embed_fn(rand_neg_ids).reshape(
                B, self._k_rand, dim
            )                                                               # (B, k_rand, dim)

            neg_vecs = torch.cat([hard_neg_vecs, rand_neg_vecs], dim=1)    # (B, k, dim)
        else:
            # Standard random negatives only
            neg_ids = torch.multinomial(
                self._neg_weights, B * self._k, replacement=True,
            )                                                               # (B*k,)
            neg_vecs = embed_fn(neg_ids).reshape(B, self._k, dim)          # (B, k, dim)

        neg_dots = torch.bmm(                                               # (B, k)
            neg_vecs, ctx_vecs.unsqueeze(2),
        ).squeeze(2)
        neg_loss = -F.logsigmoid(-neg_dots).mean()

        return pos_loss + neg_loss

    @property
    def label(self) -> str:
        if self._k_hard > 0:
            return f"SGNS(k={self._k}, hard={self._k_hard})"
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

    def compute_loss(self, ctx_vecs, pos_vecs, embed_fn=None, pos_ids=None):
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
