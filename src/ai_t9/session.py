"""T9Session: stateful dialing session that maintains a rolling context window."""

from __future__ import annotations

from collections import deque

from .predictor import RankedCandidate, T9Predictor


class T9Session:
    """A stateful T9 input session with automatic context tracking.

    Keeps a rolling window of the most recently confirmed words.  Each call to
    ``dial()`` uses the current context to re-rank candidates.  When the user
    selects a word via ``confirm()``, it is appended to the context window.

    Usage::

        session = T9Session(predictor, context_window=5)
        candidates = session.dial("4663")
        # → ["home", "gone", "good", "hood", "hone"]  (frequency-only if no model)
        session.confirm("home")

        candidates = session.dial("2")
        # → ["a", "b", "c"] — now ranked with "home" in context
        session.confirm("and")  # skipped a step, confirm manually
    """

    def __init__(
        self,
        predictor: T9Predictor,
        context_window: int = 5,
    ) -> None:
        self._predictor = predictor
        self._context: deque[str] = deque(maxlen=context_window)
        self._context_window = context_window

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def dial(
        self,
        digit_seq: str,
        top_k: int = 5,
        return_details: bool = False,
    ) -> list[str] | list[RankedCandidate]:
        """Predict words for the given digit sequence using the current context.

        Args:
            digit_seq:      T9 digit sequence, e.g. ``"4663"``
            top_k:          Number of candidates to return
            return_details: Return RankedCandidate objects with score breakdown

        Returns:
            List of predicted words (or RankedCandidate if return_details=True)
        """
        return self._predictor.predict(
            digit_seq,
            context=list(self._context),
            top_k=top_k,
            return_details=return_details,
        )

    def confirm(self, word: str) -> None:
        """Record a confirmed word (user selected it) and add it to context."""
        self._context.append(word.lower())

    def add_context(self, *words: str) -> None:
        """Pre-load context words (e.g. seed from a previous sentence)."""
        for w in words:
            self._context.append(w.lower())

    def reset(self) -> None:
        """Clear the context window (start of a new message)."""
        self._context.clear()

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def context(self) -> list[str]:
        """Current context window (oldest first)."""
        return list(self._context)

    @property
    def predictor(self) -> T9Predictor:
        return self._predictor

    def __repr__(self) -> str:
        return f"T9Session(context={list(self._context)!r})"
