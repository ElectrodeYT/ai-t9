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

    def undo_confirm(self) -> str | None:
        """Remove and return the last confirmed word from the context window.

        Useful for backspace-over-confirmed-word behaviour in a GUI: call this
        to unwind the most recent ``confirm()`` call, then re-display the
        previous context state.
        """
        if self._context:
            return self._context.pop()
        return None

    def completions(
        self,
        digit_prefix: str,
        top_k: int = 5,
        max_extra_digits: int | None = None,
        w_length: float = 0.30,
        min_model_score: float = 0.0,
        return_details: bool = False,
    ) -> list[str] | list[RankedCandidate]:
        """Predict word completions for a partially typed digit sequence.

        Like :meth:`dial` but returns words *longer* than the input, offering
        autocompletion suggestions.  Uses the current context for ranking.

        By default, the number of completions and the search depth scale
        adaptively with the prefix length (more digits typed → more and deeper
        completions, since the candidate set is smaller and more reliable).
        Pass an explicit ``max_extra_digits`` integer to override this.

        Args:
            digit_prefix:     Partial T9 digit sequence, e.g. ``"466"``
            top_k:            Maximum completions to return (adaptive mode may
                              reduce this for short prefixes)
            max_extra_digits: Maximum additional digits beyond the prefix.
                              ``None`` (default) → adaptive scaling.
            w_length:         Weight for the length bonus (higher → prefer
                              shorter completions)
            min_model_score:  Filter completions below this normalised model
                              score (0.0 = no filtering).
            return_details:   Return RankedCandidate objects with score breakdown

        Returns:
            List of predicted words (or RankedCandidate if return_details=True)
        """
        return self._predictor.predict_completions(
            digit_prefix,
            context=list(self._context),
            top_k=top_k,
            max_extra_digits=max_extra_digits,
            w_length=w_length,
            min_model_score=min_model_score,
            return_details=return_details,
        )

    def dial_with_completions(
        self,
        digit_seq: str,
        top_k: int = 5,
        completion_k: int = 3,
        max_extra_digits: int | None = None,
        w_length: float = 0.30,
        min_model_score: float = 0.0,
        return_details: bool = False,
    ) -> tuple[
        list[str] | list[RankedCandidate],
        list[str] | list[RankedCandidate],
    ]:
        """Return both exact matches and completions for the given digits.

        Convenience method combining :meth:`dial` and :meth:`completions`
        in a single call — useful for UIs that show both sections.

        ``max_extra_digits=None`` (default) uses adaptive scaling: the search
        depth and number of completions returned grow with the prefix length.

        Returns:
            ``(exact_matches, completions)`` — two separate lists.
        """
        exact = self.dial(
            digit_seq, top_k=top_k, return_details=return_details,
        )
        comps = self.completions(
            digit_seq,
            top_k=completion_k,
            max_extra_digits=max_extra_digits,
            w_length=w_length,
            min_model_score=min_model_score,
            return_details=return_details,
        )
        return exact, comps

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
