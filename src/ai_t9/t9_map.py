"""T9 digit-to-letter mapping and candidate generation."""

from itertools import product

# Standard T9 keypad layout
T9_MAP: dict[str, str] = {
    "2": "abc",
    "3": "def",
    "4": "ghi",
    "5": "jkl",
    "6": "mno",
    "7": "pqrs",
    "8": "tuv",
    "9": "wxyz",
}

# Reverse: letter → digit
LETTER_TO_DIGIT: dict[str, str] = {
    letter: digit for digit, letters in T9_MAP.items() for letter in letters
}

VALID_DIGITS = set(T9_MAP.keys())


def word_to_digits(word: str) -> str | None:
    """Convert a lowercase word to its T9 digit sequence.

    Returns None if the word contains characters not on the T9 keypad
    (digits, punctuation, uppercase, etc.).
    """
    digits = []
    for ch in word.lower():
        d = LETTER_TO_DIGIT.get(ch)
        if d is None:
            return None
        digits.append(d)
    return "".join(digits) if digits else None


def candidates_from_digits(seq: str) -> list[str]:
    """Generate all possible letter strings for a digit sequence.

    This is an exhaustive combinatorial expansion — used for testing and
    dictionary construction, not for inference (the dictionary index is used
    directly there).

    Example:
        candidates_from_digits("43") → ["gd", "ge", "gf", "hd", "he", "hf", "id", "ie", "if"]
    """
    if not seq:
        return []
    if any(d not in VALID_DIGITS for d in seq):
        raise ValueError(f"Digit sequence contains non-T9 characters: {seq!r}")
    letter_groups = [T9_MAP[d] for d in seq]
    return ["".join(combo) for combo in product(*letter_groups)]


def is_valid_digit_sequence(seq: str) -> bool:
    """Return True if seq is a non-empty string of valid T9 digits (2-9)."""
    return bool(seq) and all(d in VALID_DIGITS for d in seq)
