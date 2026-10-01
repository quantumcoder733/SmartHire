"""Text preprocessing helpers for resume data."""

import re
import unicodedata


_WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_resume_text(text: str) -> str:
    """Normalize and clean raw resume text while preserving its meaning.

    Unicode text is normalized with NFKC, whitespace sequences are collapsed
    to single spaces, and leading and trailing whitespace is removed. Punctuation,
    stop words, and other meaningful resume characters are preserved.

    Args:
        text: Raw resume text to clean.

    Returns:
        The cleaned resume text.
    """
    normalized_text = unicodedata.normalize("NFKC", text)
    return _WHITESPACE_PATTERN.sub(" ", normalized_text).strip()
