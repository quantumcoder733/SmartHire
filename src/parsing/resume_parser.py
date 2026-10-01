"""Utilities for extracting text from resume files."""

from pathlib import Path
from os import PathLike

from pypdf import PdfReader


_SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def parse_resume(file_path: str | PathLike[str]) -> str:
    """Extract and return all text from a PDF or UTF-8 TXT resume.

    Args:
        file_path: Path to the resume file to parse.

    Returns:
        The extracted resume text as a single string.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
        ValueError: If the file extension is not ``.pdf`` or ``.txt``.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Resume file not found: {path}")

    extension = path.suffix.lower()
    if extension not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported resume file type '{path.suffix or '[no extension]'}'. "
            "Supported types are PDF and TXT."
        )

    if extension == ".txt":
        return path.read_text(encoding="utf-8")

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)
