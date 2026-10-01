"""Utilities for loading SmartHire job data."""

from os import PathLike
from pathlib import Path

import pandas as pd


_REQUIRED_COLUMNS = {
    "title",
    "company",
    "location",
    "skills",
    "experience",
    "description",
    "source",
    "text",
}
_REQUIRED_RESUME_COLUMNS = {"Category", "text"}
_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
_DEFAULT_JOBS_PATH = (
    _DATA_DIR / "jobs_clean.csv"
)
_DEFAULT_RESUMES_PATH = _DATA_DIR / "resumes_clean.csv"


def load_jobs(path: str | PathLike[str] | None = None) -> pd.DataFrame:
    """Load and validate the SmartHire jobs CSV.

    Args:
        path: Optional path to a jobs CSV. When omitted, the project's
            ``data/processed/jobs_clean.csv`` file is loaded.

    Returns:
        The jobs CSV contents as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the requested CSV does not exist.
        ValueError: If required columns are missing from the CSV.
    """
    jobs_path = _DEFAULT_JOBS_PATH if path is None else Path(path)
    if not jobs_path.exists():
        raise FileNotFoundError(f"Jobs CSV not found: {jobs_path}")

    jobs = pd.read_csv(jobs_path)
    missing_columns = _REQUIRED_COLUMNS - set(jobs.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Jobs CSV is missing required columns: {missing}")

    return jobs


def load_resumes(path: str | PathLike[str] | None = None) -> pd.DataFrame:
    """Load and validate the prepared SmartHire resume CSV."""
    resumes_path = _DEFAULT_RESUMES_PATH if path is None else Path(path)
    if not resumes_path.exists():
        raise FileNotFoundError(f"Resumes CSV not found: {resumes_path}")

    resumes = pd.read_csv(resumes_path)
    missing_columns = _REQUIRED_RESUME_COLUMNS - set(resumes.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Resumes CSV is missing required columns: {missing}")

    return resumes
