"""Build and persist the reusable TF-IDF index for prepared jobs."""

import argparse
from pathlib import Path
import tempfile

import joblib
import pandas as pd
from scipy.sparse import save_npz

from src.config import (
    JOB_MATRIX_PATH,
    JOB_METADATA_PATH,
    JOB_VECTORIZER_PATH,
    JOBS_DATA_PATH,
    MODELS_DIR,
)
from src.models.recommender import create_job_vectorizer, prepare_job_text


JOB_METADATA_COLUMNS = (
    "title",
    "company",
    "location",
    "skills",
    "description",
    "experience",
    "source",
)
REQUIRED_JOB_COLUMNS = set(JOB_METADATA_COLUMNS) | {"text"}


def _stream_job_texts(
    csv_path: Path,
    metadata_temp_path: Path,
    chunksize: int,
):
    """Yield text in CSV order while writing metadata in the same order."""
    write_header = True
    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        chunk_metadata = chunk.loc[:, JOB_METADATA_COLUMNS]
        chunk_metadata.to_csv(
            metadata_temp_path,
            mode="w" if write_header else "a",
            header=write_header,
            index=False,
        )
        write_header = False
        yield from prepare_job_text(chunk)


def build_job_index_from_csv(
    csv_path: str | Path,
    vectorizer_path: str | Path,
    matrix_path: str | Path,
    metadata_path: str | Path,
    chunksize: int = 10_000,
) -> tuple[int, int]:
    """Stream prepared jobs once and persist aligned TF-IDF artifacts."""
    source_path = Path(csv_path)
    if chunksize <= 0:
        raise ValueError("chunksize must be greater than 0.")
    if not source_path.exists():
        raise FileNotFoundError(f"Jobs CSV not found: {source_path}")

    source_columns = set(pd.read_csv(source_path, nrows=0).columns)
    missing_columns = REQUIRED_JOB_COLUMNS - source_columns
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Jobs CSV is missing required columns: {missing}")

    vectorizer_output = Path(vectorizer_path)
    matrix_output = Path(matrix_path)
    metadata_output = Path(metadata_path)
    for output_path in (vectorizer_output, matrix_output, metadata_output):
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        prefix="job_metadata_",
        suffix=".csv.tmp",
        dir=metadata_output.parent,
        delete=False,
    ) as temporary_file:
        metadata_temp_path = Path(temporary_file.name)

    try:
        vectorizer = create_job_vectorizer()
        job_matrix = vectorizer.fit_transform(
            _stream_job_texts(source_path, metadata_temp_path, chunksize)
        ).tocsr()
        joblib.dump(vectorizer, vectorizer_output)
        save_npz(matrix_output, job_matrix, compressed=True)
        metadata_temp_path.replace(metadata_output)
    except Exception:
        metadata_temp_path.unlink(missing_ok=True)
        raise

    return job_matrix.shape[0], len(vectorizer.vocabulary_)


def main() -> None:
    """Build the prepared job index using bounded CSV chunks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunksize", type=int, default=10_000)
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    row_count, vocabulary_size = build_job_index_from_csv(
        JOBS_DATA_PATH,
        JOB_VECTORIZER_PATH,
        JOB_MATRIX_PATH,
        JOB_METADATA_PATH,
        chunksize=args.chunksize,
    )
    print(f"Indexed job rows: {row_count}")
    print(f"TF-IDF vocabulary size: {vocabulary_size}")
    print(f"Saved vectorizer: {JOB_VECTORIZER_PATH}")
    print(f"Saved sparse matrix: {JOB_MATRIX_PATH}")
    print(f"Saved aligned metadata: {JOB_METADATA_PATH}")


if __name__ == "__main__":
    main()