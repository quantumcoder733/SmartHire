"""High-level resume analysis using persisted SmartHire model artifacts."""

from functools import lru_cache
from os import PathLike
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    CLASSIFIER_PATH,
    JOB_CLUSTERS_PATH,
    JOB_KMEANS_PATH,
    JOB_MATRIX_PATH,
    JOB_METADATA_PATH,
    JOB_VECTORIZER_PATH,
    MODELS_DIR,
)
from src.data.load_data import load_jobs
from src.data.preprocess import clean_resume_text
from src.models.clustering import get_top_cluster_skills
from src.models.recommender import recommend_jobs_from_index
from src.models.skill_gap import extract_technical_skills
from src.parsing.resume_parser import parse_resume


_REQUIRED_METADATA_COLUMNS = {
    "title",
    "company",
    "location",
    "skills",
    "description",
    "experience",
    "source",
}


@lru_cache(maxsize=8)
def _load_joblib_artifact(path: str) -> Any:
    return joblib.load(path)


@lru_cache(maxsize=4)
def _load_sparse_matrix(path: str) -> spmatrix:
    return load_npz(path).tocsr()


@lru_cache(maxsize=4)
def _load_metadata(path: str) -> pd.DataFrame:
    metadata = pd.read_csv(path)
    missing_columns = _REQUIRED_METADATA_COLUMNS - set(metadata.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Job metadata is missing required columns: {missing}")
    return metadata


@lru_cache(maxsize=4)
def _load_cluster_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _require_artifact(path: Path, build_command: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Required runtime artifact is missing: {path}. SmartHire intentionally "
            "does not include generated job-index artifacts in a normal Git checkout. "
            f"Prepare the required source data and build it with `{build_command}`. "
            "See README.md, Large Runtime Artifacts, for the complete artifact list "
            "and deployment setup."
        )
    return path


def load_classifier(model_path: str | PathLike[str] = CLASSIFIER_PATH) -> Any:
    """Load the persisted complete resume-classifier pipeline."""
    path = _require_artifact(Path(model_path), "python -m src.train_classifier")
    return _load_joblib_artifact(str(path.resolve()))


def load_default_jobs() -> pd.DataFrame:
    """Load the prepared default SmartHire jobs DataFrame."""
    return load_jobs()


def load_job_vectorizer(
    vectorizer_path: str | PathLike[str] = JOB_VECTORIZER_PATH,
) -> TfidfVectorizer:
    """Load the saved job-corpus TF-IDF vectorizer."""
    path = _require_artifact(
        Path(vectorizer_path), "python -m src.build_job_index"
    )
    return _load_joblib_artifact(str(path.resolve()))


def load_job_tfidf_matrix(
    matrix_path: str | PathLike[str] = JOB_MATRIX_PATH,
) -> spmatrix:
    """Load the saved sparse job TF-IDF matrix."""
    path = _require_artifact(Path(matrix_path), "python -m src.build_job_index")
    return _load_sparse_matrix(str(path.resolve()))


def load_job_metadata(
    metadata_path: str | PathLike[str] = JOB_METADATA_PATH,
) -> pd.DataFrame:
    """Load saved recommendation metadata in TF-IDF row order."""
    path = _require_artifact(
        Path(metadata_path), "python -m src.build_job_index"
    )
    return _load_metadata(str(path.resolve()))


def load_job_kmeans(
    model_path: str | PathLike[str] = JOB_KMEANS_PATH,
) -> Any:
    """Load the persisted K-Means model."""
    path = _require_artifact(
        Path(model_path), "python -m src.build_job_clusters"
    )
    return _load_joblib_artifact(str(path.resolve()))


def load_job_cluster_assignments(
    clusters_path: str | PathLike[str] = JOB_CLUSTERS_PATH,
    expected_rows: int | None = None,
) -> np.ndarray:
    """Load cluster IDs and verify saved row alignment when requested."""
    path = _require_artifact(
        Path(clusters_path), "python -m src.build_job_clusters"
    )
    assignments = _load_cluster_table(str(path.resolve()))
    if not {"row_index", "cluster"}.issubset(assignments.columns):
        raise ValueError("Cluster assignments must contain row_index and cluster.")
    row_indices = assignments["row_index"].to_numpy()
    if not np.array_equal(row_indices, np.arange(len(assignments))):
        raise ValueError("Cluster assignments are not in original job row order.")
    if expected_rows is not None and len(assignments) != expected_rows:
        raise ValueError(
            "Cluster assignments, metadata, and TF-IDF matrix must be row-aligned."
        )
    return assignments["cluster"].to_numpy(dtype=int)


def _artifact_path(artifact_dir: str | PathLike[str] | None, artifact: Path) -> Path:
    if artifact_dir is None:
        return artifact
    return Path(artifact_dir) / artifact.name


def analyze_resume_unsupervised(
    resume_path: str | PathLike[str],
    jobs_df: pd.DataFrame | None = None,
    top_n: int = 5,
    n_clusters: int | None = None,
    *,
    artifact_dir: str | PathLike[str] | None = None,
    classifier_model: Any | None = None,
) -> dict[str, Any]:
    """Analyze a resume with cached, persisted classifier and job artifacts.

    ``jobs_df`` and ``n_clusters`` remain accepted for call compatibility;
    saved metadata and cluster assignments are authoritative at runtime.
    """
    del jobs_df, n_clusters
    if top_n <= 0:
        raise ValueError("top_n must be greater than 0.")

    model_dir = MODELS_DIR if artifact_dir is None else Path(artifact_dir)
    raw_text = parse_resume(resume_path)
    cleaned_text = clean_resume_text(raw_text)
    classifier = (
        load_classifier(_artifact_path(model_dir, CLASSIFIER_PATH))
        if classifier_model is None
        else classifier_model
    )
    predicted_category = classifier.predict([cleaned_text])[0]

    vectorizer = load_job_vectorizer(
        _artifact_path(model_dir, JOB_VECTORIZER_PATH)
    )
    job_matrix = load_job_tfidf_matrix(_artifact_path(model_dir, JOB_MATRIX_PATH))
    metadata = load_job_metadata(_artifact_path(model_dir, JOB_METADATA_PATH))
    cluster_assignments = load_job_cluster_assignments(
        _artifact_path(model_dir, JOB_CLUSTERS_PATH),
        expected_rows=len(metadata),
    )
    if job_matrix.shape[0] != len(metadata):
        raise ValueError(
            "Job metadata and TF-IDF matrix must contain the same number of rows."
        )

    recommendations = recommend_jobs_from_index(
        cleaned_text,
        metadata,
        vectorizer,
        job_matrix,
        cluster_assignments,
        top_n=top_n,
        candidate_skills=extract_technical_skills(cleaned_text),
    )

    clusters: list[dict[str, Any]] = []
    for cluster_id in np.unique(cluster_assignments):
        cluster_skills = metadata.loc[
            cluster_assignments == cluster_id,
            ["skills"],
        ].copy()
        cluster_skills["cluster"] = cluster_id
        top_skills = get_top_cluster_skills(
            cluster_skills,
            cluster_id=int(cluster_id),
            top_n=5,
        )
        clusters.append(
            {
                "cluster_id": int(cluster_id),
                "job_count": int(np.count_nonzero(cluster_assignments == cluster_id)),
                "top_skills": [
                    {"skill": skill, "job_count": count}
                    for skill, count in top_skills
                ],
            }
        )

    return {
        "resume": {
            "filename": Path(resume_path).name,
            "text_length": len(cleaned_text),
        },
        "predicted_category": str(predicted_category),
        "recommendations": recommendations,
        "clusters": clusters,
    }


def analyze_resume(
    resume_path: str | PathLike[str],
    classifier_model: Any | None = None,
    jobs_df: pd.DataFrame | None = None,
    top_n: int = 5,
    *,
    artifact_dir: str | PathLike[str] | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper returning the supervised recommendation view."""
    result = analyze_resume_unsupervised(
        resume_path,
        jobs_df,
        top_n=top_n,
        artifact_dir=artifact_dir,
        classifier_model=classifier_model,
    )
    return {
        "predicted_category": result["predicted_category"],
        "recommendations": result["recommendations"],
    }
