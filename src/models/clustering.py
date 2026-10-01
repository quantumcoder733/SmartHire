"""Reusable TF-IDF and K-Means job clustering utilities."""

from collections import Counter
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse, spmatrix
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from src.models.recommender import (
    create_job_vectorizer,
    prepare_job_text as _prepare_job_text,
)


def prepare_job_text(jobs_df: pd.DataFrame) -> pd.Series:
    """Prepare searchable job text using the recommender's field rules."""
    return _prepare_job_text(jobs_df)


def _validate_job_text(jobs_df: pd.DataFrame) -> pd.Series:
    if jobs_df.empty:
        raise ValueError("jobs_df must contain at least one job.")

    job_text = prepare_job_text(jobs_df)
    if not job_text.str.strip().any():
        raise ValueError("jobs_df must contain usable non-empty job text.")
    return job_text


def cluster_jobs(
    jobs_df: pd.DataFrame,
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 10,
) -> tuple[pd.DataFrame, TfidfVectorizer, KMeans]:
    """Cluster jobs with TF-IDF and K-Means while preserving row order.

    Returns a copy of ``jobs_df`` with a ``cluster`` column, the fitted
    vectorizer, and the fitted K-Means model.
    """
    job_text = _validate_job_text(jobs_df)
    if n_clusters <= 0:
        raise ValueError("n_clusters must be greater than 0.")
    if n_clusters > len(jobs_df):
        raise ValueError("n_clusters cannot exceed the number of jobs.")

    vectorizer = create_job_vectorizer()
    job_matrix = vectorizer.fit_transform(job_text)
    labels, model = cluster_job_matrix(
        job_matrix,
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )

    clustered_jobs = jobs_df.copy()
    clustered_jobs["cluster"] = labels
    return clustered_jobs, vectorizer, model


def cluster_job_matrix(
    job_matrix: spmatrix,
    n_clusters: int,
    random_state: int = 42,
    n_init: int = 10,
) -> tuple[np.ndarray, KMeans]:
    """Fit K-Means on an existing sparse job matrix."""
    if job_matrix.shape[0] == 0:
        raise ValueError("job_matrix must contain at least one job.")
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )
    labels = model.fit_predict(job_matrix)
    return labels, model


def evaluate_k_values(
    jobs_or_matrix: pd.DataFrame | spmatrix,
    k_values: Iterable[int],
    random_state: int = 42,
    n_init: int = 10,
    sample_size: int = 2_000,
) -> pd.DataFrame:
    """Evaluate K-Means on a deterministic representative sample only."""
    if isinstance(jobs_or_matrix, pd.DataFrame):
        job_text = _validate_job_text(jobs_or_matrix)
        job_matrix = create_job_vectorizer().fit_transform(job_text)
    else:
        job_matrix = jobs_or_matrix
    if job_matrix.shape[0] < 3:
        raise ValueError("At least three jobs are required to evaluate K values.")
    if sample_size < 3:
        raise ValueError("sample_size must be at least 3.")

    n_samples = min(sample_size, job_matrix.shape[0])
    sample_indices = np.random.default_rng(random_state).choice(
        job_matrix.shape[0],
        size=n_samples,
        replace=False,
    )
    sample_matrix = job_matrix[sample_indices]
    values = list(k_values)
    if not values:
        raise ValueError("k_values must contain at least one value.")

    results: list[dict[str, float | int]] = []

    for k in values:
        if k <= 1:
            raise ValueError("Each K must be at least 2 for silhouette score.")
        if k >= n_samples:
            raise ValueError(
                "Each K must be less than the evaluation sample size."
            )

        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=n_init,
        )
        labels = model.fit_predict(sample_matrix)
        if np.unique(labels).size < 2 or np.unique(labels).size >= n_samples:
            raise ValueError("K-Means produced invalid labels for silhouette scoring.")
        results.append(
            {
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette_score": float(
                    silhouette_score(sample_matrix, labels)
                ),
            }
        )

    return pd.DataFrame(results, columns=["k", "inertia", "silhouette_score"])


def reduce_to_2d(matrix: Any) -> np.ndarray:
    """Reduce a manageable sparse sample to two dimensions for visualization."""
    if matrix.shape[0] < 2:
        raise ValueError("At least two samples are required for dimensionality reduction.")

    if len(matrix.shape) != 2 or matrix.shape[1] == 0:
        raise ValueError("matrix must be a non-empty two-dimensional matrix.")
    if matrix.shape[1] < 2:
        zero_column = np.zeros((matrix.shape[0], 1), dtype=np.float32)
        matrix = hstack([matrix, zero_column]).tocsr() if issparse(matrix) else np.column_stack([matrix, zero_column])

    return TruncatedSVD(n_components=2, random_state=42).fit_transform(matrix)


def get_top_cluster_skills(
    jobs_df: pd.DataFrame,
    cluster_id: Any,
    top_n: int = 5,
) -> list[tuple[str, int]]:
    """Return skills ordered by the number of jobs containing each skill."""
    if top_n <= 0:
        raise ValueError("top_n must be greater than 0.")
    if "cluster" not in jobs_df.columns or "skills" not in jobs_df.columns:
        return []

    cluster_jobs = jobs_df.loc[
        jobs_df["cluster"] == cluster_id,
        "skills",
    ]
    skill_counts: Counter[str] = Counter()

    for skills in cluster_jobs:
        if pd.isna(skills):
            continue
        job_skills = {
            skill.strip().casefold()
            for skill in str(skills).split(",")
            if skill.strip()
        }
        skill_counts.update(job_skills)

    return sorted(
        skill_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[:top_n]
