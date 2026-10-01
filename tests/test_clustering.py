"""Focused tests for reusable job clustering utilities."""

import numpy as np
import pandas as pd
import pytest

from src.models.clustering import (
    cluster_jobs,
    evaluate_k_values,
    get_top_cluster_skills,
    reduce_to_2d,
)


def sample_jobs() -> pd.DataFrame:
    """Return small in-memory jobs for clustering tests."""
    return pd.DataFrame(
        [
            {"title": "Python Developer", "skills": "Python, SQL", "description": "Data tools"},
            {"title": "Java Developer", "skills": "Java, Spring", "description": "Backend services"},
            {"title": "Data Analyst", "skills": "Python, Excel", "description": "Business data"},
            {"title": "Frontend Developer", "skills": "JavaScript, CSS", "description": "Web interfaces"},
        ]
    )


def test_cluster_jobs_preserves_rows_and_adds_cluster() -> None:
    jobs = sample_jobs()
    clustered, vectorizer, model = cluster_jobs(jobs, n_clusters=2)

    assert len(clustered) == len(jobs)
    assert clustered["title"].tolist() == jobs["title"].tolist()
    assert "cluster" in clustered.columns
    assert clustered["cluster"].nunique() == 2
    assert len(vectorizer.get_feature_names_out()) > 0
    assert model.n_clusters == 2


def test_empty_jobs_raise_value_error() -> None:
    with pytest.raises(ValueError, match="at least one job"):
        cluster_jobs(pd.DataFrame(), n_clusters=2)


def test_invalid_cluster_counts_raise_value_error() -> None:
    jobs = sample_jobs()

    with pytest.raises(ValueError, match="greater than 0"):
        cluster_jobs(jobs, n_clusters=0)
    with pytest.raises(ValueError, match="cannot exceed"):
        cluster_jobs(jobs, n_clusters=len(jobs) + 1)


def test_invalid_text_raises_value_error() -> None:
    with pytest.raises(ValueError, match="usable text"):
        cluster_jobs(pd.DataFrame({"id": [1, 2]}), n_clusters=2)
    with pytest.raises(ValueError, match="non-empty"):
        cluster_jobs(pd.DataFrame({"title": [None, "  "]}), n_clusters=2)


def test_evaluate_k_values_returns_expected_columns() -> None:
    results = evaluate_k_values(sample_jobs(), [2, 3])

    assert list(results.columns) == ["k", "inertia", "silhouette_score"]
    assert results["k"].tolist() == [2, 3]
    assert results.shape == (2, 3)


def test_evaluate_invalid_k_raises_value_error() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        evaluate_k_values(sample_jobs(), [1])
    with pytest.raises(ValueError, match="less than"):
        evaluate_k_values(sample_jobs(), [4])


def test_reduce_to_2d_returns_two_dimensions() -> None:
    _, vectorizer, _ = cluster_jobs(sample_jobs(), n_clusters=2)
    matrix = vectorizer.transform(["Python SQL", "Java Spring", "Web CSS"])

    reduced = reduce_to_2d(matrix)

    assert isinstance(reduced, np.ndarray)
    assert reduced.shape == (3, 2)


def test_get_top_cluster_skills_counts_each_job_once() -> None:
    jobs = sample_jobs()
    jobs["cluster"] = [0, 1, 0, 1]
    jobs.loc[0, "skills"] = "Python, Python, SQL"

    top_skills = get_top_cluster_skills(jobs, cluster_id=0, top_n=2)

    assert top_skills == [("python", 2), ("excel", 1)]
