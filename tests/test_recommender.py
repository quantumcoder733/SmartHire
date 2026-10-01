"""Smoke tests for the reusable job recommendation engine."""

import pandas as pd
import pytest

from src.models.recommender import build_job_index, recommend_jobs


def sample_jobs() -> pd.DataFrame:
    """Return small in-memory jobs for isolated recommendation tests."""
    return pd.DataFrame(
        [
            {
                "title": "Python Developer",
                "company": "A",
                "skills": "Python, SQL",
                "description": "Build Python data tools",
            },
            {
                "title": "Java Developer",
                "company": "B",
                "skills": "Java, Spring",
                "description": "Build Java services",
            },
            {
                "title": "Data Analyst",
                "company": "C",
                "skills": "Python, Excel",
                "description": "Analyze business data",
            },
        ]
    )


def test_recommendations_are_ranked_and_limited() -> None:
    jobs = sample_jobs()
    vectorizer, job_matrix = build_job_index(jobs)

    recommendations = recommend_jobs(
        "Python SQL data tools",
        jobs,
        vectorizer,
        job_matrix,
        top_n=2,
    )

    assert len(recommendations) == 2
    assert recommendations[0]["title"] == "Python Developer"
    assert recommendations[0]["similarity_score"] >= recommendations[1]["similarity_score"]
    assert [job["rank"] for job in recommendations] == [1, 2]


def test_missing_optional_columns_do_not_crash() -> None:
    jobs = sample_jobs()[["title", "skills"]]
    vectorizer, job_matrix = build_job_index(jobs)

    recommendations = recommend_jobs(
        "Python",
        jobs,
        vectorizer,
        job_matrix,
    )

    assert len(recommendations) == len(jobs)


def test_empty_jobs_raise_value_error() -> None:
    with pytest.raises(ValueError, match="at least one job"):
        build_job_index(pd.DataFrame())


def test_empty_resume_raises_value_error() -> None:
    jobs = sample_jobs()
    vectorizer, job_matrix = build_job_index(jobs)

    with pytest.raises(ValueError, match="non-empty text"):
        recommend_jobs("   ", jobs, vectorizer, job_matrix)


def test_missing_all_usable_columns_raise_value_error() -> None:
    with pytest.raises(ValueError, match="usable text column"):
        build_job_index(pd.DataFrame({"id": [1]}))
