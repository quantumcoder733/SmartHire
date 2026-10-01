"""Focused tests for artifact-backed resume analysis."""

from pathlib import Path

import joblib
import pandas as pd
import pytest
from scipy.sparse import save_npz

from src.analyzer import analyze_resume_unsupervised
from src.models.recommender import build_job_index
from src.train_classifier import train_and_evaluate_classifier


def sample_jobs() -> pd.DataFrame:
    """Return small prepared jobs for artifact-backed analyzer tests."""
    return pd.DataFrame(
        [
            {
                "title": "Python Developer",
                "company": "A",
                "location": "Bengaluru",
                "skills": "Python, SQL",
                "experience": "1-3 years",
                "description": "Build Python data tools",
                "source": "fixture",
                "text": "Python Developer Bengaluru Python SQL Build Python data tools",
            },
            {
                "title": "Java Developer",
                "company": "B",
                "location": "Hyderabad",
                "skills": "Java, Spring",
                "experience": "2-5 years",
                "description": "Build Java services",
                "source": "fixture",
                "text": "Java Developer Hyderabad Java Spring Build Java services",
            },
            {
                "title": "Data Analyst",
                "company": "C",
                "location": "Pune",
                "skills": "Python, Excel",
                "experience": "1-3 years",
                "description": "Analyze business data",
                "source": "fixture",
                "text": "Data Analyst Pune Python Excel Analyze business data",
            },
            {
                "title": "Frontend Developer",
                "company": "D",
                "location": "Chennai",
                "skills": "JavaScript, CSS",
                "experience": "1-3 years",
                "description": "Build web interfaces",
                "source": "fixture",
                "text": "Frontend Developer Chennai JavaScript CSS Build web interfaces",
            },
        ]
    )


@pytest.fixture
def artifacts(tmp_path: Path) -> Path:
    """Persist tiny models and aligned job artifacts for analyzer tests."""
    resumes = pd.DataFrame(
        [
            {"Category": category, "text": text}
            for category, text in [
                ("Data", "python sql analyst"),
                ("Data", "data science python"),
                ("Data", "sql data analytics"),
                ("Data", "python statistics data"),
                ("Data", "business data sql"),
                ("Web", "javascript css frontend"),
                ("Web", "react javascript web"),
                ("Web", "css html design"),
                ("Web", "frontend web interface"),
                ("Web", "react css html"),
            ]
        ]
    )
    classifier, _ = train_and_evaluate_classifier(resumes)
    joblib.dump(classifier, tmp_path / "resume_classifier.pkl")

    jobs = sample_jobs()
    vectorizer, matrix = build_job_index(jobs)
    joblib.dump(vectorizer, tmp_path / "job_vectorizer.pkl")
    save_npz(tmp_path / "job_tfidf.npz", matrix)
    jobs.drop(columns="text").to_csv(tmp_path / "job_metadata.csv", index=False)
    pd.DataFrame(
        {"row_index": range(len(jobs)), "cluster": [0, 1, 0, 1]}
    ).to_csv(tmp_path / "job_clusters.csv", index=False)
    return tmp_path


@pytest.fixture
def resume_path(tmp_path: Path) -> Path:
    """Create a valid temporary TXT resume."""
    path = tmp_path / "sample_resume.txt"
    path.write_text("Python developer with Python and SQL experience", encoding="utf-8")
    return path


def test_analysis_returns_classifier_recommendations_and_clusters(
    resume_path: Path,
    artifacts: Path,
) -> None:
    result = analyze_resume_unsupervised(
        resume_path,
        top_n=2,
        artifact_dir=artifacts,
    )

    assert result["resume"] == {
        "filename": "sample_resume.txt",
        "text_length": len("Python developer with Python and SQL experience"),
    }
    assert result["predicted_category"] in {"Data", "Web"}
    assert len(result["recommendations"]) == 2
    assert result["recommendations"][0]["title"] == "Python Developer"
    assert {"matched_skills", "missing_skills", "cluster"} <= result[
        "recommendations"
    ][0].keys()
    assert sum(cluster["job_count"] for cluster in result["clusters"]) == 4
    assert all("top_skills" in cluster for cluster in result["clusters"])


def test_analysis_respects_top_n_and_legacy_arguments(
    resume_path: Path,
    artifacts: Path,
) -> None:
    result = analyze_resume_unsupervised(
        resume_path,
        sample_jobs(),
        top_n=1,
        n_clusters=2,
        artifact_dir=artifacts,
    )

    assert len(result["recommendations"]) == 1
    assert result["recommendations"][0]["rank"] == 1


def test_missing_artifacts_raise_clear_error(resume_path: Path, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="python -m src.train_classifier"):
        analyze_resume_unsupervised(resume_path, artifact_dir=tmp_path)


def test_top_n_must_be_positive(resume_path: Path, artifacts: Path) -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        analyze_resume_unsupervised(resume_path, top_n=0, artifact_dir=artifacts)