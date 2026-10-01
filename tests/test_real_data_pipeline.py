"""Small-fixture tests for the prepared-data ML backend."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from src.analyzer import load_classifier, load_job_cluster_assignments, load_job_kmeans
from src.build_job_clusters import build_job_clusters_from_matrix
from src.build_job_index import build_job_index_from_csv
from src.data import load_data
from src.models.recommender import build_job_index, recommend_jobs_from_index
from src.models.skill_gap import calculate_skill_gap
from src.train_classifier import train_and_evaluate_classifier


JOB_ROWS = [
    {
        "title": "Python Developer",
        "company": "A",
        "location": "Remote",
        "skills": "Python, SQL",
        "description": "Build Python data tools",
        "experience": "2 years",
        "source": "fixture-a",
        "text": "Python Developer Python SQL Build Python data tools",
    },
    {
        "title": "Data Analyst",
        "company": "B",
        "location": "Hybrid",
        "skills": "Python, Excel",
        "description": "Analyze business data",
        "experience": "1 year",
        "source": "fixture-b",
        "text": "Data Analyst Python Excel Analyze business data",
    },
    {
        "title": "Java Developer",
        "company": "C",
        "location": "Remote",
        "skills": "Java, Spring",
        "description": "Build Java services",
        "experience": "3 years",
        "source": "fixture-c",
        "text": "Java Developer Java Spring Build Java services",
    },
]


def test_loaders_validate_prepared_schemas_without_job_text(
    tmp_path: Path,
) -> None:
    jobs_path = tmp_path / "jobs.csv"
    resumes_path = tmp_path / "resumes.csv"
    pd.DataFrame(JOB_ROWS).to_csv(jobs_path, index=False)
    pd.DataFrame(
        [{"Category": "Data", "text": "Python data analyst"}]
    ).to_csv(resumes_path, index=False)

    assert load_data.load_jobs(jobs_path).columns.tolist() == list(JOB_ROWS[0])
    assert load_data.load_resumes(resumes_path).columns.tolist() == [
        "Category",
        "text",
    ]


def test_default_loader_targets_prepared_jobs_without_reading_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_paths: list[Path] = []

    def fake_read_csv(path: str | Path) -> pd.DataFrame:
        captured_paths.append(Path(path))
        return pd.DataFrame(JOB_ROWS)

    monkeypatch.setattr(load_data.pd, "read_csv", fake_read_csv)

    load_data.load_jobs()

    assert captured_paths[0].name == "jobs_clean.csv"
    assert captured_paths[0].parent.name == "processed"


def test_classifier_training_reports_metrics_and_persists_pipeline(
    tmp_path: Path,
) -> None:
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
    model, metrics = train_and_evaluate_classifier(resumes)
    model_path = tmp_path / "resume_classifier.pkl"
    joblib.dump(model, model_path)
    loaded_model = load_classifier(model_path)

    assert model.named_steps["tfidf"].max_features == 5000
    assert model.named_steps["classifier"].max_iter == 1000
    assert len(loaded_model.predict(["python SQL analyst"])) == 1
    assert metrics["confusion_matrix"].shape == (2, 2)
    assert "macro avg" in metrics["classification_report"]
    assert all(
        key in metrics
        for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
    )


def test_streamed_job_index_preserves_metadata_row_alignment(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "jobs.csv"
    metadata_path = tmp_path / "job_metadata.csv"
    pd.DataFrame(JOB_ROWS).to_csv(source_path, index=False)

    row_count, vocabulary_size = build_job_index_from_csv(
        source_path,
        tmp_path / "job_vectorizer.pkl",
        tmp_path / "job_tfidf.npz",
        metadata_path,
        chunksize=1,
    )
    metadata = pd.read_csv(metadata_path)
    vectorizer = joblib.load(tmp_path / "job_vectorizer.pkl")
    from scipy.sparse import load_npz

    matrix = load_npz(tmp_path / "job_tfidf.npz")

    assert row_count == len(JOB_ROWS) == matrix.shape[0] == len(metadata)
    assert vocabulary_size == len(vectorizer.vocabulary_)
    assert metadata["source"].tolist() == [row["source"] for row in JOB_ROWS]
    assert metadata.columns.tolist() == [
        "title",
        "company",
        "location",
        "skills",
        "description",
        "experience",
        "source",
    ]
    assert "text" not in metadata.columns


def test_sparse_index_recommendations_use_top_n_and_skill_gap() -> None:
    jobs = pd.DataFrame(JOB_ROWS)
    vectorizer, matrix = build_job_index(jobs)
    metadata = jobs.drop(columns="text")

    recommendations = recommend_jobs_from_index(
        "Python SQL developer",
        metadata,
        vectorizer,
        matrix,
        np.array([2, 2, 1]),
        top_n=2,
    )

    assert [item["title"] for item in recommendations] == [
        "Python Developer",
        "Data Analyst",
    ]
    assert [item["rank"] for item in recommendations] == [1, 2]
    assert recommendations[0]["matched_skills"] == ["Python"]
    assert recommendations[0]["missing_skills"] == []
    assert recommendations[0]["cluster"] == 2


def test_skill_gap_normalizes_and_does_not_invent_skills() -> None:
    assert calculate_skill_gap(
        {" Python ", "sql"},
        {"PYTHON", "SQL", "Tableau"},
    ) == {
        "matched_skills": ["Python", "SQL"],
        "missing_skills": ["Tableau"],
    }


def test_sampled_cluster_build_saves_aligned_assignments_and_model(
    tmp_path: Path,
) -> None:
    texts = [
        (
            "python data analytics machine learning"
            if index % 2 == 0
            else "java backend spring services"
        )
        + f" specialty{index}"
        for index in range(24)
    ]
    matrix_path = tmp_path / "job_tfidf.npz"
    save_npz(matrix_path, TfidfVectorizer().fit_transform(texts))
    model_path = tmp_path / "job_kmeans.pkl"
    clusters_path = tmp_path / "job_clusters.csv"

    evaluation, labels, model = build_job_clusters_from_matrix(
        matrix_path,
        model_path,
        clusters_path,
        sample_size=20,
        n_init=1,
    )
    loaded_labels = load_job_cluster_assignments(
        clusters_path,
        expected_rows=len(texts),
    )
    loaded_model = load_job_kmeans(model_path)

    assert len(evaluation) == 7
    assert model.n_clusters in evaluation["k"].tolist()
    assert loaded_model.n_clusters == model.n_clusters
    assert np.array_equal(loaded_labels, labels)
    with pytest.raises(ValueError, match="row-aligned"):
        load_job_cluster_assignments(clusters_path, expected_rows=len(texts) + 1)