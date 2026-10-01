"""Dataset-independent TF-IDF job recommendation utilities."""

from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.models.skill_gap import (
    calculate_skill_gap,
    extract_job_technical_skills,
    extract_technical_skills,
)


_TEXT_COLUMNS = (
    "title",
    "company",
    "location",
    "skills",
    "experience",
    "description",
)


def prepare_job_text(jobs_df: pd.DataFrame) -> pd.Series:
    """Prefer prepared job text and fall back to its searchable metadata."""
    if "text" in jobs_df.columns:
        prepared_text = jobs_df["text"].fillna("").astype(str).str.strip()
        missing_text = prepared_text.eq("")
        if not missing_text.any():
            return prepared_text
    else:
        prepared_text = pd.Series("", index=jobs_df.index, dtype="object")
        missing_text = pd.Series(True, index=jobs_df.index)

    usable_columns = [
        column for column in _TEXT_COLUMNS if column in jobs_df.columns
    ]
    if not usable_columns and missing_text.any():
        raise ValueError(
            "jobs_df must contain at least one usable text column: "
            + ", ".join(_TEXT_COLUMNS)
        )

    if usable_columns:
        fallback = (
            jobs_df.loc[missing_text, usable_columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
            .str.strip()
        )
        prepared_text.loc[missing_text] = fallback

    return prepared_text


def create_job_vectorizer() -> TfidfVectorizer:
    """Create the shared, memory-conscious job-corpus vectorizer."""
    return TfidfVectorizer(
        stop_words="english",
        max_features=50_000,
        sublinear_tf=True,
        dtype=np.float32,
    )


def build_job_index(
    jobs_df: pd.DataFrame,
) -> tuple[TfidfVectorizer, csr_matrix]:
    """Fit a TF-IDF vectorizer on the supplied jobs and return its index."""
    if jobs_df.empty:
        raise ValueError("jobs_df must contain at least one job.")

    job_text = prepare_job_text(jobs_df)
    if not job_text.str.strip().any():
        raise ValueError("jobs_df must contain usable non-empty job text.")

    vectorizer = create_job_vectorizer()
    job_matrix = vectorizer.fit_transform(job_text).tocsr()
    return vectorizer, job_matrix


def recommend_jobs(
    resume_text: str,
    jobs_df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    job_matrix: csr_matrix,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Rank jobs by raw cosine similarity to a resume."""
    if top_n <= 0:
        raise ValueError("top_n must be greater than 0.")
    if jobs_df.empty:
        raise ValueError("jobs_df must contain at least one job.")
    if not isinstance(resume_text, str) or not resume_text.strip():
        raise ValueError("resume_text must contain non-empty text.")

    resume_vector = vectorizer.transform([resume_text])
    similarity_scores = (job_matrix @ resume_vector.T).toarray().ravel()
    ranked_indices = _top_indices(similarity_scores, top_n)

    recommendations: list[dict[str, Any]] = []
    for rank, index in enumerate(ranked_indices, start=1):
        recommendation = jobs_df.iloc[index].to_dict()
        recommendation["rank"] = rank
        recommendation["similarity_score"] = float(similarity_scores[index])
        recommendations.append(recommendation)

    return recommendations


def _top_indices(scores: np.ndarray, top_n: int) -> np.ndarray:
    count = min(top_n, len(scores))
    if count == 0:
        return np.array([], dtype=int)
    candidate_indices = np.argpartition(scores, len(scores) - count)[-count:]
    order = np.lexsort((candidate_indices, -scores[candidate_indices]))
    return candidate_indices[order]


def recommend_jobs_from_index(
    resume_text: str,
    metadata: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    job_matrix: csr_matrix,
    clusters: np.ndarray,
    top_n: int = 5,
    candidate_skills: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Recommend from persisted TF-IDF rows and aligned job metadata."""
    if top_n <= 0:
        raise ValueError("top_n must be greater than 0.")
    if not isinstance(resume_text, str) or not resume_text.strip():
        raise ValueError("resume_text must contain non-empty text.")
    if len(metadata) == 0:
        raise ValueError("Job metadata must contain at least one job.")
    if job_matrix.shape[0] != len(metadata) or len(clusters) != len(metadata):
        raise ValueError("Job matrix, metadata, and clusters must be row-aligned.")

    resume_vector = vectorizer.transform([resume_text])
    similarity_scores = (job_matrix @ resume_vector.T).toarray().ravel()
    ranked_indices = _top_indices(similarity_scores, top_n)
    resume_skills = (
        extract_technical_skills(resume_text)
        if candidate_skills is None
        else candidate_skills
    )

    recommendations: list[dict[str, Any]] = []
    for rank, index in enumerate(ranked_indices, start=1):
        row = metadata.iloc[index]
        skills_value = row.get("skills", "")
        if pd.isna(skills_value):
            skills_value = ""
        job_skills = extract_job_technical_skills(row)
        skill_gap = calculate_skill_gap(resume_skills, job_skills)

        recommendations.append(
            {
                "rank": rank,
                "title": row.get("title", ""),
                "company": row.get("company", ""),
                "location": row.get("location", ""),
                "skills": skills_value,
                "experience": row.get("experience", ""),
                "similarity_score": float(similarity_scores[index]),
                "matched_skills": skill_gap["matched_skills"],
                "missing_skills": skill_gap["missing_skills"],
                "cluster": int(clusters[index]),
            }
        )

    return recommendations
