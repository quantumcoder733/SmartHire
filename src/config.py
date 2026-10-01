"""Shared paths for prepared data and persisted SmartHire artifacts."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

JOBS_DATA_PATH = PROCESSED_DATA_DIR / "jobs_clean.csv"
RESUMES_DATA_PATH = PROCESSED_DATA_DIR / "resumes_clean.csv"

CLASSIFIER_PATH = MODELS_DIR / "resume_classifier.pkl"
JOB_VECTORIZER_PATH = MODELS_DIR / "job_vectorizer.pkl"
JOB_MATRIX_PATH = MODELS_DIR / "job_tfidf.npz"
JOB_METADATA_PATH = MODELS_DIR / "job_metadata.csv"
JOB_KMEANS_PATH = MODELS_DIR / "job_kmeans.pkl"
JOB_CLUSTERS_PATH = MODELS_DIR / "job_clusters.csv"