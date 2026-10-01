"""Evaluate and persist K-Means clusters for the prepared job index."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from src.config import (
    JOB_CLUSTERS_PATH,
    JOB_KMEANS_PATH,
    JOB_MATRIX_PATH,
    MODELS_DIR,
)
from src.models.clustering import cluster_job_matrix, evaluate_k_values


def build_job_clusters_from_matrix(
    matrix_path: str | Path,
    model_path: str | Path,
    clusters_path: str | Path,
    sample_size: int = 2_000,
    n_init: int = 3,
) -> tuple[pd.DataFrame, np.ndarray, object]:
    """Select K on a deterministic sample, then label all matrix rows."""
    sparse_job_matrix = load_npz(matrix_path).tocsr()
    if sparse_job_matrix.shape[0] < 3:
        raise ValueError("At least three indexed jobs are required for clustering.")

    evaluation_sample_size = min(sample_size, sparse_job_matrix.shape[0])
    max_k = min(8, evaluation_sample_size - 1)
    if max_k < 2:
        raise ValueError("At least three jobs are required to evaluate K values.")

    evaluation = evaluate_k_values(
        sparse_job_matrix,
        range(2, max_k + 1),
        random_state=42,
        n_init=n_init,
        sample_size=evaluation_sample_size,
    )
    best_result = evaluation.loc[evaluation["silhouette_score"].idxmax()]
    chosen_k = int(best_result["k"])
    labels, model = cluster_job_matrix(
        sparse_job_matrix,
        n_clusters=chosen_k,
        random_state=42,
        n_init=n_init,
    )

    model_output = Path(model_path)
    clusters_output = Path(clusters_path)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    clusters_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_output)
    pd.DataFrame(
        {"row_index": np.arange(len(labels)), "cluster": labels}
    ).to_csv(clusters_output, index=False)

    counts = pd.Series(labels).value_counts().sort_index()
    print("Sample-based K evaluation:")
    print(evaluation.to_string(index=False))
    print(
        f"Chosen K: {chosen_k} (highest silhouette on a deterministic "
        f"sample of {evaluation_sample_size}; exploratory, not universal)."
    )
    print("Full-corpus cluster counts:")
    print(counts.to_string())
    return evaluation, labels, model


def main() -> None:
    """Build clusters from the saved sparse job matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-size", type=int, default=2_000)
    parser.add_argument("--n-init", type=int, default=3)
    args = parser.parse_args()

    if not JOB_MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"Job TF-IDF matrix not found: {JOB_MATRIX_PATH}. "
            "Run python -m src.build_job_index first."
        )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    evaluation, labels, _ = build_job_clusters_from_matrix(
        JOB_MATRIX_PATH,
        JOB_KMEANS_PATH,
        JOB_CLUSTERS_PATH,
        sample_size=args.sample_size,
        n_init=args.n_init,
    )
    chosen_k = int(evaluation.loc[evaluation["silhouette_score"].idxmax(), "k"])
    print(f"Cluster assignment rows: {len(labels)}")
    print(f"Saved K-Means model: {JOB_KMEANS_PATH}")
    print(f"Saved aligned assignments: {JOB_CLUSTERS_PATH}")
    print(f"Saved cluster count: {chosen_k}")


if __name__ == "__main__":
    main()