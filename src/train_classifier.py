"""Train and persist the SmartHire resume category classifier."""

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import CLASSIFIER_PATH
from src.data.load_data import load_resumes
from src.models.classifier import train_classifier


def train_and_evaluate_classifier(
    resumes: pd.DataFrame,
) -> tuple[Pipeline, dict[str, object]]:
    """Train and evaluate the resume classifier using a stratified holdout."""
    required_columns = {"Category", "text"}
    missing_columns = required_columns - set(resumes.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Resume data is missing required columns: {missing}")
    if resumes.empty:
        raise ValueError("Resume data must contain at least one row.")
    if resumes[["Category", "text"]].isna().any().any():
        raise ValueError("Resume categories and text must not contain missing values.")

    features = resumes["text"].astype(str)
    labels = resumes["Category"].astype(str)
    train_text, test_text, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    model = train_classifier(train_text, train_labels)
    predictions = model.predict(test_text)
    metrics: dict[str, object] = {
        "accuracy": accuracy_score(test_labels, predictions),
        "precision_macro": precision_score(
            test_labels, predictions, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            test_labels, predictions, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            test_labels, predictions, average="macro", zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            test_labels,
            predictions,
            labels=model.classes_,
        ),
        "classification_report": classification_report(
            test_labels,
            predictions,
            labels=model.classes_,
            zero_division=0,
        ),
        "labels": np.asarray(model.classes_),
    }
    return model, metrics


def main() -> None:
    """Train the prepared-data classifier and persist its complete pipeline."""
    dataset = load_resumes()
    classifier, metrics = train_and_evaluate_classifier(dataset)

    CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, CLASSIFIER_PATH)

    print("Dataset shape:", dataset.shape)
    print("Number of categories:", dataset["Category"].nunique())
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro precision: {metrics['precision_macro']:.4f}")
    print(f"Macro recall: {metrics['recall_macro']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    print("Confusion matrix (labels order):", metrics["labels"].tolist())
    print(metrics["confusion_matrix"])
    print("Classification report:")
    print(metrics["classification_report"])
    print("Saved model path:", CLASSIFIER_PATH)
    print("Holdout scores are dataset-specific and do not establish production accuracy.")


if __name__ == "__main__":
    main()
