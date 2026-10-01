"""Resume category classification utilities."""

from collections.abc import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def train_classifier(X_train: Sequence[str], y_train: Sequence[str]) -> Pipeline:
	"""Train and return a TF-IDF and logistic regression resume classifier."""
	model = Pipeline(
		steps=[
			(
				"tfidf",
				TfidfVectorizer(max_features=5000, stop_words="english"),
			),
			(
				"classifier",
				LogisticRegression(max_iter=1000, random_state=42),
			),
		]
	)
	model.fit(X_train, y_train)
	return model


def predict_categories(model: Pipeline, resumes: Sequence[str]):
	"""Predict resume categories for the supplied resume texts."""
	return model.predict(resumes)


def predict_probabilities(model: Pipeline, resumes: Sequence[str]):
	"""Return category probabilities for the supplied resume texts."""
	return model.predict_proba(resumes)
