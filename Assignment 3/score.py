import joblib
import re
import numpy as np
from sklearn.base import BaseEstimator
from typing import Tuple

# Load the saved model and vectorizer
MODEL_PATH = "best_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def preprocess_text(text: str) -> str:
    """Basic text preprocessing: lowercasing and removing special characters."""
    return re.sub(r'\W+', ' ', text.lower()).strip()

def score(text: str, model: BaseEstimator, threshold: float) -> Tuple[bool, float]:
    """
    Scores a trained model on a text sample.

    Args:
        text (str): The input text to classify.
        model (sklearn.base.BaseEstimator): A trained scikit-learn model.
        threshold (float): The decision threshold for classification.

    Returns:
        Tuple[bool, float]: A binary prediction (0 or 1) and the propensity score (probability).
    """
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
    if not isinstance(threshold, float) or not (0 <= threshold <= 1):
        raise ValueError("Threshold must be a float between 0 and 1.")

    processed_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([processed_text])
    probability = model.predict_proba(text_vectorized)[:, 1][0]  # Extract spam probability
    prediction = probability >= threshold  # Apply thresholding

    return bool(prediction), float(probability)
