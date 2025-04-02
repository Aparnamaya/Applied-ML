import pytest
import joblib
from score import score
import subprocess
import time
import requests
import app

# Load the best model from the saved pickle file
MODEL_PATH = "best_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

model = joblib.load(MODEL_PATH)

def test_score():
    """Unit tests for the score function."""

    # Smoke test: Does it run without errors?
    text = "Free money now!!!"
    prediction, propensity = score(text, model, 0.5)
    assert isinstance(prediction, bool), "Prediction should be boolean."
    assert isinstance(propensity, float), "Propensity should be a float."

    # Format tests
    assert 0 <= propensity <= 1, "Propensity score should be between 0 and 1."
    assert prediction in [0, 1], "Prediction should be either 0 or 1."

    # Edge cases: Extreme threshold values
    text = "Win a prize now!"
    assert score(text, model, 0.0)[0] == 1, "Threshold 0 should always predict 1."
    assert score(text, model, 1.0)[0] == 0, "Threshold 1 should always predict 0."

    # Domain-specific tests
    spam_text = "Congratulations! You won a lottery, claim now!"
    assert score(spam_text, model, 0.5)[0] == 1, "Obvious spam should be classified as spam."

    non_spam_text = "Let's schedule a meeting for tomorrow."
    assert score(non_spam_text, model, 0.5)[0] == 0, "Obvious non-spam should be classified as non-spam."

def test_flask():
    """Integration test for the Flask API."""
    # Start the Flask app
    process = subprocess.Popen(["python", "app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for Flask to start
    for _ in range(10):  # Try for up to 10 seconds
        try:
            response = requests.get("http://127.0.0.1:5000/")
            if response.status_code == 200:
                break
        except requests.ConnectionError:
            time.sleep(1)
    
    url = "http://127.0.0.1:5000/score"
    headers = {"Content-Type": "application/json"}

    # Send a test request
    test_data = {"text": "You won a free lottery, claim now!"}
    response = requests.post(url, json=test_data, headers=headers)

    assert response.status_code == 200, "API response should be successful."
    data = response.json()
    assert "prediction" in data and "propensity" in data, "Response should contain prediction and propensity."
    assert data["prediction"] in [0, 1], "Prediction should be either 0 or 1."
    assert 0 <= data["propensity"] <= 1, "Propensity should be between 0 and 1."

    # Stop the Flask app
    process.kill()  # Ensure it stops
    stdout, stderr = process.communicate()
    print(stdout.decode(), stderr.decode())  # Print logs if needed

if __name__ == "__main__":
    pytest.main()
