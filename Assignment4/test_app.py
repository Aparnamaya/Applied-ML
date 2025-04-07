import pytest
import joblib
from score import score
import subprocess
import time
import requests
import os
import signal
import app

# Load model
MODEL_PATH = "best_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
model = joblib.load(MODEL_PATH)

def test_score():
    """Unit tests for the score function."""
    text = "Free money now!!!"
    prediction, propensity = score(text, model, 0.5)
    assert isinstance(prediction, bool)
    assert isinstance(propensity, float)
    assert 0 <= propensity <= 1
    assert prediction in [0, 1]

    text = "Win a prize now!"
    assert score(text, model, 0.0)[0] == 1
    assert score(text, model, 1.0)[0] == 0

    assert score("Congratulations! You won a lottery", model, 0.5)[0] == 1
    assert score("Let’s catch up tomorrow", model, 0.5)[0] == 0

def test_flask():
    """Test the Flask app directly."""
    process = subprocess.Popen(["python", "app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for _ in range(10):
            try:
                r = requests.get("http://127.0.0.1:5000/")
                if r.status_code == 200:
                    break
            except requests.ConnectionError:
                time.sleep(1)

        response = requests.post("http://127.0.0.1:5000/score",
                                 json={"text": "Claim your free prize now!"},
                                 headers={"Content-Type": "application/json"})
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data and "propensity" in data
        assert data["prediction"] in [0, 1]
        assert 0 <= data["propensity"] <= 1
    finally:
        process.terminate()
        process.wait()
def test_api_no_text():
    """Test JSON request with empty text field."""
    process = subprocess.Popen(["python", "app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(2)  # Give Flask time to start

    try:
        url = "http://127.0.0.1:5000/score"
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json={}, headers=headers)
        assert response.status_code == 400
        assert "error" in response.json()
    finally:
        process.kill()


def test_form_score_no_text():
    """Test browser form POST with empty text."""
    process = subprocess.Popen(["python", "app.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    time.sleep(2)

    try:
        url = "http://127.0.0.1:5000/score"
        response = requests.post(url, data={})
        assert response.status_code == 400
        assert "Error: No text provided" in response.text
    finally:
        process.kill()


def test_docker():
    """Builds and runs Docker container, tests API, then cleans up."""
    # Step 1: Build Docker image
    build = subprocess.run(["docker", "build", "-t", "spam-detector-app", "."], capture_output=True, text=True)
    assert build.returncode == 0, f"Docker build failed:\n{build.stderr}"

    # Step 2: Run container
    run = subprocess.Popen(["docker", "run", "-p", "5000:5000", "--name", "spam_test_container", "spam-detector-app"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    try:
        # Wait for container to spin up
        for _ in range(15):
            try:
                r = requests.get("http://127.0.0.1:5000/")
                if r.status_code == 200:
                    break
            except requests.ConnectionError:
                time.sleep(1)

        # Step 3: Send POST request
        response = requests.post("http://127.0.0.1:5000/score",
                                 json={"text": "This is a limited time offer, click now!"},
                                 headers={"Content-Type": "application/json"})
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data and "propensity" in data
        assert data["prediction"] in [0, 1]
        assert 0 <= data["propensity"] <= 1

    finally:
        # Step 4: Stop & remove container
        subprocess.run(["docker", "stop", "spam_test_container"], stdout=subprocess.DEVNULL)
        subprocess.run(["docker", "rm", "spam_test_container"], stdout=subprocess.DEVNULL)
