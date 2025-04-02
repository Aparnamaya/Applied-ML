from flask import Flask, request, jsonify
import joblib
from score import score

app = Flask(__name__)

MODEL_PATH = "best_model.pkl"
model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return "Flask is running!", 200

@app.route("/score", methods=["POST"])
def score_text():
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    prediction, propensity = score(text, model, 0.5)
    return jsonify({"prediction": int(prediction), "propensity": propensity})

# Log available routes
with app.test_request_context():
    print(app.url_map)

if __name__ == "__main__":
    app.run(port=5000, debug=True)

