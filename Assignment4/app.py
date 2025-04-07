from flask import Flask, request, jsonify, render_template_string
from score import score, model

app = Flask(__name__)

# HTML template for browser interface
HTML_TEMPLATE = """
<!doctype html>
<title>Spam Detector</title>
<h2>Enter SMS text to classify:</h2>
<form method="post" action="/score">
  <textarea name="text" rows="5" cols="60" placeholder="Type your SMS message here..."></textarea><br><br>
  <input type="submit" value="Detect Spam">
</form>
{% if result %}
  <h3>Prediction: {{ result }}</h3>
  <p>Probability: {{ prob }}</p>
{% endif %}
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/score", methods=["POST"])
def score_text():
    if request.is_json:
        data = request.get_json()
        text = data.get("text", "")
    else:
        text = request.form.get("text", "")

    if not text:
        if request.is_json:
            return jsonify({"error": "No text provided"}), 400
        else:
            return render_template_string(HTML_TEMPLATE, result="Error: No text provided", prob="N/A"), 400

    prediction, propensity = score(text, model=model, threshold=0.5)
    label = "Spam" if prediction else "Not Spam"

    if not request.is_json:
        return render_template_string(HTML_TEMPLATE, result=label, prob=round(propensity, 4))

    return jsonify({
        "prediction": int(prediction),
        "propensity": propensity
    })
a=8908
b=0
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

#ghhoiijesroht3riohtoqi
#gghkl;;;



