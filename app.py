from flask import Flask, render_template, request, jsonify
import os
import joblib
import pandas as pd

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "model", "disease_model.pkl")

# If the model is not present, train_model.py must be run first.
model = joblib.load(MODEL_PATH)

LOCATION_COORDS = {
    "Mumbai": [19.0760, 72.8777],
    "Pune": [18.5204, 73.8567],
    "Delhi": [28.6139, 77.2090],
    "Ranchi": [23.3441, 85.3096],
    "Kolkata": [22.5726, 88.3639],
    "Bengaluru": [12.9716, 77.5946],
    "Chennai": [13.0827, 80.2707],
    "Lucknow": [26.8467, 80.9462],
    "Patna": [25.5941, 85.1376],
    "Nagpur": [21.1458, 79.0882],
}

MONTHS = [
    (1, "January"), (2, "February"), (3, "March"), (4, "April"),
    (5, "May"), (6, "June"), (7, "July"), (8, "August"),
    (9, "September"), (10, "October"), (11, "November"), (12, "December")
]

PREVENTION = {
    "Low": [
        "Maintain good personal hygiene.",
        "Keep drinking water and food covered.",
        "Continue routine cleaning of water-storage areas."
    ],
    "Medium": [
        "Avoid stagnant water around homes and workplaces.",
        "Use mosquito protection where appropriate.",
        "Keep surroundings clean and improve drainage."
    ],
    "High": [
        "Take extra precautions against mosquito exposure.",
        "Remove stagnant water and improve drainage promptly.",
        "If symptoms develop, consult a qualified healthcare professional."
    ]
}

@app.route("/")
def index():
    return render_template(
        "index.html",
        locations=sorted(LOCATION_COORDS),
        months=MONTHS
    )

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form

    location = data.get("location", "Mumbai")
    area = data.get("area", "Urban")
    month = int(data.get("month", 7))
    temperature = float(data.get("temperature", 28))
    humidity = float(data.get("humidity", 80))
    rainfall = float(data.get("rainfall", 200))

    input_df = pd.DataFrame([{
        "Location": location,
        "Area": area,
        "Month": month,
        "Temperature": temperature,
        "Humidity": humidity,
        "Rainfall": rainfall
    }])

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    classes = list(model.classes_)
    confidence = float(probabilities[classes.index(prediction)] * 100)

    # Educational display only; this is model confidence, not medical probability.
    if prediction == "High":
        risk_message = "Higher modeled seasonal risk"
    elif prediction == "Medium":
        risk_message = "Moderate modeled seasonal risk"
    else:
        risk_message = "Lower modeled seasonal risk"

    lat, lon = LOCATION_COORDS.get(location, [20.5937, 78.9629])

    return render_template(
        "result.html",
        prediction=prediction,
        confidence=round(confidence, 1),
        risk_message=risk_message,
        location=location,
        area=area,
        month=dict(MONTHS).get(month, "Unknown"),
        temperature=temperature,
        humidity=humidity,
        rainfall=rainfall,
        lat=lat,
        lon=lon,
        prevention=PREVENTION[prediction]
    )

@app.route("/api/locations")
def locations():
    return jsonify(LOCATION_COORDS)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    # Works locally and on hosts that provide PORT (such as Render).
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
