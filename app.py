import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model and label encoder
model = joblib.load("crop_recommendation_model.pkl")  # Ensure correct filename
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/", methods=["GET"])
def home():
    return "Flask server is running!", 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        required_keys = ["N", "P", "K", "moisture", "temperature", "humidity"]
        missing_keys = [key for key in required_keys if key not in data]

        if missing_keys:
            return jsonify({"error": f"Missing parameters: {', '.join(missing_keys)}"}), 400

        features = [[
            data["N"], data["P"], data["K"],
            data["moisture"], data["temperature"], data["humidity"]
        ]]

        predicted_label = model.predict(features)[0]
        predicted_crop = label_encoder.inverse_transform([predicted_label])[0]

        probabilities = model.predict_proba(features)[0]
        confidence = round(np.max(probabilities) * 100, 2)

        return jsonify({"crop": predicted_crop, "confidence": f"{confidence}%"})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
