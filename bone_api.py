from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# Flask Configuration
# ==========================================

app = Flask(__name__)
CORS(app)

# ==========================================
# Model Configuration
# ==========================================

N_SCANS = 22
FREQ_POINTS = 751
TOTAL_FEATURES = N_SCANS * FREQ_POINTS

frequencies = np.linspace(2.5, 10.0, FREQ_POINTS)

# ==========================================
# Load AI Assets
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCALER_PATH = os.path.join(BASE_DIR, "bone_sar_scaler.joblib")
MODEL_PATH = os.path.join(BASE_DIR, "bone_sar_ensemble_clf.joblib")

try:
    bone_scaler = joblib.load(SCALER_PATH)
    bone_model = joblib.load(MODEL_PATH)

    print("===================================")
    print(" Bone AI Model Loaded Successfully ")
    print("===================================")

except Exception as e:
    print(e)
    bone_scaler = None
    bone_model = None

# ==========================================
# Health Check
# ==========================================

@app.route("/")
def home():

    return jsonify({

        "status": "online",

        "project": "Wearable Antenna",

        "model": "Bone Fracture Detection",

        "version": "1.0"

    })
# ==========================================
# Prediction API
# ==========================================

@app.route("/predict", methods=["POST"])
def predict():

    if bone_scaler is None or bone_model is None:
        return jsonify({
            "status": "error",
            "message": "AI model is not loaded."
        }), 500

    if "file" not in request.files:
        return jsonify({
            "status": "error",
            "message": "No file uploaded."
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "status": "error",
            "message": "Please choose a CSV or Excel file."
        }), 400

    try:

        # ==========================
        # Read File
        # ==========================

        if file.filename.endswith(".csv"):
            df = pd.read_csv(file)

        elif file.filename.endswith(".xlsx"):
            df = pd.read_excel(file)

        else:
            return jsonify({
                "status": "error",
                "message": "Unsupported file format."
            }), 400

        # ==========================
        # Extract Numerical Features
        # ==========================

        values = df.select_dtypes(include=[np.number]).iloc[0].values

        if len(values) < TOTAL_FEATURES:

            return jsonify({

                "status": "error",

                "message": f"Input file must contain {TOTAL_FEATURES} numerical values."

            }), 400

        values = values[:TOTAL_FEATURES]

        # ==========================
        # Scaling
        # ==========================

        scaled = bone_scaler.transform(
            values.reshape(1, -1)
        )

        # ==========================
        # Prediction
        # ==========================

        prediction = int(
            bone_model.predict(scaled)[0]
        )

        probabilities = bone_model.predict_proba(scaled)[0]

        confidence = float(
            np.max(probabilities) * 100
        )

        labels = {

            0: "Intact Bone Structure",

            1: "Hairline / Oblique Micro-Fracture",

            2: "Complete Comminuted Fracture"

        }

        diagnosis = labels.get(
            prediction,
            "Unknown"
        )

        return jsonify({

            "status": "success",

            "prediction": prediction,

            "diagnosis": diagnosis,

            "confidence": round(confidence, 2),

            "probabilities": {

                "healthy": round(float(probabilities[0]) * 100, 2),

                "hairline": round(float(probabilities[1]) * 100, 2),

                "complete": round(float(probabilities[2]) * 100, 2)

            }

        })

    except Exception as e:

        return jsonify({

            "status": "error",

            "message": str(e)

        }), 500
    # ==========================================
# Run Server
# ==========================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({

        "status": "healthy",

        "model_loaded": bone_model is not None,

        "scaler_loaded": bone_scaler is not None,

        "features_required": TOTAL_FEATURES,

        "classes": [
            "Intact Bone Structure",
            "Hairline / Oblique Micro-Fracture",
            "Complete Comminuted Fracture"
        ]

    })


if __name__ == "__main__":

    print("=" * 50)
    print(" Wearable Antenna Bone AI API")
    print("=" * 50)
    print("Server Running...")
    print("http://127.0.0.1:5000")
    print("=" * 50)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )