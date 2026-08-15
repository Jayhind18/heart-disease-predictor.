"""
Heart Disease Prediction - Flask API

Endpoints:
    GET  /health
    GET  /features
    POST /predict

Model:
    model.pkl
"""

import os
import joblib
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify


# ============================================================
# 1. FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# 2. MODEL PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")


# ============================================================
# 3. LOAD MODEL
# ============================================================

try:
    model = joblib.load(MODEL_PATH)

    FEATURE_NAMES = list(
        getattr(model, "feature_names_in_", [])
    )

    MODEL_LOADED = True

    print("✅ Model loaded successfully")

except Exception as e:
    model = None
    FEATURE_NAMES = []
    MODEL_LOADED = False

    print("❌ Model loading failed:", str(e))


# ============================================================
# 4. HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "model_loaded": MODEL_LOADED
    })


# ============================================================
# 5. FEATURES
# ============================================================

@app.route("/features", methods=["GET"])
def features():

    if not MODEL_LOADED:
        return jsonify({
            "error": "Model is not loaded"
        }), 500

    if FEATURE_NAMES:

        return jsonify({
            "features": FEATURE_NAMES
        })

    return jsonify({
        "features": [],
        "note": (
            "Model did not expose feature_names_in_. "
            "Send features in the exact order used during training."
        )
    })


# ============================================================
# 6. PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    if not MODEL_LOADED:

        return jsonify({
            "error": "Model is not loaded"
        }), 500

    try:

        # Get JSON data
        data = request.get_json(force=True)

        if not data:

            return jsonify({
                "error": "Request body is empty"
            }), 400

        # Check features
        if "features" not in data:

            return jsonify({
                "error": "Request must contain a 'features' key"
            }), 400

        feats = data["features"]


        # ----------------------------------------------------
        # CASE 1: Features sent as dictionary
        # ----------------------------------------------------

        if isinstance(feats, dict):

            df = pd.DataFrame([feats])

            # Match training column order
            if FEATURE_NAMES:

                df = df.reindex(
                    columns=FEATURE_NAMES
                )

            X = df


        # ----------------------------------------------------
        # CASE 2: Features sent as list
        # ----------------------------------------------------

        elif isinstance(feats, list):

            X = np.array(feats).reshape(1, -1)


        # ----------------------------------------------------
        # INVALID FORMAT
        # ----------------------------------------------------

        else:

            return jsonify({
                "error": (
                    "'features' must be either "
                    "a dictionary or a list"
                )
            }), 400


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(X)

        predicted_class = int(prediction[0])

        result = {
            "prediction": predicted_class
        }


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(X)[0]

            result["probabilities"] = {
                str(cls): float(prob)
                for cls, prob in zip(
                    model.classes_,
                    probabilities
                )
            }


        return jsonify(result)


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 400


# ============================================================
# 7. RUN LOCALLY
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )