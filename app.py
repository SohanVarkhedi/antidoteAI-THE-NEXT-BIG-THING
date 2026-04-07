"""
Antidote AI — Flask Application
Adversarially Robust AI Security Middleware

Endpoints:
    POST /upload   – Upload CSV, detect poisoning, return stats
    POST /train    – Train base model on cleaned data
    POST /predict  – Run full defense pipeline on new input
"""

import os
import json
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from backend.poisoning_detector import detect_poisoning
from backend.train_model import train_model
from backend.evasion_detector import EvasionDetector
from backend.validator import validate_input
from backend.ensemble import ensemble_decision

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
FRONT_DIR  = os.path.join(BASE_DIR, "frontend")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── App ───────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=FRONT_DIR, static_url_path="")
CORS(app)

# ── In-memory state ──────────────────────────────────────────────
state = {
    "cleaned_df": None,        # pandas DataFrame after poisoning removal
    "poisoning_result": None,  # dict from poisoning detector
    "model_info": None,        # dict from training step
    "evasion_detector": EvasionDetector(),
}

# ══════════════════════════════════════════════════════════════════
#  ROUTES — Static frontend
# ══════════════════════════════════════════════════════════════════

@app.route("/")
def serve_index():
    return send_from_directory(FRONT_DIR, "index.html")


# ══════════════════════════════════════════════════════════════════
#  POST /upload
# ══════════════════════════════════════════════════════════════════

@app.route("/upload", methods=["POST"])
def upload_dataset():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files are supported."}), 400

    # Save upload
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(save_path)

    try:
        df = pd.read_csv(save_path)
    except Exception as e:
        return jsonify({"error": f"Cannot parse CSV: {e}"}), 400

    # Run poisoning detection
    result = detect_poisoning(df)

    # Persist cleaned dataframe for training
    state["cleaned_df"] = result["cleaned_df"]
    state["poisoning_result"] = {
        "total_rows": result["total_rows"],
        "suspicious_rows": result["suspicious_rows"],
        "cleaned_rows": result["cleaned_rows"],
        "suspicious_indices": result["suspicious_indices"],
    }

    return jsonify(state["poisoning_result"])


# ══════════════════════════════════════════════════════════════════
#  POST /train
# ══════════════════════════════════════════════════════════════════

@app.route("/train", methods=["POST"])
def train():
    if state["cleaned_df"] is None:
        return jsonify({"error": "No dataset uploaded yet. Upload first."}), 400

    cleaned_df = state["cleaned_df"]

    # Determine target column — use the last column if 'target' isn't present
    target_col = "target"
    if target_col not in cleaned_df.columns:
        target_col = cleaned_df.columns[-1]

    try:
        info = train_model(cleaned_df, target_column=target_col)
    except Exception as e:
        return jsonify({"error": f"Training failed: {e}"}), 500

    state["model_info"] = info

    # Fit evasion detector on the clean training features
    X_clean = cleaned_df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    state["evasion_detector"].fit(X_clean.values)
    state["evasion_detector"].save()

    return jsonify({
        "status": "Model trained successfully.",
        "accuracy": info["accuracy"],
        "n_samples": info["n_samples"],
        "n_features": info["n_features"],
    })


# ══════════════════════════════════════════════════════════════════
#  POST /predict
# ══════════════════════════════════════════════════════════════════

@app.route("/predict", methods=["POST"])
def predict():
    # Load model + meta
    model_path = os.path.join(MODEL_DIR, "base_model.pkl")
    meta_path  = os.path.join(MODEL_DIR, "meta.pkl")

    if not os.path.exists(model_path):
        return jsonify({"error": "No trained model found. Train first."}), 400

    clf  = joblib.load(model_path)
    meta = joblib.load(meta_path)

    n_features    = meta["n_features"]
    feature_names = meta["feature_names"]

    # Parse input
    data = request.get_json(force=True)
    raw_features = data.get("features")
    if raw_features is None:
        return jsonify({"error": "Missing 'features' in request body."}), 400

    # Convert to floats
    try:
        features = [float(v) for v in raw_features]
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid feature value: {e}"}), 400

    # 1 ── Validate
    val = validate_input(features, n_features)
    if not val["valid"]:
        return jsonify({
            "poisoning_risk": False,
            "evasion_risk": True,
            "model_prediction": -1,
            "decision": "BLOCK",
            "risk_score": 100,
            "details": "Validation failed: " + "; ".join(val["errors"]),
        })

    # 2 ── Evasion detection
    evasion = {"evasion_flag": False, "decision_score": 0.0}
    if state["evasion_detector"]._fitted:
        evasion = state["evasion_detector"].predict(features)
    else:
        # Try loading persisted model
        if state["evasion_detector"].load():
            evasion = state["evasion_detector"].predict(features)

    # 3 ── Base model prediction
    x = np.array(features).reshape(1, -1)
    prediction = int(clf.predict(x)[0])
    try:
        proba = clf.predict_proba(x)[0]
        confidence = float(np.max(proba))
    except Exception:
        confidence = 1.0

    # 4 ── Poisoning risk for this input (re-use session flag)
    poisoning_flag = False  # only meaningful during upload phase

    # 5 ── Ensemble decision
    result = ensemble_decision(
        poisoning_flag=poisoning_flag,
        evasion_flag=evasion["evasion_flag"],
        model_prediction=prediction,
        evasion_score=evasion["decision_score"],
        model_confidence=confidence,
    )

    return jsonify({
        "poisoning_risk": poisoning_flag,
        "evasion_risk": evasion["evasion_flag"],
        "evasion_score": evasion["decision_score"],
        "model_prediction": prediction,
        "model_confidence": round(confidence, 4),
        "decision": result["decision"],
        "risk_score": result["risk_score"],
        "details": result["details"],
    })


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app.run(debug=True, port=5000)
