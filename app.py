"""
Antidote AI — Flask Application
Adversarially Robust AI Security Middleware

Endpoints:
    POST /upload         – Upload CSV, detect poisoning, save cleaned dataset
    GET  /download-cleaned – Download cleaned CSV after poisoning removal
    POST /train          – Train ensemble models on cleaned data
    POST /predict        – Run full defense pipeline on new input
"""

import os
import json
import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

from backend.poisoning_detector import detect_poisoning
from backend.train_model import train_model
from backend.evasion_detector import EvasionDetector
from backend.evasion_ensemble import EvasionEnsemble
from backend.validator import validate_input
from backend.ensemble import ensemble_decision
from backend.ensemble_models import train_ensemble
from backend.drift_detector import DriftDetector
from backend.risk_engine import calculate_risk
from backend.explainability import explain
from backend.logger import log_poisoning, log_evasion, log_decision

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
    "evasion_ensemble": EvasionEnsemble(),
    "drift_detector": DriftDetector(),
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

    # Save cleaned dataset to CSV for download
    cleaned_csv_path = os.path.join(UPLOAD_DIR, "cleaned_dataset.csv")
    result["cleaned_df"].to_csv(cleaned_csv_path, index=False)

    # Log poisoning event
    log_poisoning(
        total_rows=result["total_rows"],
        suspicious_rows=result["suspicious_rows"],
        cleaned_rows=result["cleaned_rows"],
        filename=file.filename,
    )

    return jsonify(state["poisoning_result"])


# ══════════════════════════════════════════════════════════════════
#  GET /download-cleaned
# ══════════════════════════════════════════════════════════════════

@app.route("/download-cleaned", methods=["GET"])
def download_cleaned():
    cleaned_path = os.path.join(UPLOAD_DIR, "cleaned_dataset.csv")
    if not os.path.exists(cleaned_path):
        return jsonify({"error": "No cleaned dataset available. Upload a dataset first."}), 404

    return send_file(
        cleaned_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name="cleaned_dataset.csv",
    )


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

    # ── Train ensemble models ─────────────────────────────────
    try:
        info = train_ensemble(cleaned_df, target_column=target_col)
    except Exception as e:
        return jsonify({"error": f"Ensemble training failed: {e}"}), 500

    # ── Also train legacy base model for backward compat ──────
    try:
        train_model(cleaned_df, target_column=target_col)
    except Exception:
        pass  # Non-critical if ensemble succeeds

    state["model_info"] = info

    # Fit evasion detectors on the clean training features
    X_clean = cleaned_df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    state["evasion_detector"].fit(X_clean.values)
    state["evasion_detector"].save()

    # Fit evasion ensemble (SVM + IsoForest + Z-score)
    state["evasion_ensemble"].fit(X_clean.values)
    state["evasion_ensemble"].save()

    # Fit drift detector on the clean training features
    state["drift_detector"].fit(X_clean.values)
    state["drift_detector"].save()

    return jsonify({
        "status": "Ensemble models trained successfully.",
        "accuracy": info["accuracy"],
        "n_samples": info["n_samples"],
        "n_features": info["n_features"],
        "individual": info["individual"],
    })


# ══════════════════════════════════════════════════════════════════
#  POST /predict
# ══════════════════════════════════════════════════════════════════

@app.route("/predict", methods=["POST"])
def predict():
    # Load ensemble model + meta
    ensemble_path = os.path.join(MODEL_DIR, "ensemble_model.pkl")
    meta_path     = os.path.join(MODEL_DIR, "ensemble_meta.pkl")
    scaler_path   = os.path.join(MODEL_DIR, "scaler.pkl")

    # Fall back to base model if ensemble not available
    if not os.path.exists(ensemble_path):
        ensemble_path = os.path.join(MODEL_DIR, "base_model.pkl")
        meta_path = os.path.join(MODEL_DIR, "meta.pkl")

    if not os.path.exists(ensemble_path):
        return jsonify({"error": "No trained model found. Train first."}), 400

    clf  = joblib.load(ensemble_path)
    meta = joblib.load(meta_path)

    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

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
            "severity": "HIGH",
            "drift_flag": False,
            "drift_score": 0,
            "explanation": ["Validation failed: " + "; ".join(val["errors"])],
            "details": "Validation failed: " + "; ".join(val["errors"]),
        })

    # 2 ── Evasion ensemble detection (SVM + IsoForest + Z-score)
    evasion_ens = {
        "evasion_flag": False, "confidence": 0.0,
        "svm": "normal", "isolation_forest": "normal", "zscore": "normal",
        "votes_for": 0, "votes_total": 3,
        "svm_score": 0.0, "iforest_score": 0.0, "zscore_max": 0.0,
    }
    if state["evasion_ensemble"]._fitted:
        evasion_ens = state["evasion_ensemble"].predict(features)
    else:
        if state["evasion_ensemble"].load():
            evasion_ens = state["evasion_ensemble"].predict(features)

    # Legacy single-detector (kept for backward compat)
    evasion = {"evasion_flag": evasion_ens["evasion_flag"], "decision_score": evasion_ens.get("svm_score", 0.0)}

    log_evasion(
        input_summary=str(features[:3]) + "...",
        evasion_flag=evasion_ens["evasion_flag"],
        decision_score=evasion_ens["confidence"],
    )

    # 3 ── Drift detection
    drift = {"drift_flag": False, "drift_score": 0.0, "drifted_features": []}
    if state["drift_detector"]._fitted:
        drift = state["drift_detector"].detect(features)
    else:
        if state["drift_detector"].load():
            drift = state["drift_detector"].detect(features)

    # 4 ── Ensemble prediction
    x = np.array(features).reshape(1, -1)
    if scaler is not None:
        x_scaled = scaler.transform(x)
    else:
        x_scaled = x

    prediction = int(clf.predict(x_scaled)[0])
    try:
        proba = clf.predict_proba(x_scaled)[0]
        confidence = float(np.max(proba))
    except Exception:
        confidence = 1.0

    # 5 ── Risk scoring (use evasion confidence from ensemble)
    evasion_risk_score = evasion_ens["confidence"] * 100
    poisoning_flag = False  # only meaningful during upload phase
    poisoning_score = 100.0 if poisoning_flag else 0.0

    risk_result = calculate_risk(
        poisoning_score=poisoning_score,
        evasion_score=evasion_risk_score,
        drift_score=drift["drift_score"],
        model_confidence=confidence,
    )

    # 6 ── Explainability
    explanation_list = explain(
        x=features,
        training_data=state["drift_detector"].training_data if state["drift_detector"]._fitted else None,
        feature_names=feature_names,
    )

    # 7 ── Ensemble decision
    result = ensemble_decision(
        poisoning_flag=poisoning_flag,
        evasion_flag=evasion_ens["evasion_flag"],
        model_prediction=prediction,
        evasion_score=evasion_ens["confidence"],
        model_confidence=confidence,
        drift_flag=drift["drift_flag"],
        severity=risk_result["severity"],
        explanation=explanation_list,
    )

    # 8 ── Log decision
    log_decision(
        input_summary=str(features[:3]) + "...",
        decision=result["decision"],
        risk_score=risk_result["risk_score"],
        severity=risk_result["severity"],
        drift_flag=drift["drift_flag"],
        explanation="; ".join(explanation_list),
    )

    return jsonify({
        "poisoning_risk": poisoning_flag,
        "evasion_risk": evasion_ens["evasion_flag"],
        "evasion_confidence": evasion_ens["confidence"],
        "evasion_votes": {
            "svm": evasion_ens["svm"],
            "isolation_forest": evasion_ens["isolation_forest"],
            "zscore": evasion_ens["zscore"],
            "votes_for": evasion_ens["votes_for"],
            "votes_total": evasion_ens["votes_total"],
        },
        "model_prediction": prediction,
        "model_confidence": round(confidence, 4),
        "drift_flag": drift["drift_flag"],
        "drift_score": drift["drift_score"],
        "risk_score": risk_result["risk_score"],
        "severity": risk_result["severity"],
        "explanation": explanation_list,
        "decision": result["decision"],
        "details": result["details"],
    })


# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
