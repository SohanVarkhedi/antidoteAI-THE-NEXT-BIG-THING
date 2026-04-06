"""
ANTIDOTE-AI — Secure AI Wrapper Backend v3 (Real Model)
Dataset: Breast Cancer Wisconsin (sklearn)
Defense: Ensemble anomaly detection + threat scoring + decision trace

Setup:
    pip install fastapi uvicorn numpy scikit-learn pydantic joblib
    python train_model.py        <- run ONCE to generate model.pkl / scaler.pkl
    uvicorn main:app --reload
"""

import os, time, json, joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# ─────────────────────────────────────────────
# LOAD REAL MODEL + SCALER
# ─────────────────────────────────────────────

def _load(path, label):
    if not os.path.exists(path):
        raise RuntimeError(f"{label} not found at '{path}'. Run: python train_model.py")
    return joblib.load(path)

clf    = _load("model.pkl",  "Trained model")
scaler = _load("scaler.pkl", "Feature scaler")

with open("features.json") as f:
    FEATURE_META = json.load(f)

N_FEATURES   = FEATURE_META["n_features"]
TARGET_NAMES = FEATURE_META["target_names"]   # ["malignant", "benign"]
FEATURE_INFO = FEATURE_META["features"]


# ─────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────

class PredictRequest(BaseModel):
    features: List[float]
    attack_mode: bool = False


class PredictResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float
    is_anomaly: bool
    anomaly_score: float
    threat_level: str
    threat_score: float
    defense_strength: float
    attack_detected: bool
    pipeline_trace: List[dict]
    ensemble_votes: dict
    perturbed_features: Optional[List[float]] = None
    feature_names: List[str]


# ─────────────────────────────────────────────
# SECURE AI WRAPPER
# ─────────────────────────────────────────────

class SecureAIWrapper:
    def __init__(self, classifier, fitted_scaler, feature_info: list):
        self.clf    = classifier
        self.scaler = fitted_scaler
        self.finfo  = feature_info

        rng = np.random.default_rng(42)
        synthetic = np.column_stack([
            rng.normal(loc=fi["mean"], scale=(fi["max"] - fi["min"]) / 6, size=600)
            for fi in self.finfo
        ])
        scaled_syn = self.scaler.transform(synthetic)

        self.iforest = IsolationForest(contamination=0.12, random_state=42)
        self.iforest.fit(scaled_syn)

        self.oc_svm = OneClassSVM(nu=0.12, kernel="rbf", gamma="scale")
        self.oc_svm.fit(scaled_syn)

        self._ref_mean = np.array([fi["mean"] for fi in self.finfo])
        self._ref_std  = np.array([(fi["max"] - fi["min"]) / 4 for fi in self.finfo])

    def validate(self, features):
        if len(features) != N_FEATURES:
            return False, f"Expected {N_FEATURES} features, got {len(features)}"
        for v, fi in zip(features, self.finfo):
            lo, hi = fi["min"] * 0.5, fi["max"] * 2.0
            if not (lo <= v <= hi):
                return False, f"{fi['name']} = {v:.4f} out of range [{lo:.3f}, {hi:.3f}]"
        return True, "All features validated"

    def preprocess(self, features):
        return self.scaler.transform(np.array(features).reshape(1, -1))

    def detect_anomaly(self, scaled, raw):
        if_raw  = float(self.iforest.score_samples(scaled)[0])
        if_flag = if_raw < -0.1
        if_norm = float(np.clip(-if_raw / 0.55, 0, 1))

        svm_flag = self.oc_svm.predict(scaled)[0] == -1

        z      = np.abs((np.array(raw) - self._ref_mean) / (self._ref_std + 1e-9))
        z_max  = float(np.max(z))
        z_flag = z_max > 2.8

        votes = int(if_flag) + int(svm_flag) + int(z_flag)
        score = float(np.clip(if_norm * 0.5 + (0.25 if svm_flag else 0) + (0.25 if z_flag else 0), 0, 1))

        return {
            "is_anomaly":    votes >= 2,
            "anomaly_score": round(score, 4),
            "ensemble_votes": {
                "isolation_forest": bool(if_flag),
                "one_class_svm":    bool(svm_flag),
                "z_score_filter":   bool(z_flag),
            },
            "votes": votes,
            "z_max": round(z_max, 3),
        }

    def predict_model(self, scaled, is_anomaly):
        pred  = int(self.clf.predict(scaled)[0])
        proba = self.clf.predict_proba(scaled)[0]
        conf  = float(np.max(proba)) * (0.70 if is_anomaly else 1.0)
        return pred, round(min(conf, 0.9999), 4)

    def compute_threat(self, pred, anomaly_score, votes, attack_mode):
        raw   = (anomaly_score * 55) + (votes * 8) + (25 if pred == 0 else 0) + (18 if attack_mode else 0)
        score = round(min(raw, 100), 1)
        level = "HIGH" if score >= 65 else "MEDIUM" if score >= 32 else "LOW"
        return {
            "threat_score":      score,
            "threat_level":      level,
            "defense_strength":  round(max(100 - score * 0.42, 48.0), 1),
        }

    def perturb(self, features):
        rng  = np.random.default_rng(int(time.time() * 1000) % 99991)
        arr  = np.array(features)
        eps  = rng.uniform(0.08, 0.22) * (self._ref_std + 1e-9)
        sign = rng.choice([-1, 1], size=len(features))
        lo   = np.array([fi["min"] * 0.5 for fi in self.finfo])
        hi   = np.array([fi["max"] * 2.0 for fi in self.finfo])
        return [round(float(v), 5) for v in np.clip(arr + eps * sign, lo, hi)]

    def run(self, req: PredictRequest) -> PredictResponse:
        trace, t0, features, perturbed = [], time.monotonic(), list(req.features), None
        ms = lambda: round((time.monotonic() - t0) * 1000, 2)

        if req.attack_mode:
            perturbed = self.perturb(features)
            features  = perturbed
            trace.append({"step": "ADVERSARIAL_GENERATOR", "status": "INJECTED",
                           "detail": f"epsilon-perturbation applied across {N_FEATURES} features", "ms": ms()})

        valid, msg = self.validate(features)
        trace.append({"step": "VALIDATOR", "status": "PASS" if valid else "FAIL", "detail": msg, "ms": ms()})
        if not valid:
            return PredictResponse(
                prediction=-1, prediction_label="REJECTED", confidence=0.0,
                is_anomaly=True, anomaly_score=1.0, threat_level="HIGH",
                threat_score=100.0, defense_strength=0.0, attack_detected=True,
                pipeline_trace=trace, ensemble_votes={}, perturbed_features=perturbed,
                feature_names=[fi["name"] for fi in self.finfo],
            )

        scaled = self.preprocess(features)
        trace.append({"step": "PREPROCESSOR", "status": "PASS",
                       "detail": "StandardScaler (fit on breast cancer training set)", "ms": ms()})

        anom = self.detect_anomaly(scaled, features)
        trace.append({"step": "ANOMALY_DETECTOR", "status": "ANOMALY" if anom["is_anomaly"] else "CLEAN",
                       "detail": (
                           f"Score={anom['anomaly_score']:.4f} | "
                           f"IF={'FAIL' if anom['ensemble_votes']['isolation_forest'] else 'OK'} "
                           f"SVM={'FAIL' if anom['ensemble_votes']['one_class_svm'] else 'OK'} "
                           f"Z={'FAIL' if anom['ensemble_votes']['z_score_filter'] else 'OK'} "
                           f"z_max={anom['z_max']} votes={anom['votes']}/3"
                       ), "ms": ms()})

        pred, conf = self.predict_model(scaled, anom["is_anomaly"])
        label = TARGET_NAMES[pred] if pred < len(TARGET_NAMES) else str(pred)
        trace.append({"step": "MODEL", "status": "PASS",
                       "detail": f"RandomForest -> {label.upper()} | conf={conf:.4f}", "ms": ms()})

        threat = self.compute_threat(pred, anom["anomaly_score"], anom["votes"], req.attack_mode)
        trace.append({"step": "THREAT_ENGINE", "status": threat["threat_level"],
                       "detail": (f"score={threat['threat_score']} level={threat['threat_level']} "
                                  f"defense={threat['defense_strength']}%"), "ms": ms()})

        return PredictResponse(
            prediction=pred, prediction_label=label.upper(), confidence=conf,
            is_anomaly=anom["is_anomaly"], anomaly_score=anom["anomaly_score"],
            threat_level=threat["threat_level"], threat_score=threat["threat_score"],
            defense_strength=threat["defense_strength"],
            attack_detected=anom["is_anomaly"] or req.attack_mode,
            pipeline_trace=trace, ensemble_votes=anom["ensemble_votes"],
            perturbed_features=perturbed,
            feature_names=[fi["name"] for fi in self.finfo],
        )


# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app     = FastAPI(title="Antidote AI Firewall", version="3.0.0")
wrapper = SecureAIWrapper(clf, scaler, FEATURE_INFO)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def root():
    return {"status": "ANTIDOTE-AI v3", "dataset": FEATURE_META["dataset"],
            "n_features": N_FEATURES, "targets": TARGET_NAMES}

@app.get("/health")
def health():
    return {"status": "ok", "model": "RandomForestClassifier", "n_features": N_FEATURES}

@app.get("/features")
def features():
    return FEATURE_META

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.features) != N_FEATURES:
        raise HTTPException(422, f"Send exactly {N_FEATURES} features")
    return wrapper.run(req)