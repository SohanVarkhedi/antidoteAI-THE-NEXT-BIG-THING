"""
ANTIDOTE-AI — Enhanced Secure AI Wrapper Backend
Drops into your existing FastAPI project as a replacement for main.py
Requirements (same as before): fastapi uvicorn numpy scikit-learn pydantic
"""

import time
import random
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


# ─────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────

class PredictRequest(BaseModel):
    features: List[float]
    attack_mode: bool = False          # NEW: frontend sends True when in adversarial mode


class PredictResponse(BaseModel):
    prediction: int
    confidence: float
    is_anomaly: bool
    anomaly_score: float               # 0-1 (higher = more anomalous)
    threat_level: str                  # "LOW" | "MEDIUM" | "HIGH"
    threat_score: float                # 0-100
    defense_strength: float            # 0-100
    attack_detected: bool
    pipeline_trace: List[dict]         # NEW: step-by-step decision log
    ensemble_votes: dict               # NEW: which detectors flagged it
    perturbed_features: Optional[List[float]] = None   # if adversarial mode


# ─────────────────────────────────────────────
# SECURE AI WRAPPER
# ─────────────────────────────────────────────

class SecureAIWrapper:
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.15, random_state=42)
        self.one_class_svm = OneClassSVM(nu=0.15, kernel="rbf", gamma="auto")
        self._train_detectors()
        self.request_history: List[dict] = []

    def _train_detectors(self):
        """Train on synthetic normal distribution so detectors have a baseline."""
        rng = np.random.default_rng(42)
        normal_data = rng.normal(loc=0.5, scale=0.2, size=(500, 4)).clip(0, 1)
        scaled = self.scaler.fit_transform(normal_data)
        self.isolation_forest.fit(scaled)
        self.one_class_svm.fit(scaled)

    # ── Validation ──────────────────────────────────────────────────────────
    def validate(self, features: List[float]) -> tuple[bool, str]:
        if len(features) != 4:
            return False, f"Expected 4 features, got {len(features)}"
        for i, v in enumerate(features):
            if not isinstance(v, (int, float)):
                return False, f"Feature {i} is not numeric"
            if v < 0 or v > 1:
                return False, f"Feature {i} = {v:.3f} out of range [0, 1]"
        return True, "OK"

    # ── Preprocessing ────────────────────────────────────────────────────────
    def preprocess(self, features: List[float]) -> np.ndarray:
        arr = np.array(features).reshape(1, -1)
        return self.scaler.transform(arr)

    # ── Ensemble Anomaly Detection ───────────────────────────────────────────
    def detect_anomaly(self, scaled: np.ndarray) -> dict:
        if_score_raw = self.isolation_forest.score_samples(scaled)[0]
        svm_pred = self.one_class_svm.predict(scaled)[0]

        # Z-score on raw (pre-scaled) values — check for extreme outliers
        raw = self.scaler.inverse_transform(scaled)[0]
        z_score = float(np.max(np.abs((raw - 0.5) / 0.2)))
        z_flag = z_score > 2.5

        # Isolation Forest: negative scores → anomaly (convert to 0-1)
        if_anomaly = if_score_raw < -0.1
        if_score_norm = float(np.clip(-if_score_raw / 0.5, 0, 1))

        # SVM: -1 = anomaly, 1 = normal
        svm_anomaly = svm_pred == -1

        votes = int(if_anomaly) + int(svm_anomaly) + int(z_flag)
        is_anomaly = votes >= 2                    # majority vote

        # Final anomaly score 0-1
        anomaly_score = float(np.clip(
            (if_score_norm * 0.5) + (0.25 if svm_anomaly else 0) + (0.25 if z_flag else 0),
            0, 1
        ))

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(anomaly_score, 4),
            "ensemble_votes": {
                "isolation_forest": bool(if_anomaly),
                "one_class_svm": bool(svm_anomaly),
                "z_score_filter": bool(z_flag),
            },
            "votes": votes,
        }

    # ── Mock Model Prediction ────────────────────────────────────────────────
    def predict_model(self, features: List[float], is_anomaly: bool) -> tuple[int, float]:
        """
        Replace this with your real model. For demo: threshold on mean feature.
        Confidence is reduced when anomaly detected.
        """
        mean_val = float(np.mean(features))
        prediction = 1 if mean_val >= 0.5 else 0
        base_conf = 0.85 + 0.14 * random.random()
        confidence = base_conf * (0.65 if is_anomaly else 1.0)
        return prediction, round(min(confidence, 0.99), 4)

    # ── Threat Scoring ───────────────────────────────────────────────────────
    def compute_threat(self, anomaly_score: float, votes: int, attack_mode: bool) -> dict:
        raw = (anomaly_score * 60) + (votes * 10) + (20 if attack_mode else 0)
        threat_score = round(min(raw, 100), 1)

        if threat_score < 30:
            level = "LOW"
        elif threat_score < 65:
            level = "MEDIUM"
        else:
            level = "HIGH"

        defense_strength = round(100 - (threat_score * 0.4), 1)
        return {
            "threat_score": threat_score,
            "threat_level": level,
            "defense_strength": max(defense_strength, 50.0),
        }

    # ── Adversarial Perturbation Generator ──────────────────────────────────
    def perturb(self, features: List[float]) -> List[float]:
        """FGSM-style uniform perturbation — pushes inputs toward decision boundary."""
        rng = np.random.default_rng(int(time.time() * 1000) % 99999)
        eps = rng.uniform(0.15, 0.35, size=len(features))
        direction = rng.choice([-1, 1], size=len(features))
        perturbed = np.clip(np.array(features) + eps * direction, 0, 1)
        return [round(float(v), 4) for v in perturbed]

    # ── Full Defense Pipeline ────────────────────────────────────────────────
    def run(self, request: PredictRequest) -> PredictResponse:
        trace: List[dict] = []
        t0 = time.monotonic()

        features = request.features
        perturbed = None

        # If attack mode, generate adversarial input first
        if request.attack_mode:
            perturbed = self.perturb(features)
            features = perturbed
            trace.append({
                "step": "ADVERSARIAL_GENERATOR",
                "status": "INJECTED",
                "detail": f"Perturbation applied → features shifted by ε∈[0.15,0.35]",
                "ms": round((time.monotonic() - t0) * 1000, 1),
            })

        # 1. Validation
        valid, msg = self.validate(features)
        trace.append({
            "step": "VALIDATOR",
            "status": "PASS" if valid else "FAIL",
            "detail": msg,
            "ms": round((time.monotonic() - t0) * 1000, 1),
        })
        if not valid:
            return PredictResponse(
                prediction=-1, confidence=0.0, is_anomaly=True,
                anomaly_score=1.0, threat_level="HIGH", threat_score=100.0,
                defense_strength=0.0, attack_detected=True,
                pipeline_trace=trace,
                ensemble_votes={},
                perturbed_features=perturbed,
            )

        # 2. Preprocessing
        scaled = self.preprocess(features)
        trace.append({
            "step": "PREPROCESSOR",
            "status": "PASS",
            "detail": f"Scaled to μ=0, σ=1 (StandardScaler)",
            "ms": round((time.monotonic() - t0) * 1000, 1),
        })

        # 3. Ensemble Anomaly Detection
        anom = self.detect_anomaly(scaled)
        trace.append({
            "step": "ANOMALY_DETECTOR",
            "status": "ANOMALY" if anom["is_anomaly"] else "CLEAN",
            "detail": (
                f"Score={anom['anomaly_score']:.3f} | "
                f"IF={'✗' if anom['ensemble_votes']['isolation_forest'] else '✓'} "
                f"SVM={'✗' if anom['ensemble_votes']['one_class_svm'] else '✓'} "
                f"Z={'✗' if anom['ensemble_votes']['z_score_filter'] else '✓'} "
                f"| Votes={anom['votes']}/3"
            ),
            "ms": round((time.monotonic() - t0) * 1000, 1),
        })

        # 4. Model Inference
        pred, conf = self.predict_model(features, anom["is_anomaly"])
        trace.append({
            "step": "MODEL",
            "status": "PASS",
            "detail": f"Prediction={pred} | Confidence={conf:.4f}",
            "ms": round((time.monotonic() - t0) * 1000, 1),
        })

        # 5. Threat Assessment
        threat = self.compute_threat(
            anom["anomaly_score"], anom["votes"], request.attack_mode
        )
        trace.append({
            "step": "THREAT_ENGINE",
            "status": threat["threat_level"],
            "detail": (
                f"Score={threat['threat_score']} | "
                f"Level={threat['threat_level']} | "
                f"Defense={threat['defense_strength']}%"
            ),
            "ms": round((time.monotonic() - t0) * 1000, 1),
        })

        return PredictResponse(
            prediction=pred,
            confidence=conf,
            is_anomaly=anom["is_anomaly"],
            anomaly_score=anom["anomaly_score"],
            threat_level=threat["threat_level"],
            threat_score=threat["threat_score"],
            defense_strength=threat["defense_strength"],
            attack_detected=anom["is_anomaly"] or request.attack_mode,
            pipeline_trace=trace,
            ensemble_votes=anom["ensemble_votes"],
            perturbed_features=perturbed,
        )


# ─────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────

app = FastAPI(title="Antidote AI Firewall", version="2.0.0")
wrapper = SecureAIWrapper()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"status": "ANTIDOTE-AI v2 online", "endpoints": ["/predict", "/health"]}


@app.get("/health")
def health():
    return {"status": "ok", "detectors": ["IsolationForest", "OneClassSVM", "ZScore"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    return wrapper.run(req)