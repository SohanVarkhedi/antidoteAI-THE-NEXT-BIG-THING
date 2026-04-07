"""
Evasion Detector — Antidote AI
Uses OneClassSVM trained on clean features to flag adversarial inputs at inference.
"""

import numpy as np
from sklearn.svm import OneClassSVM
import joblib, os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class EvasionDetector:
    """Wraps a OneClassSVM to detect out-of-distribution / evasion inputs."""

    def __init__(self):
        self.model: OneClassSVM | None = None
        self._fitted = False

    # ── Training ──────────────────────────────────────────────────
    def fit(self, X_train: np.ndarray, nu: float = 0.10):
        """
        Fit OneClassSVM on the *clean* training features.

        Parameters
        ----------
        X_train : ndarray of shape (n_samples, n_features)
        nu : float – upper bound on the fraction of training errors.
        """
        self.model = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
        self.model.fit(X_train)
        self._fitted = True

    # ── Prediction ────────────────────────────────────────────────
    def predict(self, x: np.ndarray) -> dict:
        """
        Predict whether a single sample is normal or adversarial.

        Parameters
        ----------
        x : ndarray of shape (1, n_features)

        Returns
        -------
        dict  {"evasion_flag": bool, "decision_score": float}
        """
        if not self._fitted:
            raise RuntimeError("EvasionDetector not fitted yet — train first.")

        x = np.array(x).reshape(1, -1)
        label = int(self.model.predict(x)[0])        # 1 = normal, -1 = anomaly
        score = float(self.model.decision_function(x)[0])

        return {
            "evasion_flag": label == -1,
            "decision_score": round(score, 4),
        }

    # ── Persistence ───────────────────────────────────────────────
    def save(self, path: str | None = None):
        path = path or os.path.join(MODEL_DIR, "evasion_model.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: str | None = None):
        path = path or os.path.join(MODEL_DIR, "evasion_model.pkl")
        if os.path.exists(path):
            self.model = joblib.load(path)
            self._fitted = True
            return True
        return False
