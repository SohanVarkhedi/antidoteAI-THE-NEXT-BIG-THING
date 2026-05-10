"""
Evasion Ensemble — Antidote AI
Multi-model evasion detection combining One-Class SVM, Isolation Forest,
and Z-score statistical detection with majority voting.
"""

import numpy as np
import joblib
import os
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


class EvasionEnsemble:
    """Ensemble evasion detector using three anomaly detection techniques."""

    def __init__(self):
        self.svm_model: OneClassSVM | None = None
        self.iforest_model: IsolationForest | None = None
        self.train_mean: np.ndarray | None = None
        self.train_std: np.ndarray | None = None
        self._fitted = False

    # ── Training ──────────────────────────────────────────────────
    def fit(self, X_train: np.ndarray, nu: float = 0.10, zscore_threshold: float = 3.0):
        """
        Fit all three evasion detectors on the clean training data.

        Parameters
        ----------
        X_train : ndarray of shape (n_samples, n_features)
            Clean training features.
        nu : float
            Upper bound on training errors for One-Class SVM.
        zscore_threshold : float
            Z-score threshold for statistical detection (stored for predict).
        """
        X = np.array(X_train, dtype=float)

        # 1 — One-Class SVM
        self.svm_model = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
        self.svm_model.fit(X)

        # 2 — Isolation Forest
        self.iforest_model = IsolationForest(
            contamination=nu, random_state=42, n_estimators=150
        )
        self.iforest_model.fit(X)

        # 3 — Z-score statistics
        self.train_mean = np.mean(X, axis=0)
        self.train_std = np.std(X, axis=0)
        # Avoid division by zero for constant features
        self.train_std[self.train_std == 0] = 1e-10

        self.zscore_threshold = zscore_threshold
        self._fitted = True

    # ── Prediction ────────────────────────────────────────────────
    def predict(self, x: list | np.ndarray, zscore_threshold: float | None = None) -> dict:
        """
        Run all three detectors on a single sample and return voting result.

        Parameters
        ----------
        x : array-like of shape (n_features,)
            Single input sample.
        zscore_threshold : float or None
            Override z-score threshold. Uses training value if None.

        Returns
        -------
        dict with keys:
            evasion_flag   – bool (True if majority vote = anomaly)
            confidence     – float (fraction of detectors flagging, 0.0–1.0)
            svm            – "anomaly" | "normal"
            isolation_forest – "anomaly" | "normal"
            zscore         – "anomaly" | "normal"
            votes_for      – int (number of detectors flagging anomaly)
            votes_total    – int (total detectors, always 3)
            svm_score      – float (SVM decision score)
            iforest_score  – float (Isolation Forest anomaly score)
            zscore_max     – float (maximum absolute z-score across features)
        """
        if not self._fitted:
            return {
                "evasion_flag": False,
                "confidence": 0.0,
                "svm": "normal",
                "isolation_forest": "normal",
                "zscore": "normal",
                "votes_for": 0,
                "votes_total": 3,
                "svm_score": 0.0,
                "iforest_score": 0.0,
                "zscore_max": 0.0,
            }

        threshold = zscore_threshold or self.zscore_threshold
        x_arr = np.array(x, dtype=float).reshape(1, -1)

        votes = 0

        # 1 — One-Class SVM
        svm_label = int(self.svm_model.predict(x_arr)[0])  # 1=normal, -1=anomaly
        svm_score = float(self.svm_model.decision_function(x_arr)[0])
        svm_result = "anomaly" if svm_label == -1 else "normal"
        if svm_label == -1:
            votes += 1

        # 2 — Isolation Forest
        iforest_label = int(self.iforest_model.predict(x_arr)[0])  # 1=normal, -1=anomaly
        iforest_score = float(self.iforest_model.decision_function(x_arr)[0])
        iforest_result = "anomaly" if iforest_label == -1 else "normal"
        if iforest_label == -1:
            votes += 1

        # 3 — Z-score
        z_scores = np.abs((x_arr.flatten()[:len(self.train_mean)] - self.train_mean) / self.train_std)
        zscore_max = float(np.max(z_scores))
        zscore_flag = bool(np.any(z_scores > threshold))
        zscore_result = "anomaly" if zscore_flag else "normal"
        if zscore_flag:
            votes += 1

        # Voting
        total = 3
        evasion_flag = votes >= 2  # majority
        confidence = round(votes / total, 2)

        return {
            "evasion_flag": evasion_flag,
            "confidence": confidence,
            "svm": svm_result,
            "isolation_forest": iforest_result,
            "zscore": zscore_result,
            "votes_for": votes,
            "votes_total": total,
            "svm_score": round(svm_score, 4),
            "iforest_score": round(iforest_score, 4),
            "zscore_max": round(zscore_max, 4),
        }

    # ── Persistence ──────────────────────────────────────────────
    def save(self, path: str | None = None):
        """Save all three models to disk."""
        path = path or os.path.join(MODEL_DIR, "evasion_ensemble.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "svm": self.svm_model,
            "iforest": self.iforest_model,
            "mean": self.train_mean,
            "std": self.train_std,
            "zscore_threshold": self.zscore_threshold,
        }
        joblib.dump(data, path)

    def load(self, path: str | None = None) -> bool:
        """Load all three models from disk."""
        path = path or os.path.join(MODEL_DIR, "evasion_ensemble.pkl")
        if os.path.exists(path):
            data = joblib.load(path)
            self.svm_model = data["svm"]
            self.iforest_model = data["iforest"]
            self.train_mean = data["mean"]
            self.train_std = data["std"]
            self.zscore_threshold = data.get("zscore_threshold", 3.0)
            self._fitted = True
            return True
        return False
