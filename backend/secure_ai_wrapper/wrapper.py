from dataclasses import dataclass
from typing import Any, Optional
import numpy as np

from .validator import InputValidator
from .preprocessor import Preprocessor
from .anomaly_detector import AnomalyDetector
from .logger import SecureLogger


@dataclass
class PredictionResult:
    prediction: Any
    confidence: Optional[float]
    is_anomaly: bool


class SecureAIWrapper:
    """
    Orchestrates validation → preprocessing → anomaly detection → prediction.
    """

    def __init__(self, model, X_train=None):
        self.model = model

        # modules
        self.validator = InputValidator()
        self.preprocessor = Preprocessor()
        self.anomaly_detector = AnomalyDetector()
        self.logger = SecureLogger()

        # fit modules if training data available
        if X_train is not None:
            self.preprocessor.fit(X_train)
            X_processed = self.preprocessor.transform(X_train)
            self.anomaly_detector.fit(X_processed)

    def predict(self, X) -> PredictionResult:
        # validate
        X = self.validator.validate(X)

        # preprocess
        X = self.preprocessor.transform(X)

        # anomaly check
        is_anomaly = self.anomaly_detector.check(X)

        if is_anomaly:
            self.logger.warning("Anomalous input detected")

        # prediction
        prediction = self.model.predict(X)

        # confidence
        confidence = None
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            confidence = float(np.max(probs))

        return PredictionResult(
            prediction=prediction[0],
            confidence=confidence,
            is_anomaly=is_anomaly
        )