from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from secure_ai_wrapper.wrapper import SecureAIWrapper


app = FastAPI()


# ---------- load demo model ----------
X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200)
model.fit(X, y)

secure_model = SecureAIWrapper(model, X_train=X)


# ---------- request schema ----------
class PredictRequest(BaseModel):
    input: list


# ---------- route ----------
@app.post("/predict")
def predict(data: PredictRequest):
    X = np.array(data.input)

    result = secure_model.predict(X)

    return {
        "prediction": int(result.prediction),
        "confidence": result.confidence,
        "is_anomaly": result.is_anomaly
    }