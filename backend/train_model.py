"""
Model Trainer — Antidote AI
Trains a RandomForestClassifier on the cleaned dataset and persists it.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def train_model(cleaned_df: pd.DataFrame, target_column: str = "target") -> dict:
    """
    Train a RandomForestClassifier on the cleaned data.

    Parameters
    ----------
    cleaned_df : pd.DataFrame
        Cleaned dataset with features and a target column.
    target_column : str
        Name of the label column.

    Returns
    -------
    dict  {"accuracy": float, "n_samples": int, "n_features": int, "model_path": str}
    """
    if target_column not in cleaned_df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    X = cleaned_df.drop(columns=[target_column]).select_dtypes(include=[np.number])
    y = cleaned_df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "base_model.pkl")
    joblib.dump(clf, model_path)

    # Save feature count for inference validation
    meta_path = os.path.join(MODEL_DIR, "meta.pkl")
    meta = {
        "n_features": int(X.shape[1]),
        "feature_names": list(X.columns),
        "target_column": target_column,
    }
    joblib.dump(meta, meta_path)

    return {
        "accuracy": round(acc, 4),
        "n_samples": int(len(X_train)),
        "n_features": int(X.shape[1]),
        "model_path": model_path,
    }