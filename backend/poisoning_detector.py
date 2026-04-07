"""
Poisoning Detector — Antidote AI
Uses IsolationForest to detect poisoned / outlier rows in training data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


def detect_poisoning(df: pd.DataFrame, contamination: float = 0.10):
    """
    Analyse a DataFrame for poisoned (anomalous) rows.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (features + optional label column).
    contamination : float
        Expected proportion of outliers.

    Returns
    -------
    dict with keys:
        total_rows       – original row count
        suspicious_rows  – number of detected outliers
        cleaned_rows     – rows remaining after removal
        suspicious_indices – list of row indices flagged
        cleaned_df       – DataFrame with outliers removed
    """
    # Separate numeric columns only for anomaly detection
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.empty:
        return {
            "total_rows": len(df),
            "suspicious_rows": 0,
            "cleaned_rows": len(df),
            "suspicious_indices": [],
            "cleaned_df": df.copy(),
        }

    # Replace NaN/inf so IsolationForest doesn't choke
    numeric_clean = numeric_df.replace([np.inf, -np.inf], np.nan).fillna(
        numeric_df.median()
    )

    iso = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=150,
    )
    preds = iso.fit_predict(numeric_clean)  # 1 = inlier, -1 = outlier

    suspicious_mask = preds == -1
    suspicious_indices = list(df.index[suspicious_mask].tolist())

    cleaned_df = df[~suspicious_mask].reset_index(drop=True)

    return {
        "total_rows": int(len(df)),
        "suspicious_rows": int(suspicious_mask.sum()),
        "cleaned_rows": int(len(cleaned_df)),
        "suspicious_indices": suspicious_indices,
        "cleaned_df": cleaned_df,
    }
