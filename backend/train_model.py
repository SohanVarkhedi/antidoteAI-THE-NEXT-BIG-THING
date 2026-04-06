"""
ANTIDOTE-AI — Model Training Script
Dataset: Wisconsin Breast Cancer (sklearn built-in)
Model: RandomForestClassifier

Run this ONCE before starting the API server:
    python train_model.py

Outputs:
    model.pkl      — trained classifier
    scaler.pkl     — fitted StandardScaler
    features.json  — feature names & stats for the dashboard
"""

import json, joblib, numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# ── Load dataset ────────────────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target          # 569 samples, 30 features
# 0 = malignant (threat) | 1 = benign (safe)

# ── Use first 10 features (most interpretable for sliders) ──────────────────
FEATURE_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
FEATURE_NAMES   = [data.feature_names[i] for i in FEATURE_INDICES]
X = X[:, FEATURE_INDICES]

# ── Train / test split ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Scale ───────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── Train model ─────────────────────────────────────────────────────────────
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42,
    class_weight="balanced",
)
clf.fit(X_train_s, y_train)

# ── Evaluate ─────────────────────────────────────────────────────────────────
preds = clf.predict(X_test_s)
acc   = accuracy_score(y_test, preds)
print(f"\n{'='*52}")
print(f"  Breast Cancer Dataset — RandomForest")
print(f"  Test accuracy : {acc*100:.2f}%")
print(f"{'='*52}")
print(classification_report(y_test, preds, target_names=data.target_names))

# ── Save artefacts ───────────────────────────────────────────────────────────
joblib.dump(clf,    "model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Per-feature min/max in ORIGINAL (unscaled) space for slider bounds
feature_meta = []
for i, name in enumerate(FEATURE_NAMES):
    col = X[:, i]
    feature_meta.append({
        "index": i,
        "name":  name,
        "min":   float(np.percentile(col, 1)),
        "max":   float(np.percentile(col, 99)),
        "mean":  float(np.mean(col)),
    })

with open("features.json", "w") as f:
    json.dump({
        "features":      feature_meta,
        "target_names":  list(data.target_names),   # ["malignant", "benign"]
        "n_features":    len(FEATURE_NAMES),
        "dataset":       "Breast Cancer Wisconsin",
    }, f, indent=2)

print("Saved: model.pkl | scaler.pkl | features.json")
print("Run:   uvicorn main:app --reload")