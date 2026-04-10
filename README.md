# 🛡️ Antidote-AI

### Adversarially Robust AI Security Middleware

Antidote-AI is a multi-layer AI security middleware designed to protect machine learning systems against **data poisoning attacks**, **evasion attacks**, and **distribution drift**.
It sanitizes training data, monitors inference inputs, and provides explainable, risk-based decisions.

---

# 🚀 Features

## 🔐 Training-Time Protection

* Dataset upload (CSV / Excel)
* Poisoning detection using Isolation Forest
* Automatic removal of suspicious rows
* Clean dataset generation
* Download cleaned dataset

## 🧠 Model Training

* Ensemble base models:

  * RandomForest
  * LogisticRegression
  * GradientBoosting
* Training only on sanitized data
* Model persistence

## 🚨 Inference-Time Protection

* Input validation
* Multi-model evasion detection:

  * One-Class SVM
  * Isolation Forest
  * Z-score statistical detection
* Ensemble anomaly decision

## 🌊 Drift Detection

* Detects distribution shift between training and incoming data
* Statistical comparison using KS-test

## 📊 Risk Scoring Engine

* Weighted risk calculation
* Severity classification:

  * LOW
  * MEDIUM
  * HIGH

## 📥 Clean Dataset Download

* Users can download cleaned dataset after poisoning removal

## 📝 Logging System

* Poisoning attempts
* Evasion attempts
* Final decisions
* Risk scores

## 🔍 Explainability

* Displays reason for blocking
* Feature deviation explanation

---

# 🧠 System Architecture

### Training Phase

Dataset Upload → Poisoning Detection → Clean Dataset → Model Training

### Inference Phase

Input → Validator → Evasion Ensemble → Drift Detection → Model Ensemble → Risk Engine → Decision

---

# 🧩 Evasion Detection Ensemble

Antidote-AI uses multiple anomaly detectors:

* One-Class SVM (boundary anomalies)
* Isolation Forest (outlier detection)
* Z-score detection (statistical deviation)

Voting-based ensemble improves detection accuracy.

---

# 📊 Risk Score Formula

```
risk =
0.25 * poisoning_score +
0.25 * evasion_score +
0.25 * drift_score +
0.25 * model_confidence
```

Output:

* Risk Score (0–100)
* Severity Level
* Final Decision

---

# 📁 Folder Structure

```
antidote-ai/
│
├── backend/
│   ├── poisoning_detector.py
│   ├── evasion_ensemble.py
│   ├── drift_detector.py
│   ├── validator.py
│   ├── ensemble_models.py
│   ├── risk_engine.py
│   ├── logger.py
│   └── explainability.py
│
├── models/
│
├── uploads/
│
├── logs/
│
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── script.js
│
├── app.py
└── requirements.txt
```

---

# 🔌 API Endpoints

| Method | Endpoint          | Description                         |
| ------ | ----------------- | ----------------------------------- |
| POST   | /upload           | Upload dataset and detect poisoning |
| GET    | /download-cleaned | Download sanitized dataset          |
| POST   | /train            | Train ensemble models               |
| POST   | /predict          | Secure prediction pipeline          |

---

# 🖥️ UI Sections

1. Hero Section
2. Dataset Upload Panel
3. Train Model Panel
4. Inference Input Panel
5. Security Dashboard

Dashboard displays:

* Poisoning risk
* Evasion risk
* Drift status
* Model prediction
* Risk score
* Final decision

---

# 🧰 Tech Stack

### Backend

* Python
* Flask
* scikit-learn
* pandas
* numpy
* scipy
* joblib

### Frontend

* HTML
* CSS (Bauhaus Design System)
* JavaScript

### Models

* IsolationForest
* OneClassSVM
* RandomForest
* LogisticRegression
* GradientBoosting

---

# ⚙️ Installation

### Clone repository

```
git clone <repo-url>
cd antidote-ai
```

### Install dependencies

```
pip install -r requirements.txt
```

### Run server

```
python app.py
```

### Open in browser

```
http://localhost:5000
```

---

# 📥 Dataset Format

```
feature1, feature2, feature3, label
```

Example:

```
0.2,0.5,0.3,0
0.25,0.45,0.35,1
```

---

# 🛡️ Decision Logic

* **BLOCK** → High risk malicious input
* **FLAG** → Suspicious input detected
* **ALLOW** → Safe input

---

# 🌟 Key Capabilities

✔ Training data sanitization
✔ Multi-model evasion detection
✔ Drift monitoring
✔ Ensemble model prediction
✔ Risk-based decision making
✔ Clean dataset export
✔ Explainable security output
✔ Logging & monitoring

---

# 🎯 Project Goal

To provide a **robust AI middleware** that secures machine learning systems against adversarial threats across both training and inference phases.

---

# 👨‍💻 Authors

Sohan Varkhedi
