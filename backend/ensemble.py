"""
Ensemble Decision — Antidote AI
Combines poisoning, evasion, and model prediction into a final verdict + risk score.
"""


def ensemble_decision(
    poisoning_flag: bool,
    evasion_flag: bool,
    model_prediction: int,
    evasion_score: float = 0.0,
    model_confidence: float = 1.0,
) -> dict:
    """
    Produce the final security decision.

    Decision logic
    --------------
    - poisoning_flag  → BLOCK  (data was tampered with)
    - evasion_flag    → FLAG   (input looks adversarial)
    - model_prediction == 1 → BLOCK  (model says malicious)
    - otherwise       → ALLOW

    Returns
    -------
    dict with keys:
        decision   – "BLOCK" | "FLAG" | "ALLOW"
        risk_score – 0-100 integer
        details    – human-readable explanation
    """

    # ── Decision ──────────────────────────────────────────
    if poisoning_flag:
        decision = "BLOCK"
        details = "Poisoning detected in dataset — input blocked."
    elif evasion_flag:
        decision = "FLAG"
        details = "Evasion attempt detected — input flagged for review."
    elif model_prediction == 1:
        decision = "BLOCK"
        details = "Model predicts malicious class — input blocked."
    else:
        decision = "ALLOW"
        details = "All checks passed — input allowed."

    # ── Risk score (0–100) ────────────────────────────────
    risk = 0.0
    if poisoning_flag:
        risk += 45
    if evasion_flag:
        risk += 30
    if model_prediction == 1:
        risk += 25

    # Modulate by evasion confidence
    risk += abs(evasion_score) * 10

    # Lower confidence = higher risk
    risk += (1.0 - model_confidence) * 15

    risk_score = int(min(max(risk, 0), 100))

    return {
        "decision": decision,
        "risk_score": risk_score,
        "details": details,
    }
