"""
Input Validator — Antidote AI
Checks missing values, value ranges, and type consistency before inference.
"""

import numpy as np


def validate_input(features: list, feature_count: int, feature_ranges: dict | None = None) -> dict:
    """
    Validate a single inference input vector.

    Parameters
    ----------
    features : list
        Raw feature values from the user.
    feature_count : int
        Expected number of features.
    feature_ranges : dict | None
        Optional dict mapping feature index → (min, max) for range checks.

    Returns
    -------
    dict  {"valid": bool, "errors": list[str]}
    """
    errors: list[str] = []

    # 1 ── Length check
    if len(features) != feature_count:
        errors.append(f"Expected {feature_count} features, got {len(features)}.")
        return {"valid": False, "errors": errors}

    # 2 ── Missing / non-numeric check
    for i, v in enumerate(features):
        if v is None:
            errors.append(f"Feature {i} is missing (None).")
        elif isinstance(v, (int, float)):
            if np.isnan(v) or np.isinf(v):
                errors.append(f"Feature {i} is NaN or Inf.")
        else:
            try:
                float(v)
            except (ValueError, TypeError):
                errors.append(f"Feature {i} has invalid type: {type(v).__name__}.")

    # 3 ── Range check (if ranges supplied)
    if feature_ranges and not errors:
        for idx, (lo, hi) in feature_ranges.items():
            idx = int(idx)
            if idx < len(features):
                val = float(features[idx])
                if val < lo or val > hi:
                    errors.append(
                        f"Feature {idx} value {val:.4f} out of range [{lo:.4f}, {hi:.4f}]."
                    )

    return {"valid": len(errors) == 0, "errors": errors}
