"""Small helpers shared by training and prediction."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ensemble_predict(bundle: dict, X: pd.DataFrame) -> np.ndarray:
    members = bundle["ensemble_members"]
    preds = np.column_stack([bundle["models"][m].predict(X) for m in members])
    weights = np.array([bundle["weights"][m] for m in members], dtype=float)
    return np.clip(preds @ (weights / weights.sum()), 0, None)


def quantile_predict(bundle: dict, X: pd.DataFrame) -> dict[float, np.ndarray]:
    """Quantile forecasts, rescaled by the calibration factors learned from out-of-fold data
    (a simple multiplicative form of conformalised quantile regression)."""
    calib = bundle.get("calibration", {})
    return {q: np.clip(m.predict(X), 0, None) * calib.get(q, 1.0) for q, m in bundle["quantile_models"].items()}


def finalize_intervals(expected, q_low, q_safe, q_high):
    """Quantile models are trained separately so they can occasionally cross each other.
    Enforce  low <= expected <= recommended <= high  so the numbers shown to staff always make sense."""
    expected = np.asarray(expected, float)
    low = np.minimum(q_low, expected)
    rec = np.maximum(q_safe, expected)
    high = np.maximum(q_high, rec)
    return low, rec, high
