"""Metrics. We report numbers a canteen owner can understand, not just R^2."""
from __future__ import annotations

import numpy as np


def regression_metrics(y, p) -> dict[str, float]:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    err = p - y
    nz = y > 0
    return {
        "mae": float(np.mean(np.abs(err))),  # "on average we're off by N students"
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "wape_pct": float(np.sum(np.abs(err)) / max(np.sum(y), 1e-9) * 100),  # total error as % of total demand
        "within_10pct": float(np.mean(np.abs(err[nz]) <= 0.10 * y[nz]) * 100) if nz.any() else float("nan"),
        "bias": float(np.mean(err)),  # >0 = over-predicts on average
    }


def business_impact(y, prep) -> dict[str, float]:
    """What would have happened if the kitchen had cooked `prep` portions every time?"""
    y = np.asarray(y, float)
    prep = np.asarray(prep, float)
    over = np.clip(prep - y, 0, None)
    under = np.clip(y - prep, 0, None)
    return {
        "wastage_pct": float(over.sum() / max(prep.sum(), 1e-9) * 100),  # % of cooked food left over
        "shortage_services_pct": float(np.mean(y > prep) * 100),  # % of meal services that ran out
        "unmet_demand_pct": float(under.sum() / max(y.sum(), 1e-9) * 100),  # % of students who found no food
    }


def interval_metrics(y, low, safe, high, safety_q: float, interval: tuple[float, float]) -> dict[str, float]:
    y = np.asarray(y, float)
    return {
        "interval_nominal_pct": float((interval[1] - interval[0]) * 100),
        "interval_coverage_pct": float(np.mean((y >= low) & (y <= high)) * 100),
        "safety_nominal_pct": float(safety_q * 100),
        "safety_coverage_pct": float(np.mean(y <= safe) * 100),
        "avg_interval_width": float(np.mean(np.asarray(high) - np.asarray(low))),
    }
