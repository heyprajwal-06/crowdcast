"""Turn a headcount forecast into dishes (portions) and then into ingredient quantities.

This part is plain arithmetic on the recipes in the college config - no ML involved, and that is on purpose:
the ML predicts *how many people*, the recipe book decides *how much rice that means*.
"""
from __future__ import annotations

import math

import pandas as pd

from canteen_ai.config import CollegeConfig


def dish_plan(cfg: CollegeConfig, forecast: pd.DataFrame, column: str = "recommended_prep") -> pd.DataFrame:
    rows = []
    for r in forecast.itertuples():
        for item in cfg.menu.get(r.meal, []):
            portions = int(math.ceil(getattr(r, column) * float(item["portions_per_head"])))
            rows.append({"date": r.date, "meal": r.meal, "dish": item["dish"], "portions": portions})
    return pd.DataFrame(rows, columns=["date", "meal", "dish", "portions"])


def ingredient_plan(cfg: CollegeConfig, dishes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-dish breakdown, per-day shopping list)."""
    buf = 1.0 + cfg.wastage_buffer
    rows = []
    for r in dishes.itertuples():
        for ing, (qty, unit) in cfg.recipes[r.dish].items():
            rows.append(
                {"date": r.date, "meal": r.meal, "dish": r.dish, "ingredient": ing, "unit": unit,
                 "quantity": r.portions * qty * buf}
            )
    detail = pd.DataFrame(rows, columns=["date", "meal", "dish", "ingredient", "unit", "quantity"])
    if detail.empty:
        return detail, detail.copy()
    shopping = detail.groupby(["date", "ingredient", "unit"], as_index=False)["quantity"].sum()
    shopping["quantity"] = [
        math.ceil(q) if u == "pcs" else round(q, 2) for q, u in zip(shopping["quantity"], shopping["unit"])
    ]
    detail["quantity"] = detail["quantity"].round(2)
    return detail, shopping
