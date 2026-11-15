"""Turn a room-wise timetable into per-(weekday, meal) 'how many students are free to eat' signals."""
from __future__ import annotations

import pandas as pd

from canteen_ai.config import CollegeConfig, hm_to_min

TT_FEATURES = ["tt_load_day", "tt_load_before_meal", "tt_load_during_meal"]


def timetable_features(tt: pd.DataFrame, cfg: CollegeConfig) -> pd.DataFrame:
    """One row per (day_of_week, meal) with:

    tt_load_day          expected students sitting in lectures that day  (sum of capacity x attendance)
    tt_load_before_meal  students whose class ends shortly before the meal starts (they walk to the canteen)
    tt_load_during_meal  students who are in class while the canteen is serving (they can't eat now)
    """
    t = tt.copy()
    t["start_min"] = t["start_time"].map(hm_to_min)
    t["end_min"] = t["end_time"].map(hm_to_min)
    t["seats"] = t["room_capacity"] * t["avg_attendance"]
    rows = []
    for dow, g in t.groupby("day_of_week"):
        load_day = float(g["seats"].sum())
        for meal, (m_start, m_end) in cfg.meals.items():
            before = g[(g["end_min"] >= m_start - 90) & (g["end_min"] <= m_start + 15)]["seats"].sum()
            during = g[(g["start_min"] < m_end) & (g["end_min"] > m_start)]["seats"].sum()
            rows.append(
                {
                    "day_of_week": int(dow),
                    "meal": meal,
                    "tt_load_day": load_day,
                    "tt_load_before_meal": float(before),
                    "tt_load_during_meal": float(during),
                }
            )
    return pd.DataFrame(rows)
