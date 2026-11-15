"""Feature engineering: turns raw CSVs into one model-ready table (one row per date x meal).

The SAME functions are used for training and for forecasting, which is what prevents the classic
"works in the notebook, breaks in production" bug (training/serving skew).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from canteen_ai.config import CollegeConfig
from canteen_ai.data.loader import CollegeData
from canteen_ai.features.timetable import TT_FEATURES, timetable_features

CALENDAR_FEATURES = ["meal_idx", "dow", "month", "day_of_month", "week_of_year"]
WEATHER_FEATURES = ["temp_c", "rain_mm"]
HISTORY_FEATURES = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "roll_mean_7", "roll_mean_14", "roll_mean_28", "roll_std_7",
    "dow_mean_4w", "trend_7_28",
]
WARMUP_DAYS = 28  # first N days have no history to learn from


def climatology(weather: pd.DataFrame) -> pd.DataFrame:
    w = weather.copy()
    w["month"] = w["date"].dt.month
    return w.groupby("month", as_index=False)[["temp_c", "rain_mm"]].mean()


def assemble(
    data: CollegeData,
    cfg: CollegeConfig,
    horizon_dates=None,
    weather_future: pd.DataFrame | None = None,
    overrides: dict | None = None,
) -> pd.DataFrame:
    """Full (meal x date) grid from first history day to the last horizon day.

    footfall is NaN for days without actuals (future days, or days the canteen was closed).
    """
    hist = data.footfall
    start = hist["date"].min()
    end = hist["date"].max()
    if horizon_dates is not None and len(horizon_dates):
        end = max(end, pd.Timestamp(max(horizon_dates)))
    dates = pd.date_range(start, end, freq="D")
    meals = cfg.meal_names

    grid = pd.MultiIndex.from_product([meals, dates], names=["meal", "date"]).to_frame(index=False)
    grid["meal_idx"] = grid["meal"].map({m: i for i, m in enumerate(meals)}).astype(int)
    grid = grid.merge(hist[["date", "meal", "footfall"]], on=["date", "meal"], how="left")
    grid["footfall"] = grid["footfall"].astype(float)

    # ---- calendar
    grid["dow"] = grid["date"].dt.dayofweek
    grid["month"] = grid["date"].dt.month
    grid["day_of_month"] = grid["date"].dt.day
    grid["week_of_year"] = grid["date"].dt.isocalendar().week.astype(int)

    # ---- timetable
    if data.timetable is not None:
        ttf = timetable_features(data.timetable, cfg).rename(columns={"day_of_week": "dow"})
        grid = grid.merge(ttf, on=["dow", "meal"], how="left")
        grid[TT_FEATURES] = grid[TT_FEATURES].fillna(0.0)  # no classes that day -> 0

    # ---- weather (observed > forecast > monthly climatology)
    if data.weather is not None:
        parts = [data.weather[["date", "temp_c", "rain_mm"]].assign(weather_source="observed")]
        if weather_future is not None and len(weather_future):
            parts.append(weather_future[["date", "temp_c", "rain_mm"]].assign(weather_source="forecast"))
        w = pd.concat(parts, ignore_index=True).drop_duplicates("date", keep="last")
        wf = pd.DataFrame({"date": dates}).merge(w, on="date", how="left")
        wf["month"] = wf["date"].dt.month
        clim = climatology(data.weather).rename(columns={"temp_c": "c_temp", "rain_mm": "c_rain"})
        wf = wf.merge(clim, on="month", how="left")
        wf["weather_source"] = wf["weather_source"].fillna("climatology")
        wf["temp_c"] = wf["temp_c"].fillna(wf["c_temp"])
        wf["rain_mm"] = wf["rain_mm"].fillna(wf["c_rain"])
        grid = grid.merge(wf[["date", "temp_c", "rain_mm", "weather_source"]], on="date", how="left")

    # ---- events -> 0/1 flags
    for et in cfg.event_types:
        if data.events is not None:
            flagged = set(data.events.loc[data.events["event_type"] == et, "date"])
            grid[f"evt_{et}"] = grid["date"].isin(flagged).astype(int)
        else:
            grid[f"evt_{et}"] = 0

    # ---- what-if overrides, e.g. {Timestamp('2026-10-24'): {'rain_mm': 30, 'evt_exam': 1}}
    if overrides:
        for d, changes in overrides.items():
            mask = grid["date"] == pd.Timestamp(d).normalize()
            for col, val in changes.items():
                if col in grid.columns:
                    grid.loc[mask, col] = val

    return grid.sort_values(["meal_idx", "date"]).reset_index(drop=True)


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag / rolling features per meal. Uses ONLY information from before each row's date."""
    out = df.copy()
    g = out.groupby("meal", sort=False)["footfall"]
    for k in (1, 7, 14, 21, 28):
        out[f"lag_{k}"] = g.shift(k)

    shifted = g.shift(1)  # yesterday and earlier -> no leakage of today's value
    by_meal = out["meal"]
    out["roll_mean_7"] = shifted.groupby(by_meal).transform(lambda s: s.rolling(7, min_periods=3).mean())
    out["roll_mean_14"] = shifted.groupby(by_meal).transform(lambda s: s.rolling(14, min_periods=5).mean())
    out["roll_mean_28"] = shifted.groupby(by_meal).transform(lambda s: s.rolling(28, min_periods=7).mean())
    out["roll_std_7"] = shifted.groupby(by_meal).transform(lambda s: s.rolling(7, min_periods=3).std())

    out["dow_mean_4w"] = out[["lag_7", "lag_14", "lag_21", "lag_28"]].mean(axis=1)
    out["trend_7_28"] = out["roll_mean_7"] / out["roll_mean_28"].replace(0, np.nan)
    return out.drop(columns=["lag_21"])


def select_features(cfg: CollegeConfig, data: CollegeData) -> list[str]:
    cols = list(CALENDAR_FEATURES)
    if data.timetable is not None:
        cols += TT_FEATURES
    if data.weather is not None:
        cols += WEATHER_FEATURES
    cols += [f"evt_{e}" for e in cfg.event_types]
    cols += HISTORY_FEATURES
    return cols


def training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows we can learn from: actual known, and past the warm-up window."""
    cutoff = df["date"].min() + pd.Timedelta(days=WARMUP_DAYS)
    return df[df["footfall"].notna() & (df["date"] >= cutoff)].reset_index(drop=True)
