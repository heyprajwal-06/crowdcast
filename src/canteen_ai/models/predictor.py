"""Forecaster: loads a trained bundle and produces multi-day forecasts.

Multi-day forecasting is RECURSIVE: predict tomorrow from real history, then pretend that prediction
happened and use it as 'yesterday' for the day after, and so on. Accuracy therefore decays with the horizon
(the Model report page shows exactly how fast), and the intervals for later days are optimistic.
"""
from __future__ import annotations

import dataclasses

import joblib
import pandas as pd

from canteen_ai.config import CollegeConfig, model_dir
from canteen_ai.data.loader import CollegeData, load_college_data
from canteen_ai.features.build import add_history_features, assemble
from canteen_ai.models.ensemble import ensemble_predict, finalize_intervals, quantile_predict
from canteen_ai.weather import fetch_forecast

MODEL_FILE = "model.joblib"


class Forecaster:
    def __init__(self, cfg: CollegeConfig, bundle: dict, data: CollegeData | None = None):
        self.cfg = cfg
        self.bundle = bundle
        self.data = data if data is not None else load_college_data(cfg)

    @classmethod
    def load(cls, cfg: CollegeConfig, data: CollegeData | None = None) -> "Forecaster":
        path = model_dir(cfg.college_id) / MODEL_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model for '{cfg.college_id}'. Train one:  python -m canteen_ai train --college {cfg.college_id}"
            )
        return cls(cfg, joblib.load(path), data)

    @property
    def last_actual_date(self) -> pd.Timestamp:
        return self.data.footfall["date"].max()

    def forecast(
        self,
        dates,
        overrides: dict | None = None,
        fetch_weather: bool = True,
        as_of=None,
    ) -> pd.DataFrame:
        """Forecast the given dates.

        overrides  what-if changes per date, e.g. {date: {"rain_mm": 40, "evt_exam": 1}}
        as_of      pretend we are standing at this date (history before it only) - used for backtests
        """
        dates = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize().sort_values().unique()
        hist = self.data.footfall
        if as_of is None and dates.min() <= hist["date"].max():
            as_of = dates.min()  # asked about a day that already has actuals -> backtest style
        if as_of is not None:
            hist = hist[hist["date"] < pd.Timestamp(as_of).normalize()]
        if hist.empty:
            raise ValueError("No history before the requested date.")
        last_known = hist["date"].max()
        future = pd.date_range(last_known + pd.Timedelta(days=1), dates.max(), freq="D")

        weather_future = None
        if fetch_weather and self.data.weather is not None:
            today = pd.Timestamp.today().normalize()
            w_start, w_end = max(future.min(), today), min(future.max(), today + pd.Timedelta(days=15))
            if w_start <= w_end:
                weather_future = fetch_forecast(self.cfg, w_start, w_end)

        data = dataclasses.replace(self.data, footfall=hist)
        df = assemble(data, self.cfg, horizon_dates=future, weather_future=weather_future, overrides=overrides)
        cols = self.bundle["feature_cols"]
        keep_dates = set(dates)
        out = []
        for d in future:
            feats = add_history_features(df)
            mask = (feats["date"] == d).to_numpy()
            X = feats.loc[mask, cols]
            expected = ensemble_predict(self.bundle, X)
            q = quantile_predict(self.bundle, X)
            df.loc[mask, "footfall"] = expected  # recursion: today's forecast becomes tomorrow's 'yesterday'
            if d in keep_dates:
                lo_q, hi_q = self.cfg.interval
                low, rec, high = finalize_intervals(expected, q[lo_q], q[self.cfg.safety_quantile], q[hi_q])
                part = feats.loc[mask, ["date", "meal", "meal_idx"]].copy()
                part["expected"], part["low"], part["recommended_prep"], part["high"] = expected, low, rec, high
                for c in ("temp_c", "rain_mm", "weather_source"):
                    if c in feats.columns:
                        part[c] = feats.loc[mask, c].to_numpy()
                part["days_ahead"] = (d - last_known).days
                out.append(part)
        res = pd.concat(out, ignore_index=True).sort_values(["date", "meal_idx"]).reset_index(drop=True)
        for c in ("expected", "low", "recommended_prep", "high"):
            res[c] = res[c].round().astype(int)
        return res.drop(columns="meal_idx")
