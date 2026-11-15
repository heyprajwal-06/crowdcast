"""Weather helpers using Open-Meteo (free, no API key). Every function fails SOFT: returns None on any
network problem, and the pipeline then falls back to monthly climatology from the historical data."""
from __future__ import annotations

import pandas as pd

from canteen_ai.config import CollegeConfig

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_DAILY = "temperature_2m_mean,precipitation_sum"


def _call(url: str, cfg: CollegeConfig, start, end, timeout: float) -> pd.DataFrame | None:
    if cfg.latitude is None or cfg.longitude is None:
        return None
    try:
        import requests

        r = requests.get(
            url,
            params={
                "latitude": cfg.latitude,
                "longitude": cfg.longitude,
                "daily": _DAILY,
                "timezone": cfg.timezone,
                "start_date": pd.Timestamp(start).strftime("%Y-%m-%d"),
                "end_date": pd.Timestamp(end).strftime("%Y-%m-%d"),
            },
            timeout=timeout,
        )
        r.raise_for_status()
        d = r.json()["daily"]
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(d["time"]),
                "temp_c": d["temperature_2m_mean"],
                "rain_mm": d["precipitation_sum"],
            }
        ).dropna()
        return df if len(df) else None
    except Exception:  # noqa: BLE001 - offline, blocked, API change ... all handled the same way
        return None


def fetch_forecast(cfg: CollegeConfig, start, end, timeout: float = 6.0) -> pd.DataFrame | None:
    """Daily forecast, Open-Meteo supports up to ~16 days ahead."""
    return _call(FORECAST_URL, cfg, start, end, timeout)


def fetch_history(cfg: CollegeConfig, start, end, timeout: float = 30.0) -> pd.DataFrame | None:
    """Past daily weather - handy to build weather.csv for a real college."""
    return _call(ARCHIVE_URL, cfg, start, end, timeout)
