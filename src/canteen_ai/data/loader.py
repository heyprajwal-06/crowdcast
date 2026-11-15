"""Read, validate and write the CSV files that describe one college.

Files live in data/raw/<college_id>/ :

  footfall.csv   date,meal,footfall                         (REQUIRED - the thing we predict)
  timetable.csv  day_of_week,start_time,end_time,room_capacity,avg_attendance   (optional)
  weather.csv    date,temp_c,rain_mm                         (optional, daily mean temp & rainfall)
  events.csv     date,event_type                             (optional - exams, holidays, fests ...)
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from canteen_ai.config import CollegeConfig, data_dir

FOOTFALL_COLS = ["date", "meal", "footfall"]
TIMETABLE_COLS = ["day_of_week", "start_time", "end_time", "room_capacity", "avg_attendance"]
WEATHER_COLS = ["date", "temp_c", "rain_mm"]
EVENTS_COLS = ["date", "event_type"]

FILE_COLUMNS = {
    "footfall.csv": FOOTFALL_COLS,
    "timetable.csv": TIMETABLE_COLS,
    "weather.csv": WEATHER_COLS,
    "events.csv": EVENTS_COLS,
}


@dataclass
class CollegeData:
    footfall: pd.DataFrame
    timetable: pd.DataFrame | None
    weather: pd.DataFrame | None
    events: pd.DataFrame | None
    source: str = "unknown"  # "synthetic" | "real" | "unknown"


# --------------------------------------------------------------------------- readers
def _read(path: Path, cols: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing column(s) {missing}. Expected columns: {cols}")
    return df


def _dates(series: pd.Series, fname: str) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="raise").dt.normalize()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"{fname}: could not read the 'date' column ({exc}). Use YYYY-MM-DD.") from exc


def read_footfall(path: Path, cfg: CollegeConfig) -> pd.DataFrame:
    df = _read(path, FOOTFALL_COLS)[FOOTFALL_COLS].copy()
    if df.empty:
        raise ValueError(
            f"{path.name} has no rows yet. Add daily footfall (or generate demo data first)."
        )
    df["date"] = _dates(df["date"], path.name)
    df["meal"] = df["meal"].astype(str).str.strip().str.lower()
    unknown = sorted(set(df["meal"]) - set(cfg.meal_names))
    if unknown:
        raise ValueError(
            f"{path.name}: unknown meal name(s) {unknown}. Meals in the config: {cfg.meal_names}"
        )
    df["footfall"] = pd.to_numeric(df["footfall"], errors="coerce")
    if df["footfall"].isna().any():
        raise ValueError(f"{path.name}: 'footfall' must be numbers (found blanks or text)")
    if (df["footfall"] < 0).any():
        raise ValueError(f"{path.name}: 'footfall' cannot be negative")
    dup = df.duplicated(["date", "meal"], keep="last")
    if dup.any():
        warnings.warn(f"{path.name}: {int(dup.sum())} duplicate (date, meal) rows - kept the last")
        df = df[~dup]
    return df.sort_values(["date", "meal"]).reset_index(drop=True)


def read_timetable(path: Path) -> pd.DataFrame:
    df = _read(path, TIMETABLE_COLS)[TIMETABLE_COLS].copy()
    if df.empty:
        raise ValueError(f"{path.name} has no rows")
    for col in ("day_of_week", "room_capacity", "avg_attendance"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["day_of_week", "room_capacity", "avg_attendance"]].isna().any().any():
        raise ValueError(f"{path.name}: day_of_week, room_capacity, avg_attendance must be numeric")
    if not df["day_of_week"].between(0, 6).all():
        raise ValueError(f"{path.name}: day_of_week must be 0 (Mon) .. 6 (Sun)")
    if df["avg_attendance"].max() > 1.5:  # looks like percentages
        warnings.warn(f"{path.name}: avg_attendance looks like a percentage - dividing by 100")
        df["avg_attendance"] = df["avg_attendance"] / 100.0
    df["day_of_week"] = df["day_of_week"].astype(int)
    return df.reset_index(drop=True)


def read_weather(path: Path) -> pd.DataFrame:
    df = _read(path, WEATHER_COLS)[WEATHER_COLS].copy()
    if df.empty:
        raise ValueError(f"{path.name} has no rows")
    df["date"] = _dates(df["date"], path.name)
    df["temp_c"] = pd.to_numeric(df["temp_c"], errors="coerce")
    df["rain_mm"] = pd.to_numeric(df["rain_mm"], errors="coerce")
    return df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def read_events(path: Path, cfg: CollegeConfig) -> pd.DataFrame:
    df = _read(path, EVENTS_COLS)[EVENTS_COLS].copy()
    if df.empty:
        raise ValueError(f"{path.name} has no rows")
    df["date"] = _dates(df["date"], path.name)
    df["event_type"] = df["event_type"].astype(str).str.strip().str.lower()
    unknown = sorted(set(df["event_type"]) - set(cfg.event_types))
    if unknown:
        warnings.warn(f"{path.name}: ignoring unknown event type(s) {unknown}. "
                      f"Known: {cfg.event_types} (add new ones under 'event_types' in the config)")
        df = df[df["event_type"].isin(cfg.event_types)]
    return df.drop_duplicates().sort_values("date").reset_index(drop=True)


def _optional(path: Path, reader):
    """Optional files: missing or header-only -> None; malformed -> raise."""
    if not path.exists():
        return None
    try:
        return reader(path)
    except ValueError as exc:
        if "has no rows" in str(exc):
            return None
        raise


def read_data_source(cfg: CollegeConfig) -> str:
    p = data_dir(cfg.college_id) / "data_source.json"
    if p.exists():
        try:
            return json.loads(p.read_text()).get("type", "unknown")
        except Exception:  # noqa: BLE001
            return "unknown"
    return "unknown"


def set_data_source(cfg: CollegeConfig, kind: str, **extra) -> None:
    d = data_dir(cfg.college_id)
    d.mkdir(parents=True, exist_ok=True)
    payload = {"type": kind, "updated_at": datetime.now().isoformat(timespec="seconds"), **extra}
    (d / "data_source.json").write_text(json.dumps(payload, indent=2))


def load_college_data(cfg: CollegeConfig) -> CollegeData:
    d = data_dir(cfg.college_id)
    f = d / "footfall.csv"
    if not f.exists():
        raise FileNotFoundError(
            f"{f} not found. Generate demo data:  python -m canteen_ai generate-data --college "
            f"{cfg.college_id}   (or put your real footfall.csv there)"
        )
    return CollegeData(
        footfall=read_footfall(f, cfg),
        timetable=_optional(d / "timetable.csv", read_timetable),
        weather=_optional(d / "weather.csv", read_weather),
        events=_optional(d / "events.csv", lambda p: read_events(p, cfg)),
        source=read_data_source(cfg),
    )


# --------------------------------------------------------------------------- writers
def save_actuals(cfg: CollegeConfig, date, values: dict[str, int]) -> Path:
    """Upsert one day's actual footfall (used by the 'Log actuals' page)."""
    d = data_dir(cfg.college_id)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "footfall.csv"
    date = pd.Timestamp(date).normalize()
    new = pd.DataFrame(
        [{"date": date, "meal": m.lower(), "footfall": int(v)} for m, v in values.items()]
    )
    if path.exists() and path.stat().st_size > 0:
        old = pd.read_csv(path)
        if len(old):
            old["date"] = pd.to_datetime(old["date"]).dt.normalize()
            old = old[~((old["date"] == date) & (old["meal"].isin(new["meal"])))]
            new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values(["date", "meal"])
    new["date"] = pd.to_datetime(new["date"]).dt.strftime("%Y-%m-%d")
    new.to_csv(path, index=False)
    return path


def add_event(cfg: CollegeConfig, date, event_type: str) -> None:
    event_type = event_type.lower()
    if event_type not in cfg.event_types:
        raise ValueError(f"Unknown event type '{event_type}'. Known: {cfg.event_types}")
    d = data_dir(cfg.college_id)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "events.csv"
    row = pd.DataFrame([{"date": pd.Timestamp(date).strftime("%Y-%m-%d"), "event_type": event_type}])
    if path.exists() and path.stat().st_size > 0:
        old = pd.read_csv(path)
        if len(old):
            row = pd.concat([old, row], ignore_index=True).drop_duplicates()
    row.sort_values("date").to_csv(path, index=False)


def write_templates(cfg: CollegeConfig) -> None:
    """Create empty CSVs with the right headers (used when onboarding a new college)."""
    d = data_dir(cfg.college_id)
    d.mkdir(parents=True, exist_ok=True)
    for fname, cols in FILE_COLUMNS.items():
        p = d / fname
        if not p.exists():
            pd.DataFrame(columns=cols).to_csv(p, index=False)
