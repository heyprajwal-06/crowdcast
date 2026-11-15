"""Demo-data simulator.

IMPORTANT - READ THIS: the numbers produced here are SIMULATED. The simulator bakes in plausible
real-world effects (exams, holidays, rain, class timetable, day-to-day persistence ...) so that the
whole pipeline can be built and tested before real canteen data exists. A model trained on this data
only proves that the *code* works; it says nothing about how well the system works on a real campus.
Swap in real footfall.csv as soon as you have it - no other code changes needed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from canteen_ai.config import CollegeConfig, data_dir
from canteen_ai.data.loader import set_data_source
from canteen_ai.features.timetable import timetable_features

SLOT_STARTS = ["09:00", "10:00", "11:00", "13:30", "14:30", "15:30"]  # 1-hour lectures
DAY_FACTOR = {0: 1.0, 1: 1.0, 2: 0.88, 3: 1.0, 4: 0.82, 5: 0.30}  # Sat = few morning lectures

VACATION = [((12, 22), (12, 31)), ((1, 1), (1, 6)), ((5, 5), (7, 5))]
EXAMS = [((9, 20), (9, 30)), ((11, 25), (12, 12)), ((2, 15), (2, 25)), ((4, 20), (5, 4))]
HOLIDAYS = [(1, 26), (3, 8), (4, 14), (5, 1), (8, 15), (9, 7), (10, 2), (10, 20), (11, 9), (12, 25)]
FEST_DAYS = [(2, 10), (2, 11), (2, 12), (9, 12), (9, 13)]

MONTH_TEMP = {1: 21, 2: 24, 3: 27.5, 4: 30, 5: 30.5, 6: 27.5, 7: 25.5, 8: 25, 9: 25.5, 10: 25.5, 11: 23, 12: 21}
MONTH_RAIN_P = {1: .02, 2: .02, 3: .03, 4: .04, 5: .08, 6: .55, 7: .75, 8: .70, 9: .60, 10: .25, 11: .05, 12: .02}

RAIN_BOOST = {"breakfast": 1.03, "lunch": 1.06, "snacks": 1.10}  # rain keeps people on campus


def _in_window(d: pd.Timestamp, windows) -> bool:
    md = (d.month, d.day)
    return any(a <= md <= b for a, b in windows)


# --------------------------------------------------------------------------- pieces
def _make_timetable(cfg: CollegeConfig, rng: np.random.Generator) -> pd.DataFrame:
    rooms_per_slot = max(8, int(cfg.total_students / (90 * 0.8) * 0.9))
    rows = []
    for dow, factor in DAY_FACTOR.items():
        slots = SLOT_STARTS if dow < 5 else SLOT_STARTS[:3]
        for s in slots:
            hh, mm = map(int, s.split(":"))
            end = f"{hh + 1:02d}:{mm:02d}"
            n_rooms = max(1, int(rooms_per_slot * factor * rng.uniform(0.9, 1.1)))
            caps = rng.choice([60, 90, 120], size=n_rooms, p=[0.35, 0.45, 0.20])
            att = np.clip(rng.normal(0.80, 0.07, size=n_rooms), 0.5, 0.97)
            for c, a in zip(caps, att):
                rows.append((dow, s, end, int(c), round(float(a), 2)))
    return pd.DataFrame(rows, columns=["day_of_week", "start_time", "end_time", "room_capacity", "avg_attendance"])


def _make_events(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for d in dates:
        md = (d.month, d.day)
        vac = _in_window(d, VACATION)
        hol = md in HOLIDAYS
        exam = _in_window(d, EXAMS)
        weekday = d.dayofweek < 5
        if vac:
            rows.append((d, "vacation"))
        if hol:
            rows.append((d, "holiday"))
        if exam and not vac:
            rows.append((d, "exam"))
        if md in FEST_DAYS and not vac:
            rows.append((d, "fest"))
        if weekday and not (vac or hol or exam) and rng.random() < 0.07:
            rows.append((d, "placement_drive"))
        if weekday and not (vac or hol):
            p = 0.70 if d.dayofweek == 4 else 0.03
            if rng.random() < p:
                rows.append((d, "special_menu"))
    return pd.DataFrame(rows, columns=["date", "event_type"])


def _make_weather(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    rows, z = [], 0.0
    for d in dates:
        z = 0.7 * z + rng.normal(0, 1.2)
        rainy = rng.random() < MONTH_RAIN_P[d.month]
        rain = float(rng.gamma(1.3, 11.0)) if rainy else 0.0
        temp = MONTH_TEMP[d.month] + z - (2.0 if rainy else 0.0)
        rows.append((d, round(temp, 1), round(rain, 1)))
    return pd.DataFrame(rows, columns=["date", "temp_c", "rain_mm"])


def _timetable_multiplier(ttd, mean_before, dow: int, meal: str) -> float:
    """Day-scholars eat more when lectures end just before the meal and less when they're in class."""
    if (dow, meal) not in ttd.index:
        return 1.0
    r = ttd.loc[(dow, meal)]
    if r["tt_load_day"] <= 0:
        return 1.0
    mb = mean_before.get(meal, 0.0)
    rel = float(r["tt_load_before_meal"] / mb) if mb > 0 else 1.0
    frac_during = float(r["tt_load_during_meal"] / r["tt_load_day"])
    return (0.85 + 0.15 * float(np.clip(rel, 0, 2))) * (1 - 0.8 * float(np.clip(frac_during, 0, 0.6)))


def _simulate(cfg, dates, tt_feats, events, weather, rng) -> pd.DataFrame:
    syn = cfg.synthetic
    n_total = cfg.total_students
    hostel = n_total * float(syn.get("hostel_ratio", 0.55))
    day_scholars = n_total - hostel
    eat_cfg = syn.get("eat_prob", {})

    ev = {et: set(events.loc[events["event_type"] == et, "date"]) for et in cfg.event_types}
    w = weather.set_index("date")
    ttd = tt_feats.set_index(["day_of_week", "meal"])
    load_by_dow = tt_feats.groupby("day_of_week")["tt_load_day"].first()
    max_load = float(load_by_dow.max())
    mean_before = tt_feats.groupby("meal")["tt_load_before_meal"].mean().to_dict()

    rows, z = [], 0.0
    for d in dates:
        dow = d.dayofweek
        z = 0.55 * z + rng.normal(0, 0.045)  # shared day-level mood -> persistence between days
        is_vac = d in ev.get("vacation", set())
        is_hol = d in ev.get("holiday", set())
        is_exam = d in ev.get("exam", set())
        is_fest = d in ev.get("fest", set())
        is_place = d in ev.get("placement_drive", set())
        is_special = d in ev.get("special_menu", set())

        if is_vac:
            hostel_p, day_p = hostel * rng.uniform(0.04, 0.08), 0.0
        elif is_hol:
            hostel_p, day_p = hostel * 0.80, day_scholars * 0.02
        elif dow == 6:
            hostel_p, day_p = hostel * 0.78, day_scholars * 0.02
        else:
            hostel_p = hostel * 0.93
            load = float(load_by_dow.get(dow, 0.0))
            day_p = day_scholars * float(np.clip(0.9 * load / max_load, 0.05, 0.95))
            if is_exam:
                day_p *= 0.72
        rain, temp = float(w.loc[d, "rain_mm"]), float(w.loc[d, "temp_c"])
        if rain > 15:
            day_p *= 0.92

        for meal in cfg.meal_names:
            probs = eat_cfg.get(meal, {"hostel": 0.15, "day": 0.25})
            mu = hostel_p * probs["hostel"] * (0.92 if is_exam else 1.0)
            mu += day_p * probs["day"] * _timetable_multiplier(ttd, mean_before, dow, meal)
            mu *= RAIN_BOOST.get(meal, 1.0) if rain > 2 else 1.0
            if temp > 34 and meal == "snacks":
                mu *= 0.93
            if meal == "lunch" and is_place:
                mu *= 0.90
            if meal == "lunch" and is_special:
                mu *= 1.12
            if is_fest and meal != "breakfast":
                mu = mu * 1.12 + rng.uniform(150, 350)
            mu *= float(np.exp(z + rng.normal(0, 0.04)))
            count = int(max(0, round(rng.normal(mu, 1.8 * np.sqrt(max(mu, 1.0))))))
            if rng.random() < 0.012:  # a few days nobody logged the count
                continue
            rows.append((d, meal, count))
    return pd.DataFrame(rows, columns=["date", "meal", "footfall"])


# --------------------------------------------------------------------------- public API
def generate_demo_dataset(
    cfg: CollegeConfig, days: int = 540, end_date=None, seed: int | None = None
) -> dict[str, Path]:
    seed = int(seed if seed is not None else cfg.synthetic.get("seed", 42))
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end_date).normalize() if end_date else pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    dates = pd.date_range(end - pd.Timedelta(days=days - 1), end, freq="D")

    timetable = _make_timetable(cfg, rng)
    tt_feats = timetable_features(timetable, cfg)
    events = _make_events(dates, rng)
    weather = _make_weather(dates, rng)
    footfall = _simulate(cfg, dates, tt_feats, events, weather, rng)

    out = data_dir(cfg.college_id)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "footfall": out / "footfall.csv",
        "timetable": out / "timetable.csv",
        "weather": out / "weather.csv",
        "events": out / "events.csv",
    }
    for frame in (footfall, weather, events):
        frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    footfall.to_csv(paths["footfall"], index=False)
    timetable.to_csv(paths["timetable"], index=False)
    weather.to_csv(paths["weather"], index=False)
    events.to_csv(paths["events"], index=False)
    set_data_source(cfg, "synthetic", seed=seed, days=days)
    return paths
