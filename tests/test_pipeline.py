"""End-to-end: simulate data -> train (fast) -> forecast -> ingredient plan."""
import numpy as np
import pandas as pd
import pytest

from canteen_ai.config import load_config, model_dir
from canteen_ai.data.loader import load_college_data
from canteen_ai.data.synthetic import generate_demo_dataset
from canteen_ai.features.build import add_history_features, assemble
from canteen_ai.models.predictor import Forecaster
from canteen_ai.models.trainer import date_block_splits, train_college_model
from canteen_ai.planning.bom import dish_plan, ingredient_plan

END = "2026-03-31"  # a normal teaching period: holdout contains exams + a holiday, training saw a vacation


@pytest.fixture()
def trained(home):
    cfg = load_config("demo_college")
    generate_demo_dataset(cfg, days=260, end_date=END, seed=1)
    summary = train_college_model(cfg, fast=True, log=lambda *_: None)
    return cfg, summary


def test_no_leakage_in_history_features(home):
    """Changing a day's footfall must not change that same day's features."""
    cfg = load_config("demo_college")
    generate_demo_dataset(cfg, days=120, end_date=END, seed=2)
    data = load_college_data(cfg)
    base = add_history_features(assemble(data, cfg))
    changed = data.footfall.copy()
    target = changed["date"].max()
    changed.loc[changed["date"] == target, "footfall"] += 5000
    data2 = type(data)(changed, data.timetable, data.weather, data.events, data.source)
    alt = add_history_features(assemble(data2, cfg))
    lag_cols = [c for c in base.columns if c.startswith(("lag_", "roll_", "dow_mean", "trend"))]
    row_b = base[base["date"] == target][lag_cols].reset_index(drop=True)
    row_a = alt[alt["date"] == target][lag_cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(row_b, row_a)


def test_cv_splits_never_train_on_the_future():
    dates = pd.Series(pd.date_range("2025-01-01", periods=300).repeat(3))
    for tr, va in date_block_splits(dates):
        assert dates.iloc[tr].max() < dates.iloc[va].min()


def test_training_produces_artifacts_and_sane_metrics(trained):
    cfg, s = trained
    out = model_dir(cfg.college_id)
    for f in ("model.joblib", "metrics.json", "holdout_predictions.csv", "feature_importance.csv"):
        assert (out / f).exists()
    assert s["metrics"]["ensemble"]["mae"] < s["metrics"]["baseline_same_day_last_week"]["mae"]
    assert 0 < s["intervals"]["safety_coverage_pct"] <= 100


def test_forecast_shape_and_ordering(trained):
    cfg, _ = trained
    fc = Forecaster.load(cfg)
    res = fc.forecast(pd.date_range("2026-04-01", periods=3), fetch_weather=False)
    assert len(res) == 3 * len(cfg.meal_names)
    assert (res["low"] <= res["expected"]).all()
    assert (res["expected"] <= res["recommended_prep"]).all()
    assert (res["recommended_prep"] <= res["high"]).all()
    assert res["days_ahead"].min() == 1


def test_what_if_rain_changes_forecast(trained):
    cfg, _ = trained
    fc = Forecaster.load(cfg)
    d = pd.Timestamp("2026-04-01")
    a = fc.forecast([d], fetch_weather=False)
    b = fc.forecast([d], overrides={d: {"rain_mm": 80.0, "evt_exam": 1}}, fetch_weather=False)
    assert not np.array_equal(a["expected"].to_numpy(), b["expected"].to_numpy())


def test_backtest_style_forecast_for_past_date(trained):
    cfg, _ = trained
    res = Forecaster.load(cfg).forecast([pd.Timestamp("2026-03-20")], fetch_weather=False)
    assert res["days_ahead"].iloc[0] == 1  # pretends to stand at 19 Mar


def test_bom_scales_with_headcount(trained):
    cfg, _ = trained
    fc = Forecaster.load(cfg)
    res = fc.forecast(pd.date_range("2026-04-01", periods=1), fetch_weather=False)
    d1 = dish_plan(cfg, res)
    doubled = res.assign(recommended_prep=res["recommended_prep"] * 2)
    d2 = dish_plan(cfg, doubled)
    assert d2["portions"].sum() >= 2 * d1["portions"].sum() - len(d1)  # ceil() rounding slack
    _, shop1 = ingredient_plan(cfg, d1)
    _, shop2 = ingredient_plan(cfg, d2)
    assert shop2["quantity"].sum() > shop1["quantity"].sum() * 1.8


def test_too_little_data_gives_clear_message(home):
    cfg = load_config("demo_college")
    generate_demo_dataset(cfg, days=60, end_date=END, seed=3)
    with pytest.raises(ValueError, match="need at least"):
        train_college_model(cfg, fast=True, log=lambda *_: None)
