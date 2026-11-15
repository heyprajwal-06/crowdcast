"""Training pipeline.

What happens when you run `python -m canteen_ai train`:

 1. Build features (calendar, timetable, weather, events, lags/rolling stats).
 2. Keep the LAST `holdout_days` aside - the model never sees them while learning/tuning.
 3. Tune three strong tree models (RandomForest, XGBoost, LightGBM) with *time-aware* cross-validation
    (always train on the past, validate on the following weeks - never the other way round).
 4. Blend them into an ensemble (weights from cross-validated error).
 5. Train LightGBM *quantile* models -> likely range + a 'cook for' number with a chosen safety level.
 6. Score everything on the untouched holdout vs simple baselines (what a canteen does today).
 7. Backtest 1..7-day-ahead accuracy, then re-train on ALL data and save the bundle.
"""
from __future__ import annotations

import json
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from canteen_ai.config import CollegeConfig, model_dir
from canteen_ai.data.loader import CollegeData, load_college_data
from canteen_ai.features.build import add_history_features, assemble, select_features, training_rows
from canteen_ai.models.ensemble import ensemble_predict, finalize_intervals, quantile_predict
from canteen_ai.models.evaluate import business_impact, interval_metrics, regression_metrics
from canteen_ai.models.predictor import MODEL_FILE, Forecaster

ENSEMBLE_MEMBERS = ["random_forest", "xgboost", "lightgbm"]
MIN_DAYS = 120


# --------------------------------------------------------------------------- CV splitting
def date_block_splits(dates: pd.Series, n_splits: int = 4, val_days: int = 28, min_train_days: int = 90):
    """Expanding-window splits on whole days:  [train ....][val] -> [train ........][val] -> ..."""
    values = pd.Series(dates).reset_index(drop=True).to_numpy()
    uniq = np.sort(np.unique(values))
    n = len(uniq)
    n_splits = max(1, min(n_splits, (n - min_train_days) // val_days))
    pos = np.searchsorted(uniq, values)
    splits = []
    for i in range(n_splits):
        val_end = n - (n_splits - 1 - i) * val_days
        val_start = val_end - val_days
        splits.append((np.where(pos < val_start)[0], np.where((pos >= val_start) & (pos < val_end))[0]))
    return splits


# --------------------------------------------------------------------------- model zoo
def _make_xgb(seed, **p):
    return XGBRegressor(tree_method="hist", random_state=seed, n_jobs=-1, verbosity=0, **p)


def _make_lgbm(seed, **p):
    return LGBMRegressor(random_state=seed, verbose=-1, subsample_freq=1, importance_type="gain", **p)


def _search_spaces(seed: int):
    return {
        "random_forest": (
            RandomForestRegressor(random_state=seed, n_jobs=-1),
            {"n_estimators": [200, 400], "max_depth": [None, 8, 12, 16],
             "min_samples_leaf": [1, 2, 4], "max_features": [0.5, 0.7, 1.0]},
        ),
        "xgboost": (
            _make_xgb(seed),
            {"n_estimators": [200, 400, 700], "learning_rate": [0.02, 0.05, 0.1], "max_depth": [3, 4, 5, 6],
             "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 3, 5],
             "reg_lambda": [0.5, 1.0, 3.0], "objective": ["reg:squarederror", "count:poisson"]},
        ),
        "lightgbm": (
            _make_lgbm(seed),
            {"n_estimators": [200, 400, 700], "learning_rate": [0.02, 0.05, 0.1], "num_leaves": [15, 31, 63],
             "min_child_samples": [10, 20, 40], "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.6, 0.8, 1.0],
             "reg_lambda": [0.0, 1.0, 3.0], "objective": ["regression", "poisson"]},
        ),
    }


def _fill_baseline(frame: pd.DataFrame, col: str) -> np.ndarray:
    s = frame[col]
    for fallback in ("dow_mean_4w", "roll_mean_7", "roll_mean_28"):
        s = s.fillna(frame[fallback])
    return s.fillna(frame["footfall"].mean()).to_numpy()


# --------------------------------------------------------------------------- main entry
def train_college_model(cfg: CollegeConfig, data: CollegeData | None = None, fast: bool = False, log=print) -> dict:
    t0 = time.time()
    seed = 42
    n_iter = 4 if fast else 16
    data = data if data is not None else load_college_data(cfg)

    df = add_history_features(assemble(data, cfg))
    feature_cols = select_features(cfg, data)
    labelled = training_rows(df)
    n_days = labelled["date"].nunique()
    if n_days < MIN_DAYS:
        raise ValueError(
            f"Only {n_days} usable days of footfall after the 28-day warm-up; need at least {MIN_DAYS} "
            "(about 4-5 months). Collect more data, or use the demo data generator."
        )

    holdout_days = int(min(cfg.holdout_days, max(14, n_days // 4)))
    cutoff = labelled["date"].max() - pd.Timedelta(days=holdout_days - 1)
    train = labelled[labelled["date"] < cutoff].reset_index(drop=True)
    test = labelled[labelled["date"] >= cutoff].reset_index(drop=True)
    X_tr, y_tr = train[feature_cols], train["footfall"]
    X_te, y_te = test[feature_cols], test["footfall"].to_numpy()
    log(f"Data: {len(labelled)} rows / {n_days} days | train {len(train)} rows, "
        f"holdout {len(test)} rows (last {holdout_days} days) | {len(feature_cols)} features")

    splits = date_block_splits(train["date"])
    log(f"Time-series CV: {len(splits)} expanding folds")

    # ---- 1) linear reference + tuned tree models
    ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=3.0)).fit(X_tr, y_tr)
    fitted: dict[str, object] = {"ridge_reference": ridge}
    cv_mae: dict[str, float] = {}
    best_params: dict[str, dict] = {}
    for name, (est, space) in _search_spaces(seed).items():
        t = time.time()
        rs = RandomizedSearchCV(est, space, n_iter=n_iter, scoring="neg_mean_absolute_error", cv=splits,
                                random_state=seed, n_jobs=1, refit=True)
        rs.fit(X_tr, y_tr)
        fitted[name] = rs.best_estimator_
        cv_mae[name] = float(-rs.best_score_)
        best_params[name] = rs.best_params_
        log(f"  tuned {name:<14} CV MAE {cv_mae[name]:7.1f}   ({time.time() - t:.0f}s)")

    # ---- 2) ensemble weights ~ 1 / CV error
    inv = np.array([1.0 / cv_mae[m] for m in ENSEMBLE_MEMBERS])
    weights = dict(zip(ENSEMBLE_MEMBERS, (inv / inv.sum()).tolist()))

    # ---- 3) quantile models (likely range + safety level)
    lgbm_params = {k: v for k, v in best_params["lightgbm"].items() if k != "objective"}
    quantile_models = {
        q: _make_lgbm(seed, objective="quantile", alpha=q, **lgbm_params).fit(X_tr, y_tr) for q in cfg.quantiles
    }
    calibration = _calibrate_quantiles(cfg, X_tr, y_tr, splits, lgbm_params, seed)
    log("Quantile calibration factors: " + ", ".join(f"q{int(q * 100)}x{c:.3f}" for q, c in calibration.items()))
    train_bundle = {"feature_cols": feature_cols, "models": fitted, "ensemble_members": ENSEMBLE_MEMBERS,
                    "weights": weights, "quantile_models": quantile_models, "calibration": calibration}

    # ---- 4) honest scoring on the untouched holdout
    preds = {name: np.clip(m.predict(X_te), 0, None) for name, m in fitted.items()}
    preds["ensemble"] = ensemble_predict(train_bundle, X_te)
    base_last_week = _fill_baseline(test, "lag_7")
    base_avg4w = _fill_baseline(test, "dow_mean_4w")
    lo_q, hi_q = cfg.interval
    q = quantile_predict(train_bundle, X_te)
    low, safe, high = finalize_intervals(preds["ensemble"], q[lo_q], q[cfg.safety_quantile], q[hi_q])

    model_metrics = {
        "baseline_same_day_last_week": regression_metrics(y_te, base_last_week),
        "baseline_avg_same_weekday_4w": regression_metrics(y_te, base_avg4w),
        **{name: regression_metrics(y_te, p) for name, p in preds.items()},
    }
    for name in cv_mae:
        model_metrics[name]["cv_mae"] = cv_mae[name]
    intervals = interval_metrics(y_te, low, safe, high, cfg.safety_quantile, cfg.interval)
    impact = {
        "same_day_last_week_+10%": business_impact(y_te, base_last_week * 1.10),
        "avg_same_weekday_4w_+10%": business_impact(y_te, base_avg4w * 1.10),
        "model_expected": business_impact(y_te, preds["ensemble"]),
        f"model_recommended_p{int(cfg.safety_quantile * 100)}": business_impact(y_te, safe),
    }

    holdout_df = test[["date", "meal"]].copy()
    holdout_df["actual"] = y_te
    holdout_df["ensemble"], holdout_df["low"], holdout_df["recommended"], holdout_df["high"] = preds["ensemble"], low, safe, high
    holdout_df["baseline_last_week"], holdout_df["baseline_avg_4w"] = base_last_week, base_avg4w

    # ---- 5) how fast does accuracy decay for 1..7 days ahead? (recursive forecast backtest)
    horizon_df = _horizon_backtest(cfg, data, train_bundle, cutoff)

    # ---- 6) final model: same settings, re-trained on ALL data
    X_all, y_all = labelled[feature_cols], labelled["footfall"]
    final_models = {name: clone(fitted[name]).fit(X_all, y_all) for name in fitted}
    final_q = {q_: clone(m).fit(X_all, y_all) for q_, m in quantile_models.items()}
    imp = pd.DataFrame({
        "feature": feature_cols,
        "xgboost": final_models["xgboost"].feature_importances_,
        "lightgbm": final_models["lightgbm"].feature_importances_,
    })
    for c in ("xgboost", "lightgbm"):
        imp[c] = imp[c] / imp[c].sum()
    imp["importance"] = imp[["xgboost", "lightgbm"]].mean(axis=1)
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)

    summary = {
        "college_id": cfg.college_id,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "data_source": data.source,
        "n_rows": int(len(labelled)), "n_days": int(n_days), "holdout_days": holdout_days,
        "holdout_start": str(cutoff.date()), "holdout_end": str(labelled["date"].max().date()),
        "last_actual_date": str(data.footfall["date"].max().date()),
        "features": feature_cols, "best_params": {k: {a: _py(b) for a, b in v.items()} for k, v in best_params.items()},
        "ensemble_weights": weights, "metrics": model_metrics, "intervals": intervals, "business_impact": impact,
        "safety_quantile": cfg.safety_quantile, "interval": list(cfg.interval),
        "train_seconds": round(time.time() - t0, 1),
    }
    bundle = {**train_bundle, "models": final_models, "quantile_models": final_q, "meta": summary}  # calibration carried over

    out = model_dir(cfg.college_id)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / MODEL_FILE)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=_py))
    holdout_df.to_csv(out / "holdout_predictions.csv", index=False)
    imp.to_csv(out / "feature_importance.csv", index=False)
    horizon_df.to_csv(out / "horizon_backtest.csv", index=False)

    _print_report(log, summary, horizon_df)
    return summary


# --------------------------------------------------------------------------- helpers
def _calibrate_quantiles(cfg, X, y, splits, lgbm_params, seed) -> dict[float, float]:
    """Raw quantile models are usually a bit over-confident. Fit them on each CV fold, look at how often the
    unseen validation days actually fell below each quantile, and learn one scale factor per quantile so that
    the share really is ~q (e.g. 80% of days at or below the 'cook for' number).

    Only 'normal operating days' are used for this (predicted level >= 50% of that meal's typical level),
    because vacation / closed days have far noisier *relative* errors and would inflate the intervals."""
    typical = y.groupby(X["meal_idx"]).median()
    oof: dict[float, list] = {q: [] for q in cfg.quantiles}
    ys, normal = [], []
    for tr_idx, va_idx in splits:
        ys.append(y.iloc[va_idx].to_numpy())
        level = X["meal_idx"].iloc[va_idx].map(typical).to_numpy()
        top = None
        for q in cfg.quantiles:
            m = _make_lgbm(seed, objective="quantile", alpha=q, **lgbm_params)
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            pred = np.clip(m.predict(X.iloc[va_idx]), 1.0, None)
            oof[q].append(pred)
            top = pred if top is None else np.maximum(top, pred)
        normal.append(top >= 0.5 * level)
    keep = np.concatenate(normal)
    if keep.sum() < 30:  # tiny datasets: fall back to using everything
        keep[:] = True
    y_all = np.concatenate(ys)[keep]
    return {q: float(np.quantile(y_all / np.concatenate(oof[q])[keep], q)) for q in cfg.quantiles}


def _horizon_backtest(cfg, data, bundle, cutoff, max_h: int = 7) -> pd.DataFrame:
    """Rolling-origin test: every 3 days inside the holdout, forecast the next 7 days recursively."""
    fc = Forecaster(cfg, bundle, data)
    actual = data.footfall.rename(columns={"footfall": "actual"})
    last = data.footfall["date"].max()
    rows = []
    for origin in pd.date_range(cutoff, last - pd.Timedelta(days=max_h), freq="3D"):
        res = fc.forecast(pd.date_range(origin, periods=max_h), fetch_weather=False, as_of=origin)
        rows.append(res.merge(actual, on=["date", "meal"], how="inner"))
    if not rows:
        return pd.DataFrame(columns=["horizon", "mae", "wape_pct", "n"])
    allr = pd.concat(rows, ignore_index=True)
    allr["horizon"] = allr["days_ahead"]
    allr["abs_err"] = (allr["expected"] - allr["actual"]).abs()
    g = allr.groupby("horizon")
    return pd.DataFrame({"mae": g["abs_err"].mean(), "wape_pct": g["abs_err"].sum() / g["actual"].sum() * 100,
                         "n": g.size()}).reset_index()


def _py(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def _print_report(log, s: dict, horizon: pd.DataFrame) -> None:
    m = s["metrics"]
    log("\n=== Holdout results (days the model never saw) ===")
    log(f"{'model':<30}{'MAE':>8}{'WAPE%':>9}{'within10%':>11}")
    for name, v in m.items():
        log(f"{name:<30}{v['mae']:8.1f}{v['wape_pct']:9.1f}{v['within_10pct']:11.1f}")
    i = s["intervals"]
    log(f"\nLikely-range coverage: {i['interval_coverage_pct']:.0f}% (target {i['interval_nominal_pct']:.0f}%) | "
        f"safety level hit: {i['safety_coverage_pct']:.0f}% (target {i['safety_nominal_pct']:.0f}%)")
    log("\n=== What-if the kitchen had cooked this much (holdout period) ===")
    log(f"{'strategy':<34}{'wasted%':>9}{'ran-out%':>10}{'unmet%':>8}")
    for k, v in s["business_impact"].items():
        log(f"{k:<34}{v['wastage_pct']:9.1f}{v['shortage_services_pct']:10.1f}{v['unmet_demand_pct']:8.1f}")
    if len(horizon):
        log("\n=== Error by forecast horizon (days ahead) ===")
        log("  ".join(f"D+{int(r.horizon)}: {r.wape_pct:.1f}%" for r in horizon.itertuples()))
    log(f"\nSaved model + reports to models/{s['college_id']}/   ({s['train_seconds']}s)")
