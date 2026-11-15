"""Canteen AI dashboard.   Run:  streamlit run app/streamlit_app.py"""
from __future__ import annotations

import datetime as dt
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # works even if the package isn't pip-installed

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from canteen_ai.config import (  # noqa: E402
    CollegeConfig, data_dir, list_colleges, load_config, model_dir,
)
from canteen_ai.data import loader  # noqa: E402
from canteen_ai.models.predictor import MODEL_FILE, Forecaster  # noqa: E402
from canteen_ai.planning.bom import dish_plan, ingredient_plan  # noqa: E402

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Canteen AI", page_icon="🍽️", layout="wide")

MEAL_ICON = {"breakfast": "🌅", "lunch": "🍛", "snacks": "🥪", "dinner": "🌙"}


# --------------------------------------------------------------------------- cached helpers
def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_resource(show_spinner=False)
def _load_forecaster(college_id: str, model_stamp: float, data_stamp: float):
    cfg = load_config(college_id)
    return Forecaster.load(cfg)


def get_forecaster(cfg: CollegeConfig):
    mpath = model_dir(cfg.college_id) / MODEL_FILE
    if not mpath.exists():
        return None
    dpaths = [data_dir(cfg.college_id) / f for f in ("footfall.csv", "events.csv", "weather.csv", "timetable.csv")]
    return _load_forecaster(cfg.college_id, _mtime(mpath), sum(_mtime(p) for p in dpaths))


def read_metrics(cfg: CollegeConfig) -> dict | None:
    p = model_dir(cfg.college_id) / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def pretty(name: str) -> str:
    return name.replace("_", " ").title()


def csv_download(df: pd.DataFrame, label: str, fname: str) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=fname, mime="text/csv")


# --------------------------------------------------------------------------- banners
def data_banner(cfg: CollegeConfig) -> None:
    src = loader.read_data_source(cfg)
    if src == "synthetic":
        st.warning(
            "**Demo mode - this college is using SIMULATED data.** The forecasts below show that the system "
            "works, not how accurate it would be on a real campus. Upload real footfall under *Setup & data*.",
            icon="⚠️",
        )


# --------------------------------------------------------------------------- page: forecast
def page_forecast(cfg: CollegeConfig) -> None:
    st.title("🍽️ What to cook")
    data_banner(cfg)
    fc = get_forecaster(cfg)
    if fc is None:
        st.info("No trained model yet. Go to **Setup & data** to load data, then **Model report** to train.")
        return

    c1, c2, c3 = st.columns([1, 1, 2])
    start = c1.date_input("From", value=dt.date.today() + dt.timedelta(days=1))
    days = c2.slider("Days ahead", 1, 7, 3)
    live = c3.checkbox("Use live weather forecast (needs internet)", value=True)
    dates = pd.date_range(pd.Timestamp(start), periods=days)

    overrides: dict = {}
    with st.expander("🔧 What-if: change the weather or add an event for one day"):
        w_date = st.selectbox("Day", [d.date() for d in dates])
        use = st.checkbox("Apply these changes")
        a, b = st.columns(2)
        rain = a.slider("Rain (mm)", 0, 100, 0)
        temp = b.slider("Temperature (°C)", 5, 45, 28)
        forced = st.multiselect("Treat this day as", [pretty(e) for e in cfg.event_types])
        if use:
            ch = {"rain_mm": float(rain), "temp_c": float(temp)}
            for e in cfg.event_types:
                if pretty(e) in forced:
                    ch[f"evt_{e}"] = 1
            overrides[pd.Timestamp(w_date)] = ch

    try:
        with st.spinner("Forecasting..."):
            res = fc.forecast(dates, overrides=overrides or None, fetch_weather=live)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not forecast: {exc}")
        return

    if res["days_ahead"].min() > 3:
        st.warning(
            f"The last logged actual counts are from **{fc.last_actual_date:%d %b %Y}**. "
            "Forecasts for later days lean on earlier estimates - log recent counts on *Log actuals* for better accuracy."
        )
    if (res["weather_source"] == "climatology").any() if "weather_source" in res else False:
        st.caption("Weather: some days use typical monthly weather because no live forecast was available.")

    pct = int(cfg.safety_quantile * 100)
    for d, g in res.groupby("date"):
        st.subheader(f"{d:%A, %d %B}")
        cols = st.columns(len(g))
        for col, r in zip(cols, g.itertuples()):
            with col:
                st.metric(f"{MEAL_ICON.get(r.meal, '🍴')} {pretty(r.meal)} - cook for", f"{r.recommended_prep:,}",
                          help=f"Covers demand on about {pct} of 100 days. Expected {r.expected:,}.")
                st.caption(f"Expected **{r.expected:,}** · likely range {r.low:,} - {r.high:,}")
        if "temp_c" in g.columns and g["temp_c"].notna().any():
            r0 = g.iloc[0]
            st.caption(f"Weather assumed: {r0.temp_c:.0f}°C, rain {r0.rain_mm:.0f} mm ({r0.weather_source})")

    fig = go.Figure()
    for meal, g in res.groupby("meal"):
        fig.add_trace(go.Bar(
            name=pretty(meal), x=g["date"].dt.strftime("%a %d %b"), y=g["expected"],
            error_y=dict(type="data", symmetric=False, array=(g["high"] - g["expected"]), arrayminus=(g["expected"] - g["low"])),
        ))
    fig.update_layout(barmode="group", yaxis_title="Students", height=340, margin=dict(t=20, b=10), legend_title="")
    st.plotly_chart(fig, width="stretch")

    st.markdown("### 🧾 What to prepare")
    dishes = dish_plan(cfg, res)
    detail, shopping = ingredient_plan(cfg, dishes)
    pick = st.selectbox("Day", ["All days"] + [d.strftime("%a %d %b") for d in dates])
    t1, t2 = st.tabs(["Dishes (portions)", "Ingredients to arrange"])
    with t1:
        v = dishes.copy()
        if pick != "All days":
            v = v[v["date"].dt.strftime("%a %d %b") == pick]
        v = v.assign(date=v["date"].dt.strftime("%a %d %b"), meal=v["meal"].map(pretty))
        st.dataframe(v.rename(columns=str.title), hide_index=True, width="stretch")
        csv_download(v, "Download dish plan (CSV)", "dish_plan.csv")
    with t2:
        s = shopping.copy()
        if pick != "All days":
            s = s[s["date"].dt.strftime("%a %d %b") == pick]
        else:
            s = s.groupby(["ingredient", "unit"], as_index=False)["quantity"].sum()
        s = s.assign(ingredient=s["ingredient"].map(pretty))
        if "date" in s.columns:
            s = s.assign(date=s["date"].dt.strftime("%a %d %b"))
        st.dataframe(s.rename(columns=str.title), hide_index=True, width="stretch")
        csv_download(s, "Download shopping list (CSV)", "ingredients.csv")
        st.caption(f"Includes a {cfg.wastage_buffer:.0%} wastage buffer on top of the recipe quantities.")


# --------------------------------------------------------------------------- page: log actuals
def page_log(cfg: CollegeConfig) -> None:
    st.title("📝 Log actual footfall")
    st.write("Enter how many students were actually served. Every entry makes the next training round smarter.")
    day = st.date_input("Date", value=dt.date.today(), max_value=dt.date.today())
    existing = {}
    fpath = data_dir(cfg.college_id) / "footfall.csv"
    if fpath.exists() and fpath.stat().st_size:
        try:
            ff = loader.read_footfall(fpath, cfg)
            existing = ff[ff["date"] == pd.Timestamp(day)].set_index("meal")["footfall"].to_dict()
        except ValueError:
            pass
    with st.form("log_form"):
        vals = {m: st.number_input(f"{MEAL_ICON.get(m, '🍴')} {pretty(m)} - students served", min_value=0, step=1,
                                   value=int(existing.get(m, 0))) for m in cfg.meal_names}
        ok = st.form_submit_button("Save")
    if ok:
        loader.save_actuals(cfg, day, vals)
        if loader.read_data_source(cfg) == "synthetic":
            loader.set_data_source(cfg, "mixed")
        st.success(f"Saved counts for {day:%d %b %Y}.")
        fc = get_forecaster(cfg)
        if fc is not None:
            try:
                prev = fc.forecast([pd.Timestamp(day)], fetch_weather=False, as_of=pd.Timestamp(day))
                cmp = prev[["meal", "expected", "recommended_prep"]].copy()
                cmp["actual"] = cmp["meal"].map(vals)
                cmp["difference"] = cmp["actual"] - cmp["expected"]
                cmp["meal"] = cmp["meal"].map(pretty)
                st.write("**How the model would have done** (using only data from before that day):")
                st.dataframe(cmp.rename(columns=lambda c: pretty(c)), hide_index=True)
            except Exception:  # noqa: BLE001
                pass
    with st.expander("Bulk upload past footfall (CSV with date, meal, footfall)"):
        up = st.file_uploader("footfall.csv", type="csv", key="bulk_log")
        if up is not None and st.button("Merge into history"):
            try:
                tmp = data_dir(cfg.college_id) / "_upload.csv"
                tmp.write_bytes(up.getvalue())
                new = loader.read_footfall(tmp, cfg)
                tmp.unlink()
                for d, g in new.groupby("date"):
                    loader.save_actuals(cfg, d, dict(zip(g["meal"], g["footfall"])))
                st.success(f"Merged {len(new)} rows.")
            except ValueError as exc:
                st.error(str(exc))


# --------------------------------------------------------------------------- page: model report
def page_report(cfg: CollegeConfig) -> None:
    st.title("📈 Model report")
    data_banner(cfg)
    c1, c2, _ = st.columns([1, 1, 3])
    fast = c1.checkbox("Fast mode", value=True, help="Fewer tuning rounds: ~1 minute instead of a few.")
    if c2.button("🔁 Train / retrain", type="primary"):
        from canteen_ai.models.trainer import train_college_model

        try:
            with st.spinner("Training (tuning RandomForest, XGBoost, LightGBM...)"):
                train_college_model(cfg, fast=fast, log=lambda *_: None)
            st.cache_resource.clear()
            st.success("Done.")
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
    m = read_metrics(cfg)
    if not m:
        st.info("No report yet - train the model first.")
        return
    base = m["metrics"]["baseline_avg_same_weekday_4w"]
    ens = m["metrics"]["ensemble"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Typical error (MAE)", f"{ens['mae']:.0f} students", f"{ens['mae'] - base['mae']:.0f} vs simple average", delta_color="inverse")
    k2.metric("Total error", f"{ens['wape_pct']:.1f}%", f"{ens['wape_pct'] - base['wape_pct']:.1f} pts vs simple average", delta_color="inverse")
    k3.metric("Within ±10%", f"{ens['within_10pct']:.0f}% of meals")
    k4.metric("Safety level hit", f"{m['intervals']['safety_coverage_pct']:.0f}%", f"target {m['intervals']['safety_nominal_pct']:.0f}%", delta_color="off")
    st.caption(f"Tested on {m['holdout_days']} days the model never saw ({m['holdout_start']} to {m['holdout_end']}). "
               f"Trained {m['trained_at']} on {m['n_days']} days of **{m['data_source']}** data.")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Models compared", "Actual vs predicted", "Cook-for strategies", "Accuracy by horizon", "What drives footfall"])
    mdir = model_dir(cfg.college_id)
    with tab1:
        rows = [{"model": pretty(k), "MAE": v["mae"], "RMSE": v["rmse"], "WAPE %": v["wape_pct"], "within 10 %": v["within_10pct"]}
                for k, v in m["metrics"].items()]
        df = pd.DataFrame(rows).round(1)
        st.dataframe(df, hide_index=True, width="stretch")
        st.plotly_chart(px.bar(df, x="model", y="MAE", title="Lower is better (average students off per meal)"), width="stretch")
        st.caption("'Ridge reference' is a plain linear model, shown so you can see how much the tree-based models add.")
    with tab2:
        hp = pd.read_csv(mdir / "holdout_predictions.csv", parse_dates=["date"])
        meal = st.selectbox("Meal", cfg.meal_names, format_func=pretty)
        h = hp[hp["meal"] == meal]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=h["date"], y=h["high"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=h["date"], y=h["low"], fill="tonexty", line=dict(width=0), name="Likely range", fillcolor="rgba(99,110,250,0.18)"))
        fig.add_trace(go.Scatter(x=h["date"], y=h["actual"], name="Actual", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=h["date"], y=h["ensemble"], name="Predicted", mode="lines"))
        fig.update_layout(height=380, yaxis_title="Students", margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
    with tab3:
        bi = pd.DataFrame(m["business_impact"]).T.reset_index()
        bi.columns = ["Strategy", "Food wasted %", "Meals that ran out %", "Students unserved %"]
        st.dataframe(bi.round(1), hide_index=True, width="stretch")
        st.caption("Replay of the held-out days: what if the kitchen had cooked this many portions each time? "
                   "'+10%' rows are what a careful cook does today (last week's number plus a cushion).")
    with tab4:
        hz = mdir / "horizon_backtest.csv"
        if hz.exists() and hz.stat().st_size > 30:
            h = pd.read_csv(hz)
            st.plotly_chart(px.line(h, x="horizon", y="wape_pct", markers=True, labels={"horizon": "Days ahead", "wape_pct": "Total error %"}), width="stretch")
            st.caption("Multi-day forecasts reuse earlier forecasts as inputs, so error can grow with the horizon (here it uses perfect weather).")
    with tab5:
        fi = pd.read_csv(mdir / "feature_importance.csv").head(15).iloc[::-1]
        st.plotly_chart(px.bar(fi, x="importance", y="feature", orientation="h", height=420), width="stretch")


# --------------------------------------------------------------------------- page: setup
def page_setup(cfg: CollegeConfig) -> None:
    st.title("⚙️ Setup & data")
    data_banner(cfg)
    st.markdown(f"**{cfg.name}** · {cfg.total_students:,} students · meals: " + ", ".join(pretty(m) for m in cfg.meal_names))
    d = data_dir(cfg.college_id)

    st.subheader("Data status")
    rows = []
    for fname in loader.FILE_COLUMNS:
        p = d / fname
        n = max(len(pd.read_csv(p)), 0) if p.exists() else 0
        rows.append({"file": fname, "rows": n, "status": "✅ ready" if n else ("⬜ optional / empty" if fname != "footfall.csv" else "❌ required")})
    st.dataframe(pd.DataFrame(rows), hide_index=True)

    st.subheader("Load data")
    with st.expander("🧪 Generate SIMULATED demo data (overwrites this college's CSVs)"):
        st.warning("Synthetic data only proves the system works. Don't present results from it as real-world accuracy.")
        n_days = st.slider("Days of history", 200, 720, 540)
        if st.button("Generate demo data"):
            from canteen_ai.data.synthetic import generate_demo_dataset

            generate_demo_dataset(cfg, days=n_days)
            st.success("Demo data created. Now train the model on the *Model report* page.")
    for fname, label in [("footfall.csv", "Footfall (date, meal, footfall)"), ("timetable.csv", "Timetable (day_of_week, start_time, end_time, room_capacity, avg_attendance)"),
                         ("weather.csv", "Weather (date, temp_c, rain_mm)"), ("events.csv", "Events (date, event_type)")]:
        with st.expander(f"⬆️ Upload {label}"):
            up = st.file_uploader(fname, type="csv", key=f"up_{fname}")
            if up is not None and st.button(f"Validate & save {fname}", key=f"btn_{fname}"):
                tmp = d / f"_{fname}"
                d.mkdir(parents=True, exist_ok=True)
                tmp.write_bytes(up.getvalue())
                try:
                    {"footfall.csv": lambda p: loader.read_footfall(p, cfg), "timetable.csv": loader.read_timetable,
                     "weather.csv": loader.read_weather, "events.csv": lambda p: loader.read_events(p, cfg)}[fname](tmp)
                    tmp.replace(d / fname)
                    if fname == "footfall.csv":
                        loader.set_data_source(cfg, "real")
                    st.success(f"{fname} saved.")
                except ValueError as exc:
                    tmp.unlink(missing_ok=True)
                    st.error(str(exc))

    st.subheader("Calendar events")
    st.caption("Tell the model about upcoming exams, holidays, fests... it learns how each one changes footfall.")
    e1, e2, e3 = st.columns([1, 1, 1])
    ev_date = e1.date_input("Date", value=dt.date.today() + dt.timedelta(days=1), key="ev_date")
    ev_type = e2.selectbox("Type", cfg.event_types, format_func=pretty)
    if e3.button("Add event"):
        loader.add_event(cfg, ev_date, ev_type)
        st.success("Added.")
    evp = d / "events.csv"
    if evp.exists() and evp.stat().st_size > 20:
        ev = pd.read_csv(evp)
        st.dataframe(ev.tail(15).iloc[::-1], hide_index=True)

    st.subheader("Menu & recipes")
    menu_rows = [{"meal": pretty(m), "dish": i["dish"], "portions per head": i["portions_per_head"]} for m, items in cfg.menu.items() for i in items]
    st.dataframe(pd.DataFrame(menu_rows), hide_index=True, width="stretch")
    st.caption(f"Edit `configs/colleges/{cfg.college_id}.yaml` to change dishes, recipes or meal timings.")

    st.subheader("Add another college")
    with st.form("onboard"):
        n1, n2 = st.columns(2)
        cid = n1.text_input("College id (lowercase, e.g. vit_pune)")
        cname = n2.text_input("Display name")
        n3, n4, n5 = st.columns(3)
        studs = n3.number_input("Total students", 100, 100000, 3000, step=100)
        lat = n4.number_input("Latitude", value=18.52, format="%.4f")
        lon = n5.number_input("Longitude", value=73.86, format="%.4f")
        if st.form_submit_button("Create college"):
            from canteen_ai.onboard import onboard_college

            try:
                onboard_college(cid.strip(), cname.strip() or None, int(studs), float(lat), float(lon))
                st.success(f"Created '{cid}'. Pick it in the sidebar, then upload its data.")
            except (ValueError, FileExistsError) as exc:
                st.error(str(exc))


# --------------------------------------------------------------------------- main
def main() -> None:
    colleges = list_colleges()
    if not colleges:
        st.error("No college configs found in configs/colleges/.")
        return
    with st.sidebar:
        st.markdown("## 🍽️ Canteen AI")
        college = st.selectbox("College", colleges, format_func=lambda c: c.replace("_", " ").title())
        page = st.radio("Go to", ["What to cook", "Log actuals", "Model report", "Setup & data"], label_visibility="collapsed")
    try:
        cfg = load_config(college)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Config problem: {exc}")
        return
    {"What to cook": page_forecast, "Log actuals": page_log, "Model report": page_report, "Setup & data": page_setup}[page](cfg)


main()
