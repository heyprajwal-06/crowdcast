"""Command line interface:  python -m canteen_ai <command> --college <id>"""
from __future__ import annotations

import argparse
import sys
import warnings

import pandas as pd

from canteen_ai.config import list_colleges, load_config


def _cfg(args):
    return load_config(args.college)


def cmd_generate(args) -> None:
    from canteen_ai.data.synthetic import generate_demo_dataset

    cfg = _cfg(args)
    paths = generate_demo_dataset(cfg, days=args.days, seed=args.seed)
    print(f"Generated SIMULATED data for '{cfg.college_id}' ({args.days} days):")
    for k, p in paths.items():
        print(f"  {k:<10} -> {p}")
    print("Reminder: this is synthetic. Replace footfall.csv with real counts when you have them.")


def cmd_train(args) -> None:
    from canteen_ai.models.trainer import train_college_model

    train_college_model(_cfg(args), fast=args.fast)


def cmd_forecast(args) -> None:
    from canteen_ai.models.predictor import Forecaster
    from canteen_ai.planning.bom import dish_plan, ingredient_plan

    cfg = _cfg(args)
    fc = Forecaster.load(cfg)
    start = pd.Timestamp(args.date) if args.date else pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    res = fc.forecast(pd.date_range(start, periods=args.days), fetch_weather=not args.offline)
    if res["days_ahead"].min() > 3:
        print(f"! Last logged actuals are from {fc.last_actual_date.date()} - forecasts rely on estimates. "
              "Log recent counts for better accuracy.\n")
    show = res[["date", "meal", "expected", "low", "high", "recommended_prep", "weather_source"]].copy()
    show["date"] = show["date"].dt.strftime("%a %d %b")
    print(show.to_string(index=False))
    dishes = dish_plan(cfg, res)
    _, shopping = ingredient_plan(cfg, dishes)
    if not shopping.empty:
        shopping["date"] = shopping["date"].dt.strftime("%a %d %b")
        print("\nIngredients to arrange (recommended prep + wastage buffer):")
        print(shopping.to_string(index=False))


def cmd_onboard(args) -> None:
    from canteen_ai.onboard import onboard_college

    path = onboard_college(args.college_id, args.name, args.students, args.lat, args.lon)
    print(f"Created {path}\nNext: put CSVs in data/raw/{args.college_id}/, edit the menu/recipes in the YAML, then train.")


def cmd_fetch_weather(args) -> None:
    from canteen_ai.data.loader import read_footfall
    from canteen_ai.config import data_dir
    from canteen_ai.weather import fetch_history

    cfg = _cfg(args)
    ff = read_footfall(data_dir(cfg.college_id) / "footfall.csv", cfg)
    w = fetch_history(cfg, ff["date"].min(), min(ff["date"].max(), pd.Timestamp.today().normalize() - pd.Timedelta(days=2)))
    if w is None:
        sys.exit("Could not download weather (offline, blocked, or no latitude/longitude in the config).")
    w["date"] = w["date"].dt.strftime("%Y-%m-%d")
    out = data_dir(cfg.college_id) / "weather.csv"
    w.to_csv(out, index=False)
    print(f"Saved {len(w)} days of weather -> {out}")


def cmd_quickstart(args) -> None:
    cmd_generate(args)
    cmd_train(args)
    args.days = 3
    cmd_forecast(args)


def main(argv=None) -> None:
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(prog="canteen_ai", description="Canteen footfall forecasting")
    sub = p.add_subparsers(dest="cmd", required=True)

    def college_arg(sp):
        sp.add_argument("--college", default="demo_college", help=f"college id (available: {list_colleges()})")

    g = sub.add_parser("generate-data", help="create SIMULATED demo data"); college_arg(g)
    g.add_argument("--days", type=int, default=540); g.add_argument("--seed", type=int, default=None)
    g.set_defaults(fn=cmd_generate)

    t = sub.add_parser("train", help="train + evaluate + save the model"); college_arg(t)
    t.add_argument("--fast", action="store_true", help="fewer tuning rounds (quicker, slightly less accurate)")
    t.set_defaults(fn=cmd_train)

    f = sub.add_parser("forecast", help="forecast footfall + ingredient list"); college_arg(f)
    f.add_argument("--date", help="first date YYYY-MM-DD (default: tomorrow)")
    f.add_argument("--days", type=int, default=3)
    f.add_argument("--offline", action="store_true", help="don't try to download the weather forecast")
    f.set_defaults(fn=cmd_forecast)

    o = sub.add_parser("onboard", help="add a new college"); o.add_argument("--college-id", required=True)
    o.add_argument("--name"); o.add_argument("--students", type=int)
    o.add_argument("--lat", type=float); o.add_argument("--lon", type=float)
    o.set_defaults(fn=cmd_onboard)

    w = sub.add_parser("fetch-weather", help="download historical weather into weather.csv"); college_arg(w)
    w.set_defaults(fn=cmd_fetch_weather)

    q = sub.add_parser("quickstart", help="generate data -> train -> forecast"); college_arg(q)
    q.add_argument("--days", type=int, default=540, help="days of demo history to simulate")
    q.add_argument("--seed", type=int, default=None)
    q.add_argument("--fast", action="store_true"); q.add_argument("--date", default=None)
    q.add_argument("--offline", action="store_true", help="don't try to download the weather forecast")
    q.set_defaults(fn=cmd_quickstart)

    args = p.parse_args(argv)
    try:
        args.fn(args)
    except (FileNotFoundError, ValueError, FileExistsError) as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
