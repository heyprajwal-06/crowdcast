import pandas as pd
import pytest

from canteen_ai.config import data_dir, load_config
from canteen_ai.data.loader import load_college_data, read_footfall, save_actuals


def test_unknown_meal_rejected(home):
    cfg = load_config("demo_college")
    p = home / "f.csv"
    pd.DataFrame({"date": ["2026-01-01"], "meal": ["brunch"], "footfall": [10]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="brunch"):
        read_footfall(p, cfg)


def test_negative_and_text_rejected(home):
    cfg = load_config("demo_college")
    p = home / "f.csv"
    pd.DataFrame({"date": ["2026-01-01"], "meal": ["lunch"], "footfall": [-5]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="negative"):
        read_footfall(p, cfg)
    pd.DataFrame({"date": ["2026-01-01"], "meal": ["lunch"], "footfall": ["many"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="numbers"):
        read_footfall(p, cfg)


def test_missing_footfall_file_explains_fix(home):
    with pytest.raises(FileNotFoundError, match="generate-data"):
        load_college_data(load_config("demo_college"))


def test_save_actuals_upserts(home):
    cfg = load_config("demo_college")
    save_actuals(cfg, "2026-03-02", {"lunch": 100, "snacks": 50})
    save_actuals(cfg, "2026-03-02", {"lunch": 120})  # correction
    df = pd.read_csv(data_dir("demo_college") / "footfall.csv")
    assert len(df) == 2
    assert df.loc[df.meal == "lunch", "footfall"].iloc[0] == 120
