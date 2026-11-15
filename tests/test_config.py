import pytest
import yaml

from canteen_ai.config import ConfigError, _validate, config_dir, load_config


def test_demo_config_loads(home):
    cfg = load_config("demo_college")
    assert cfg.meal_names == ["breakfast", "lunch", "snacks"]
    assert cfg.quantiles == sorted(cfg.quantiles)
    assert cfg.recipes["Poha"]["onion"] == (0.02, "kg")


def test_missing_college_gives_helpful_error(home):
    with pytest.raises(FileNotFoundError, match="onboard"):
        load_config("nope")


def test_dish_without_recipe_is_rejected(home):
    raw = yaml.safe_load((config_dir() / "demo_college.yaml").read_text())
    raw["menu"]["lunch"].append({"dish": "Mystery Stew", "portions_per_head": 0.1})
    with pytest.raises(ConfigError, match="Mystery Stew"):
        _validate(raw)


def test_inconsistent_units_rejected(home):
    raw = yaml.safe_load((config_dir() / "demo_college.yaml").read_text())
    raw["recipes"]["Upma"]["onion"] = [0.02, "g"]  # onion is kg elsewhere
    with pytest.raises(ConfigError, match="onion"):
        _validate(raw)
