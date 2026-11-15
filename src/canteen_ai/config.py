"""Load and validate per-college configuration (configs/colleges/<college_id>.yaml)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a college config file is missing something or is inconsistent."""


# --------------------------------------------------------------------------- paths
def project_root() -> Path:
    """Repo root. Can be overridden with the CANTEEN_AI_HOME env var (used by tests)."""
    env = os.environ.get("CANTEEN_AI_HOME")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    return project_root() / "configs" / "colleges"


def data_dir(college_id: str) -> Path:
    return project_root() / "data" / "raw" / college_id


def model_dir(college_id: str) -> Path:
    return project_root() / "models" / college_id


def list_colleges() -> list[str]:
    if not config_dir().exists():
        return []
    return sorted(p.stem for p in config_dir().glob("*.yaml"))


# --------------------------------------------------------------------------- helpers
def hm_to_min(value: Any) -> int:
    """'12:30' -> 750 (minutes since midnight)."""
    try:
        h, m = str(value).strip().split(":")[:2]
        return int(h) * 60 + int(m)
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Bad time '{value}' - expected HH:MM (24h)") from exc


@dataclass
class CollegeConfig:
    raw: dict[str, Any]

    # ---- identity
    @property
    def college_id(self) -> str:
        return str(self.raw["college_id"])

    @property
    def name(self) -> str:
        return str(self.raw.get("name", self.college_id))

    @property
    def total_students(self) -> int:
        return int(self.raw.get("total_students", 0))

    @property
    def latitude(self) -> float | None:
        return (self.raw.get("location") or {}).get("latitude")

    @property
    def longitude(self) -> float | None:
        return (self.raw.get("location") or {}).get("longitude")

    @property
    def timezone(self) -> str:
        return (self.raw.get("location") or {}).get("timezone", "Asia/Kolkata")

    # ---- meals / events
    @property
    def meals(self) -> dict[str, tuple[int, int]]:
        return {
            str(k).lower(): (hm_to_min(v["start"]), hm_to_min(v["end"]))
            for k, v in self.raw["meals"].items()
        }

    @property
    def meal_names(self) -> list[str]:
        return list(self.meals)

    @property
    def event_types(self) -> list[str]:
        return [str(e).lower() for e in self.raw.get("event_types", [])]

    # ---- forecast settings
    @property
    def _fc(self) -> dict[str, Any]:
        return self.raw.get("forecast") or {}

    @property
    def holdout_days(self) -> int:
        return int(self._fc.get("holdout_days", 60))

    @property
    def interval(self) -> tuple[float, float]:
        lo, hi = self._fc.get("interval", [0.10, 0.90])
        return float(lo), float(hi)

    @property
    def safety_quantile(self) -> float:
        return float(self._fc.get("safety_quantile", 0.80))

    @property
    def quantiles(self) -> list[float]:
        return sorted({self.interval[0], self.safety_quantile, self.interval[1]})

    # ---- planning
    @property
    def wastage_buffer(self) -> float:
        return float((self.raw.get("planning") or {}).get("wastage_buffer", 0.05))

    @property
    def menu(self) -> dict[str, list[dict[str, Any]]]:
        return {str(k).lower(): v for k, v in (self.raw.get("menu") or {}).items()}

    @property
    def recipes(self) -> dict[str, dict[str, tuple[float, str]]]:
        out: dict[str, dict[str, tuple[float, str]]] = {}
        for dish, ings in (self.raw.get("recipes") or {}).items():
            out[dish] = {ing: (float(q[0]), str(q[1])) for ing, q in ings.items()}
        return out

    @property
    def synthetic(self) -> dict[str, Any]:
        return self.raw.get("synthetic") or {}


def _validate(raw: dict[str, Any]) -> None:
    for key in ("college_id", "meals"):
        if key not in raw:
            raise ConfigError(f"Config is missing required key '{key}'")
    cfg = CollegeConfig(raw)
    meals = cfg.meals  # also validates the time strings
    if not meals:
        raise ConfigError("'meals' must contain at least one meal")
    for name, (s, e) in meals.items():
        if s >= e:
            raise ConfigError(f"Meal '{name}': start must be before end")
    lo, hi = cfg.interval
    if not (0 < lo < cfg.safety_quantile < hi < 1) and not (0 < lo < hi < 1):
        raise ConfigError("forecast.interval must be two quantiles like [0.1, 0.9]")
    if not 0.5 <= cfg.safety_quantile < 1:
        raise ConfigError("forecast.safety_quantile must be between 0.5 and 1")
    recipes = cfg.recipes
    for meal, items in cfg.menu.items():
        if meal not in meals:
            raise ConfigError(f"menu has meal '{meal}' which is not in 'meals'")
        for item in items:
            dish = item.get("dish")
            if dish not in recipes:
                raise ConfigError(f"Dish '{dish}' (meal '{meal}') has no entry under 'recipes'")
            if float(item.get("portions_per_head", 0)) <= 0:
                raise ConfigError(f"Dish '{dish}': portions_per_head must be > 0")
    units: dict[str, str] = {}
    for dish, ings in recipes.items():
        for ing, (_, unit) in ings.items():
            if units.setdefault(ing, unit) != unit:
                raise ConfigError(
                    f"Ingredient '{ing}' uses two different units ({units[ing]} and {unit}) - "
                    "use one unit per ingredient"
                )


def load_config(college_id: str) -> CollegeConfig:
    path = config_dir() / f"{college_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No config for '{college_id}'. Available: {list_colleges() or 'none'}. "
            f"Create one with:  python -m canteen_ai onboard --college-id {college_id}"
        )
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    _validate(raw)
    return CollegeConfig(raw)
