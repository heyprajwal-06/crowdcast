"""Add a new college: creates its config (copied from the demo college) and empty CSV templates."""
from __future__ import annotations

import re
from pathlib import Path

from canteen_ai.config import config_dir, data_dir, load_config
from canteen_ai.data.loader import set_data_source, write_templates

_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]{1,40}$")


def onboard_college(
    college_id: str,
    name: str | None = None,
    total_students: int | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    template_id: str = "demo_college",
) -> Path:
    if not _ID_RE.match(college_id):
        raise ValueError("college_id must be lower-case letters/digits/underscore, e.g. 'vit_pune'")
    target = config_dir() / f"{college_id}.yaml"
    if target.exists():
        raise FileExistsError(f"{target} already exists")
    text = (config_dir() / f"{template_id}.yaml").read_text(encoding="utf-8")

    def sub(pattern: str, repl: str) -> None:
        nonlocal text
        text = re.sub(pattern, repl, text, count=1, flags=re.M)

    sub(r"^college_id:.*$", f"college_id: {college_id}")
    sub(r"^name:.*$", f"name: {name or college_id.replace('_', ' ').title()}")
    if total_students:
        sub(r"^total_students:\s*\d+", f"total_students: {int(total_students)}")
    if latitude is not None:
        sub(r"^(\s+latitude:\s*)[-\d.]+", rf"\g<1>{latitude}")
    if longitude is not None:
        sub(r"^(\s+longitude:\s*)[-\d.]+", rf"\g<1>{longitude}")
    target.write_text(text, encoding="utf-8")

    cfg = load_config(college_id)  # validates what we just wrote
    write_templates(cfg)
    set_data_source(cfg, "unknown")
    (data_dir(college_id) / "README.txt").write_text(
        "Put your real data here (see README.md > 'Using your own college data').\n"
        "  footfall.csv   date,meal,footfall                        <- required\n"
        "  timetable.csv  day_of_week,start_time,end_time,room_capacity,avg_attendance\n"
        "  weather.csv    date,temp_c,rain_mm   (or run: python -m canteen_ai fetch-weather)\n"
        "  events.csv     date,event_type\n"
        f"Then edit configs/colleges/{college_id}.yaml (meals, menu, recipes) and run:\n"
        f"  python -m canteen_ai train --college {college_id}\n",
        encoding="utf-8",
    )
    return target
