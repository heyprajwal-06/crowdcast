import pytest

from canteen_ai.config import data_dir, load_config
from canteen_ai.onboard import onboard_college


def test_onboard_creates_config_and_templates(home):
    path = onboard_college("my_college", "My College", total_students=2500, latitude=19.07, longitude=72.87)
    cfg = load_config("my_college")
    assert path.exists() and cfg.name == "My College" and cfg.total_students == 2500
    assert cfg.latitude == 19.07 and cfg.longitude == 72.87
    for f in ("footfall.csv", "timetable.csv", "weather.csv", "events.csv", "README.txt"):
        assert (data_dir("my_college") / f).exists()


def test_onboard_rejects_bad_or_duplicate_ids(home):
    with pytest.raises(ValueError):
        onboard_college("Bad Id!")
    onboard_college("ok_id")
    with pytest.raises(FileExistsError):
        onboard_college("ok_id")
