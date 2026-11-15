import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture()
def home(tmp_path, monkeypatch):
    """An isolated copy of the project folder so tests never touch your real data/models."""
    (tmp_path / "configs" / "colleges").mkdir(parents=True)
    shutil.copy(REPO / "configs" / "colleges" / "demo_college.yaml", tmp_path / "configs" / "colleges")
    monkeypatch.setenv("CANTEEN_AI_HOME", str(tmp_path))
    return tmp_path
