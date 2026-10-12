from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def configs_dir() -> Path:
    return REPO_ROOT / "configs"
