import os
from pathlib import Path

DATA_DIR = Path(os.getenv("RESILENS_DATA_DIR", "data"))
SCENARIOS_DIR = DATA_DIR / "scenarios"
UPLOADS_DIR = DATA_DIR / "uploads"
WEIGHTS_DIR = Path(os.getenv("RESILENS_WEIGHTS_DIR", "weights"))
RUNS_DIR = Path(os.getenv("RESILENS_RUNS_DIR", "runs"))
CONFIGS_DIR = Path(os.getenv("RESILENS_CONFIGS_DIR", "configs"))

API_URL = os.getenv("RESILENS_API_URL", "http://127.0.0.1:8000")
