# Resilens v2.0

Resilens is a platform for reproducible experiments with object detection and multi-object tracking in video under adverse conditions (fog, rain, low light, motion blur). It compares processing modes A–E by tracking quality, latency and resource usage.

## Running locally

Requires Python 3.13.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Put model weights into `weights/` and videos and scenarios into `data/` (both folders are excluded from git).

```powershell
uvicorn resilens.api.app:app --reload    # REST API: http://127.0.0.1:8000/docs
streamlit run src/resilens/ui/app.py     # Web UI:   http://localhost:8501
resilens run path/to/video.mp4 --mode A --profile minimal   # CLI
pytest                                   # tests
ruff check .                             # linter
```

Or run everything in Docker: `docker compose up --build`.
