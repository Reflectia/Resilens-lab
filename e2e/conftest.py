import os
import shutil
import socket
import subprocess
import sys
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from pathlib import Path
from time import monotonic, sleep

import httpx
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
HOST = "127.0.0.1"


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("Resilens browser tests")
    group.addoption("--e2e-video", type=Path, help="Short control video")
    group.addoption("--e2e-coco", type=Path, help="COCO JSON matching the control video")
    group.addoption("--e2e-mot", type=Path, help="MOT TXT matching the control video")
    group.addoption(
        "--e2e-weights", type=Path, default=ROOT / "weights/yolox_nano_coco.pth", help="YOLOX-Nano weights file"
    )


def input_file(config: pytest.Config, option: str) -> Path:
    value = config.getoption(option)
    if value is None or not value.is_file():
        pytest.fail(f"{option} must point to an existing file (got {value})", pytrace=False)
    return value.resolve()


@pytest.fixture(scope="session")
def control_video(pytestconfig: pytest.Config) -> Path:
    return input_file(pytestconfig, "--e2e-video")


@pytest.fixture(scope="session")
def annotations(pytestconfig: pytest.Config) -> tuple[Path, Path]:
    return input_file(pytestconfig, "--e2e-coco"), input_file(pytestconfig, "--e2e-mot")


@contextmanager
def local_address() -> Generator[tuple[int, str]]:
    with socket.socket() as reservation:
        reservation.bind((HOST, 0))
        port = reservation.getsockname()[1]
        yield port, f"http://{HOST}:{port}"


def available_address() -> tuple[int, str]:
    with local_address() as address:
        return address


@contextmanager
def running_service(
    module: str, args: list[str], health_url: str, env: dict[str, str], log_path: Path
) -> Generator[None]:
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", module, *args], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            with httpx.Client(timeout=1, trust_env=False) as probe:
                deadline = monotonic() + 60
                while monotonic() < deadline and process.poll() is None:
                    try:
                        if probe.get(health_url).is_success:
                            break
                    except httpx.RequestError:
                        pass
                    sleep(0.1)
                else:
                    output = log_path.read_text(encoding="utf-8", errors="replace")
                    pytest.fail(f"{module} did not become ready at {health_url}\n{output}", pytrace=False)
            yield
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@pytest.fixture
def api_server(request: pytest.FixtureRequest, pytestconfig: pytest.Config, tmp_path: Path) -> Iterator[str]:
    scenario = getattr(request, "param", "healthy")
    if scenario == "offline":
        with local_address() as (_, api_url):
            yield api_url
        return

    configs = tmp_path / "configs"
    shutil.copytree(ROOT / "configs", configs)
    mode_file = configs / "modes/A.yaml"
    mode = yaml.safe_load(mode_file.read_text(encoding="utf-8"))
    weights = (
        tmp_path / "missing-yolox.pth" if scenario == "missing_weights" else input_file(pytestconfig, "--e2e-weights")
    )
    mode["detector_params"]["weights_path"] = str(weights)
    mode_file.write_text(yaml.safe_dump(mode), encoding="utf-8")
    env = {
        **os.environ,
        "RESILENS_CONFIGS_DIR": str(configs),
        "RESILENS_DATA_DIR": str(tmp_path / "data"),
        "RESILENS_RUNS_DIR": str(tmp_path / "runs"),
    }
    port, api_url = available_address()
    args = ["resilens.api.app:app", "--host", HOST, "--port", str(port)]
    with running_service("uvicorn", args, f"{api_url}/modes", env, tmp_path / "api.log"):
        yield api_url


@pytest.fixture
def api(api_server: str) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=api_server, trust_env=False) as client:
        yield client


@pytest.fixture
def ui_url(api_server: str, tmp_path: Path) -> Iterator[str]:
    port, url = available_address()
    args = [
        "run",
        "src/resilens/ui/app.py",
        "--server.address",
        HOST,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--server.fileWatcherType",
        "none",
        "--browser.gatherUsageStats",
        "false",
    ]
    env = {**os.environ, "RESILENS_API_URL": api_server, "NO_PROXY": HOST}
    with running_service("streamlit", args, f"{url}/_stcore/health", env, tmp_path / "ui.log"):
        yield url
