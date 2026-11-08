"""Web UI against the real REST API (stub Run Manager): the API client and a smoke run of the Streamlit page."""

from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import streamlit as st
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

import resilens.ui.api_client as api_client
from resilens.api.app import create_app
from resilens.testing.runs import STUB_METRICS, StubRunManager
from resilens.ui.api_client import ApiClient, ApiError

APP = Path(__file__).resolve().parents[2] / "src" / "resilens" / "ui" / "app.py"
# TestClient warns about per-request timeouts, which the real httpx client needs for uploads and rendering
pytestmark = pytest.mark.filterwarnings("ignore:You should not use the .timeout. argument")

TRACKS = "1,1,5,5,20,20,0.9,-1,-1,-1\n2,1,6,5,20,20,0.9,-1,-1,-1\n"


@pytest.fixture
def api(run_manager: StubRunManager, tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(run_manager, tmp_path / "uploads")) as test_client:
        yield test_client


@pytest.fixture
def client(api: TestClient) -> ApiClient:
    return ApiClient(base_url="http://testserver", http=api)


def start_run(client: ApiClient, api: TestClient, video: Path, mode: str = "A") -> str:
    run_id = client.create_run(mode, "minimal", "fog_l2.mp4", video.read_bytes())["run_id"]
    api.app.state.queue.join()
    run_dir = api.app.state.manager.store.run_dir(run_id)
    (run_dir / "tracks.txt").write_bytes(TRACKS.encode())
    (run_dir / "detections.txt").write_bytes(TRACKS.encode())
    return run_id


@pytest.fixture
def ui(monkeypatch: pytest.MonkeyPatch, client: ApiClient) -> Iterator[AppTest]:
    """The Streamlit page with its ApiClient talking to the test API."""
    monkeypatch.setattr(api_client, "ApiClient", lambda: client)
    st.cache_resource.clear()
    st.cache_data.clear()
    yield AppTest.from_file(str(APP), default_timeout=60)
    st.cache_resource.clear()
    st.cache_data.clear()


def test_client_starts_a_run_and_reads_its_results(client, api, make_video) -> None:
    run_id = start_run(client, api, make_video("clip.mp4", frames=3))

    assert client.get_run(run_id)["status"] == "completed"
    assert client.get_report(run_id)["metrics"] == STUB_METRICS
    assert "tracks.txt" in client.list_files(run_id)
    assert client.get_file(run_id, "tracks.txt").decode() == TRACKS
    assert client.get_video(run_id)[:4] == bytes.fromhex("1a45dfa3")
    assert [run["run_id"] for run in client.list_runs()] == [run_id]
    assert len(client.list_modes()) == 5 and client.list_profiles()


def test_client_raises_api_errors_with_detail(client, make_video) -> None:
    with pytest.raises(ApiError, match="Unknown mode 'Z'") as error:
        client.create_run("Z", "minimal", "clip.mp4", make_video("clip.mp4").read_bytes())
    assert error.value.status_code == 422

    with pytest.raises(ApiError, match="not found") as error:
        client.get_run("f" * 32)
    assert error.value.status_code == 404


def test_client_reports_unreachable_api() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = ApiClient(
        "http://api:8000", http=httpx.Client(base_url="http://api:8000", transport=httpx.MockTransport(refuse))
    )

    with pytest.raises(ApiError, match="Cannot reach the Resilens API at http://api:8000") as error:
        client.list_runs()
    assert error.value.status_code is None


def test_page_without_runs(ui: AppTest) -> None:
    ui.run()

    assert not ui.exception
    assert [tab.label for tab in ui.tabs] == ["New run", "Runs", "Comparison"]
    assert [box.label for box in ui.selectbox] == ["Mode", "Resource profile"]
    assert ui.selectbox[1].value == "minimal"
    assert any("No runs yet" in info.value for info in ui.info)


def test_page_shows_run_results_and_comparison(ui: AppTest, client, api, make_video) -> None:
    video = make_video("clip.mp4", frames=3)
    first = start_run(client, api, video, mode="A")
    start_run(client, api, video, mode="D")

    ui.run()

    assert not ui.exception, ui.exception
    assert {metric.label for metric in ui.metric} >= {"FPS", "Peak RAM, MB", "Processed frames"}
    run_details = next(box for box in ui.selectbox if box.label == "Run details")
    assert run_details.value != first  # newest run first
    assert any("no ground-truth annotations" in caption.value for caption in ui.caption)
    compare = next(select for select in ui.multiselect if select.label == "Runs to compare")
    assert len(compare.value) == 2
    reference = next(box for box in ui.selectbox if box.label.startswith("Reference run"))
    assert reference.value.startswith("A ·")


def test_page_shows_api_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    offline = ApiClient(
        "http://api:8000", http=httpx.Client(base_url="http://api:8000", transport=httpx.MockTransport(refuse))
    )
    monkeypatch.setattr(api_client, "ApiClient", lambda: offline)
    st.cache_resource.clear()
    page = AppTest.from_file(str(APP), default_timeout=60)

    page.run()

    assert not page.exception
    assert len(page.error) == 3 and "Cannot reach the Resilens API" in page.error[0].value
    st.cache_resource.clear()
