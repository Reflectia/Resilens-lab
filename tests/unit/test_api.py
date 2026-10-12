"""The API contract is reachable and returns data."""

from fastapi.testclient import TestClient

from resilens.api.app import create_app

client = TestClient(create_app())


def test_create_run() -> None:
    response = client.post("/runs", data={"mode": "A", "profile": "minimal"})
    assert response.status_code == 202
    assert response.json()["status"] == "queued"


def test_get_run_and_report() -> None:
    assert client.get("/runs/mock-run-0001").status_code == 200
    assert "metrics" in client.get("/runs/mock-run-0001/report").json()


def test_list_scenarios() -> None:
    assert isinstance(client.get("/scenarios").json(), list)
