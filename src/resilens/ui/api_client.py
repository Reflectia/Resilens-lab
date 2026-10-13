"""Httpx client for the REST API"""

from pathlib import Path

from resilens.config import config


class ApiClient:
    def __init__(self, base_url: str = config.API_URL) -> None:
        self.base_url = base_url

    def create_run(self, mode: str, profile: str, video: Path | None = None, scenario_id: str | None = None) -> dict:
        pass

    def get_run(self, run_id: str) -> dict:
        pass

    def get_report(self, run_id: str) -> dict:
        pass

    def list_scenarios(self) -> list[dict]:
        pass
