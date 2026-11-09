"""Httpx client for the REST API."""

import httpx

from resilens.config import config


class ApiError(Exception):
    """The API answered with an error, or could not be reached (status_code None)."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class ApiClient:
    def __init__(self, base_url: str = config.API_URL, http: httpx.Client | None = None, timeout: float = 30) -> None:
        self.base_url = base_url
        self.http = http if http is not None else httpx.Client(base_url=base_url, timeout=timeout)

    def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            response = self.http.request(method, path, **kwargs)
        except httpx.HTTPError as error:
            raise ApiError(f"Cannot reach the Resilens API at {self.base_url}: {error}") from error
        if response.is_error:
            try:
                detail = response.json()["detail"]
            except (ValueError, KeyError, TypeError):
                detail = response.text or response.reason_phrase
            raise ApiError(str(detail), response.status_code)
        return response

    def list_modes(self) -> list[dict]:
        return self._request("GET", "/modes").json()

    def list_profiles(self) -> list[dict]:
        return self._request("GET", "/profiles").json()

    def create_run(self, mode: str, profile: str, video_name: str, video: bytes, replay_mode: str = "offline") -> dict:
        """Upload a video and queue a run; returns {"run_id", "status"}."""
        return self._request(
            "POST",
            "/runs",
            data={"mode": mode, "profile": profile, "replay_mode": replay_mode},
            files={"video": (video_name, video, "application/octet-stream")},
            timeout=300,  # large uploads are validated before the API answers
        ).json()

    def list_runs(self) -> list[dict]:
        return self._request("GET", "/runs").json()

    def get_run(self, run_id: str) -> dict:
        return self._request("GET", f"/runs/{run_id}").json()

    def get_report(self, run_id: str) -> dict:
        return self._request("GET", f"/runs/{run_id}/report").json()

    def list_files(self, run_id: str) -> list[str]:
        return self._request("GET", f"/runs/{run_id}/files").json()

    def get_file(self, run_id: str, name: str) -> bytes:
        return self._request("GET", f"/runs/{run_id}/files/{name}").content

    def get_video(self, run_id: str) -> bytes:
        """Annotated WebM video; the first request renders it, which takes a while on long videos."""
        return self._request("GET", f"/runs/{run_id}/video", timeout=None).content
