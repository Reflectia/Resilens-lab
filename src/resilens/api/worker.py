from resilens.runs.manager import RunManager


class RunQueue:
    def __init__(self, manager: RunManager) -> None:
        self.manager = manager

    def start(self) -> None:
        pass

    def submit(self, run_id: str) -> None:
        pass

    def stop(self) -> None:
        pass
