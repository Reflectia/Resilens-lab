import logging
from queue import Queue
from threading import Event, Thread

from resilens.runs.manager import RunManager
from resilens.runs.models import RunStatus

logger = logging.getLogger(__name__)


class RunQueue:
    """Executes submitted runs one at a time in a background thread.
    Runs execute in submission order, so measurements of one run are
    not disturbed by another.
    """

    def __init__(self, manager: RunManager) -> None:
        self.manager = manager
        self._queue: Queue[str | None] = Queue()
        self._stopping = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stopping.clear()
        self._thread = Thread(target=self._work, name="run-queue", daemon=True)
        self._thread.start()

    def submit(self, run_id: str) -> None:
        self._queue.put(run_id)

    def join(self) -> None:
        """Block until every submitted run has been processed."""
        self._queue.join()

    def stop(self) -> None:
        """Finish the run in progress and stop; runs still waiting stay QUEUED."""
        if self._thread is None:
            return
        self._stopping.set()
        self._queue.put(None)  # wake the worker if it is waiting for a run
        self._thread.join()
        self._thread = None

    def _work(self) -> None:
        while not self._stopping.is_set():
            run_id = self._queue.get()
            try:
                if run_id is not None and not self._stopping.is_set():
                    self._execute(run_id)
            finally:
                self._queue.task_done()

    def _execute(self, run_id: str) -> None:
        try:
            self.manager.execute(run_id)
        except Exception as error:  # a failing run must not stop the worker
            logger.exception("Run %s failed", run_id)
            try:
                self.manager.store.update_status(run_id, RunStatus.FAILED, error=str(error) or type(error).__name__)
            except Exception:
                logger.exception("Could not mark run %s as failed", run_id)
