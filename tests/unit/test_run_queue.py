from pathlib import Path
from threading import Event, Thread

import pytest

from resilens.api.worker import RunQueue
from resilens.config.loader import build_run_config
from resilens.runs.models import RunStatus
from resilens.testing.runs import StubRunManager


@pytest.fixture
def queue(run_manager: StubRunManager):
    run_queue = RunQueue(run_manager)
    run_queue.start()
    yield run_queue
    if run_manager.release is not None:
        run_manager.release.set()
    run_queue.stop()


@pytest.fixture
def video(make_video) -> Path:
    return make_video()


def submit(run_manager: StubRunManager, queue: RunQueue, video: Path) -> str:
    run_id = run_manager.submit(build_run_config("A", "minimal", video))
    queue.submit(run_id)
    return run_id


def test_runs_execute_in_submission_order(queue, run_manager, video) -> None:
    run_ids = [submit(run_manager, queue, video) for _ in range(3)]
    queue.join()

    assert run_manager.executed == run_ids
    assert all(run_manager.store.get(run_id).status == RunStatus.COMPLETED for run_id in run_ids)


def test_exception_marks_run_failed_and_worker_continues(queue, run_manager, video) -> None:
    run_manager.fail_with = ValueError()
    failed_id = submit(run_manager, queue, video)
    queue.join()
    run_manager.fail_with = None
    next_id = submit(run_manager, queue, video)
    queue.join()

    failed = run_manager.store.get(failed_id)
    assert (failed.status, failed.error) == (RunStatus.FAILED, "ValueError")  # no message: type name is kept
    assert run_manager.store.get(next_id).status == RunStatus.COMPLETED


def test_stop_finishes_current_run_and_leaves_the_rest_queued(run_manager, video) -> None:
    queue = RunQueue(run_manager)
    queue.start()
    run_manager.release = Event()
    current, waiting = submit(run_manager, queue, video), submit(run_manager, queue, video)
    assert run_manager.started.wait(timeout=5)

    stopper = Thread(target=queue.stop)
    stopper.start()
    run_manager.release.set()
    stopper.join(timeout=5)

    assert not stopper.is_alive()
    assert run_manager.store.get(current).status == RunStatus.COMPLETED
    assert run_manager.store.get(waiting).status == RunStatus.QUEUED


def test_start_twice_keeps_one_worker(queue, run_manager, video) -> None:
    queue.start()
    run_ids = [submit(run_manager, queue, video) for _ in range(2)]
    queue.join()

    assert run_manager.executed == run_ids
