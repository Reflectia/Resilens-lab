from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from resilens import __version__
from resilens.api.routes import router
from resilens.api.worker import RunQueue
from resilens.config.config import RUNS_DIR, UPLOADS_DIR
from resilens.runs.manager import RunManager
from resilens.runs.store import RunStore


def create_app(manager: RunManager | None = None, uploads_dir: Path = UPLOADS_DIR) -> FastAPI:
    manager = manager if manager is not None else RunManager(RunStore(RUNS_DIR))
    queue = RunQueue(manager)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        queue.start()
        yield
        queue.stop()

    application = FastAPI(title="Resilens API", version=__version__, lifespan=lifespan)
    application.state.manager = manager
    application.state.queue = queue
    application.state.uploads_dir = uploads_dir
    application.include_router(router)
    return application


app = create_app()
