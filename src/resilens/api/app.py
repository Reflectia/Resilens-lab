from fastapi import FastAPI

from resilens import __version__
from resilens.api.routes import router


def create_app() -> FastAPI:
    application = FastAPI(title="Resilens API", version=__version__)
    application.include_router(router)
    return application


app = create_app()
