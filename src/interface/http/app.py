from src.interface.http.state import AppState

from .handlers import configure_routes
from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.config import config

from src.pipeline.temporal import build_temporal_executor
from src.interface.http.error_handling import register_error_handlers
from src.interface.http.metrics import configure_metrics
from src.interface.http.middlewares import register_middlewares


def build_app() -> FastAPI:
    app = FastAPI(
        lifespan=lifespan_manager,
        root_path=config.server.root_path,
    )
    app.include_router(configure_routes())
    app = register_error_handlers(app)
    app = register_middlewares(app)
    if config.metrics.enabled:
        app = configure_metrics(app)
    return app


@asynccontextmanager
async def lifespan_manager(app: FastAPI):
    pipeline_executor = await build_temporal_executor(
        config.temporal.address, config.temporal.namespace, config.temporal.task_queue
    )
    state = AppState(pipeline_executor)
    app.state = state
    yield
