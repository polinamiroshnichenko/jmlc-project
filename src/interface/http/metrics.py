from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
from src.config import config


def configure_metrics(app: FastAPI) -> FastAPI:
    instrumentator = Instrumentator(
        should_group_status_codes=config.metrics.should_group_status_codes,
        should_ignore_untemplated=config.metrics.should_ignore_untemplated,
        should_respect_env_var=config.metrics.should_respect_env_var,
        should_instrument_requests_inprogress=config.metrics.should_instrument_requests_inprogress,
        excluded_handlers=config.metrics.excluded_handlers,
        inprogress_name=config.metrics.inprogress_name,
        inprogress_labels=config.metrics.inprogress_labels,
    )
    instrumentator.instrument(app).expose(app)
    return app
