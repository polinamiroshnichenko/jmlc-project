import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from temporalio import exceptions

logger = logging.getLogger(__name__)


def register_error_handlers(app: FastAPI):
    @app.exception_handler(exceptions.WorkflowAlreadyStartedError)
    def workflow_already_started_error(_req: Request, exc: exceptions.WorkflowAlreadyStartedError):
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"error": str(exc)})

    @app.exception_handler(Exception)
    async def unhandled_exception(_req: Request, exc: Exception):
        logger.exception("Unhandled exception on %s %s", _req.method, _req.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "Internal Server Error"},
        )

    return app
