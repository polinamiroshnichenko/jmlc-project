import time

from fastapi import FastAPI, Request, Response


import logging

logger = logging.getLogger()


def register_middlewares(app: FastAPI) -> FastAPI:
    @app.middleware("http")
    async def log_access(request: Request, call_next):
        start_time = time.perf_counter()
        response: Response = await call_next(request)
        request_time = time.perf_counter() - start_time

        logger.info(
            "http_request",
            extra={
                "client_ip": request.client.host,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration": request_time,
            },
        )
        return response

    return app
