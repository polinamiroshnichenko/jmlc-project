import logging
import logging.config


from src.interface.http.app import build_app
from src.config import config as app_config

logging.config.dictConfig(app_config.logging)

logger = logging.getLogger()

app = build_app()


if __name__ == "__main__":
    import asyncio

    from hypercorn.asyncio import serve
    from hypercorn.config import Config

    config = Config()
    config.errorlog = None
    config.bind = [f"{app_config.server.addr}:{app_config.server.port}"]
    logger.info(f"Starting API server on {app_config.server.addr}:{app_config.server.port}")
    asyncio.run(serve(app, config))
