import asyncio
import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class BaseHTTPClient:
    def __init__(self):
        self._client = httpx.AsyncClient()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _send_request(
        self,
        request: Any,
        url: str,
        params: Optional[Dict[str, str]] = None,
        retry_count: int = 3,
        retry_delay: int = 1,
    ) -> Optional[Any]:
        for attempt in range(retry_count):
            try:
                response = await self._client.post(url, json=request, params=params)
                response.raise_for_status()
                data = response.json()
                if data is not None:
                    return data
                logger.warning("Received None response, retrying...")
            except (httpx.HTTPError, ValueError) as e:
                logger.error(f"Request error (attempt {attempt + 1}/{retry_count}): {e}")

            if attempt < retry_count - 1:
                await asyncio.sleep(retry_delay)

        logger.warning("All retry attempts exhausted, returning None")
        return None
