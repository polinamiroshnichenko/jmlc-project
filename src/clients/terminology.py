import asyncio
import logging
from typing import Dict, List, Optional
from urllib.parse import quote

import aiohttp

logger = logging.getLogger(__name__)


class TerminologyClient:
    """Client for terminology service API."""

    def __init__(self, base_url: str = "https://terminology-v2-stage.example.com/api/v1"):
        self.base_url = base_url
        self._results_cache = {}  # Кеш результатов для этого экземпляра

    async def _send_request(
        self, url: str, retry_count: int = 3, retry_delay: int = 3
    ) -> Optional[List[Dict]]:
        """Send GET request to terminology service with retries."""
        for attempt in range(retry_count):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        if response.status == 200:
                            response_data = await response.json()
                            if response_data is None:
                                logger.warning("Received None response, retrying...")
                            else:
                                return response_data
                        else:
                            logger.warning(
                                f"Received status {response.status} for URL {url}, retrying..."
                            )

            except aiohttp.ClientError as e:
                logger.error(f"Request Error (retry {attempt + 1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    logger.warning("There are no more retries left")
                    return None
            except ValueError as ve:
                logger.error(f"JSON decoding error: {ve}, retrying...")
                if attempt < retry_count - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    logger.warning("There are no more retries left")
                    return None

        return None

    async def search_loinc(
        self, content: str, loinc_url: str = "http://loinc.org"
    ) -> Optional[List[Dict]]:
        """
        Search LOINC concepts by content.
        Использует lru_cache для кеширования результатов.

        Args:
            content: Search query (will be converted to lowercase)
            loinc_url: LOINC URL (default: "http://loinc.org")

        Returns:
            List of concept dictionaries or None if request failed
        """
        # Content should always be lowercase
        content_lower = content.lower().strip()
        if not content_lower:
            return None

        # Проверяем кеш
        cache_key = (self.base_url, content_lower, loinc_url)
        if cache_key in self._results_cache:
            cached_result = self._results_cache[cache_key]
            logger.debug(f"Cached result found for content '{content_lower}'")
            return cached_result

        # URL encode the content
        content_encoded = quote(content_lower)
        url = f"{self.base_url}/Concepts?Url={quote(loinc_url)}&Content={content_encoded}"

        try:
            result = await self._send_request(url)

            # Сохраняем результат в кеш
            self._results_cache[cache_key] = result

            logger.debug(f"cache_key = {cache_key}, result cached")
            return result
        except Exception as e:
            logger.error(f"Error searching LOINC for content '{content_lower}': {e}")
            return None

    async def search_loinc_multiple(
        self, contents: List[str], loinc_url: str = "http://loinc.org"
    ) -> Dict[str, Optional[List[Dict]]]:
        """
        Search LOINC concepts for multiple contents in batches of 5 (parallel within batch).

        Args:
            contents: List of search queries
            loinc_url: LOINC URL (default: "http://loinc.org")

        Returns:
            Dictionary mapping content to search results
        """
        if not contents:
            return {}

        batch_size = 5
        results = {}

        for i in range(0, len(contents), batch_size):
            batch = contents[i : i + batch_size]
            # Обрабатываем батч параллельно
            tasks = [self.search_loinc(content, loinc_url) for content in batch]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Формируем словарь результатов
            for j in range(len(batch)):
                content = batch[j]
                result = batch_results[j]
                if isinstance(result, Exception):
                    logger.error(f"Error searching LOINC for content '{content}': {result}")
                    results[content] = None
                else:
                    results[content] = result

        return results
