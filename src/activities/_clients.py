"""Lazily-built, process-wide client/agent singletons shared by activities.

Construction is deferred to first call (not import time) so that importing an
activity module never reaches into config or pulls in heavy agent modules — this
keeps the Temporal workflow sandbox clean. ``lru_cache`` provides the
single-instance-per-process behavior the old hand-rolled ``global`` blocks did,
but thread-safely and without the boilerplate.
"""

from functools import lru_cache
from typing import TypeVar

T = TypeVar("T")


@lru_cache(maxsize=1)
def get_llm_client():
    from src.clients.llm_client import LLMClient
    from src.config import config

    return LLMClient(
        base_url=config.llm.base_url,
        api_key=config.llm.token,
    )


@lru_cache(maxsize=1)
def get_labtest():
    from src.clients.labtest_recognition import LabtestRecognitionClient
    from src.config import config

    return LabtestRecognitionClient(base_url=config.labtest_recognition.url)


@lru_cache(maxsize=None)
def get_agent(agent_cls: type[T]) -> T:
    return agent_cls()
