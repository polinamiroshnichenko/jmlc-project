import contextvars
import time
from functools import wraps
from typing import Optional

from fastapi import HTTPException
from prometheus_client import Counter, Histogram

# Context variable for passing pipeline kind through async context
pipeline_kind_context: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "pipeline_kind", default=None
)

#  Unified metric for all pipelines with kind label
# Buckets optimized for LLM request durations (typically 1-60 seconds, sometimes up to 2 minutes)
# Default buckets [0.005, 0.01, ..., 10.0] are too granular for fast requests and too coarse for slow LLM requests
duration_seconds = Histogram(
    "duration_seconds",
    "Duration of pipeline requests in seconds",
    ["kind"],
    buckets=[
        0.1,  # 100ms
        0.5,  # 500ms
        1.0,  # 1s
        2.0,  # 2s
        5.0,  # 5s
        10.0,  # 10s
        15.0,  # 15s
        20.0,  # 20s
        25.0,  # 25s
        30.0,  # 30s
        35.0,  # 35s
        40.0,  # 40s
        50.0,  # 50s
        60.0,  # 60s
        120.0,  # 2min
        float("inf"),  # +Inf
    ],
)

# Token metrics for all pipelines
llm_input_tokens = Counter(
    "llm_input_tokens",
    "Total number of input tokens",
    ["prompt_type"],
)

llm_output_tokens = Counter(
    "llm_output_tokens",
    "Total number of output tokens",
    ["prompt_type"],
)

llm_reasoning_tokens = Counter(
    "llm_reasoning_tokens",
    "Total number of reasoning tokens",
    ["prompt_type"],
)

llm_cached_tokens = Counter(
    "llm_cached_tokens",
    "Total number of cache tokens",
    ["prompt_type"],
)

# Empty responses counter for all pipelines
llm_empty_responses = Counter(
    "llm_empty_responses",
    "Total number of empty responses from model (None, empty array, or empty dict)",
    ["prompt_type"],
)

# Total requests counter for all pipelines
llm_requests = Counter(
    "llm_requests",
    "Total number of LLM requests",
    ["prompt_type"],
)


def record_token_metrics(
    kind: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    reason_tokens: int = 0,
    cache_tokens: int = 0,
):
    """
    Record token usage metrics for a pipeline.

    Args:
        kind: Pipeline name (e.g., "checkup", "catalog", "general", etc.)
        prompt_tokens: Number of input/prompt tokens
        completion_tokens: Number of output/completion tokens
        reason_tokens: Number of reasoning tokens (for reasoning models)
        cache_tokens: Number of cache tokens
    """
    llm_input_tokens.labels(prompt_type=kind).inc(prompt_tokens)
    llm_output_tokens.labels(prompt_type=kind).inc(completion_tokens)
    llm_reasoning_tokens.labels(prompt_type=kind).inc(reason_tokens)
    llm_cached_tokens.labels(prompt_type=kind).inc(cache_tokens)


def record_empty_response(kind: str):
    """
    Record an empty response metric for a pipeline.

    Args:
        kind: Pipeline name (e.g., "checkup", "catalog", "general", etc.)
    """
    llm_empty_responses.labels(prompt_type=kind).inc()


def record_llm_request(kind: str):
    """
    Record an LLM request metric for a pipeline.

    Args:
        kind: Pipeline name (e.g., "checkup", "catalog", "general", etc.)
    """
    llm_requests.labels(prompt_type=kind).inc()


def is_empty_response(response) -> bool:
    """
    Check if a response is considered empty.

    Args:
        response: The response to check

    Returns:
        True if response is None, empty list, or empty dict
    """
    if response is None:
        return True
    if isinstance(response, list) and len(response) == 0:
        return True
    if isinstance(response, dict) and len(response) == 0:
        return True
    return False


def with_metrics(kind: str):
    """
    Decorator for tracking request duration and status codes for pipelines.
    Uses unified metrics duration_seconds and status_code_total with kind label.
    Also sets the pipeline kind in context for token metrics collection.

    Args:
        kind: Pipeline name (e.g., "checkup", "catalog", "general", etc.)
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Set the pipeline kind in context
            token = pipeline_kind_context.set(kind)
            start_time = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                raise
            except Exception:
                raise
            finally:
                duration = time.perf_counter() - start_time
                duration_seconds.labels(kind=kind).observe(duration)
                # Reset context
                pipeline_kind_context.reset(token)

        return wrapper

    return decorator
