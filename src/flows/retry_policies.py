from datetime import timedelta

from temporalio.common import RetryPolicy

NO_RETRY = RetryPolicy(maximum_attempts=1)
LLM_RETRY = RetryPolicy(maximum_attempts=2, initial_interval=timedelta(seconds=5))
RETRY_3 = RetryPolicy(
    maximum_attempts=3, backoff_coefficient=2.0, initial_interval=timedelta(seconds=1)
)
