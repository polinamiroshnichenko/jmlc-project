from datetime import timedelta
from typing import Callable, Optional

from pydantic import BaseModel
from temporalio import workflow

from src.flows.retry_policies import RETRY_3

with workflow.unsafe.imports_passed_through():
    from src.activities.callbacks import PostCallbackParams, post_callback
    from src.schemas.chat import Topic


async def send_callback(
    callback_url: Optional[str],
    topic: Optional[Topic],
    response: BaseModel,
    convert_function: Callable[[Topic, BaseModel], Topic],
) -> None:
    if not callback_url:
        return
    if topic is not None:
        converted = convert_function(topic.model_copy(deep=True), response)
        result = converted.model_dump(mode="json", by_alias=True, exclude_none=True)
    else:
        result = response.model_dump(exclude_none=True)
    await workflow.execute_activity(
        post_callback,
        PostCallbackParams(
            url=callback_url,
            workflow_id=workflow.info().workflow_id,
            result=result,
        ),
        start_to_close_timeout=timedelta(seconds=10),
        retry_policy=RETRY_3,
    )
