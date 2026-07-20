from pydantic import BaseModel
from temporalio import activity

from src.clients.callback_client import CallbackClient

_callback_client: CallbackClient | None = None


def _get_client() -> CallbackClient:
    global _callback_client
    if _callback_client is None:
        _callback_client = CallbackClient()
    return _callback_client


class PostCallbackParams(BaseModel):
    url: str
    workflow_id: str
    result: dict


@activity.defn(name="post_callback")
async def post_callback(params: PostCallbackParams) -> None:
    activity.logger.info(f"post_callback: url={params.url} workflow_id={params.workflow_id}")
    await _get_client().post_result(params.url, params.workflow_id, params.result)
