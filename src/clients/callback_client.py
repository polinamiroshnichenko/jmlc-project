from temporalio.exceptions import ApplicationError

from src.clients.base_http_client import BaseHTTPClient


class CallbackClient(BaseHTTPClient):
    async def post_result(self, url: str, workflow_id: str, result: dict) -> None:
        payload = {
            "workflow_id": workflow_id,
            "status": "completed",
            "result": result,
        }
        response = await self._client.post(url, json=payload, timeout=10)
        if 400 <= response.status_code < 500:
            raise ApplicationError(
                f"Callback rejected with {response.status_code}", non_retryable=True
            )
        response.raise_for_status()
