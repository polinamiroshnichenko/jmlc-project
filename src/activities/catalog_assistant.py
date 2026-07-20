from pydantic import BaseModel
from temporalio import activity
from temporalio.exceptions import ApplicationError

from src.activities._clients import get_agent, get_llm_client
from src.clients.errors import LLMParseError
from src.schemas.agents.catalog_assistant import CatalogAssistantResponse
from src.schemas.medical_assistants import MedicalAssistantRequest


def _get_clients():
    from src.agents.catalog_assistant import CatalogAssistantAgent

    return get_agent(CatalogAssistantAgent), get_llm_client()


class CallLLMParams(BaseModel):
    request: MedicalAssistantRequest


@activity.defn(name="catalog_call_llm")
async def call_llm(params: CallLLMParams) -> CatalogAssistantResponse:
    activity.logger.info(f"call_llm: patient={params.request.patient.id}")
    agent, llm_client = _get_clients()
    messages = agent.get_messages(params.request)
    try:
        return await llm_client.chat_completion_request(
            model=agent.llm_request_params["model"],
            messages=messages,
            response_format=CatalogAssistantResponse,
            temperature=agent.llm_request_params.get("temperature"),
            max_tokens=agent.llm_request_params.get("max_tokens"),
            extra_body=agent.llm_request_params.get("extra_body"),
            retry_count=agent.llm_request_params.get("retry_count", 1),
        )
    except LLMParseError as e:
        raise ApplicationError(str(e), non_retryable=True) from e
