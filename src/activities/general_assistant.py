from typing import List

from pydantic import BaseModel
from temporalio import activity
from temporalio.exceptions import ApplicationError

from src.activities._clients import get_agent, get_llm_client
from src.clients.errors import LLMParseError
from src.schemas.agents.general_assistant import (
    GeneralAssistantResponse,
    GeneralAssistantRouterResponse,
)
from src.schemas.medical_assistants import MedicalAssistantRequest


def _get_clients():
    from src.agents.general_assistant import (
        GeneralAssistantAgent,
        GeneralAssistantRouterAgent,
    )

    return (
        get_agent(GeneralAssistantRouterAgent),
        get_agent(GeneralAssistantAgent),
        get_llm_client(),
    )


class RouterCallLLMParams(BaseModel):
    request: MedicalAssistantRequest


class GeneratorCallLLMParams(BaseModel):
    request: MedicalAssistantRequest
    context_needed: List[str] = []


@activity.defn
async def router_call_llm(
    params: RouterCallLLMParams,
) -> GeneralAssistantRouterResponse:
    activity.logger.info(f"router_call_llm: patient={params.request.patient.id}")
    router_agent, _, llm_client = _get_clients()
    messages = router_agent.get_messages(params.request)
    try:
        return await llm_client.chat_completion_request(
            model=router_agent.llm_request_params["model"],
            messages=messages,
            response_format=GeneralAssistantRouterResponse,
            temperature=router_agent.llm_request_params.get("temperature"),
            max_tokens=router_agent.llm_request_params.get("max_tokens"),
            extra_body=router_agent.llm_request_params.get("extra_body"),
            retry_count=router_agent.llm_request_params.get("retry_count", 1),
        )
    except LLMParseError as e:
        raise ApplicationError(str(e), non_retryable=True) from e


@activity.defn
async def generator_call_llm(
    params: GeneratorCallLLMParams,
) -> GeneralAssistantResponse:
    activity.logger.info(
        f"generator_call_llm: patient={params.request.patient.id}, context={params.context_needed}"
    )
    _, generator_agent, llm_client = _get_clients()
    messages = generator_agent.get_messages(params.request, context_needed=params.context_needed)
    try:
        return await llm_client.chat_completion_request(
            model=generator_agent.llm_request_params["model"],
            messages=messages,
            response_format=GeneralAssistantResponse,
            temperature=generator_agent.llm_request_params.get("temperature"),
            max_tokens=generator_agent.llm_request_params.get("max_tokens"),
            extra_body=generator_agent.llm_request_params.get("extra_body"),
            retry_count=generator_agent.llm_request_params.get("retry_count", 1),
        )
    except LLMParseError as e:
        raise ApplicationError(str(e), non_retryable=True) from e
