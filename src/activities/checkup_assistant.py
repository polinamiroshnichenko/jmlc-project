from typing import List, Optional

from pydantic import BaseModel
from temporalio import activity
from temporalio.exceptions import ApplicationError

from src.activities._clients import get_agent, get_labtest, get_llm_client
from src.clients.errors import LLMParseError
from src.schemas.agents.checkup_assistant import (
    CheckupAssistantRecommendation,
    CheckupAssistantResponse,
)
from src.schemas.chat import Code
from src.schemas.common import Patient
from src.schemas.medical_assistants import MedicalAssistantRequest


def _get_clients():
    from src.agents.checkup_assistant import CheckupAssistantAgent

    return get_agent(CheckupAssistantAgent), get_llm_client(), get_labtest()


class CallLLMParams(BaseModel):
    request: MedicalAssistantRequest


class NormalizeParams(BaseModel):
    recommendation: CheckupAssistantRecommendation
    patient: Patient
    normalize_only_lab: bool
    initiator: Optional[str] = None


class FilterParams(BaseModel):
    response: CheckupAssistantResponse


class PostprocessParams(BaseModel):
    response: CheckupAssistantResponse
    is_chat: bool
    output_format: str = "text"


@activity.defn
async def call_llm(params: CallLLMParams) -> CheckupAssistantResponse:
    activity.logger.info(f"call_llm: patient={params.request.patient.id}")
    agent, llm_client, _ = _get_clients()
    messages = agent.get_messages(params.request)
    try:
        return await llm_client.chat_completion_request(
            model=agent.llm_request_params["model"],
            messages=messages,
            response_format=CheckupAssistantResponse,
            temperature=agent.llm_request_params.get("temperature"),
            max_tokens=agent.llm_request_params.get("max_tokens"),
            extra_body=agent.llm_request_params.get("extra_body"),
            retry_count=agent.llm_request_params.get("retry_count", 1),
        )
    except LLMParseError as e:
        raise ApplicationError(str(e), non_retryable=True) from e


@activity.defn
async def normalize_recommendation(params: NormalizeParams) -> List[Code]:
    activity.logger.info(f"normalize_recommendation: {params.recommendation.name!r}")
    _, _, labtest = _get_clients()
    return await labtest.normalization_request(
        analysis=params.recommendation,
        patient=params.patient,
        normalize_only_lab=params.normalize_only_lab,
        initiator=params.initiator,
    )


@activity.defn
async def filter_recommendations(params: FilterParams) -> CheckupAssistantResponse:
    activity.logger.info("filter_recommendations")
    agent, _, labtest = _get_clients()
    return await agent.filter_recommendations(params.response, labtest.filter_intersected_analyses)


@activity.defn
async def postprocess(params: PostprocessParams) -> CheckupAssistantResponse:
    activity.logger.info(f"postprocess: is_chat={params.is_chat}")
    agent, _, _ = _get_clients()
    if params.is_chat:
        return agent.chat_postprocess(params.response)
    return agent.postprocess(params.response, params.output_format)
