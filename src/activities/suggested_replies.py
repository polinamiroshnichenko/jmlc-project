from typing import List

from pydantic import BaseModel
from temporalio import activity

from src.activities._clients import get_agent, get_llm_client
from src.schemas.llm import TextContent
from src.schemas.medical_assistants import MedicalAssistantRequest


def _get_clients():
    from src.agents.suggested_replies import SuggestedRepliesAgent

    return get_agent(SuggestedRepliesAgent), get_llm_client()


class CallLLMParams(BaseModel):
    request: MedicalAssistantRequest


@activity.defn(name="suggested_replies_call_llm")
async def call_llm(params: CallLLMParams) -> List[TextContent]:
    activity.logger.info(f"call_llm: patient={params.request.patient.id}")
    agent, llm_client = _get_clients()
    messages = agent.get_messages(params.request)

    raw = await llm_client.chat_completion_request(
        model=agent.llm_request_params["model"],
        messages=messages,
        response_format=None,
        temperature=agent.llm_request_params.get("temperature"),
        max_tokens=agent.llm_request_params.get("max_tokens"),
        extra_body=agent.llm_request_params.get("extra_body"),
        retry_count=agent.llm_request_params.get("retry_count", 1),
    )

    return [TextContent(text=part.strip()) for part in raw.split(";") if part.strip()]
