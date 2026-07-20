from pydantic import BaseModel
from temporalio import activity

from src.activities._clients import get_agent, get_llm_client
from src.schemas.agents.recommendation_editor import RecommendationEditorResponse
from src.schemas.chat import Patient


def _get_clients():
    from src.agents.recommendation_editor import RecommendationEditorAgent

    return get_agent(RecommendationEditorAgent), get_llm_client()


class CallLLMParams(BaseModel):
    recommendation_text: str
    patient: Patient


@activity.defn(name="recommendation_editor_call_llm")
async def call_llm(params: CallLLMParams) -> RecommendationEditorResponse:
    activity.logger.info(f"call_llm: patient={params.patient.id}")
    agent, llm_client = _get_clients()
    messages = agent.get_messages(
        recommendation_text=params.recommendation_text,
        patient=params.patient,
    )

    raw = await llm_client.chat_completion_request(
        model=agent.llm_request_params["model"],
        messages=messages,
        response_format=None,
        temperature=agent.llm_request_params.get("temperature"),
        max_tokens=agent.llm_request_params.get("max_tokens"),
        extra_body=agent.llm_request_params.get("extra_body"),
        retry_count=agent.llm_request_params.get("retry_count", 1),
    )

    return RecommendationEditorResponse(text=raw.strip())
