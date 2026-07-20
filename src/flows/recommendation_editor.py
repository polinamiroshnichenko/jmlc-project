from datetime import timedelta
from typing import Optional

from pydantic import BaseModel
from temporalio import workflow

from src.flows.retry_policies import LLM_RETRY

with workflow.unsafe.imports_passed_through():
    from src.activities.recommendation_editor import CallLLMParams, call_llm
    from src.converters.chat_converter import convert_recommendation_editor_response
    from src.flows.callback import send_callback
    from src.schemas.agents.recommendation_editor import RecommendationEditorResponse
    from src.schemas.chat import Topic


class RecommendationEditorWorkflowParams(BaseModel):
    recommendation_text: str
    topic: Topic
    callback_url: Optional[str] = None
    llm_timeout_seconds: int = 120


def _extract_action_request(topic: Topic):
    for msg in reversed(topic.history):
        if msg.content and getattr(msg.content, "type", None) == "ActionRequest":
            return msg.content
    return None


def build_params(
    topic: Topic, initiator: Optional[str], is_chat: bool
) -> RecommendationEditorWorkflowParams:
    action_content = _extract_action_request(topic)
    if action_content is None:
        raise ValueError("No ActionRequest found in topic history")

    return RecommendationEditorWorkflowParams(
        recommendation_text=action_content.text,
        topic=topic,
        callback_url=topic.service_attributes.get("callback_url"),
    )


@workflow.defn
class RecommendationEditorFlow:
    @workflow.run
    async def run(self, params: RecommendationEditorWorkflowParams) -> RecommendationEditorResponse:
        workflow.logger.info(f"RecommendationEditorFlow started: patient={params.topic.patient.id}")

        response: RecommendationEditorResponse = await workflow.execute_activity(
            call_llm,
            CallLLMParams(
                recommendation_text=params.recommendation_text,
                patient=params.topic.patient,
            ),
            start_to_close_timeout=timedelta(seconds=params.llm_timeout_seconds),
            retry_policy=LLM_RETRY,
        )

        await send_callback(
            params.callback_url,
            params.topic,
            response,
            convert_recommendation_editor_response,
        )

        workflow.logger.info("RecommendationEditorFlow completed")
        return response
