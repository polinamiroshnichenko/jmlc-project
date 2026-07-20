from datetime import timedelta
from typing import List, Optional

from pydantic import BaseModel
from temporalio import workflow

from src.flows.retry_policies import LLM_RETRY

with workflow.unsafe.imports_passed_through():
    from src.activities.recommendation_enricher import CallLLMParams, call_llm
    from src.converters.chat_converter import convert_recommendation_enricher_response
    from src.flows.callback import send_callback
    from src.schemas.agents.recommendation_editor import RecommendationEditorResponse
    from src.schemas.chat import ReportBlock, ReportContent, Topic


class RecommendationEnricherWorkflowParams(BaseModel):
    recommendation_text: str
    topic: Topic
    callback_url: Optional[str] = None
    checkup_recommendations: List[str] = []
    summary: Optional[str] = None
    llm_timeout_seconds: int = 120


def _extract_action_request(topic: Topic):
    for msg in reversed(topic.history):
        if msg.content and getattr(msg.content, "type", None) == "ActionRequest":
            return msg.content
    return None


def _extract_checkup_recommendations(topic: Topic) -> List[str]:
    recommendations: List[str] = []
    for msg in topic.history:
        if msg.model != "checkup-assistant":
            continue
        content = msg.content
        if not isinstance(content, ReportContent):
            continue
        if not content.report or not content.report.blocks:
            continue
        for block in content.report.blocks:
            _collect_block_titles(block, recommendations)
    return recommendations


def _collect_block_titles(block: ReportBlock, out: List[str]) -> None:
    if block.items:
        for item in block.items:
            if item.title:
                out.append(item.title)
    if block.blocks:
        for sub in block.blocks:
            _collect_block_titles(sub, out)


def build_params(
    topic: Topic, initiator: Optional[str], is_chat: bool
) -> RecommendationEnricherWorkflowParams:
    action_content = _extract_action_request(topic)
    if action_content is None:
        raise ValueError("No ActionRequest found in topic history")

    return RecommendationEnricherWorkflowParams(
        recommendation_text=action_content.text,
        topic=topic,
        callback_url=topic.service_attributes.get("callback_url"),
        checkup_recommendations=_extract_checkup_recommendations(topic),
        summary=topic.summary,
    )


@workflow.defn
class RecommendationEnricherFlow:
    @workflow.run
    async def run(
        self, params: RecommendationEnricherWorkflowParams
    ) -> RecommendationEditorResponse:
        workflow.logger.info(
            f"RecommendationEnricherFlow started: patient={params.topic.patient.id}"
        )

        response: RecommendationEditorResponse = await workflow.execute_activity(
            call_llm,
            CallLLMParams(
                recommendation_text=params.recommendation_text,
                patient=params.topic.patient,
                checkup_recommendations=params.checkup_recommendations,
                summary=params.summary,
            ),
            start_to_close_timeout=timedelta(seconds=params.llm_timeout_seconds),
            retry_policy=LLM_RETRY,
        )

        await send_callback(
            params.callback_url,
            params.topic,
            response,
            convert_recommendation_enricher_response,
        )

        workflow.logger.info("RecommendationEnricherFlow completed")
        return response
