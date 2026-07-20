from datetime import timedelta
from typing import List, Optional

from pydantic import BaseModel
from temporalio import workflow

from src.flows.retry_policies import LLM_RETRY

with workflow.unsafe.imports_passed_through():
    from src.activities.suggested_replies import CallLLMParams, call_llm
    from src.schemas.chat import Topic
    from src.schemas.llm import ChatMessage, TextContent
    from src.schemas.medical_assistants import MedicalAssistantRequest
    from src.utils import convert_patient


class SuggestedRepliesWorkflowParams(BaseModel):
    request: MedicalAssistantRequest
    llm_timeout_seconds: int = 120


def _topic_to_messages(topic: Topic) -> List[ChatMessage]:
    messages = []
    for msg in topic.history:
        if msg.role not in ("User", "Assistant") or msg.content is None:
            continue
        role = "user" if msg.role == "User" else "assistant"
        text = getattr(msg.content, "text", None)
        if text:
            messages.append(ChatMessage(role=role, content=text))
    return messages


def build_params(
    topic: Topic, initiator: Optional[str], is_chat: bool
) -> SuggestedRepliesWorkflowParams:
    return SuggestedRepliesWorkflowParams(
        request=MedicalAssistantRequest(
            patient=convert_patient(topic.patient),
            messages=_topic_to_messages(topic),
        ),
    )


@workflow.defn
class SuggestedRepliesFlow:
    @workflow.run
    async def run(self, params: SuggestedRepliesWorkflowParams) -> List[TextContent]:
        workflow.logger.info(f"SuggestedRepliesFlow started: patient={params.request.patient.id}")

        replies: List[TextContent] = await workflow.execute_activity(
            call_llm,
            CallLLMParams(request=params.request),
            start_to_close_timeout=timedelta(seconds=params.llm_timeout_seconds),
            retry_policy=LLM_RETRY,
        )

        workflow.logger.info("SuggestedRepliesFlow completed")
        return replies
