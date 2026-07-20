from datetime import timedelta
from typing import List, Optional

from pydantic import BaseModel
from temporalio import workflow

from src.flows.retry_policies import LLM_RETRY

with workflow.unsafe.imports_passed_through():
    from src.activities.catalog_assistant import CallLLMParams, call_llm
    from src.converters.chat_converter import convert_catalog_response
    from src.flows.callback import send_callback
    from src.schemas.agents.catalog_assistant import CatalogAssistantResponse
    from src.schemas.chat import Topic
    from src.schemas.llm import ChatMessage
    from src.schemas.medical_assistants import MedicalAssistantRequest
    from src.utils import convert_patient


class CatalogWorkflowParams(BaseModel):
    request: MedicalAssistantRequest
    callback_url: Optional[str] = None
    llm_timeout_seconds: int = 120
    topic: Optional[Topic] = None


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


def build_params(topic: Topic, initiator: Optional[str], is_chat: bool) -> CatalogWorkflowParams:
    request = MedicalAssistantRequest(
        patient=convert_patient(topic.patient),
        messages=_topic_to_messages(topic),
        callback_url=topic.service_attributes.get("callback_url"),
    )
    return CatalogWorkflowParams(
        request=request,
        callback_url=topic.service_attributes.get("callback_url"),
        topic=topic if is_chat else None,
    )


@workflow.defn
class CatalogAssistantFlow:
    @workflow.run
    async def run(self, params: CatalogWorkflowParams) -> CatalogAssistantResponse:
        workflow.logger.info(f"CatalogAssistantFlow started: patient={params.request.patient.id}")

        response: CatalogAssistantResponse = await workflow.execute_activity(
            call_llm,
            CallLLMParams(request=params.request),
            start_to_close_timeout=timedelta(seconds=params.llm_timeout_seconds),
            retry_policy=LLM_RETRY,
        )

        await send_callback(params.callback_url, params.topic, response, convert_catalog_response)

        workflow.logger.info("CatalogAssistantFlow completed")
        return response
