from datetime import timedelta
from typing import List, Optional

from pydantic import BaseModel
from temporalio import workflow

from src.flows.retry_policies import LLM_RETRY

with workflow.unsafe.imports_passed_through():
    from src.activities.general_assistant import (
        GeneratorCallLLMParams,
        RouterCallLLMParams,
        generator_call_llm,
        router_call_llm,
    )
    from src.converters.chat_converter import convert_general_response
    from src.flows.callback import send_callback
    from src.schemas.agents.general_assistant import (
        GeneralAssistantResponse,
        GeneralAssistantRouterResponse,
        RouterRedirect,
    )
    from src.schemas.chat import AgentType, Topic
    from src.schemas.llm import ChatMessage
    from src.schemas.medical_assistants import MedicalAssistantRequest
    from src.utils import convert_patient

_EXTERNAL_REDIRECTS = {
    RouterRedirect.checkup_assistant,
    RouterRedirect.catalog_assistant,
    RouterRedirect.interpretation_assistant,
    RouterRedirect.prescription_recognizer,
}


class GeneralAssistantWorkflowParams(BaseModel):
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


def build_params(
    topic: Topic, initiator: Optional[str], is_chat: bool
) -> GeneralAssistantWorkflowParams:
    request = MedicalAssistantRequest(
        patient=convert_patient(topic.patient),
        messages=_topic_to_messages(topic),
        callback_url=topic.service_attributes.get("callback_url"),
    )
    return GeneralAssistantWorkflowParams(
        request=request,
        callback_url=topic.service_attributes.get("callback_url"),
        topic=topic if is_chat else None,
    )


@workflow.defn
class GeneralAssistantFlow:
    @workflow.run
    async def run(self, params: GeneralAssistantWorkflowParams) -> GeneralAssistantResponse:
        workflow.logger.info(f"GeneralAssistantFlow started: patient={params.request.patient.id}")

        router_result: GeneralAssistantRouterResponse = await workflow.execute_activity(
            router_call_llm,
            RouterCallLLMParams(request=params.request),
            start_to_close_timeout=timedelta(seconds=params.llm_timeout_seconds),
            retry_policy=LLM_RETRY,
        )

        external = [r for r in router_result.redirects if r in _EXTERNAL_REDIRECTS]
        if external:
            response = GeneralAssistantResponse(
                message=router_result.message,
                redirect_to=AgentType(external[0].value),
                codes=[],
            )
            await send_callback(
                params.callback_url, params.topic, response, convert_general_response
            )
            workflow.logger.info(f"GeneralAssistantFlow: external redirect to {external[0].value}")
            return response

        response: GeneralAssistantResponse = await workflow.execute_activity(
            generator_call_llm,
            GeneratorCallLLMParams(
                request=params.request,
                context_needed=self._build_context_needed(router_result.redirects),
            ),
            start_to_close_timeout=timedelta(seconds=params.llm_timeout_seconds),
            retry_policy=LLM_RETRY,
        )

        await send_callback(params.callback_url, params.topic, response, convert_general_response)

        workflow.logger.info("GeneralAssistantFlow completed")
        return response

    @staticmethod
    def _build_context_needed(redirects: List[RouterRedirect]) -> List[str]:
        context_needed = []
        for r in redirects:
            if r == RouterRedirect.general_assistant_codes:
                context_needed.extend(["doctors", "instrumental"])
            elif r == RouterRedirect.general_assistant_faq:
                context_needed.append("faq")
            elif r == RouterRedirect.general_assistant_bm:
                context_needed.append("bm_instructions")
        return context_needed
