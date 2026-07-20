from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    import asyncio
    from datetime import timedelta
    from typing import List, Optional
    from src.flows.retry_policies import LLM_RETRY, NO_RETRY, RETRY_3
    from pydantic import BaseModel
    from src.activities.checkup_assistant import (
        CallLLMParams,
        FilterParams,
        NormalizeParams,
        PostprocessParams,
        call_llm,
        filter_recommendations,
        normalize_recommendation,
        postprocess,
    )
    from src.converters.chat_converter import convert_checkup_response
    from src.flows.callback import send_callback
    from src.schemas.agents.checkup_assistant import CheckupAssistantResponse
    from src.schemas.chat import Topic
    from src.schemas.llm import ChatMessage
    from src.schemas.medical_assistants import MedicalAssistantRequest
    from src.utils import convert_patient


class CheckupWorkflowParams(BaseModel):
    request: MedicalAssistantRequest
    is_chat: bool = False
    normalize_only_lab: bool = True
    initiator: Optional[str] = None
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


def build_params(topic: Topic, initiator: Optional[str], is_chat: bool) -> CheckupWorkflowParams:
    request = MedicalAssistantRequest(
        patient=convert_patient(topic.patient),
        messages=_topic_to_messages(topic),
        callback_url=topic.service_attributes.get("callback_url"),
    )
    return CheckupWorkflowParams(
        request=request,
        is_chat=is_chat,
        normalize_only_lab=True if is_chat else initiator != "budzdorov",
        initiator=initiator,
        callback_url=topic.service_attributes.get("callback_url"),
        topic=topic if is_chat else None,
    )


@workflow.defn
class CheckupAssistantFlow:
    @workflow.run
    async def run(self, params: CheckupWorkflowParams) -> CheckupAssistantResponse:
        workflow.logger.info(f"CheckupAssistantFlow started: patient={params.request.patient.id}")

        response = await workflow.execute_activity(
            call_llm,
            CallLLMParams(request=params.request),
            start_to_close_timeout=timedelta(seconds=params.llm_timeout_seconds),
            retry_policy=LLM_RETRY,
        )

        if not response.is_finished or not response.recommendations:
            workflow.logger.info("LLM response not finished — skipping normalization")
            await send_callback(
                params.callback_url, params.topic, response, convert_checkup_response
            )
            return response

        response = await self._normalize_recommendations(response, params)

        response = await workflow.execute_activity(
            filter_recommendations,
            FilterParams(response=response),
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RETRY_3,
        )

        response = await workflow.execute_activity(
            postprocess,
            PostprocessParams(
                response=response,
                is_chat=params.is_chat,
                output_format=params.request.format,
            ),
            start_to_close_timeout=timedelta(seconds=10),
            retry_policy=NO_RETRY,
        )

        await send_callback(params.callback_url, params.topic, response, convert_checkup_response)

        workflow.logger.info("CheckupAssistantFlow completed")
        return response

    async def _normalize_recommendations(
        self,
        response: CheckupAssistantResponse,
        params: CheckupWorkflowParams,
    ) -> CheckupAssistantResponse:
        norm_tasks = [
            workflow.execute_activity(
                normalize_recommendation,
                NormalizeParams(
                    recommendation=rec,
                    patient=params.request.patient,
                    normalize_only_lab=params.normalize_only_lab,
                    initiator=params.initiator,
                ),
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RETRY_3,
            )
            for rec in response.recommendations
        ]
        norm_results = await asyncio.gather(*norm_tasks)

        for rec, codes in zip(response.recommendations, norm_results):
            rec.hxids = codes

        return response
