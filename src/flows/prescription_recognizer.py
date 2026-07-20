from datetime import timedelta
from typing import Optional

from pydantic import BaseModel
from temporalio import workflow

from src.flows.retry_policies import LLM_RETRY

with workflow.unsafe.imports_passed_through():
    from src.activities.prescription_recognizer import (
        PrescriptionCallLLMParams,
        PrescriptionNormalizeParams,
        call_llm,
        normalize,
    )
    from src.converters.chat_converter import convert_prescription_response
    from src.flows.callback import send_callback
    from src.schemas.agents.prescription_recognizer import (
        PrescriptionRecognitionNormalizedResult,
    )
    from src.schemas.chat import Topic


class PrescriptionWorkflowParams(BaseModel):
    topic: Topic
    is_chat: bool = False
    callback_url: Optional[str] = None
    llm_timeout_seconds: int = 120


def build_params(
    topic: Topic, initiator: Optional[str], is_chat: bool
) -> PrescriptionWorkflowParams:
    return PrescriptionWorkflowParams(
        topic=topic,
        is_chat=is_chat,
        callback_url=topic.service_attributes.get("callback_url"),
    )


@workflow.defn
class PrescriptionRecognizerFlow:
    @workflow.run
    async def run(
        self, params: PrescriptionWorkflowParams
    ) -> PrescriptionRecognitionNormalizedResult:
        workflow.logger.info(
            f"PrescriptionRecognizerFlow started: patient={params.topic.patient.id}"
        )

        topic_for_callback = params.topic if params.is_chat else None

        recognition = await workflow.execute_activity(
            call_llm,
            PrescriptionCallLLMParams(topic=params.topic),
            start_to_close_timeout=timedelta(seconds=params.llm_timeout_seconds),
            retry_policy=LLM_RETRY,
        )

        if not recognition.items:
            workflow.logger.info("No items recognized — skipping normalization")
            empty = PrescriptionRecognitionNormalizedResult(items=[])
            await send_callback(
                params.callback_url,
                topic_for_callback,
                empty,
                convert_prescription_response,
            )
            return empty

        normalized = await workflow.execute_activity(
            normalize,
            PrescriptionNormalizeParams(recognition_result=recognition),
            start_to_close_timeout=timedelta(seconds=params.llm_timeout_seconds),
            retry_policy=LLM_RETRY,
        )

        await send_callback(
            params.callback_url,
            topic_for_callback,
            normalized,
            convert_prescription_response,
        )

        workflow.logger.info("PrescriptionRecognizerFlow completed")
        return normalized
