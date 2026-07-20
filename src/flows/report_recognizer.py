from datetime import timedelta
from typing import Optional

from pydantic import BaseModel
from temporalio import workflow

from src.flows.retry_policies import LLM_RETRY, RETRY_3

with workflow.unsafe.imports_passed_through():
    from src.activities.report_recognizer import (
        BuildStructuredOutputParams,
        ReportRecognizerActivities,
        ReportRecognizerCallLLMParams,
    )
    from src.schemas.chat import Topic
    from src.schemas.recognition import ReportRecognitionResult


class ReportRecognizerParams(BaseModel):
    topic: Topic
    is_chat: bool = False
    callback_url: Optional[str] = None
    llm_timeout_seconds: int = 120
    fhir_timeout_seconds: int = 180


def build_params(topic: Topic, initiator: Optional[str], is_chat: bool) -> ReportRecognizerParams:
    return ReportRecognizerParams(
        topic=topic,
        is_chat=is_chat,
        callback_url=topic.service_attributes.get("callback_url"),
    )


@workflow.defn
class ReportRecognizerFlow:
    @workflow.run
    async def run(self, params: ReportRecognizerParams) -> ReportRecognitionResult:
        workflow.logger.info(f"ReportRecognizerFlow started: patient={params.topic.patient.id}")

        recognition = await workflow.execute_activity(
            ReportRecognizerActivities.call_llm,
            ReportRecognizerCallLLMParams(topic=params.topic),
            start_to_close_timeout=timedelta(seconds=params.llm_timeout_seconds),
            retry_policy=LLM_RETRY,
        )

        # Normalize recognized tables (LOINC/UCUM) and build FHIR resources.
        if recognition.items:
            structured_output = await workflow.execute_activity(
                ReportRecognizerActivities.build_structured_output,
                BuildStructuredOutputParams(recognition_result=recognition),
                start_to_close_timeout=timedelta(seconds=params.fhir_timeout_seconds),
                retry_policy=RETRY_3,
            )
            recognition.structured_output = structured_output

        workflow.logger.info(f"ReportRecognizerFlow completed: items={len(recognition.items)}")
        return recognition
