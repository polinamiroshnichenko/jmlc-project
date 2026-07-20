from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from datetime import timedelta
    from src.schemas.flow_inputs.interpretation_with_recomendations import (
        InterpretationWithRecomendationsParams,
    )
    from typing import Optional

    from src.activities.interpretation_recommendations_activities import (
        AnnotateRecomendationsReassignedProps,
        BuildCheckupDialogProps,
    )
    from src.converters.chat_converter import (
        build_context_from_topic,
        convert_interpretation_with_recommendations_callback,
    )
    from src.flows.callback import send_callback
    from src.flows.checkup_assistant import CheckupWorkflowParams
    from src.flows.diagnostic_report_interpreter import DiagnosticReportInterpreterProps
    from src.flows.report_recognizer import ReportRecognizerParams
    from src.flows.retry_policies import RETRY_3
    from src.schemas.chat import Topic
    from src.schemas.medical_assistants import MedicalAssistantRequest
    from src.schemas.recognition import InterpretationWithRecommendationsResult
    from src.utils import convert_patient


LOS_CLIENT_INITIATOR = "Los_client"


def build_params(
    topic: Topic, initiator: Optional[str], is_chat: bool
) -> InterpretationWithRecomendationsParams:
    return InterpretationWithRecomendationsParams(
        topic=topic,
        context=build_context_from_topic(topic),
        is_chat=is_chat,
        callback_url=topic.service_attributes.get("callback_url"),
    )


@workflow.defn(name="InterpretationWithRecomendationsFlow")
class InterpretationWithRecomendationsFlow:

    @workflow.run
    async def run(self, params: InterpretationWithRecomendationsParams) -> Topic:
        topic = params.topic
        initiator = topic.initiator

        # Recognize lab report from the topic via child workflow
        recognition_result = await workflow.execute_child_workflow(
            "ReportRecognizerFlow",
            ReportRecognizerParams(topic=topic, is_chat=params.is_chat),
            retry_policy=RETRY_3,
        )

        # Get interpretation result from child workflow
        interpretation_result = await workflow.execute_child_workflow(
            "DiagnosticReportInterpreterFlow",
            DiagnosticReportInterpreterProps(
                recognition_result=recognition_result,
                context=params.context,
                initiator=initiator,
                topic_id=topic.id,
                patient=convert_patient(topic.patient).model_dump(),
            ),
            retry_policy=RETRY_3,
        )

        if interpretation_result is None:
            raise ValueError("Failed to interpret results")

        # Build dialog for checkup assistant
        dialog = await workflow.execute_activity(
            "build_checkup_dialog",
            BuildCheckupDialogProps(
                interpretation_result=interpretation_result,
                initiator=initiator,
                recognition_result=recognition_result,
            ),
            schedule_to_close_timeout=timedelta(seconds=5),
            retry_policy=RETRY_3,
        )

        # Convert dialog to ChatMessages
        chat_messages = await workflow.execute_activity(
            "convert_dialog_to_chat_messages",
            dialog,
            schedule_to_close_timeout=timedelta(seconds=5),
            retry_policy=RETRY_3,
        )

        # Call checkup assistant via child workflow
        checkup_request = MedicalAssistantRequest(
            patient=convert_patient(topic.patient),
            messages=chat_messages,
        )
        checkup_params = CheckupWorkflowParams(
            request=checkup_request,
            is_chat=False,
            normalize_only_lab=True,
            initiator=initiator,
        )

        checkup_response = await workflow.execute_child_workflow(
            "CheckupAssistantFlow", checkup_params, retry_policy=RETRY_3
        )

        # Annotate recommendations with reassignment status if LOS client initiator
        if initiator == LOS_CLIENT_INITIATOR and checkup_response:
            await workflow.execute_activity(
                "annotate_recommendations_reassigned",
                AnnotateRecomendationsReassignedProps(
                    checkup_response=checkup_response,
                    recognition_result=recognition_result,
                ),
                schedule_to_close_timeout=timedelta(minutes=10),
                retry_policy=RETRY_3,
            )

        result = InterpretationWithRecommendationsResult(
            interpretation=interpretation_result,
            checkup_recommendations=checkup_response,
        )

        updated_topic = convert_interpretation_with_recommendations_callback(
            topic.model_copy(deep=True), result
        )

        await send_callback(
            params.callback_url,
            topic,
            result,
            convert_interpretation_with_recommendations_callback,
        )

        return updated_topic
