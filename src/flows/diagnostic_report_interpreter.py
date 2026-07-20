from temporalio import workflow
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from pydantic import BaseModel, Field
    import json
    import asyncio
    from datetime import timedelta
    from src.flows.retry_policies import LLM_RETRY, RETRY_3
    from src.activities.template import RenderPromptProps

    from src.activities.llm_activities import RequestLLMProps
    from src.activities.recognition_data_processing import PostprocessInterpretationProps

    from src.activities.mcp import GetMcpTables
    from src.schemas.chat import InitiatorType
    from src.schemas.recognition import ReportInterpretatedResult, ReportRecognitionResult
    from src.services.prompts_service import prompts_service
    from src.utils import LLM_PIPELINE_CONFIGS
    from src.config import config


class DiagnosticReportInterpreterProps(BaseModel):
    recognition_result: ReportRecognitionResult
    context: str = Field(default_factory=str)
    initiator: InitiatorType | None = None
    topic_id: str | None = None
    patient: dict | None = None


@workflow.defn
class DiagnosticReportInterpreterFlow:

    @workflow.run
    async def run(self, props: DiagnosticReportInterpreterProps):

        prompt_config = prompts_service.prompts.get("report_interpretation")
        llm_config = LLM_PIPELINE_CONFIGS.get("report_interpretation")
        if not props.recognition_result:
            return ReportInterpretatedResult()
        historical_tables_json = "[]"
        has_historical_data = False

        if config.mcp.enabled and props.topic_id:
            historical_tables: str | None = await workflow.execute_activity(
                "get_mcp_tables",
                GetMcpTables(
                    recognition_result=props.recognition_result,
                    topic_id=props.topic_id,
                    patient_info=props.patient,
                ),
                schedule_to_close_timeout=timedelta(minutes=1),
                retry_policy=RETRY_3,
            )
            if historical_tables:
                has_historical_data = True
            historical_tables_json = historical_tables or historical_tables_json

        preprocessed_data: list[dict] = await workflow.execute_activity(
            "preprocess_recognition_data",
            props.recognition_result.items,
            schedule_to_close_timeout=timedelta(seconds=5),
            retry_policy=RETRY_3,
        )

        current_tables_json = json.dumps(preprocessed_data, ensure_ascii=False, indent=2)

        base_prompt, user_prompt = await asyncio.gather(
            workflow.start_activity(
                "render_prompt",
                RenderPromptProps(
                    template_str=prompt_config.base_prompt,
                    initiator=props.initiator,
                    has_historical_data=has_historical_data,
                ),
                schedule_to_close_timeout=timedelta(seconds=10),
                retry_policy=RETRY_3,
            ),
            workflow.start_activity(
                "render_prompt",
                RenderPromptProps(
                    template_str=prompt_config.template,
                    initiator=props.initiator,
                    has_historical_data=has_historical_data,
                    historical_tables=historical_tables_json,
                    context=props.context,
                    tables=current_tables_json,
                ),
                schedule_to_close_timeout=timedelta(seconds=10),
                retry_policy=RETRY_3,
            ),
        )

        interpreter_output = await workflow.execute_activity(
            "request_llm",
            RequestLLMProps(
                model_name=llm_config["model_name"],
                response_format="src.schemas.recognition.ReportInterpretatedResult",
                base_prompt=base_prompt,
                user_prompt=user_prompt,
            ),
            schedule_to_close_timeout=timedelta(minutes=1),
            retry_policy=LLM_RETRY,
        )

        if interpreter_output is None:
            msg = "LLM returned no result for interpretation"
            workflow.logger.error(msg)
            raise ApplicationError(msg)

        interpreted_result = await workflow.execute_activity(
            "postprocess_interpretation",
            PostprocessInterpretationProps(
                raw_data=preprocessed_data,
                interpreted_data=interpreter_output,
            ),
            schedule_to_close_timeout=timedelta(seconds=10),
            retry_policy=RETRY_3,
        )

        interpreted_result["structured_output"] = (
            props.recognition_result.structured_output
            if props.recognition_result.structured_output
            else {"resources": None}
        )

        return interpreted_result
