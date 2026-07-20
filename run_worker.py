import asyncio
import logging
import logging.config


from prometheus_client import start_http_server
from temporalio.client import Client
from temporalio.worker import Worker
from temporalio.contrib.pydantic import pydantic_data_converter

from src.activities.llm_activities import LLMActivities
from src.activities.report_recognizer import ReportRecognizerActivities
from src.agents.registry import AGENTS, get_all_activities, get_all_workflows
from src.activities.interpretation_recommendations_activities import (
    InterpretationRecommendationsActivities,
)
from src.activities.template import render_prompt
from src.adapters.fs_assets_repository import FSAssetsRepository
from src.clients.fhir_converter import FhirToReportResultConverter
from src.clients.mcp import McpClient
from src.clients.mcp_enrichment import McpEnrichmentService
from src.clients.llm_client import LLMClient
from src.clients.terminology import TerminologyClient
from src.config import config
from src.flows.diagnostic_report_interpreter import DiagnosticReportInterpreterFlow
from src.activities.recognition_data_processing import (
    postprocess_interpretation,
    preprocess_recognition_data,
)
from src.activities.mcp import McpActivities

logging.config.dictConfig(config.logging)

logger = logging.getLogger()


async def main():
    client = await Client.connect(
        config.temporal.address,
        namespace=config.temporal.namespace,
        data_converter=pydantic_data_converter,
    )
    assets = FSAssetsRepository(config.root_dir / "assets")
    llm_client = LLMClient(config.llm.base_url, config.llm.token)
    interpretation_activities = InterpretationRecommendationsActivities(
        FhirToReportResultConverter(),
        assets,
    )

    report_recognizer_activities = ReportRecognizerActivities(llm_client, assets)
    llm = LLMActivities(llm_client)
    activities = [
        *get_all_activities(),
        *interpretation_activities.get_activities(),
        *report_recognizer_activities.get_activities(),
        *llm.get_activities(),
        postprocess_interpretation,
        preprocess_recognition_data,
        render_prompt,
    ]
    if config.mcp.enabled:
        activities.append(
            McpActivities(
                McpEnrichmentService(
                    mcp_client=McpClient(
                        config.mcp.server_url, lab_history_tool=config.mcp.tool_name
                    ),
                    llm_client=llm_client,
                    terminology_client=TerminologyClient(base_url=config.terminology.url),
                )
            )
        )
    worker = Worker(
        client,
        task_queue=config.temporal.task_queue,
        workflows=[
            *get_all_workflows(),
            DiagnosticReportInterpreterFlow,
        ],
        activities=activities,
        max_concurrent_activities=config.worker.max_concurrent_activities,
        max_concurrent_workflow_tasks=config.worker.max_concurrent_workflow_tasks,
    )
    logger.info(f"Worker started on queue: {config.temporal.task_queue}")
    logger.info(f"Registered agents: {', '.join(AGENTS)}")
    start_http_server(9000)
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
