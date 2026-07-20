import asyncio
from typing import Callable

from pydantic import BaseModel
from temporalio import activity
from temporalio.exceptions import ApplicationError

from src.clients.errors import LLMParseError
from src.clients.fhir_converter import ReportResultToFhirConverter
from src.clients.normalizer import LabTestRecognitionNormalizer
from src.clients.llm_client import LLMClient
from src.clients.terminology import TerminologyClient
from src.ports.assets_repository import AssetsRepository
from src.schemas.chat import Topic
from src.schemas.recognition import (
    RecognizedTable,
    ReportNormalizedResult,
    ReportRecognitionResult,
)


class ReportRecognizerCallLLMParams(BaseModel):
    topic: Topic


class BuildStructuredOutputParams(BaseModel):
    recognition_result: ReportRecognitionResult


class ReportRecognizerActivities:
    def __init__(self, llm_client: LLMClient, assets: AssetsRepository):
        from src.agents.report_recognizer import ReportRecognizerAgent

        self._agent = ReportRecognizerAgent()
        self._llm_client = llm_client
        self._normalizer = LabTestRecognitionNormalizer(
        llm_client=llm_client,
            terminology_client=TerminologyClient(),
            assets=assets,
        )
        self._fhir_converter = ReportResultToFhirConverter(assets=assets)

    def get_activities(self) -> list[Callable]:
        return [self.call_llm, self.build_structured_output]

    @activity.defn(name="report_recognizer_call_llm")
    async def call_llm(self, params: ReportRecognizerCallLLMParams) -> ReportRecognitionResult:
        activity.logger.info(f"report_recognizer_call_llm: patient={params.topic.patient.id}")
        messages = self._agent.get_recognition_messages(params.topic)
        try:
            return await self._llm_client.chat_completion_request(
                model=self._agent.llm_request_params["model"],
                messages=messages,
                response_format=ReportRecognitionResult,
                temperature=self._agent.llm_request_params.get("temperature"),
                max_tokens=self._agent.llm_request_params.get("max_tokens"),
                retry_count=self._agent.llm_request_params.get("retry_count", 1),
            )
        except LLMParseError as e:
            raise ApplicationError(str(e), non_retryable=True) from e

    async def _normalize_single_table(self, table: RecognizedTable) -> ReportNormalizedResult:
        try:
            return await self._normalizer([table])
        except Exception as e:
            activity.logger.error(f"Failed to normalize single table: {e}")
            return ReportNormalizedResult(items=[])

    async def _normalize_recognition_results(
        self, tables: list[RecognizedTable]
    ) -> ReportNormalizedResult:
        if not tables:
            return ReportNormalizedResult(items=[])

        results = await asyncio.gather(
            *(self._normalize_single_table(table) for table in tables),
            return_exceptions=True,
        )

        normalized_tables = []
        for result in results:
            if isinstance(result, ReportNormalizedResult) and result.items:
                normalized_tables.extend(result.items)
            else:
                activity.logger.error(f"Normalization failed: {result}")

        return ReportNormalizedResult(items=normalized_tables)

    @activity.defn(name="report_recognizer_build_structured_output")
    async def build_structured_output(self, params: BuildStructuredOutputParams) -> dict:
        """Normalize recognized tables (LOINC/UCUM) and build FHIR resources."""
        items = params.recognition_result.items or []
        if not items:
            return {"resources": None}

        normalized_result = await self._normalize_recognition_results(items)

        if not normalized_result.items:
            activity.logger.warning("Normalization produced no items; skipping FHIR generation")
            return {"resources": None}

        fhir_bundles = self._fhir_converter.convert_normalized_result_to_fhir(normalized_result)
        fhir_bundles_dicts = [
            bundle.dict(by_alias=True, exclude_none=True) for bundle in fhir_bundles
        ]

        activity.logger.info(f"Generated {len(fhir_bundles_dicts)} FHIR bundles")
        return {"resources": fhir_bundles_dicts}
