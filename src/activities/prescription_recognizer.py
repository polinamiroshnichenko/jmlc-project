from pydantic import BaseModel
from temporalio import activity
from temporalio.exceptions import ApplicationError

from src.activities._clients import get_agent, get_llm_client
from src.clients.errors import LLMParseError
from src.schemas.agents.prescription_recognizer import (
    PrescriptionNormalizationLLMResult,
    PrescriptionRecognitionNormalizedResult,
    PrescriptionRecognitionResult,
)
from src.schemas.chat import Topic


def _get_clients():
    from src.agents.prescription_recognizer import PrescriptionRecognizerAgent

    return get_agent(PrescriptionRecognizerAgent), get_llm_client()


class PrescriptionCallLLMParams(BaseModel):
    topic: Topic


class PrescriptionNormalizeParams(BaseModel):
    recognition_result: PrescriptionRecognitionResult


@activity.defn(name="prescription_call_llm")
async def call_llm(params: PrescriptionCallLLMParams) -> PrescriptionRecognitionResult:
    activity.logger.info(f"prescription_call_llm: patient={params.topic.patient.id}")
    agent, llm_client = _get_clients()
    messages = agent.get_recognition_messages(params.topic)
    try:
        return await llm_client.chat_completion_request(
            model=agent.llm_request_params["model"],
            messages=messages,
            response_format=PrescriptionRecognitionResult,
            temperature=agent.llm_request_params.get("temperature"),
            max_tokens=agent.llm_request_params.get("max_tokens"),
            retry_count=agent.llm_request_params.get("retry_count", 1),
        )
    except LLMParseError as e:
        raise ApplicationError(str(e), non_retryable=True) from e


@activity.defn(name="prescription_normalize")
async def normalize(
    params: PrescriptionNormalizeParams,
) -> PrescriptionRecognitionNormalizedResult:
    activity.logger.info("prescription_normalize")
    from src.agents.prescription_recognizer import _load_normalizer_data

    agent, llm_client = _get_clients()
    name2code, code2name, complex_to_parts = _load_normalizer_data()
    messages = agent.get_normalization_messages(params.recognition_result, name2code)
    try:
        llm_result: PrescriptionNormalizationLLMResult = await llm_client.chat_completion_request(
            model=agent.normalizer_llm_params["model"],
            messages=messages,
            response_format=PrescriptionNormalizationLLMResult,
            temperature=agent.normalizer_llm_params.get("temperature"),
            max_tokens=agent.normalizer_llm_params.get("max_tokens"),
            retry_count=agent.normalizer_llm_params.get("retry_count", 1),
        )
    except LLMParseError as e:
        raise ApplicationError(str(e), non_retryable=True) from e

    processed = agent.postprocess_normalization(llm_result.items, code2name, complex_to_parts)
    return PrescriptionRecognitionNormalizedResult(items=processed)
