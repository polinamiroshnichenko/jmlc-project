from dataclasses import dataclass
from typing import Callable, Dict, Tuple

from src.activities.callbacks import post_callback
from src.activities.general_assistant import (
    generator_call_llm,
    router_call_llm,
)
from src.activities.checkup_assistant import (
    call_llm,
    filter_recommendations,
    normalize_recommendation,
    postprocess,
)
from src.activities.catalog_assistant import call_llm as catalog_call_llm
from src.activities.prescription_recognizer import (
    call_llm as prescription_call_llm,
    normalize as prescription_normalize,
)


from src.activities.suggested_replies import call_llm as suggested_replies_call_llm


from src.flows.catalog_assistant import (
    CatalogAssistantFlow,
    build_params as catalog_build_params,
)
from src.activities.recommendation_editor import (
    call_llm as recommendation_editor_call_llm,
)
from src.activities.recommendation_enricher import (
    call_llm as recommendation_enricher_call_llm,
)
from src.flows.general_assistant import (
    GeneralAssistantFlow,
    build_params as general_assistant_build_params,
)
from src.flows.checkup_assistant import (
    CheckupAssistantFlow,
    build_params as checkup_build_params,
)
from src.flows.interpretation_recomendations import (
    InterpretationWithRecomendationsFlow,
    build_params as interpretation_build_params,
)
from src.flows.prescription_recognizer import (
    PrescriptionRecognizerFlow,
    build_params as prescription_build_params,
)
from src.flows.report_recognizer import (
    ReportRecognizerFlow,
    build_params as report_recognizer_build_params,
)
from src.flows.suggested_replies import (
    SuggestedRepliesFlow,
    build_params as suggested_replies_build_params,
)
from src.flows.recommendation_editor import (
    RecommendationEditorFlow,
    build_params as recommendation_editor_build_params,
)
from src.flows.recommendation_enricher import (
    RecommendationEnricherFlow,
    build_params as recommendation_enricher_build_params,
)


@dataclass(frozen=True)
class AgentSpec:
    workflow: type
    activities: Tuple[Callable, ...]
    build_params: Callable
    sync: bool = False


AGENTS: Dict[str, AgentSpec] = {
    "general-assistant": AgentSpec(
        workflow=GeneralAssistantFlow,
        activities=(
            router_call_llm,
            generator_call_llm,
            post_callback,
        ),
        build_params=general_assistant_build_params,
    ),
    "checkup-assistant": AgentSpec(
        workflow=CheckupAssistantFlow,
        activities=(
            call_llm,
            normalize_recommendation,
            filter_recommendations,
            postprocess,
            post_callback,
        ),
        build_params=checkup_build_params,
    ),
    "catalog-assistant": AgentSpec(
        workflow=CatalogAssistantFlow,
        activities=(
            catalog_call_llm,
            post_callback,
        ),
        build_params=catalog_build_params,
    ),
    "prescription-recognizer": AgentSpec(
        workflow=PrescriptionRecognizerFlow,
        activities=(
            prescription_call_llm,
            prescription_normalize,
            post_callback,
        ),
        build_params=prescription_build_params,
    ),
    "suggested-replies": AgentSpec(
        workflow=SuggestedRepliesFlow,
        activities=(suggested_replies_call_llm,),
        build_params=suggested_replies_build_params,
        sync=True,
    ),
    "report-recognizer": AgentSpec(
        workflow=ReportRecognizerFlow,
        activities=tuple(),
        build_params=report_recognizer_build_params,
    ),
    "interpretation-with-recomendations": AgentSpec(
        workflow=InterpretationWithRecomendationsFlow,
        activities=tuple(),
        build_params=interpretation_build_params,
    ),
    "recommendation-editor": AgentSpec(
        workflow=RecommendationEditorFlow,
        activities=(
            recommendation_editor_call_llm,
            post_callback,
        ),
        build_params=recommendation_editor_build_params,
    ),
    "recommendation-enricher": AgentSpec(
        workflow=RecommendationEnricherFlow,
        activities=(
            recommendation_enricher_call_llm,
            post_callback,
        ),
        build_params=recommendation_enricher_build_params,
    ),
}


def get_all_workflows() -> list:
    workflows = [spec.workflow for spec in AGENTS.values()]

    return workflows


def get_all_activities() -> list:
    seen: Dict[int, Callable] = {}
    for spec in AGENTS.values():
        for fn in spec.activities:
            seen[id(fn)] = fn
    # Add child workflow activities instance
    return list(seen.values())
