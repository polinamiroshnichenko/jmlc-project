import json
from pathlib import Path
from typing import List, Optional

from src.agents.base_agent import BaseAgent
from src.schemas.llm import ChatMessage, TextContent
from src.schemas.medical_assistants import MedicalAssistantRequest

_ASSETS_PATH = Path(__file__).parent.parent / "assets"

_catalog_cache: Optional[List[dict]] = None
_faq_cache: Optional[str] = None
_bm_instructions_cache: Optional[str] = None


def _load_catalog_items() -> List[dict]:
    global _catalog_cache
    if _catalog_cache is None:
        with open(_ASSETS_PATH / "catalog.json") as f:
            _catalog_cache = json.load(f)
    return _catalog_cache


def _load_faq() -> str:
    global _faq_cache
    if _faq_cache is None:
        with open(_ASSETS_PATH / "faq.json") as f:
            items = json.load(f)
        _faq_cache = "\n".join(
            f"{item.get('question', '')} | {item.get('answer', '')}" for item in items
        )
    return _faq_cache


def _load_bm_instructions() -> str:
    global _bm_instructions_cache
    if _bm_instructions_cache is None:
        with open(_ASSETS_PATH / "bm_instructions.json") as f:
            items = json.load(f)
        _bm_instructions_cache = "\n".join(
            f"{item.get('name', '')} | {item.get('instruction', '')}" for item in items
        )
    return _bm_instructions_cache


def _format_catalog_items(items: List[dict]) -> str:
    return "\n".join(
        f"{item.get('name', '')} | {item.get('code', '')} | {item.get('type', '')}"
        for item in items
    )


class GeneralAssistantRouterAgent(BaseAgent):
    config_name = "general_assistant_router"

    def get_messages(self, request: MedicalAssistantRequest, **kwargs) -> List[ChatMessage]:
        input_data = self._build_patient_context(request.patient)

        base_text = self._base_template.render(**input_data)
        body_text = self._body_template.render(**input_data)

        prompt_message = ChatMessage(
            role="user",
            content=[
                TextContent(text=base_text, cache_control={"type": "ephemeral"}),
                TextContent(text=body_text),
            ],
        )

        messages: List[ChatMessage] = [prompt_message]
        if request.messages:
            messages.extend(request.messages)
        return messages


class GeneralAssistantAgent(BaseAgent):
    config_name = "general_assistant"

    def get_messages(
        self,
        request: MedicalAssistantRequest,
        context_needed: Optional[List[str]] = None,
        **kwargs,
    ) -> List[ChatMessage]:
        input_data = self._build_patient_context(request.patient)

        doctors_str = ""
        instrumental_str = ""
        faq_str = ""
        bm_instructions_str = ""

        if context_needed:
            catalog = _load_catalog_items()
            if "doctors" in context_needed:
                doctors_str = _format_catalog_items(
                    [c for c in catalog if c.get("type") == "doctor"]
                )
            if "instrumental" in context_needed:
                instrumental_str = _format_catalog_items(
                    [c for c in catalog if c.get("type") == "instrumental_analysis"]
                )
            if "faq" in context_needed:
                faq_str = _load_faq()
            if "bm_instructions" in context_needed:
                bm_instructions_str = _load_bm_instructions()

        input_data.update(
            {
                "doctors": doctors_str,
                "instrumental_analyses": instrumental_str,
                "faq": faq_str,
                "bm_instructions": bm_instructions_str,
            }
        )

        base_text = self._base_template.render(**input_data)
        body_text = self._body_template.render(**input_data)

        prompt_message = ChatMessage(
            role="user",
            content=[
                TextContent(text=base_text, cache_control={"type": "ephemeral"}),
                TextContent(text=body_text),
            ],
        )

        messages: List[ChatMessage] = [prompt_message]
        if request.messages:
            messages.extend(request.messages)
        return messages
