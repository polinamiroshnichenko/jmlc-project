import json
from pathlib import Path
from typing import List

from src.agents.base_agent import BaseAgent
from src.schemas.llm import ChatMessage, TextContent
from src.schemas.medical_assistants import MedicalAssistantRequest

_CATALOG_PATH = Path(__file__).parent.parent / "assets" / "catalog.json"

_catalog_str: str | None = None


def _load_catalog() -> str:
    global _catalog_str
    if _catalog_str is None:
        with open(_CATALOG_PATH) as f:
            _catalog_str = json.dumps(json.load(f), ensure_ascii=False)
    return _catalog_str


class CatalogAssistantAgent(BaseAgent):
    config_name = "catalog_assistant"

    def get_messages(self, request: MedicalAssistantRequest, **kwargs) -> List[ChatMessage]:
        input_data = self._build_patient_context(request.patient)
        input_data["catalog"] = _load_catalog()

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
            conv_messages = list(request.messages)
            if conv_messages and conv_messages[-1].role == "assistant":
                conv_messages = conv_messages[:-1]
            messages.extend(conv_messages)

        return messages
