import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from jinja2 import Template

from src.agents.base_agent import BaseAgent
from src.agents.config_loader import (
    _CONFIGS_DIR,
    load_agent_config,
)
from src.schemas.agents.prescription_recognizer import (
    NormalizedPrescriptionItem,
    ProcessedPrescriptionItem,
    PrescriptionRecognitionResult,
)
from src.schemas.chat import Topic
from src.schemas.llm import ChatMessage, ImageContent, ImageUrl, TextContent

_NORMALIZER_DATA_PATH = Path(__file__).parent.parent / "assets" / "normalizer_data.json"
_COMPLEXES_PATH = Path(__file__).parent.parent / "assets" / "complexes.json"

_name2code: Optional[Dict[str, Dict[str, str]]] = None
_code2name: Optional[Dict[Tuple[str, str, str], str]] = None
_complex_to_parts: Optional[Dict[str, List[str]]] = None


def _load_normalizer_data() -> Tuple[
    Dict[str, Dict[str, str]],
    Dict[Tuple[str, str, str], str],
    Dict[str, List[str]],
]:
    global _name2code, _code2name, _complex_to_parts
    if _name2code is None:
        with open(_NORMALIZER_DATA_PATH) as f:
            data = json.load(f)
        _name2code = {
            item["name"]: {
                "code": item.get("code", ""),
                "codesystem": item.get("codesystem", "hxid"),
                "type": item.get("type", ""),
            }
            for item in data
            if item.get("name")
        }
        _code2name = {
            (item["code"], item["codesystem"], item["type"]): item.get("name", "")
            for item in data
            if item.get("code") and item.get("codesystem") and item.get("type")
        }
    if _complex_to_parts is None:
        with open(_COMPLEXES_PATH) as f:
            complexes = json.load(f)
        _complex_to_parts = {
            complex_id: [item["code"] for item in data.get("part_items", [])]
            for complex_id, data in complexes.items()
        }
    return _name2code, _code2name, _complex_to_parts


class PrescriptionRecognizerAgent(BaseAgent):
    config_name = "prescription_recognizer"

    def __init__(self) -> None:
        super().__init__()  # loads llm_request_params, _base_template, _body_template
        full_cfg = load_agent_config(self.config_name)
        self.normalizer_llm_params: Dict[str, Any] = full_cfg["normalizer_params"]
        normalizer_md = _CONFIGS_DIR / self.config_name / "normalizer.md"
        self._normalization_template = Template(normalizer_md.read_text(encoding="utf-8").strip())
        self._normalization_prompt: Optional[str] = None

    def get_recognition_messages(self, topic: Topic) -> List[ChatMessage]:
        base_text = self._base_template.render()
        messages: List[ChatMessage] = [
            ChatMessage(
                role="user",
                content=[TextContent(text=base_text, cache_control={"type": "ephemeral"})],
            )
        ]

        for msg in topic.history:
            if msg.role not in ("User", "Assistant"):
                continue

            role = "user" if msg.role == "User" else "assistant"
            text = msg.content.text if msg.content else ""

            if role == "user" and msg.attachment:
                content_parts: List[Any] = []
                for att in msg.attachment:
                    if att.link:
                        content_parts.append(ImageContent(image_url=ImageUrl(url=att.link)))
                if text:
                    content_parts.append(TextContent(text=text))
                if content_parts:
                    messages.append(ChatMessage(role="user", content=content_parts))
            elif text:
                messages.append(ChatMessage(role=role, content=text))

        return messages

    def get_normalization_messages(
        self,
        recognition_result: PrescriptionRecognitionResult,
        name2code: Dict[str, Dict[str, str]],
    ) -> List[ChatMessage]:
        if self._normalization_prompt is None:
            name2code_json = json.dumps(name2code, ensure_ascii=False)
            self._normalization_prompt = self._normalization_template.render(
                name2code=name2code_json
            )
        base_text = self._normalization_prompt

        lines = [f"{idx + 1} - {item.text}" for idx, item in enumerate(recognition_result.items)]
        user_message = "Описание:\n" + "\n".join(lines)

        return [
            ChatMessage(
                role="user",
                content=[TextContent(text=base_text, cache_control={"type": "ephemeral"})],
            ),
            ChatMessage(role="user", content=user_message),
        ]

    @staticmethod
    def postprocess_normalization(
        items: List[NormalizedPrescriptionItem],
        code2name: Dict[Tuple[str, str, str], str],
        complex_to_parts: Dict[str, List[str]],
    ) -> List[ProcessedPrescriptionItem]:
        result: List[ProcessedPrescriptionItem] = []
        for item in items:
            if not item.code or not item.codesystem:
                continue
            key = (item.code, item.codesystem, item.type)
            display_name = code2name.get(key, "")
            result.append(
                ProcessedPrescriptionItem(
                    name=item.name,
                    code=item.code,
                    codesystem=item.codesystem,
                    display_name=display_name,
                    type=item.type,
                )
            )

        all_codes: Set[str] = {i.code for i in result}
        parts_to_remove: Set[str] = set()
        for code in all_codes:
            if code in complex_to_parts:
                parts_to_remove.update(complex_to_parts[code])

        return [i for i in result if i.code not in parts_to_remove]
