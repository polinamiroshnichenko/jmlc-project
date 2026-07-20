import logging
from typing import Dict, List, Literal

from src.agents.base_agent import BaseAgent
from src.schemas.llm import ChatMessage, TextContent
from src.schemas.agents.checkup_assistant import (
    CheckupAssistantRecommendation,
    CheckupAssistantResponse,
)
from src.schemas.medical_assistants import MedicalAssistantRequest

logger = logging.getLogger(__name__)


def _format_chat_recommendation(index: int, rec: CheckupAssistantRecommendation) -> str:
    desc = rec.reason if rec.reason.endswith(".") else rec.reason + "."
    return f"**{index}. {rec.name}.** {desc}"


def _format_recommendations(
    title: str,
    recs: List[CheckupAssistantRecommendation],
    is_markdown: bool,
) -> str:
    if not recs:
        return ""
    lines = [f"\n{title}"]
    for i, rec in enumerate(recs, start=1):
        if is_markdown:
            lines.append(
                f"**{i}. {rec.name}**  \nПоказания: {rec.reason}.  \nСрочность: {rec.urgency}."
            )
        else:
            lines.append(
                f"{i}. {rec.name}  \nПоказания: {rec.reason}.  \nСрочность: {rec.urgency}."
            )
    return "\n".join(lines)


class CheckupAssistantAgent(BaseAgent):
    config_name = "checkup_assistant"

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
            conv_messages = list(request.messages)

            if conv_messages and conv_messages[-1].role == "assistant":
                conv_messages = conv_messages[:-1]
            messages.extend(conv_messages)

        return messages

    @staticmethod
    async def filter_recommendations(
        response: CheckupAssistantResponse,
        filter_fn,
    ) -> CheckupAssistantResponse:
        lab_codes = [
            code.code
            for rec in response.recommendations
            if rec.type == "lab_analysis"
            for code in rec.hxids
        ]

        if len(lab_codes) <= 1:
            return response

        codes_to_keep = set(await filter_fn(lab_codes))

        for rec in response.recommendations:
            if rec.type == "lab_analysis":
                rec.hxids = [c for c in rec.hxids if c.code in codes_to_keep]

        return response

    @staticmethod
    def chat_postprocess(
        response: CheckupAssistantResponse,
    ) -> CheckupAssistantResponse:
        by_type: Dict[str, List] = {
            "lab_analysis": [],
            "instrumental_analysis": [],
            "doctor": [],
        }
        for r in response.recommendations:
            if r.type in by_type:
                by_type[r.type].append(r)

        non_lab = by_type["instrumental_analysis"] + by_type["doctor"]
        lines = ["### Что делать дальше"]
        idx = 1

        if by_type["lab_analysis"]:
            lines.append(f"**{idx}. Сдать анализы из списка ниже.**")
            idx += 1

        for rec in non_lab:
            lines.append(_format_chat_recommendation(idx, rec))
            idx += 1

        response.message = f"{response.message}\n" + "\n".join(lines)
        return response

    @staticmethod
    def postprocess(
        response: CheckupAssistantResponse,
        output_format: Literal["text", "markdown"],
    ) -> CheckupAssistantResponse:
        by_triage: Dict[str, List] = {"recommended": [], "optional": []}
        for r in response.recommendations:
            if r.triage in by_triage:
                by_triage[r.triage].append(r)

        is_md = output_format == "markdown"
        sections = [
            (
                "### Основные рекомендации" if is_md else "Основные рекомендации",
                by_triage["recommended"],
            ),
            (
                "### Дополнительные рекомендации" if is_md else "Дополнительные рекомендации",
                by_triage["optional"],
            ),
        ]

        all_text = "\n".join(
            _format_recommendations(title, recs, is_md) for title, recs in sections if recs
        ).strip()

        if all_text:
            response.message = f"{response.message.strip()}\n{all_text}"

        return response
