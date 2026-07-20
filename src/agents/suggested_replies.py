from typing import List

from src.agents.base_agent import BaseAgent
from src.schemas.llm import ChatMessage, TextContent
from src.schemas.medical_assistants import MedicalAssistantRequest
from src.utils import remove_agent_metadata


class SuggestedRepliesAgent(BaseAgent):
    config_name = "suggested_replies"

    @staticmethod
    def _message_text(message: ChatMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content
        for item in content:
            if getattr(item, "type", None) == "text":
                return item.text
        return ""

    def _build_context(self, request: MedicalAssistantRequest) -> str:
        lines: List[str] = []
        last_idx = len(request.messages) - 1
        for idx, msg in enumerate(request.messages):
            if msg.role == "user":
                prefix = "Пользователь:"
            elif msg.role == "assistant":
                prefix = "▶ ПОСЛЕДНЕЕ Помощник:" if idx == last_idx else "Помощник:"
            else:
                prefix = ""

            text = remove_agent_metadata(self._message_text(msg))
            lines.append(f"{prefix} {text}")

        return "\n".join(lines).strip()

    def get_messages(self, request: MedicalAssistantRequest, **kwargs) -> List[ChatMessage]:
        input_data = {"context": self._build_context(request)}

        base_text = self._base_template.render(**input_data)
        body_text = self._body_template.render(**input_data)

        return [
            ChatMessage(
                role="user",
                content=[
                    TextContent(text=base_text, cache_control={"type": "ephemeral"}),
                    TextContent(text=body_text),
                ],
            )
        ]
