from typing import Any, List

from src.agents.base_agent import BaseAgent
from src.schemas.chat import Topic
from src.schemas.llm import ChatMessage, ImageContent, ImageUrl, TextContent


class ReportRecognizerAgent(BaseAgent):
    config_name = "report_recognizer"

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
