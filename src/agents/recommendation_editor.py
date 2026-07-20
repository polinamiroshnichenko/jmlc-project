from typing import Dict, List

from src.agents.base_agent import BaseAgent
from src.schemas.chat import Patient
from src.schemas.llm import ChatMessage, TextContent


class RecommendationEditorAgent(BaseAgent):
    config_name = "recommendation_editor"

    def get_messages(
        self,
        recommendation_text: str,
        patient: Patient,
        **kwargs,
    ) -> List[ChatMessage]:
        input_data: Dict[str, str] = {
            "recommendation_text": recommendation_text,
            **self._build_patient_context(patient),
        }

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
