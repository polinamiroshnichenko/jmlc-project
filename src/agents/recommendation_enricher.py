from typing import Any, Dict, List, Optional

from src.agents.base_agent import BaseAgent
from src.schemas.chat import Patient
from src.schemas.llm import ChatMessage, TextContent


class RecommendationEnricherAgent(BaseAgent):
    config_name = "recommendation_enricher"

    def get_messages(
        self,
        recommendation_text: str,
        patient: Patient,
        checkup_recommendations: Optional[List[str]] = None,
        summary: Optional[str] = None,
        **kwargs,
    ) -> List[ChatMessage]:
        input_data: Dict[str, Any] = {
            "recommendation_text": recommendation_text,
            "checkup_recommendations": checkup_recommendations or [],
            "summary": summary or "",
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
