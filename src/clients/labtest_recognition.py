import logging
from typing import List, Optional

from pydantic import ValidationError

from src.clients.base_http_client import BaseHTTPClient
from src.schemas.agents.checkup_assistant import CheckupAssistantRecommendation
from src.schemas.chat import Code, InitiatorType
from src.schemas.common import Patient

logger = logging.getLogger(__name__)


class LabtestRecognitionClient(BaseHTTPClient):
    def __init__(self, base_url: str):
        super().__init__()
        self.normalization_url = base_url + "/normalize"
        self.analysis_filtration_url = base_url + "/filter-intersected-analyses"

    async def normalization_request(
        self,
        analysis: CheckupAssistantRecommendation,
        patient: Patient,
        normalize_only_lab: bool,
        initiator: Optional[InitiatorType] = None,
    ) -> List[Code]:
        if normalize_only_lab and analysis.type != "lab_analysis":
            return []

        request_data = {
            "items": [{"text": analysis.name, "type": analysis.type}],
            "comment": "",
            "gender": patient.gender,
        }

        params = {"initiator": initiator} if initiator else None
        response = await self._send_request(request_data, self.normalization_url, params=params)
        if response:
            return [Code(**c) for c in response["items"]]
        return []

    async def filter_intersected_analyses(self, codes: List[str]) -> List[str]:
        response = await self._send_request(codes, self.analysis_filtration_url)
        if response:
            return response
        return []
