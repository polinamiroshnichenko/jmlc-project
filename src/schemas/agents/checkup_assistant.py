from typing import List, Literal

from pydantic import BaseModel, ConfigDict

from src.schemas.chat import Code


class CheckupAssistantRecommendation(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    type: Literal["lab_analysis", "instrumental_analysis", "doctor"]
    reason: str
    urgency: Literal["routine", "urgent", "asap"]
    triage: Literal["recommended", "optional"]
    hxids: List[Code] = []


class CheckupAssistantResponse(BaseModel):
    message: str
    recommendations: List[CheckupAssistantRecommendation] = []
    is_finished: bool
