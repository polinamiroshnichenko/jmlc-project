from typing import List, Literal, Optional

from pydantic import BaseModel

from src.schemas.common import Patient
from src.schemas.llm import ChatMessage


class MedicalAssistantRequest(BaseModel):
    patient: Patient
    messages: List[ChatMessage] = []
    format: Literal["text", "markdown"] = "text"
    callback_url: Optional[str] = None
