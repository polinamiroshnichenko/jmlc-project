from enum import Enum
from typing import List

from pydantic import BaseModel

from src.schemas.chat import AgentType, Code


class RouterRedirect(str, Enum):
    checkup_assistant = "checkup-assistant"
    catalog_assistant = "catalog-assistant"
    interpretation_assistant = "interpretation-assistant"
    prescription_recognizer = "prescription-recognizer"
    general_assistant = "general-assistant"
    general_assistant_faq = "general-assistant-faq"
    general_assistant_codes = "general-assistant-codes"
    general_assistant_bm = "general-assistant-bm"


class GeneralAssistantRouterResponse(BaseModel):
    redirects: List[RouterRedirect]
    message: str = ""


class GeneralAssistantResponse(BaseModel):
    message: str
    redirect_to: AgentType
    codes: List[Code] = []
