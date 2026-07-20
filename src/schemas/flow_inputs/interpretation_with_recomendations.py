from typing import Optional

from pydantic import BaseModel, Field

from src.schemas.chat import InitiatorType, Topic
from src.schemas.recognition import ReportRecognitionResult


class InterpretationWithRecomendationsProps(BaseModel):
    recognition_result: ReportRecognitionResult
    context: str = Field(default_factory=str)
    initiator: InitiatorType | None = None
    patient: dict | None = None


class InterpretationWithRecomendationsParams(BaseModel):
    """Вход InterpretationWithRecomendationsFlow при вызове из HTTP.

    Содержит исходный Topic — распознавание отчёта выполняется внутри flow через
    child-workflow ReportRecognizerFlow. Контекст диалога собирается в build_params.
    """

    topic: Topic
    context: str = Field(default_factory=str)
    is_chat: bool = False
    callback_url: Optional[str] = None
