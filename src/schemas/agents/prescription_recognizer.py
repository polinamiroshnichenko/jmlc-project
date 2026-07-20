from typing import List, Literal

from pydantic import BaseModel, Field


class Prescription(BaseModel):
    text: str = Field(description="Назначение, как оно указано в тексте")
    type: Literal["lab_analysis", "instrumental_analysis", "doctor"] = Field(
        description="Тип назначения"
    )


class PrescriptionRecognitionResult(BaseModel):
    items: List[Prescription] = Field(default=[])


class NormalizedPrescriptionItem(BaseModel):
    id: int = 0
    name: str = ""
    code: str = ""
    codesystem: str = ""
    type: str = ""


class PrescriptionNormalizationLLMResult(BaseModel):
    items: List[NormalizedPrescriptionItem] = Field(default=[])


class ProcessedPrescriptionItem(BaseModel):
    name: str
    code: str
    codesystem: str
    display_name: str = ""
    type: str = ""


class PrescriptionRecognitionNormalizedResult(BaseModel):
    items: List[ProcessedPrescriptionItem] = Field(default=[])
