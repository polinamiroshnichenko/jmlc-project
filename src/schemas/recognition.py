import datetime
from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, validator


class Document(BaseModel):
    data: bytes
    name: str


class NormalizedReferenceValues(BaseModel):
    """Normalized reference values for laboratory parameters."""

    low: Optional[Union[str, float]] = Field(None, description="Lower bound of reference range")
    high: Optional[Union[str, float]] = Field(None, description="Upper bound of reference range")
    text: str = Field(
        "",
        description="Textual reference values (e.g., 'positive', 'negative', 'detected')",
    )


class RecognizedRow(BaseModel):
    labname: Optional[str]
    result: Optional[str]
    measure: Optional[str]
    ref_value: Optional[str]
    comment: Optional[str]


class RecognizedTable(BaseModel):
    date_of_birth: Optional[str]
    gender: Optional[Literal["male", "female"]]
    laboratory: Optional[str]
    request_date: Optional[str]
    result_date: Optional[str]
    biomaterial: Optional[str]
    name: Optional[str]
    code: Optional[str] = None
    codesystem: Optional[str] = None
    rows: Optional[List[RecognizedRow]]
    comment: Optional[str]

    @field_validator("gender", mode="before")
    def set_default_gender(cls, v):
        if v not in ["male", "female"]:
            return None
        return v

    @field_validator("code", mode="before")
    def validate_code(cls, v):
        if v is None:
            return ""
        return v

    @field_validator("codesystem", mode="before")
    def validate_codesystem(cls, v):
        if v is None:
            return ""
        return v


class ReportRecognitionResult(BaseModel):
    items: List[RecognizedTable] = []
    structured_output: Optional[Dict] = None


class Prescription(BaseModel):
    text: str = Field(
        description="Назначение, как оно указано в тексте",
    )
    type: Literal["lab_analysis", "instrumental_analysis", "doctor"] = Field(
        description="Тип назначения"
    )


class PrescriptionRecognitionResult(BaseModel, extra="allow"):
    items: List[Prescription] = Field(
        default=[],
        description="Список всех анализов, найденных на изображении",
    )


class NormalizedPrescription(BaseModel, extra="allow"):
    name: str = Field("", description="Назначение, как оно указано в тексте")
    code: str = Field("", description="Код назначения из словаря с наименованиями и кодами")
    codesystem: Literal["hxid", "docdoc", "budzdorov"] = Field(description="Кодовая система")
    type: Literal["lab_analysis", "instrumental_analysis", "doctor"] = Field(
        description="Тип назначения"
    )

    @validator("codesystem", pre=True, always=True)
    def set_default_codesystem(cls, v):
        if v == "" or v is None:
            return "hxid"
        return v


class PrescriptionRecognitionNormalizedResult(BaseModel):
    items: List[NormalizedPrescription] = Field(
        default=[],
        description="Список всех нормализованных анализов, найденных в тексте",
    )


class InterpretatedRow(BaseModel):
    id: Optional[str] = None
    labname: str
    result: str = ""
    ref_value: str = ""
    is_norm: bool = True
    description: str = ""
    interpretation: str = ""

    @validator("ref_value", "result", "description", "interpretation", pre=True, always=True)
    def validate_values(cls, v):
        if v is None:
            return ""
        return v

    @validator("is_norm", pre=True, always=True)
    def validate_norm(cls, v):
        if v is None:
            return True
        return v


class InterpretatedTable(BaseModel):
    name: str = ""
    code: str = ""
    codesystem: str = ""
    request_date: Optional[datetime.date] = None
    result_date: Optional[datetime.date] = None
    rows: List[InterpretatedRow] = []

    @field_validator("name", mode="before")
    def validate_name(v):
        if v is None:
            return ""
        return v

    @field_validator("request_date", "result_date", mode="before")
    def validate_date(v):
        return None if (v == "" or v == "null") else v

    @field_validator("code", mode="before")
    def validate_code(v):
        if v is None:
            return ""
        return v

    @field_validator("codesystem", mode="before")
    def validate_codesystem(v):
        if v is None:
            return ""
        return v


class ReportInterpretatedResult(BaseModel):
    introduction: str = ""
    tables: List[InterpretatedTable] = []
    interpretation_summary: str = "Не удалось интерпретировать результаты анализов"


class RecognizedRowNormalized(BaseModel):
    """Normalized result row."""

    labname: str = Field("", description="Normalized labname parameter")
    result: Union[str, float] = Field("", description="Test result value")
    measure: str = Field("", description="Measurement unit")
    measure_code: str = Field("", description="unique identifier of measurement unit")
    ref_value: NormalizedReferenceValues = Field(
        default=NormalizedReferenceValues(),
        description="Normalized reference values for the parameter",
    )
    comment: str = Field("", description="Comment for the parameter")
    is_norm: bool = Field(True, description="Whether the result is within normal range")
    interpretation_code: Literal["N", "L", "H", "A"] = Field("N", description="Interpretation code")
    code: str = Field("", description="LOINC code or hash code")
    codesystem: str = Field("", description="Code system URL")
    display: str = Field("", description="Display name (normalized value)")
    text: str = Field("", description="Original text (recognized value)")

    @field_validator(
        "labname",
        "result",
        "measure",
        "measure_code",
        "code",
        "codesystem",
        "display",
        "text",
        mode="before",
    )
    def validate(v):
        if v is None:
            return ""
        return v


class RecognizedTableNormalized(BaseModel):
    """Normalized table."""

    name: str = Field("", description="Normalized test name")
    request_date: str = Field("", description="Test request date in YYYY-MM-DD format")
    result_date: str = Field("", description="Test result date in YYYY-MM-DD format")
    laboratory: str = Field("", description="Normalized laboratory name")
    biomaterial: str = Field("", description="Normalized biomaterial type")
    comment: str = Field("", description="Test comment")
    rows: List[RecognizedRowNormalized] = Field(default=[], description="List of test results")


class ReportNormalizedResult(BaseModel):
    """Result containing normalized tables."""

    items: List[RecognizedTableNormalized] = Field(
        default=[],
        description="List of normalized tables",
    )


class LoincSearchVariantsResult(BaseModel):
    """Result containing LOINC search variants for multiple labnames."""

    items: Dict[str, List[str]] = Field(
        default={},
        description="Dictionary mapping 'table_name|labname' to list of search variants",
    )


class Code(BaseModel):
    name: str
    code: str
    codesystem: Literal["hxid", "docdoc", "budzdorov"]
    display_name: str = ""
    type: Literal["lab_analysis", "instrumental_analysis", "doctor"]


class CheckupAssistantRecommendation(BaseModel):
    name: str
    reason: str
    urgency: str
    triage: Literal["recommended", "optional"]
    type: Literal["lab_analysis", "instrumental_analysis", "doctor"]
    hxids: List[Code] = []
    reassigned: Optional[Literal["available", "unavailable"]] = None


class CheckupAssistantResponse(BaseModel):
    message: str
    recommendations: List[CheckupAssistantRecommendation] = []
    is_finished: bool


class InterpretationWithRecommendationsResult(BaseModel):
    """Result containing interpretation, recognition data, and checkup recommendations."""

    interpretation: ReportInterpretatedResult = Field(
        ..., description="Interpretation result (dictionary from _postprocessing_data)"
    )
    checkup_recommendations: Optional[CheckupAssistantResponse] = Field(
        None, description="Recommendations from checkup assistant"
    )
