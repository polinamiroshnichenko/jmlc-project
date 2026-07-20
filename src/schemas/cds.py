from datetime import datetime
import logging
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from fhir.resources.activitydefinition import ActivityDefinition
from fhir.resources.basic import Basic
from fhir.resources.bundle import Bundle
from fhir.resources.condition import Condition
from fhir.resources.fhirresourcemodel import FHIRResourceModel
from fhir.resources.observation import Observation
from fhir.resources.patient import Patient
from pydantic import (
    UUID4,
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)
from src import utils
from src.constants import assets_path

logger = logging.getLogger(__name__)

CONDITIONS = open(assets_path + "/conditions_in_use.txt").read().splitlines()


ConditionEnum = Enum("ConditionEnum", {condition: condition for condition in CONDITIONS}, type=str)


class Prefetch(BaseModel):
    diagnoses: Optional[Bundle]
    symptoms: Optional[Bundle]
    recommendations: Optional[Bundle]
    complaints: Optional[Basic]
    anamnesis: Optional[Basic]
    patient: Optional[Patient]
    dialogue: Optional[Bundle]

    @classmethod
    def check_bundle_type(cls, bundle: Bundle, expected_fhir_model: FHIRResourceModel):
        if bundle is not None:
            for entry in bundle.entry:
                if not isinstance(entry.resource, expected_fhir_model):
                    raise ValueError(
                        f"All resources in bundle must be of type {expected_fhir_model.__name__}"
                    )
        return bundle

    @classmethod
    def check_purpose(cls, bundle: Bundle):
        if bundle is not None:
            _purposes = []
            for entry in bundle.entry:
                if not hasattr(entry.resource, "purpose"):
                    raise ValueError("All resources in bundle must have a purpose")
                _purposes.append(getattr(entry.resource, "purpose", None))
            assert any(_purposes), "At least one recommendation must have not null purpose value."
        return bundle

    @field_validator("diagnoses")
    @classmethod
    def check_diagnoses(cls, v):
        return cls.check_bundle_type(v, Condition)

    @field_validator("symptoms")
    @classmethod
    def check_symptoms(cls, v):
        return cls.check_bundle_type(v, Observation)

    @field_validator("recommendations")
    @classmethod
    def check_recommendations(cls, v):
        bundle = cls.check_bundle_type(v, ActivityDefinition)
        bundle = cls.check_purpose(v)
        return bundle


class Context(BaseModel):
    patientId: UUID4


class RawCodeableMedicalReport(BaseModel):
    hook: str
    hookInstance: UUID4 = Field(default_factory=uuid4)
    fhirServer: Optional[str]
    fhirAuthorization: Optional[dict]
    context: Optional[Context]
    prefetch: Prefetch

    @classmethod
    def check_resourses(cls, hook: str, prefetch: Prefetch):
        if prefetch:
            if "symptom-checker" in hook:
                return
            elif "symptoms-diagnoses" not in hook:
                if not getattr(prefetch, "recommendations"):
                    raise ValueError("Prefetch must have recommendations")
            else:
                if not getattr(prefetch, "symptoms"):
                    raise ValueError("Prefetch must have symptoms")

    @model_validator()
    def check_request(self):
        prefetch = self.prefetch
        hook = self.hook
        self.check_resourses(hook, prefetch)
        return self


class Action(BaseModel):
    type: Literal["create", "update", "remove"]
    description: str
    resource: Basic
    resourceId: Optional[str]


class Source(BaseModel):
    label: str
    url: Optional[str]
    icon: Optional[str]


class Suggestion(BaseModel):
    label: str
    uuid: UUID4 = Field(default_factory=uuid4)
    actions: Optional[List[Action]]


class Card(BaseModel):
    uuid: UUID4 = Field(default_factory=uuid4)
    summary: str
    detail: Optional[str]
    indicator: Optional[str]
    source: Dict[str, Optional[str]]
    suggestions: List[Suggestion]
    extensions: Optional[Dict]


class CDSResponse(BaseModel):
    cards: List[Card]


class SymptomCheckerCondition(BaseModel):
    name: ConditionEnum
    reason: str
    triage: Literal["info", "warning", "critical"]


class SymptomCheckerMedicalResponse(BaseModel):
    question: str
    options: List[str]
    reason: str
    conditions: Optional[List[SymptomCheckerCondition]]


class SymptomCheckerPatientMedicalResponse(BaseModel):
    response: str


class DocumentReference(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    url: AnyHttpUrl = Field(..., alias="link")
    name: Optional[str] = Field(default="")
    mime_type: str

    @field_validator("mime_type", mode="before")
    @classmethod
    def validate_mime_type(cls, v):
        if v not in utils.ALLOWED_MIME_TYPES:
            raise ValueError("Unsupported file extension")
        return v


class RecognitionSubmission(BaseModel):
    kind: Literal[
        "report",
        "prescription",
        "interpretation",
        "interpretation_with_recommendations",
    ]
    callback_url: Optional[AnyHttpUrl]
    references: List[DocumentReference]
    context: Optional[str] = ""


class DocumentStatus(BaseModel):
    document_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    context: dict | None


class FinishedTask(BaseModel):
    task_id: str
    status: str
    result: Any
