from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from fhir.resources import get_fhir_model_class
from fhir.resources.questionnaireresponse import QuestionnaireResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from typing_extensions import Literal

InitiatorType = Literal[
    "chat_backend_app",
    "Los_client",
    "budzdorov",
    "helzy-doc-llm",
    "ml",
]


class DownloadedFile(BaseModel):
    content: bytes
    content_length: Optional[int]


class ActionType(str, Enum):
    general_assistant = "StartGeneralAssistantChat"
    checkup_assistant = "GetLabtestCheckup"
    catalog_assistant = "GetCatalogInfo"
    prescription_recognizer = "RecognizePrescription"
    interpretation_assistant = "ExplainLabtestResult"

    interpretation_with_recommendations = "InterpretateWithRecommendations"
    fix_recommendation_text = "FixRecommendationText"
    expand_recommendation_text = "ExpandRecommendationText"


class AgentType(str, Enum):
    general_assistant = "general-assistant"
    checkup_assistant = "checkup-assistant"
    catalog_assistant = "catalog-assistant"
    prescription_recognizer = "prescription-recognizer"
    interpretation_assistant = "interpretation-assistant"
    interpretation_with_recommendations = "interpretation-with-recommendations-assistant"
    personal_recommendations_agent = "personal-recommendations-agent"

    def to_action(self) -> Optional[ActionType]:
        mapping = {
            AgentType.general_assistant: ActionType.general_assistant,
            AgentType.checkup_assistant: ActionType.checkup_assistant,
            AgentType.catalog_assistant: ActionType.catalog_assistant,
            AgentType.prescription_recognizer: ActionType.prescription_recognizer,
            AgentType.interpretation_assistant: ActionType.interpretation_assistant,
            AgentType.interpretation_with_recommendations: (
                ActionType.interpretation_with_recommendations
            ),
        }
        return mapping.get(self)


class MessageAttachment(BaseModel):
    id: Optional[str] = None
    mime_type: Optional[str] = Field(
        None,
        description='Attachment type, same as https mime types, eg "image/jpg", "application/json", "application/pdf"',
        alias="mimeType",
    )
    link: Optional[str] = Field(
        None,
        description="Full s3 link to file with data. Should not require auth",
    )


class Patient(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., min_length=1)
    gender: Optional[Literal["Male", "Female"]] = Field(None)
    age: Optional[int] = Field(None)
    birth_date: Optional[date] = Field(None, alias="birthDate")
    is_pregnant: Optional[bool] = Field(None, alias="isPregnant")
    last_menstrual_date: Optional[date] = Field(None, alias="lastMenstrualDate")


class Coding(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    code_system_url: str | None = Field(default=None, min_length=1, alias="system")
    code: str | None = Field(default=None, min_length=1)
    display: Optional[str] = Field(None)


class CodedConcept(BaseModel):
    codes: List[Coding] = Field(...)


class ReportBlockItem(BaseModel):
    id: Optional[str] = Field(None)
    type: Optional[
        Literal[
            "Laboratory",
            "Instrumental",
            "Consultation",
            "Telemedicine",
            "Behavior",
            "OtherTextual",
        ]
    ] = Field(None)
    code: Optional[CodedConcept] = Field(None)
    title: Optional[str] = Field(None)
    description: Optional[str] = Field(None)
    triage: Optional[Literal["recommended", "optional"]] = Field(None)
    urgency: Optional[str] = Field(None)
    reassigned: Optional[Literal["available", "unavailable"]] = Field(None)


class ReportBlock(BaseModel):
    title: Optional[str] = Field(None)
    description: Optional[str] = Field(None)
    items: Optional[List[ReportBlockItem]] = Field(None)
    blocks: Optional[List["ReportBlock"]] = Field(None)
    type: Literal["Recommendation", "ObservationInterpretation"] = Field(None)


ReportBlock.model_rebuild()


class StructuredOutput(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    resources: Optional[List[Dict[str, Any]]] = Field(None)
    custom_object: Optional[Any] = Field(None, alias="customObject")

    @field_validator("resources", mode="before")
    @classmethod
    def _parse_fhir_resource(cls, v):
        if not isinstance(v, dict):
            return v
        fhir_model_class = get_fhir_model_class(v["resourceType"])
        return fhir_model_class(**v).model_dump(exclude_none=True)


class LongTermArtifact(BaseModel):
    output: Optional[StructuredOutput] = Field(None)


class Report(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    blocks: Optional[List[ReportBlock]] = Field(None)
    structured_output: Optional[StructuredOutput] = Field(None, alias="structuredOutput")
    long_term_artifact: Optional[LongTermArtifact] = Field(None, alias="longTermArtifact")


class OrderItem(BaseModel):
    code: CodedConcept = Field(...)
    source: Optional[str] = Field(None)
    price: float = Field(...)
    discount: Optional[float] = Field(None)


class LabOrder(BaseModel):
    order_items: List[OrderItem] = Field(..., alias="orderItems")
    result: Optional[List[MessageAttachment]] = Field(None)


class Order(BaseModel):
    order_id: str = Field(..., min_length=1, alias="orderId")
    order_date: date = Field(..., alias="orderDate")
    questionnaire_response: Optional[QuestionnaireResponse] = Field(
        None, alias="questionnaireResponse"
    )
    lab_orders: List[LabOrder] = Field(..., alias="labOrders")


class UserActivityContent(BaseModel):
    text: str
    type: Literal["UserActivity"] = "UserActivity"
    orders: List[Order]


class TextContent(BaseModel):
    text: str
    type: Literal["Text"] = "Text"


class ActionContent(BaseModel):
    action: ActionType
    text: str
    type: Literal["ActionRequest"] = "ActionRequest"


class ReportContent(BaseModel):
    text: str
    report: Report
    type: Literal["Report"] = "Report"


class TopicMessage(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str
    role: Literal["User", "Assistant", "Service"]
    content: Optional[Union[ActionContent, TextContent, ReportContent, UserActivityContent]] = None
    attachment: Optional[List[MessageAttachment]] = None
    created_at: datetime = Field(..., alias="createdAt")
    model: Optional[Union[AgentType, str]] = ""


class Topic(BaseModel):
    id: str
    state: Literal["Active", "Resolved", "Archived"]
    history: List[TopicMessage]
    summary: Optional[str] = None
    patient: Patient
    language: Literal["ru-ru", "en-en"]
    initiator: Optional[InitiatorType] = None
    tools: List[str] = Field(default_factory=list)
    service_attributes: Dict[str, Any] = Field(default_factory=dict, alias="serviceAttributes")


class CallAgentResponse(BaseModel):
    task_id: str
    status: str


class Code(BaseModel):
    name: str
    code: str
    codesystem: Literal["hxid", "docdoc", "budzdorov"]
    display_name: str = ""
    type: str = ""


class ChatFinishedTask(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    taskId: str
    status: str
    result: Optional[Topic] = Field(None)
