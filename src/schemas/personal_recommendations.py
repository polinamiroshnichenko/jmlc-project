from typing import List, Literal, Optional, Union

from pydantic import BaseModel

from src.schemas.common import Code, Patient


class QuestionnaireResponse(BaseModel):
    answer: str
    question: str


class OrderItem(BaseModel):
    codes: List[Code]
    source: str = ""
    price: float
    discount: float = 0.0


class LabResultRow(BaseModel):
    labname: str
    result: str
    measure: str
    ref_value: str
    comment: str


class ReportResult(BaseModel):
    result_date: Optional[str] = None
    request_date: Optional[str] = None
    name: str
    biomaterial: str
    rows: List[LabResultRow] = []
    comment: str
    laboratory: str


class ReportRecognitionResult(BaseModel):
    items: List[ReportResult] = []


class ImageUrl(BaseModel):
    url: str


class LabResult(BaseModel):
    image_url: ImageUrl
    converted_data: Optional[ReportRecognitionResult] = None


class LabOrder(BaseModel):
    order_items: List[OrderItem] = []
    result: Union[LabResult, List[ReportResult]]


class Order(BaseModel):
    order_id: str
    order_date: str
    questionnaire_response: List[QuestionnaireResponse] = []
    lab_orders: List[LabOrder] = []


class PersonalRecommendationsRequest(BaseModel):
    patient: Patient
    orders: List[Order] = []


class Recommendation(BaseModel):
    name: str
    recommendation_reason: str
    type: Literal["lab_analysis", "instrumental_analysis", "doctor"] = "lab_analysis"
    hxids: List[Code] = []


class Profile(BaseModel):
    profile_name: str
    status: str
    recommendations: List[Recommendation] = []


class PersonalRecommendationsResponse(BaseModel):
    summary: str
    profiles: List[Profile] = []
