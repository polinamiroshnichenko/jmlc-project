from pydantic import BaseModel


class SymptomsDiagnosesMergerResponse(BaseModel):
    message: str


class BehavioursMergerResponse(BaseModel):
    message: str


class ExaminationCommentsMergerResponse(BaseModel):
    message: str
