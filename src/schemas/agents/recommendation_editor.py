from pydantic import BaseModel


class RecommendationEditorResponse(BaseModel):
    text: str
