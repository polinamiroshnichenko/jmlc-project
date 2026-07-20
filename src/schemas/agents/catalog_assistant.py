from typing import List, Literal

from pydantic import BaseModel

from src.schemas.common import Code


class CatalogSearchResult(BaseModel):
    name: str
    type: Literal["lab_analysis", "instrumental_analysis", "doctor"]
    code: Code


class CatalogAssistantResponse(BaseModel):
    message: str
    catalog_search_result: List[CatalogSearchResult] = []
    is_finished: bool
