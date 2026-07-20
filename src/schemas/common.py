from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel


class Patient(BaseModel):
    id: str
    gender: Optional[Literal["female", "male"]] = None
    date_of_birth: Optional[date] = None
    age: Optional[int] = None
    is_pregnant: Optional[bool] = None


class Code(BaseModel):
    name: str
    code: str
    codesystem: Literal["hxid", "docdoc", "budzdorov"]
    display_name: str = ""
    type: str = ""
