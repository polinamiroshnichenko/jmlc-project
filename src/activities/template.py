from datetime import date

from temporalio import activity
from pydantic import BaseModel, Field

from jinja2 import Template

from src.schemas.chat import InitiatorType


class RenderPromptProps(BaseModel):
    template_str: str
    tables: str = Field(default_factory=str)
    context: str = Field(default_factory=str)
    initiator: InitiatorType | None = None
    has_historical_data: bool = False
    historical_tables: str = Field(default="[]")


@activity.defn
async def render_prompt(props: RenderPromptProps) -> str:
    return Template(props.template_str).render({"date": date.today(), **props.model_dump()})
