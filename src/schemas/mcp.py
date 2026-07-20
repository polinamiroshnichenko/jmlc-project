"""
MCP (Model Context Protocol) schemas for lab history enrichment.

These schemas are used for requesting and processing lab history data
from MCP servers to enrich interpretation results.
"""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class DateFilter(BaseModel):
    """Date filter for MCP lab history requests."""

    class Config:
        allow_population_by_field_name = True

    filter_type: Literal["Date"] = Field(default="Date", alias="type")
    date_from: Optional[str] = Field(default=None, alias="from")
    date_to: Optional[str] = Field(default=None, alias="to")


class Coding(BaseModel):
    """Coding for filters."""

    code: Optional[str] = None
    system: Optional[str] = Field(None, alias="codeSystemUrl")
    display: Optional[str] = None


class NameFilter(BaseModel):
    """Name filter for MCP lab history requests."""

    filter_type: Literal["Name"] = Field(default="Name", alias="type")
    name: Coding


class InterpretationFilter(BaseModel):
    """Interpretation filter for MCP lab history requests."""

    filter_type: Literal["Interpretation"] = Field(default="Interpretation", alias="type")
    interpretation: Coding


class SpecimenFilter(BaseModel):
    """Specimen filter for MCP lab history requests."""

    filter_type: Literal["Specimen"] = Field(default="Specimen", alias="type")
    specimen: Coding


# Order matters for Pydantic Union validation - put NameFilter first to avoid DateFilter matching
McpFilter = Union[NameFilter, DateFilter, InterpretationFilter, SpecimenFilter]


class McpLabHistoryRequest(BaseModel):
    """Request arguments for MCP tool."""

    topicId: Optional[str] = None
    filters: List[McpFilter] = Field(default_factory=list)


class AdditionalLabsPlan(BaseModel):
    """LLM decision about whether additional lab history is needed."""

    need_tool: bool = Field(description="Нужно ли обращаться к MCP tool за историей анализов")
    requests: List[McpLabHistoryRequest] = Field(
        default_factory=list,
        description="Список запросов на получение анализов из истории",
    )


class McpToolCallResult(BaseModel):
    """Result of MCP tool call for a single request."""

    request: McpLabHistoryRequest
    structured_content: Optional[Dict[str, Any]] = None
    text_content: List[str] = Field(default_factory=list)
    is_error: bool = False
