from typing import List, Union

from fastapi import APIRouter

from src.schemas.chat import CallAgentResponse
from src.schemas.llm import TextContent

from .call_agent import handle


def configure_routes() -> APIRouter:
    router = APIRouter()

    router.add_api_route(
        "/{call_type}/{agent_name}",
        handle,
        methods=["POST"],
        response_model=Union[List[TextContent], CallAgentResponse],
        response_model_exclude_none=True,
    )

    return router
