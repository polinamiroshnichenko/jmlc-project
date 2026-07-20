import logging
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from fastapi import Depends

from src.agents.registry import AGENTS
from src.interface.http.di import get_pipeline_executor
from src.pipeline.base import AbstractPipelineExecutor
from src.schemas.chat import CallAgentResponse, Topic

logger = logging.getLogger(__name__)

Agent = Enum("Agent", {name: name for name in AGENTS})


async def handle(
    data: Topic,
    agent_name: Agent,
    call_type: Literal["chat", "cds-services"],
    pipeline_executor: AbstractPipelineExecutor = Depends(get_pipeline_executor),
) -> Any:
    spec = AGENTS[agent_name.value]
    is_chat = call_type == "chat"
    workflow_params = spec.build_params(data, data.initiator, is_chat)
    workflow_id = f"{agent_name.value}-{data.patient.id}-{uuid4()}"

    if spec.sync:
        logger.info(f"Starting sync flow for agent: {agent_name.value}")
        return await pipeline_executor.execute_pipeline_sync(
            workflow_id, spec.workflow.__name__, workflow_params
        )

    logger.info(f"Starting async flow for agent: {agent_name.value}")
    await pipeline_executor.execute_pipeline(workflow_id, spec.workflow.__name__, workflow_params)

    return CallAgentResponse(task_id=workflow_id, status="started")
