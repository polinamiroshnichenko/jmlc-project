from fastapi import Request
from src.pipeline.base import AbstractPipelineExecutor


async def get_pipeline_executor(request: Request) -> AbstractPipelineExecutor:
    return request.app.state.pipeline_executor
