from temporalio.client import Client
from temporalio.contrib.pydantic import pydantic_data_converter
from src.pipeline.base import AbstractPipelineExecutor


class TemporalExecutor(AbstractPipelineExecutor):
    def __init__(self, temporal_client: Client, task_queue: str):
        self.temporal_client: Client = temporal_client
        self.task_queue: str = task_queue

    async def execute_pipeline(self, execution_id: str, pipeline_definition: str, payload):
        args = []
        if payload:
            args.append(payload)
        await self.temporal_client.start_workflow(
            pipeline_definition, *args, id=execution_id, task_queue=self.task_queue
        )

    async def execute_pipeline_sync(self, execution_id: str, pipeline_definition: str, payload):
        args = []
        if payload:
            args.append(payload)
        return await self.temporal_client.execute_workflow(
            pipeline_definition, *args, id=execution_id, task_queue=self.task_queue
        )


async def build_temporal_executor(
    temporal_address: str, namespace: str, task_queue: str
) -> TemporalExecutor:
    temporal_client = await Client.connect(
        temporal_address, namespace=namespace, data_converter=pydantic_data_converter
    )
    return TemporalExecutor(temporal_client, task_queue)
