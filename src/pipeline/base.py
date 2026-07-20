from abc import ABC, abstractmethod


class AbstractPipelineExecutor(ABC):
    @abstractmethod
    async def execute_pipeline(self, execution_id: str, pipeline_definition_name: str, payload):
        raise NotImplementedError("Method 'execute_pipeline' must be implemented by subclasses.")

    @abstractmethod
    async def execute_pipeline_sync(
        self, execution_id: str, pipeline_definition_name: str, payload
    ):
        """Start the pipeline and wait for its result, returning it to the caller."""
        raise NotImplementedError(
            "Method 'execute_pipeline_sync' must be implemented by subclasses."
        )
