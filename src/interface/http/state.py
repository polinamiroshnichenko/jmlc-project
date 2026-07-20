from src.pipeline.base import AbstractPipelineExecutor


class AppState:
    def __init__(self, pipeline_executor: AbstractPipelineExecutor):
        self.pipeline_executor: AbstractPipelineExecutor = pipeline_executor
