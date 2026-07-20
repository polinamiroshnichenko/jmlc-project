from importlib import import_module
from typing import Callable

from temporalio import activity
from pydantic import BaseModel

from src.clients.llm_client import LLMClient
from src.schemas.chat import Topic
from src.schemas.llm import ChatMessage


def import_class(path: str) -> type[BaseModel]:
    module_name, class_name = path.rsplit(".", 1)
    module = import_module(module_name)
    cls = getattr(module, class_name)
    return cls


class CallLLMProps(BaseModel):
    model_name: str
    response_format: type[BaseModel]
    base_prompt: str
    user_prompt: str
    temperature: float
    timeout: int
    max_tokens: int
    retry_count: int
    topic: Topic


class RequestLLMProps(BaseModel):
    model_name: str
    response_format: str
    base_prompt: str | None = None
    user_prompt: str | None = None
    base64_images: list[str] | None = None
    texts: list[str] | None = None
    prompt_type: str | None = None


class LLMActivities:

    def __init__(self, client: LLMClient):
        self.client = client

    def get_activities(self) -> list[Callable]:
        return [self.call_llm, self.request_llm]

    @activity.defn(name="new_call_llm")
    async def call_llm(self, props: CallLLMProps) -> BaseModel | str:
        activity.logger.info(f"call llm: patient={props.topic.patient.id}")
        return await self.client.chat_completion_request(
            model=props.model_name, messages=props.topic.history
        )

    @activity.defn
    async def request_llm(self, props: RequestLLMProps) -> BaseModel | None:

        response = await self.client.request(
            **props.model_dump(exclude="response_format"),
            response_format=import_class(props.response_format),
        )
        activity.logger.info(f"{response.model_dump()}")
        return response
