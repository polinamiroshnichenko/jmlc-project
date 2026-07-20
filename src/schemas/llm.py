from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, field_validator


class TextContent(BaseModel):
    type: str = "text"
    text: str
    cache_control: Optional[Dict[str, Any]] = None


class ImageUrl(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_data_url(cls, v: str) -> str:
        if v.startswith("http://") or v.startswith("https://"):
            return v
        raise ValueError("An Image URL must be an http(s) link")


class Base64Image(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_base64_url(cls, v: str) -> str:
        if not (v.startswith("data:image/png;base64,") or v.startswith("data:image/jpeg;base64,")):
            raise ValueError(
                "Base64Image url must start with 'data:image/png;base64,' or 'data:image/jpeg;base64,'"
            )
        return v


class ImageContent(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: Union[ImageUrl, Base64Image]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[str, List[Union[TextContent, ImageContent]]]


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: List[ChatMessage]
    response_format: Optional[Any] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    extra_body: Optional[Dict[str, Any]] = None
