import asyncio
import json
import logging
import re
from typing import Dict, List, Optional, Type, Union

from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion

from pydantic import BaseModel, ValidationError

from src.clients.errors import LLMParseError
from src.schemas.llm import ChatMessage

logger = logging.getLogger()


# Re-exported for backward compatibility
__all__ = ["LLMParseError", "LLMClient"]


class LLMClient:
    def __init__(self, base_url: str, api_key: str):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    def _extract_json(self, content: str) -> str:
        if not content:
            return content

        cleaned = content.strip()
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"^json\s*", "", cleaned, flags=re.IGNORECASE)

        match = re.search(r"[\[{]", cleaned)
        if match:
            cleaned = cleaned[match.start() :]

        return cleaned.strip()

    def _is_empty_response(self, response_data: Dict) -> bool:
        """Check if response data is empty (empty dict, empty list, or None)."""
        if not response_data:
            return True

        if isinstance(response_data, dict):
            for value in response_data.values():
                if value:
                    return False
            return True

        return False

    def _parse_response(
        self,
        response,
        response_format: Optional[Type[BaseModel]],
    ) -> Union[BaseModel, str]:
        try:
            content = response.choices[0].message.content
            if not content:
                raise LLMParseError("Empty content in LLM response")

            if response_format is None:
                return content.strip()

            content = self._extract_json(content)
            response_data = json.loads(content)

            if isinstance(response_data, list) and len(response_data) == 1:
                response_data = response_data[0]

            return response_format.model_validate(response_data)
        except LLMParseError:
            raise
        except (AttributeError, IndexError, json.JSONDecodeError) as e:
            raise LLMParseError(f"Error parsing response structure: {e}") from e
        except ValidationError as e:
            raise LLMParseError(f"Cannot convert LLM response to Pydantic model: {e}") from e

    async def chat_completion_request(
        self,
        model: str,
        messages: List[ChatMessage],
        response_format: Optional[Type[BaseModel]],
        retry_count: int = 3,
        retry_delay: int = 1,
        **kwargs,
    ) -> Union[BaseModel, str]:
        api_response_format = None
        if response_format is not None:
            api_response_format = {"type": "json_object"}

        serializable_messages = [
            m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else m
            for m in messages
        ]

        request_params: Dict = {"model": model, "messages": serializable_messages}
        if api_response_format:
            request_params["response_format"] = api_response_format
        request_params.update(kwargs)

        last_exc: Exception = RuntimeError("All retry attempts exhausted")
        for attempt in range(retry_count):
            try:
                response = await self.client.chat.completions.create(**request_params)
                if response is None:
                    last_exc = RuntimeError("Received None response from API")
                    logger.warning("Received None response, retrying...")
                    if attempt < retry_count - 1:
                        await asyncio.sleep(retry_delay)
                    continue

                return self._parse_response(response, response_format)

            except Exception as e:
                logger.error(f"Request error (attempt {attempt + 1}/{retry_count}): {e}")
                last_exc = e
                if attempt < retry_count - 1:
                    await asyncio.sleep(retry_delay)

        raise last_exc

    async def _parse_response_async(
        self,
        response: ChatCompletion,
        response_format: Type[BaseModel],
        prompt_type: Optional[str] = None,
    ) -> Optional[BaseModel]:
        """Parse ChatCompletion response into Pydantic model."""
        try:
            content = response.choices[0].message.content
            if not content or not content.strip():
                return

            content = self._extract_json(content)
            if not content or not content.strip():
                return

            response_data = json.loads(content)
            logger.info(f"response_data: {response_data}")

            # Check if response data is empty before parsing
            # is_empty = self._is_empty_response(response_data)

            parsed_response = response_format.model_validate(response_data)

            # Check if parsed response is empty (has empty items/list)
            if parsed_response is not None:
                if hasattr(parsed_response, "items"):
                    if not parsed_response.items or len(parsed_response.items) == 0:
                        # is_empty = True
                        ...
                elif hasattr(parsed_response, "__dict__"):
                    # Check if all attributes are empty
                    if all(
                        (not v or (isinstance(v, (list, dict)) and len(v) == 0))
                        for v in parsed_response.__dict__.values()
                    ):
                        # is_empty = True
                        ...

            return parsed_response
        except (AttributeError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing response structure: {e}")
            # Don't count parsing errors as empty responses
            return None
        except ValidationError as e:
            logger.error(f"Cannot convert model response to Pydantic model: {e}")
            # Don't count validation errors as empty responses
            return None

    async def _send_request(
        self,
        request_params: Dict,
        response_format: Type[BaseModel],
        prompt_type: Optional[str] = None,
        retry_count: int = 3,
        retry_delay: int = 1,
    ) -> Optional[BaseModel]:
        for attempt in range(retry_count):
            try:
                response: ChatCompletion = await self.client.chat.completions.create(
                    **request_params
                )

                parsed_response = await self._parse_response_async(
                    response, response_format, prompt_type
                )
                if parsed_response is not None:

                    return parsed_response

                logger.warning(f"Failed to parse response, attempt {attempt + 1}/{retry_count}")

            except Exception as e:
                logger.error(f"Request Error (attempt {attempt + 1}/{retry_count}): {e}")

            if attempt < retry_count - 1:
                await asyncio.sleep(retry_delay)

        logger.warning("All retry attempts failed")
        return None

    async def send_request_and_parse_response(
        self,
        request: Dict,
        response_format: Type[BaseModel],
        prompt_type: Optional[str] = None,
    ) -> Optional[BaseModel]:
        try:
            response = await self._send_request(request, response_format, prompt_type)
            if response is None:
                logger.warning("Max retries reached, returning None.")
            return response
        except Exception as e:
            logger.error(f"Unexpected error in send_request_and_parse_response: {e}")
            return None

    async def request(
        self,
        model_name: str,
        response_format: Type[BaseModel],
        base_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        base64_images: Optional[List[str]] = None,
        texts: Optional[List[str]] = None,
        prompt_type: Optional[str] = None,
        **kwargs,
    ) -> Optional[BaseModel]:
        """
        Sends a request to the LLM API with support for images and text content.

        Args:
            model_name: The name of the model
            response_format: Pydantic model for parsing the response
            base_prompt: Optional base/system prompt
            user_prompt: Optional user prompt
            base64_images: Optional list of base64-encoded PNG images (without data URI prefix)
            texts: Optional list of additional text content
            **kwargs: Additional parameters (temperature, max_tokens etc.)

        Returns:
            An instance of response_format or None in case of an error
        """
        try:
            api_response_format = None
            if response_format is not None:
                api_response_format = {"type": "json_object"}

            base64_images = base64_images or []
            texts = texts or []

            role = "user"

            messages: List[Dict] = []

            if base_prompt:
                messages.append(
                    {
                        "role": role,
                        "content": [
                            {
                                "type": "text",
                                "text": base_prompt,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    }
                )

            content_blocks: List[Dict] = []
            if user_prompt:
                content_blocks.append({"type": "text", "text": user_prompt})

            for text in texts:
                content_blocks.append({"type": "text", "text": text})
            for image in base64_images:
                # Ensure base64 image has proper data URI prefix
                image_url = (
                    image
                    if image.startswith("data:image/png;base64,")
                    else f"data:image/png;base64,{image}"
                )
                content_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    }
                )

            if content_blocks:
                messages.append({"role": role, "content": content_blocks})

            request_params = {
                "model": model_name,
                "messages": messages,
            }

            if api_response_format:
                request_params["response_format"] = api_response_format

            request_params.update(kwargs)

            return await self.send_request_and_parse_response(
                request_params, response_format, prompt_type
            )
        except Exception as e:
            logger.error(f"Unexpected error in request: {e}")
            return None
