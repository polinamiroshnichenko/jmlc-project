"""
MCP (Model Context Protocol) client for lab history requests.

This client communicates with MCP servers using JSON-RPC 2.0 protocol
to retrieve lab history data for enriching interpretation results.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx

from src.schemas.mcp import McpLabHistoryRequest, McpToolCallResult

logger = logging.getLogger()

_MAX_RETRIES = 2
_RETRY_DELAY_S = 1.0
_LIST_TIMEOUT = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)
_CALL_TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)
_DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)


def _decode_for_log(raw: Any) -> str:
    """Decode nested JSON strings in MCP result for readable log output."""
    try:
        import copy

        raw_copy = copy.deepcopy(raw)
        for item in raw_copy.get("content") or []:
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                try:
                    item["text"] = json.loads(item["text"])
                except Exception:
                    pass
        return json.dumps(raw_copy, ensure_ascii=False)
    except Exception:
        return repr(raw)


class McpClient:
    """Client for communicating with MCP servers via JSON-RPC 2.0."""

    def __init__(
        self,
        server_url: str,
        lab_history_tool: str,
    ):
        self.server_url = server_url
        self.lab_history_tool = lab_history_tool
        self.headers = self._build_headers()
        self._tools_cache: Optional[List[Dict[str, Any]]] = None
        self._http = httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)

    async def close(self):
        """Close the underlying HTTP client. Call on shutdown."""
        await self._http.aclose()

    async def list_tools(self) -> List[Dict[str, Any]]:
        """Return list of tools from MCP server (name/description/input_schema).

        Results are cached after the first successful call.
        """
        if self._tools_cache is not None:
            return self._tools_cache

        result = await self._jsonrpc(
            method="tools/list", params={}, request_id=1, timeout=_LIST_TIMEOUT
        )
        tools = None
        if isinstance(result, dict):
            tools = result.get("tools")
        if tools is None:
            tools = result
        if not isinstance(tools, list):
            raise RuntimeError(f"Unexpected tools/list result: {result}")

        self._tools_cache = [self._tool_to_dict(t) for t in tools]
        return self._tools_cache

    async def call_lab_history(
        self,
        requests: List[McpLabHistoryRequest],
    ) -> Tuple[str, List[McpToolCallResult]]:
        """Call MCP tool for each request and return results.

        Returns:
            Tuple of (resolved tool_name, list of results)
        """
        tools = await self.list_tools()
        tool_names = [t.get("name") for t in tools if t.get("name")]
        tool_name = self._choose_tool_name(
            self.lab_history_tool, [t for t in tool_names if isinstance(t, str)]
        )

        if not requests:
            return tool_name, []

        coros = [
            self._jsonrpc(
                method="tools/call",
                params={"name": tool_name, "arguments": req.model_dump(by_alias=True)},
                request_id=idx,
                timeout=_CALL_TIMEOUT,
            )
            for idx, req in enumerate(requests, start=1)
        ]
        raw_results: List[Any] = await asyncio.gather(*coros)

        results: List[McpToolCallResult] = []
        for req, raw in zip(requests, raw_results):
            logger.debug(f"result _jsonrpc {_decode_for_log(raw)}")
            results.append(self._parse_call_tool_result(req, raw))

        return tool_name, results

    @staticmethod
    def _build_headers() -> Dict[str, str]:
        """Build optional headers for MCP server (e.g., Authorization).

        Supports two variants:
        - MCP_AUTHORIZATION="Bearer ..."
        - MCP_HEADERS_JSON='{"Header-Name": "value", ...}'
        """
        headers: Dict[str, str] = {}

        auth = (os.getenv("MCP_AUTHORIZATION", "") or "").strip()
        if auth:
            headers["Authorization"] = auth

        raw = (os.getenv("MCP_HEADERS_JSON", "") or "").strip()
        if raw:
            try:
                extra = json.loads(raw)
                if isinstance(extra, dict):
                    for k, v in extra.items():
                        if k and v is not None:
                            headers[str(k)] = str(v)
            except Exception as e:
                logger.warning(f"Failed to parse MCP_HEADERS_JSON: {e}")

        return headers

    async def _jsonrpc(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        request_id: int = 1,
        timeout: Optional[httpx.Timeout] = None,
    ) -> Any:
        """JSON-RPC 2.0 call with retry on transient transport errors."""
        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return await self._do_jsonrpc(method, params, request_id, timeout)
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_exc = e
                logger.warning(
                    f"MCP transport error (attempt {attempt + 1}/" f"{_MAX_RETRIES + 1}): {e}"
                )
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(_RETRY_DELAY_S)
        raise RuntimeError(
            f"MCP request '{method}' failed after {_MAX_RETRIES + 1} attempts"
        ) from last_exc

    async def _do_jsonrpc(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        request_id: int = 1,
        timeout: Optional[httpx.Timeout] = None,
    ) -> Any:
        """
        Minimal JSON-RPC 2.0 client over HTTP POST.
        """
        payload: Dict[str, Any] = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        req_kwargs: Dict[str, Any] = {
            "json": payload,
            "headers": self.headers,
        }
        if timeout:
            req_kwargs["timeout"] = timeout

        async with self._http.stream("POST", self.server_url, **req_kwargs) as resp:
            resp.raise_for_status()
            content_type = (resp.headers.get("content-type") or "").lower()

            if "text/event-stream" in content_type:
                data = None
                async for line in resp.aiter_lines():
                    line = (line or "").strip()
                    if not line.startswith("data:"):
                        continue
                    raw = line[len("data:") :].strip()
                    if not raw:
                        continue
                    try:
                        parsed = json.loads(raw)
                    except Exception:
                        continue
                    if isinstance(parsed, dict) and parsed.get("id") == request_id:
                        data = parsed
                        break
                if data is None:
                    raise RuntimeError(
                        "Unexpected SSE response from MCP server: "
                        f"no JSON-RPC response with id={request_id} in `data:`"
                    )
            else:
                body = await resp.aread()
                try:
                    data = json.loads(body)
                except Exception as e:
                    raise RuntimeError(
                        f"Unexpected non-JSON response from MCP " f"server: {body[:200]!r}"
                    ) from e

        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"MCP JSON-RPC error: {data['error']}")
        if not isinstance(data, dict) or "result" not in data:
            raise RuntimeError(f"Unexpected MCP JSON-RPC response: {data}")
        return data["result"]

    @staticmethod
    def _choose_tool_name(tool_name: Optional[str] = None, tool_names: List[str] = None) -> str:
        """Choose tool name, validating it exists on the server."""
        if tool_name is not None:
            if tool_name not in tool_names:
                raise RuntimeError(
                    f"MCP_TOOL_NAME='{tool_name}' not found on the server. "
                    f"Available: {tool_names}"
                )
            return tool_name
        if len(tool_names) == 1:
            return tool_names[0]
        raise RuntimeError(f"Found tools={tool_names} on the server. Specify MCP_TOOL_NAME.")

    @staticmethod
    def _parse_call_tool_result(req: McpLabHistoryRequest, result: Any) -> McpToolCallResult:
        """Parse result from JSON-RPC tools/call into our structure.

        Handles different server conventions:
        - structuredContent / structured_content
        - isError / is_error
        - content: [{type:"text", text:"..."}, ...] or just string
        """
        text_blocks: List[str] = []
        structured: Optional[Dict[str, Any]] = None
        is_error = False

        if isinstance(result, dict):
            structured_val = result.get("structured_content", None)
            if structured_val is None:
                structured_val = result.get("structuredContent", None)
            if isinstance(structured_val, str):
                try:
                    structured_val = json.loads(structured_val)
                except Exception:
                    structured_val = {"raw": structured_val}
            if structured_val is not None and not isinstance(structured_val, dict):
                structured_val = {"raw": structured_val}
            structured = structured_val

            is_error = bool(result.get("isError", False) or result.get("is_error", False))

            content = result.get("content", None)
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        if c.get("text") is not None:
                            text_blocks.append(str(c.get("text")))
                    else:
                        text_blocks.append(str(c))
            elif content is not None:
                text_blocks.append(str(content))
        else:
            text_blocks.append(str(result))

        return McpToolCallResult(
            request=req,
            structured_content=structured,
            text_content=text_blocks,
            is_error=is_error,
        )

    @staticmethod
    def _tool_to_dict(tool: Any) -> Dict[str, Any]:
        """Unify tool fields between SDK versions."""
        if isinstance(tool, dict):
            input_schema = tool.get("input_schema", None)
            if input_schema is None:
                input_schema = tool.get("inputSchema", None)
            return {
                "name": tool.get("name", None),
                "description": tool.get("description", None),
                "input_schema": input_schema,
            }

        input_schema = getattr(tool, "input_schema", None)
        if input_schema is None:
            input_schema = getattr(tool, "inputSchema", None)
        return {
            "name": getattr(tool, "name", None),
            "description": getattr(tool, "description", None),
            "input_schema": input_schema,
        }
