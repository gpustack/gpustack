import asyncio
from typing import AsyncGenerator, List, Optional, Tuple, Any, Dict
import aiohttp
import logging
import json

from fastapi import APIRouter, Query, Request, Response, status
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import (
    BadRequestException,
    NotFoundException,
    InternalServerErrorException,
    ServiceUnavailableException,
    GatewayTimeoutException,
    ForbiddenException,
)
from gpustack.api.responses import StreamingResponseWithStatusCode
from gpustack.config.envs import PROXY_TIMEOUT
from gpustack.http_proxy.load_balancer import LoadBalancer
from gpustack.schemas.models import BackendEnum, Model
from gpustack.server.db import get_engine
from gpustack.server.services import ModelInstanceService, ModelService, WorkerService

logger = logging.getLogger(__name__)

load_balancer = LoadBalancer()

router = APIRouter()

# Mapping of Anthropic endpoints to OpenAI endpoints
ENDPOINT_MAPPING = {
    "messages": "chat/completions",
    # Can extend other endpoints in the future
    # "completions": "completions",
    # "embeddings": "embeddings",
}


@router.post("/messages")
async def create_message(request: Request):
    return await proxy_anthropic_request(request, "messages")


async def proxy_anthropic_request(request: Request, anthropic_endpoint: str):
    """
    Proxy Anthropic requests to OpenAI-compatible backends.
    Convert Anthropic format to OpenAI format, call backend,
    then convert back to Anthropic format.

    Args:
        request: FastAPI request object
        anthropic_endpoint: Anthropic endpoint name (e.g., "messages")
    """
    # Get corresponding OpenAI endpoint
    openai_endpoint = ENDPOINT_MAPPING.get(anthropic_endpoint)
    if not openai_endpoint:
        raise BadRequestException(
            message=f"Unsupported endpoint: {anthropic_endpoint}",
            is_anthropic_exception=True,
        )

    allowed_model_names = getattr(
        request.state, "user_allow_model_names", set()
    )
    model_name, stream, anthropic_body = await parse_request_body(
        request,
        anthropic_endpoint
    )

    if model_name not in allowed_model_names:
        raise ForbiddenException(
            message="Model not found",
            is_anthropic_exception=True,
        )

    # Get model information
    async with AsyncSession(get_engine()) as session:
        model = await ModelService(session).get_by_name(model_name)

        if not model:
            raise NotFoundException(
                message="Model not found",
                is_anthropic_exception=True,
            )

        request.state.model = model
        request.state.stream = stream
        request.state.anthropic_request = anthropic_body

        # Get running instance
        instance = await get_running_instance(session, model.id)
        worker = await WorkerService(session).get_by_id(instance.worker_id)
        if not worker:
            raise InternalServerErrorException(
                message=f"Worker with ID {instance.worker_id} not found",
                is_anthropic_exception=True,
            )

    # Convert Anthropic request to OpenAI format
    openai_body = convert_anthropic_to_openai(
        anthropic_body, anthropic_endpoint
    )

    # Build proxy URL
    url = f"http://{instance.worker_ip}:{worker.port}/proxy/v1/{openai_endpoint}"
    token = worker.token
    extra_headers = {
        "X-Target-Port": str(instance.port),
        "Authorization": f"Bearer {token}",
    }

    if model.backend == BackendEnum.ASCEND_MINDIE:
        # Connectivity to the loopback address via worker proxy does not work for Ascend MindIE.
        # Bypassing the worker proxy and directly connecting to the instance as a workaround.
        url = f"http://{instance.worker_ip}:{instance.port}/v1/{openai_endpoint}"
        extra_headers = {}

    logger.debug(f"proxying to {url}, instance port: {instance.port}")

    try:
        if stream:
            return await handle_anthropic_streaming_request(
                request, url, openai_body, extra_headers
            )
        else:
            return await handle_anthropic_standard_request(
                request, url, openai_body, extra_headers
            )
    except asyncio.TimeoutError as e:
        error_message = f"Request to {url} timed out"
        if str(e):
            error_message += f": {e}"
        raise GatewayTimeoutException(
            message=error_message,
            is_anthropic_exception=True,
        )
    except Exception as e:
        error_message = "An unexpected error occurred"
        if str(e):
            error_message += f": {e}"
        raise ServiceUnavailableException(
            message=error_message,
            is_anthropic_exception=True,
        )


async def parse_request_body(request: Request, endpoint: str):
    """
    Parse Anthropic request body.

    Args:
        request: FastAPI request object
        endpoint: Anthropic endpoint name

    Returns:
        Tuple[str, bool, Dict]: (model_name, stream, body_json)

    Note:
        Anthropic API only supports JSON format, no need to handle multipart/form-data.
        Images and other content are passed via base64 encoding in JSON.
    """
    body_json, model_name, stream = await parse_json_body(request)

    # Validate request body
    validate_anthropic_request(body_json, endpoint)

    return model_name, stream, body_json


async def parse_json_body(request: Request):
    """
    Parse JSON request body.

    Returns:
        Tuple[Dict, str, bool]: (body_json, model_name, stream)
    """
    try:
        body_json = await request.json()
        model_name = body_json.get("model")
        stream = body_json.get("stream", False)
        return body_json, model_name, stream
    except Exception as e:
        raise BadRequestException(
            message=f"We could not parse the JSON body of your request: {e}",
            is_anthropic_exception=True,
        )


def validate_anthropic_request(
        anthropic_body: Dict[str, Any], endpoint: str
):
    """
    Validate required fields in Anthropic request body.

    Args:
        anthropic_body: Anthropic request body
        endpoint: Endpoint name
    """
    # All endpoints require model
    if not anthropic_body.get("model"):
        raise BadRequestException(
            message="Missing 'model' field",
            is_anthropic_exception=True,
        )

    # Specific validation for messages endpoint
    if endpoint == "messages":
        if not anthropic_body.get("messages"):
            raise BadRequestException(
                message="Missing 'messages' field",
                is_anthropic_exception=True,
            )
        if not anthropic_body.get("max_tokens"):
            raise BadRequestException(
                message="Missing 'max_tokens' field",
                is_anthropic_exception=True,
            )
    # Can add validation for other endpoints here in the future


def convert_anthropic_to_openai(
        anthropic_body: Dict[str, Any], endpoint: str
) -> Dict[str, Any]:
    """
    Convert Anthropic format request to OpenAI format.

    Args:
        anthropic_body: Anthropic request body
        endpoint: Anthropic endpoint name
    """
    if endpoint == "messages":
        return convert_messages_to_openai(anthropic_body)
    # Can add conversion logic for other endpoints here in the future
    else:
        raise BadRequestException(
            message=f"Unsupported endpoint conversion: {endpoint}",
            is_anthropic_exception=True,
        )


def convert_messages_to_openai(anthropic_body: Dict[str, Any]) -> Dict[
    str, Any]:
    """
    Convert Anthropic Messages format to OpenAI Chat Completions format.
    """
    openai_body = {
        "model": anthropic_body["model"],
        "messages": [],
        "max_tokens": anthropic_body["max_tokens"],
        "stream": anthropic_body.get("stream", False),
    }

    # Handle system parameter
    system = anthropic_body.get("system")
    if system:
        openai_body["messages"].append({"role": "system", "content": system})

    # Convert message format
    for msg in anthropic_body["messages"]:
        openai_msg = {"role": msg["role"]}

        # Anthropic's content can be a string or object array
        content = msg.get("content")
        if isinstance(content, str):
            openai_msg["content"] = content
        elif isinstance(content, list):
            # Extract text content
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            openai_msg["content"] = "".join(text_parts)

        openai_body["messages"].append(openai_msg)

    # Pass other optional parameters
    optional_params = [
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
    ]
    for param in optional_params:
        if param in anthropic_body:
            # Map stop_sequences to stop
            if param == "stop_sequences":
                openai_body["stop"] = anthropic_body[param]
            else:
                openai_body[param] = anthropic_body[param]

    return openai_body


def convert_openai_to_anthropic(
        openai_response: Dict[str, Any], request_id: str = None
) -> Dict[str, Any]:
    """
    Convert OpenAI format response to Anthropic format.
    """
    choice = openai_response["choices"][0]
    message = choice["message"]

    anthropic_response = {
        "id": openai_response.get("id", "msg_unknown"),
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": message.get("content", "")}],
        "model": openai_response.get("model", ""),
        "stop_reason": map_finish_reason(choice.get("finish_reason")),
    }

    # Add usage information
    usage = openai_response.get("usage")
    if usage:
        anthropic_response["usage"] = {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }

    return anthropic_response


def map_finish_reason(openai_reason: Optional[str]) -> str:
    """
    Map OpenAI's finish_reason to Anthropic's stop_reason.
    """
    mapping = {
        "stop": "end_turn",
        "length": "max_tokens",
        "content_filter": "stop_sequence",
        None: "end_turn",
    }
    return mapping.get(openai_reason, "end_turn")


async def handle_anthropic_standard_request(
        request: Request,
        url: str,
        openai_body: Dict[str, Any],
        extra_headers: Optional[Dict[str, str]] = None,
):
    """
    Handle non-streaming Anthropic requests.
    """
    headers = filter_headers(request.headers)
    if extra_headers:
        headers.update(extra_headers)

    http_client: aiohttp.ClientSession = request.app.state.http_client
    timeout = aiohttp.ClientTimeout(total=PROXY_TIMEOUT)

    async with http_client.request(
            method="POST",
            url=url,
            headers=headers,
            json=openai_body,
            timeout=timeout,
    ) as response:
        content = await response.read()

        if response.status >= 400:
            return Response(
                status_code=response.status,
                headers=dict(response.headers),
                content=content,
            )

        # Convert response to Anthropic format
        try:
            openai_response = json.loads(content)
            anthropic_response = convert_openai_to_anthropic(openai_response)
            return Response(
                status_code=200,
                headers={"content-type": "application/json"},
                content=json.dumps(anthropic_response, ensure_ascii=False),
            )
        except Exception as e:
            logger.error(f"Error converting response format: {e}")
            raise InternalServerErrorException(
                message=f"Failed to convert response format: {e}",
                is_anthropic_exception=True,
            )


async def handle_anthropic_streaming_request(
        request: Request,
        url: str,
        openai_body: Dict[str, Any],
        extra_headers: Optional[Dict[str, str]] = None,
):
    """
    Handle streaming Anthropic requests.
    """
    headers = filter_headers(request.headers)
    if extra_headers:
        headers.update(extra_headers)

    # Add stream_options to get usage information
    if "stream_options" not in openai_body:
        openai_body["stream_options"] = {"include_usage": True}

    async def stream_generator():
        message_id = f"msg_{id(request)}"
        message_started = False
        content_block_started = False

        try:
            http_client: aiohttp.ClientSession = request.app.state.http_client
            timeout = aiohttp.ClientTimeout(total=PROXY_TIMEOUT)

            async with http_client.request(
                    method="POST",
                    url=url,
                    headers=headers,
                    json=openai_body,
                    timeout=timeout,
            ) as resp:
                if resp.status >= 400:
                    content = await resp.read()
                    yield content, resp.headers, resp.status
                    return

                async for openai_chunk in stream_openai_chunks(resp):
                    # Convert each OpenAI chunk to Anthropic event
                    anthropic_events = convert_openai_chunk_to_anthropic(
                        openai_chunk,
                        message_id,
                        message_started,
                        content_block_started,
                    )

                    if anthropic_events:
                        for event in anthropic_events:
                            if event["type"] == "message_start":
                                message_started = True
                            elif event["type"] == "content_block_start":
                                content_block_started = True

                            event_data = (
                                f"event: {event['type']}\n"
                                f"data: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
                            )
                            yield (
                                event_data.encode("utf-8"),
                                resp.headers,
                                resp.status,
                            )

                # Send message_stop event
                stop_event = (
                    "event: message_stop\n"
                    "data: {}\n\n"
                )
                yield stop_event.encode("utf-8"), resp.headers, resp.status

        except Exception as e:
            logger.error(f"Streaming response processing error: {e}")
            error_event = {
                "type": "error",
                "error": {
                    "type": "internal_error",
                    "message": str(e),
                },
            }
            yield (
                f"event: error\ndata: {json.dumps(error_event, ensure_ascii=False)}\n\n".encode(
                    "utf-8"
                ),
                {},
                500,
            )

    return StreamingResponseWithStatusCode(
        stream_generator(), media_type="text/event-stream"
    )


async def stream_openai_chunks(resp: aiohttp.ClientResponse):
    """
    Extract and parse data chunks from OpenAI SSE stream.
    """
    chunk_size = 4096
    chunk_buffer = b""

    async for data in resp.content.iter_chunked(chunk_size):
        lines = (chunk_buffer + data).split(b"\n")
        chunk_buffer = lines.pop(-1)

        for line_bytes in lines:
            if not line_bytes:
                continue

            line = line_bytes.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue

            data_str = line[6:]  # Remove "data: " prefix
            if data_str == "[DONE]":
                continue

            try:
                yield json.loads(data_str)
            except json.JSONDecodeError:
                logger.warning(f"Unable to parse JSON: {data_str}")
                continue

    if chunk_buffer:
        line = chunk_buffer.decode("utf-8").strip()
        if line.startswith("data: ") and line[6:] != "[DONE]":
            try:
                yield json.loads(line[6:])
            except json.JSONDecodeError:
                pass


def convert_openai_chunk_to_anthropic(
        openai_chunk: Dict[str, Any],
        message_id: str,
        message_started: bool,
        content_block_started: bool,
) -> List[Dict[str, Any]]:
    """
    Convert OpenAI streaming response chunk to Anthropic event.
    """
    events = []

    # If message hasn't started, send message_start
    if not message_started:
        events.append({
            "type": "message_start",
            "data": {
                "type": "message_start",
                "message": {
                    "id": message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": openai_chunk.get("model", ""),
                    "stop_reason": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
        })

    # Process content
    if "choices" in openai_chunk and openai_chunk["choices"]:
        choice = openai_chunk["choices"][0]
        delta = choice.get("delta", {})
        content = delta.get("content")

        if content:
            # If content block hasn't started, send content_block_start
            if not content_block_started:
                events.append({
                    "type": "content_block_start",
                    "data": {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                })

            # Send content_block_delta
            events.append({
                "type": "content_block_delta",
                "data": {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": content},
                },
            })

        # Check if completed
        finish_reason = choice.get("finish_reason")
        if finish_reason:
            # Send content_block_stop
            events.append({
                "type": "content_block_stop",
                "data": {"type": "content_block_stop", "index": 0},
            })

            # Send message_delta with usage
            usage = openai_chunk.get("usage", {})
            events.append({
                "type": "message_delta",
                "data": {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": map_finish_reason(finish_reason)
                    },
                    "usage": {
                        "output_tokens": usage.get("completion_tokens", 0)
                    },
                },
            })

    return events


async def get_running_instance(session: AsyncSession, model_id: int):
    """
    Get running model instance.
    """
    running_instances = await ModelInstanceService(
        session).get_running_instances(
        model_id
    )
    if not running_instances:
        raise ServiceUnavailableException(
            message="No available running instances",
            is_anthropic_exception=True,
        )
    return await load_balancer.get_instance(running_instances)


def filter_headers(headers):
    """
    Filter request headers.
    """
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in [
            "content-length",
            "host",
            "content-type",
            "transfer-encoding",
            "authorization",
            "anthropic-version",
        ]
    }
