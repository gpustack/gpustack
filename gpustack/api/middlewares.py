from datetime import date, datetime, timezone
import json
import logging
import time
from typing import Type, Union
from fastapi import Request, Response, status
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from jwt import DecodeError, ExpiredSignatureError
from starlette.middleware.base import BaseHTTPMiddleware
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types import CompletionUsage
from openai.types.audio.transcription_create_response import (
    Transcription,
)
from openai.types.create_embedding_response import (
    Usage as EmbeddingUsage,
)
from gpustack.api.exceptions import ErrorResponse
from gpustack.routes.rerank import RerankResponse, RerankUsage
from gpustack.schemas.images import ImageGenerationChunk, ImagesResponse
from gpustack.schemas.model_usage import ModelUsage, OperationEnum
from gpustack.schemas.models import Model
from gpustack.schemas.users import User
from gpustack.security import JWTManager
from gpustack import envs
from gpustack.api.auth import SESSION_COOKIE_NAME
from gpustack.server.db import async_session

from gpustack.server.services import ModelUsageService
from gpustack.api.types.openai_ext import CreateEmbeddingResponseExt, CompletionExt


logger = logging.getLogger(__name__)


class RequestTimeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request.state.start_time = datetime.now(timezone.utc)
        try:
            response = await call_next(request)
        except Exception as e:
            response = JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    reason="Internal Server Error",
                    message=f"Unexpected error occurred: {e}",
                ).model_dump(),
            )
        return response


class ModelUsageMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if response.status_code == 200:
            path = request.url.path
            if path == "/v1-openai/chat/completions" or path == "/v1/chat/completions":
                return await process_request(
                    request, response, ChatCompletion, OperationEnum.CHAT_COMPLETION
                )
            elif path == "/v1-openai/completions" or path == "/v1/completions":
                return await process_request(
                    request, response, CompletionExt, OperationEnum.COMPLETION
                )
            elif path == "/v1-openai/embeddings" or path == "/v1/embeddings":
                return await process_request(
                    request,
                    response,
                    CreateEmbeddingResponseExt,
                    OperationEnum.EMBEDDING,
                )
            elif (
                path == "/v1-openai/images/generations"
                or path == "/v1/images/generations"
                or path == "/v1-openai/images/edits"
                or path == "/v1/images/edits"
            ):
                return await process_request(
                    request,
                    response,
                    ImagesResponse,
                    OperationEnum.IMAGE_GENERATION,
                )
            elif path == "/v1-openai/audio/speech" or path == "/v1/audio/speech":
                return await process_request(
                    request,
                    response,
                    FileResponse,
                    OperationEnum.AUDIO_SPEECH,
                )
            elif (
                path == "/v1-openai/audio/transcriptions"
                or path == "/v1/audio/transcriptions"
            ):
                return await process_request(
                    request,
                    response,
                    Transcription,
                    OperationEnum.AUDIO_TRANSCRIPTION,
                )
            elif request.url.path == "/v1/rerank":
                return await process_request(
                    request,
                    response,
                    RerankResponse,
                    OperationEnum.RERANK,
                )

        return response


async def process_request(
    request: Request,
    response: StreamingResponse,
    response_class: Type[
        Union[
            ChatCompletion,
            CompletionExt,
            CreateEmbeddingResponseExt,
            RerankResponse,
            ImagesResponse,
            FileResponse,
            Transcription,
        ]
    ],
    operation: OperationEnum,
):
    stream: bool = getattr(request.state, "stream", False)
    if stream:
        if response_class == ChatCompletion:
            response_class = ChatCompletionChunk
        if response_class == ImagesResponse:
            response_class = ImageGenerationChunk
        return await handle_streaming_response(
            request, response, response_class, operation
        )
    else:
        response_body = b"".join([chunk async for chunk in response.body_iterator])
        try:
            usage = None
            if (
                response.headers.get("content-type")
                .lower()
                .startswith("application/json")
            ):
                response_dict = json.loads(response_body)
                response_instance = response_class(**response_dict)
                if hasattr(response_instance, "usage"):
                    usage = response_instance.usage

            await record_model_usage(request, usage, operation)
        except Exception as e:
            logger.error(f"Error processing model usage: {e}")
        response = Response(content=response_body, headers=dict(response.headers))

    return response


async def record_model_usage(
    request: Request,
    usage: Union[CompletionUsage, EmbeddingUsage, RerankUsage, None],
    operation: OperationEnum,
):
    total_tokens = getattr(usage, 'total_tokens', 0) or 0
    prompt_tokens = getattr(usage, 'prompt_tokens', total_tokens) or total_tokens
    completion_tokens = (
        getattr(usage, 'completion_tokens', total_tokens - prompt_tokens)
        or total_tokens - prompt_tokens
    )

    user: User = request.state.user
    model: Model = request.state.model
    fields = {
        "user_id": user.id,
        "model_id": model.id,
        "model_name": model.name,
        "date": date.today(),
        "operation": operation,
    }
    model_usage = ModelUsage(
        **fields,
        completion_token_count=completion_tokens,
        prompt_token_count=prompt_tokens,
        request_count=1,
    )
    async with async_session() as session:
        model_usage_service = ModelUsageService(session)
        current_model_usage = await model_usage_service.get_by_fields(fields)
        if current_model_usage:
            await model_usage_service.update(
                current_model_usage, completion_tokens, prompt_tokens
            )
        else:
            await model_usage_service.create(model_usage)


async def handle_streaming_response(
    request: Request,
    response: StreamingResponse,
    response_class: Type[
        Union[ChatCompletionChunk, CompletionExt, ImageGenerationChunk]
    ],
    operation: OperationEnum,
):
    async def streaming_generator():
        async for chunk in response.body_iterator:
            try:
                async for processed_chunk in process_chunk(
                    chunk, request, response_class, operation
                ):
                    yield processed_chunk
            except Exception as e:
                logger.error(f"Error processing streaming response: {e}")
                yield chunk

    return StreamingResponse(streaming_generator(), headers=response.headers)


async def process_chunk(
    chunk,
    request,
    response_class,
    operation: OperationEnum,
):
    if not hasattr(request.state, 'first_token_time'):
        request.state.first_token_time = datetime.now(timezone.utc)

    # each chunk may contain multiple data lines
    lines = chunk.decode("utf-8").split("\n\n")
    for line in lines[:-1]:
        if not line.startswith('data: '):
            # skip non-data SSE messages
            yield f"{line}\n\n".encode("utf-8")
            continue

        data = line.split('data: ')[-1]
        if data.startswith('[DONE]'):
            yield "data: [DONE]\n\n".encode("utf-8")
            continue

        if '"usage":' in data:
            response_dict = None
            try:
                response_dict = json.loads(data.strip())
            except Exception as e:
                raise e
            response_chunk = response_class(**response_dict)

            if is_usage_chunk(response_chunk):
                await record_model_usage(request, response_chunk.usage, operation)

                # Fill rate metrics. These are extended info not included in OAI APIs.
                # llama-box provides them out-of-the-box. Align with other backends here.
                if should_add_metrics(response_dict):
                    add_metrics(response_dict, request, response_chunk)

            yield f"data: {json.dumps(response_dict, separators=(',', ':'))}\n\n".encode(
                "utf-8"
            )
        else:
            yield f"{line}\n\n".encode("utf-8")


def should_add_metrics(response_dict):
    if not isinstance(response_dict, dict):
        return False

    usage = response_dict.get('usage', {})

    return 'prompt_tokens' in usage and 'tokens_per_second' not in usage


def add_metrics(response_dict, request, response_chunk):
    now = datetime.now(timezone.utc)
    time_to_first_token_ms = (
        request.state.first_token_time - request.state.start_time
    ).total_seconds() * 1000

    tokens_after_first = max(response_chunk.usage.completion_tokens - 1, 1)
    time_per_output_token_ms = (
        (now - request.state.first_token_time).total_seconds()
        * 1000
        / tokens_after_first
    )

    tokens_per_second = (
        1000 / time_per_output_token_ms if time_per_output_token_ms > 0 else 0
    )

    response_dict['usage'].update(
        {
            "time_to_first_token_ms": time_to_first_token_ms,
            "time_per_output_token_ms": time_per_output_token_ms,
            "tokens_per_second": tokens_per_second,
        }
    )


class RefreshTokenMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        jwt_manager: JWTManager = request.app.state.jwt_manager
        token = request.cookies.get(SESSION_COOKIE_NAME)

        if token:
            try:
                payload = jwt_manager.decode_jwt_token(token)
                if payload:
                    # Check if the token is about to expire (less than 15 minutes left)
                    if payload['exp'] - time.time() < 15 * 60:
                        new_token = jwt_manager.create_jwt_token(
                            username=payload['sub']
                        )
                        response.set_cookie(
                            key=SESSION_COOKIE_NAME,
                            value=new_token,
                            httponly=True,
                            max_age=envs.JWT_TOKEN_EXPIRE_MINUTES * 60,
                            expires=envs.JWT_TOKEN_EXPIRE_MINUTES * 60,
                        )
            except (ExpiredSignatureError, DecodeError):
                pass

        return response


def is_usage_chunk(
    chunk: Union[ChatCompletionChunk, CompletionExt, ImageGenerationChunk],
) -> bool:
    choices = getattr(chunk, "choices", None)

    if not choices and chunk.usage:
        return True

    for choice in choices or []:
        if choice.finish_reason is not None and chunk.usage:
            return True

    return False
