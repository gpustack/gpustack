from datetime import date, datetime, timezone
import json
import logging
import time
from typing import Type, Union
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from jwt import DecodeError, ExpiredSignatureError
from starlette.middleware.base import BaseHTTPMiddleware
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types import Completion, CompletionUsage
from openai.types.create_embedding_response import (
    CreateEmbeddingResponse,
    Usage as EmbeddingUsage,
)
from gpustack.routes.rerank import RerankResponse, RerankUsage
from gpustack.schemas.model_usage import ModelUsage, OperationEnum
from gpustack.schemas.models import Model
from gpustack.schemas.users import User
from gpustack.security import JWT_TOKEN_EXPIRE_MINUTES, JWTManager
from gpustack.server.auth import SESSION_COOKIE_NAME
from gpustack.server.db import get_engine
from sqlmodel.ext.asyncio.session import AsyncSession

logger = logging.getLogger(__name__)


class RequestTimeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request.state.start_time = datetime.now(timezone.utc)
        response = await call_next(request)
        return response


class ModelUsageMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if response.status_code == 200:
            if request.url.path == "/v1-openai/chat/completions":
                return await self.process_request(
                    request,
                    response,
                    self.record_chat_completions_model_usage,
                    ChatCompletion,
                )
            elif request.url.path == "/v1-openai/completions":
                return await self.process_request(
                    request, response, self.record_completions_model_usage, Completion
                )
            elif request.url.path == "/v1-openai/embeddings":
                return await self.process_request(
                    request,
                    response,
                    self.record_embeddings_model_usage,
                    CreateEmbeddingResponse,
                )
            elif request.url.path == "/v1/rerank":
                return await self.process_request(
                    request,
                    response,
                    self.record_rerank_model_usage,
                    RerankResponse,
                )

        return response

    async def process_request(
        self,
        request: Request,
        response: StreamingResponse,
        record_usage,
        response_class: Type[
            Union[ChatCompletion, Completion, CreateEmbeddingResponse, RerankResponse]
        ],
    ):
        stream: bool = getattr(request.state, "stream", False)
        if stream:
            if response_class == ChatCompletion:
                response_class = ChatCompletionChunk
            return await self.handle_streaming_response(
                request, response, record_usage, response_class
            )
        else:
            response_body = b"".join([chunk async for chunk in response.body_iterator])
            try:
                response_dict = json.loads(response_body)
                response_instance = response_class(**response_dict)
                await record_usage(request, response_instance.usage)
            except Exception as e:
                logger.error(f"Error processing model usage: {e}")
            response = Response(content=response_body, headers=dict(response.headers))

        return response

    async def handle_streaming_response(
        self,
        request: Request,
        response: StreamingResponse,
        record_usage,
        response_class: Type[Union[ChatCompletionChunk, Completion]],
    ):
        async def streaming_generator():
            try:
                async for chunk in response.body_iterator:
                    if not hasattr(request.state, 'first_token_time'):
                        request.state.first_token_time = datetime.now(timezone.utc)
                    data = chunk.decode("utf-8").split('data: ')[-1]
                    if not data.startswith('[DONE]'):
                        response_dict = json.loads(data.strip())
                        response_chunk = response_class(**response_dict)
                        is_usage_chunk = len(response_chunk.choices) == 0
                        if is_usage_chunk:
                            await record_usage(request, response_chunk.usage)

                            if 'tokens_per_second' not in response_dict['usage']:
                                # Fill rate metrics. These are extended info not included in OAI APIs.
                                # llama-box provides them out-of-the-box. Align with other backends here.
                                now = datetime.now(timezone.utc)
                                time_to_first_token_ms = (
                                    request.state.first_token_time
                                    - request.state.start_time
                                ).total_seconds() * 1000
                                time_per_output_token_ms = (
                                    (
                                        now - request.state.first_token_time
                                    ).total_seconds()
                                    * 1000
                                    / max(response_chunk.usage.completion_tokens, 1)
                                )
                                tokens_per_second = (
                                    1000 / time_per_output_token_ms
                                    if time_per_output_token_ms > 0
                                    else 0
                                )

                                response_dict['usage'][
                                    "time_to_first_token_ms"
                                ] = time_to_first_token_ms
                                response_dict['usage'][
                                    "time_per_output_token_ms"
                                ] = time_per_output_token_ms
                                response_dict['usage'][
                                    "tokens_per_second"
                                ] = tokens_per_second

                                json_str = json.dumps(
                                    response_dict, separators=(',', ':')
                                )
                                chunk = f"data: {json_str}\n\n".encode("utf-8")

                    yield chunk
            except Exception as e:
                logger.error(f"Error processing streaming response: {e}")

        return StreamingResponse(streaming_generator(), headers=dict(response.headers))

    async def record_model_usage(
        self,
        request: Request,
        usage: Union[CompletionUsage, EmbeddingUsage, RerankUsage],
        operation: OperationEnum,
    ):
        prompt_tokens = usage.prompt_tokens
        total_tokens = usage.total_tokens
        completion_tokens = getattr(
            usage, 'completion_tokens', total_tokens - prompt_tokens
        )
        user: User = request.state.user
        model: Model = request.state.model
        fields = {
            "user_id": user.id,
            "model_id": model.id,
            "date": date.today(),
            "operation": operation,
        }
        model_usage = ModelUsage(
            **fields,
            completion_token_count=completion_tokens,
            prompt_token_count=prompt_tokens,
            request_count=1,
        )
        async with AsyncSession(get_engine()) as session:
            current_model_usage = await ModelUsage.one_by_fields(session, fields)
            if current_model_usage:
                current_model_usage.completion_token_count += completion_tokens
                current_model_usage.prompt_token_count += prompt_tokens
                current_model_usage.request_count += 1
                await current_model_usage.update(session)
            else:
                await ModelUsage.create(session, model_usage)

    async def record_chat_completions_model_usage(
        self, request: Request, usage: CompletionUsage
    ):
        await self.record_model_usage(request, usage, OperationEnum.CHAT_COMPLETION)

    async def record_completions_model_usage(
        self, request: Request, usage: CompletionUsage
    ):
        await self.record_model_usage(request, usage, OperationEnum.COMPLETION)

    async def record_embeddings_model_usage(
        self, request: Request, usage: EmbeddingUsage
    ):
        await self.record_model_usage(request, usage, OperationEnum.EMBEDDING)

    async def record_rerank_model_usage(self, request: Request, usage: RerankUsage):
        await self.record_model_usage(request, usage, OperationEnum.RERANK)


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
                            max_age=JWT_TOKEN_EXPIRE_MINUTES * 60,
                            expires=JWT_TOKEN_EXPIRE_MINUTES * 60,
                        )
            except (ExpiredSignatureError, DecodeError):
                pass

        return response
