from datetime import date
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
from gpustack.schemas.model_usage import ModelUsage, OperationEnum
from gpustack.schemas.models import Model
from gpustack.schemas.users import User
from gpustack.security import JWT_TOKEN_EXPIRE_MINUTES, JWTManager
from gpustack.server.auth import SESSION_COOKIE_NAME
from gpustack.server.db import get_engine
from sqlmodel.ext.asyncio.session import AsyncSession

logger = logging.getLogger(__name__)


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

        return response

    async def process_request(
        self,
        request: Request,
        response: StreamingResponse,
        record_usage,
        response_class: Type[
            Union[ChatCompletion, Completion, CreateEmbeddingResponse]
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
                    data = chunk.decode("utf-8")
                    yield chunk
                    if '"completion_tokens":' in data:
                        lines = data.split('\n')
                        json_line = None
                        for line in lines:
                            if '"completion_tokens":' in line:
                                json_line = line
                                break
                        if json_line:
                            response_dict = json.loads(json_line.split('data: ')[-1])
                            response_chunk = response_class(**response_dict)
                            await record_usage(request, response_chunk.usage)
                        break

                async for chunk in response.body_iterator:
                    yield chunk
            except Exception as e:
                logger.error(f"Error processing streaming response: {e}")

        return StreamingResponse(streaming_generator(), headers=dict(response.headers))

    async def record_model_usage(
        self,
        request: Request,
        usage: Union[CompletionUsage, EmbeddingUsage],
        operation: OperationEnum,
    ):
        prompt_tokens = usage.prompt_tokens
        completion_tokens = getattr(usage, 'completion_tokens', 0)
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
