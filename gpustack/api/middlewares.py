from datetime import date
import json
import logging
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from gpustack.schemas.model_usage import ModelUsage
from gpustack.schemas.models import Model
from gpustack.schemas.users import User
from gpustack.server.db import get_engine
from sqlmodel.ext.asyncio.session import AsyncSession

logger = logging.getLogger(__name__)


# class ModelUsageMiddleware(BaseHTTPMiddleware):
#     async def dispatch(self, request: Request, call_next):
#         response = await call_next(request)
#         if (
#             not request.url.path == "/v1-openai/chat/completions"
#             or response.status_code != 200
#         ):
#             return response

#         stream: bool = request.state.stream
#         if stream:
#             return response
#             # response = await self.handle_streaming_response(response)
#         else:
#             response_body = b"".join([chunk async for chunk in response.body_iterator])
#             completion_dict = json.loads(response_body)
#             chat_completion = ChatCompletion(**completion_dict)
#             completion_tokens = chat_completion.usage.completion_tokens
#             prompt_tokens = chat_completion.usage.prompt_tokens
#             user: User = request.state.user
#             model: Model = request.state.model
#             fields = {
#                 "user_id": user.id,
#                 "model_id": model.id,
#                 "date": date.today(),
#             }
#             model_usage = ModelUsage(
#                 **fields,
#                 completion_token_count=completion_tokens,
#                 prompt_token_count=prompt_tokens,
#                 request_count=1,
#                 operation="chat_completion",
#             )
#             async with AsyncSession(get_engine()) as session:
#                 current_model_usage = await ModelUsage.one_by_fields(session, fields)
#                 if current_model_usage:
#                     current_model_usage.completion_token_count += completion_tokens
#                     current_model_usage.prompt_token_count += prompt_tokens
#                     current_model_usage.request_count += 1
#                     await current_model_usage.update(session)
#                 else:
#                     await ModelUsage.create(session, model_usage)

#             response = Response(content=response_body, headers=dict(response.headers))

#         return response

#     async def handle_streaming_response(self, response: StreamingResponse):
#         pass


class ModelUsageMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        if (
            not request.url.path == "/v1-openai/chat/completions"
            or response.status_code != 200
        ):
            return response

        stream: bool = getattr(request.state, "stream", False)
        if stream:
            response = await self.handle_streaming_response(request, response)
        else:
            response_body = b"".join([chunk async for chunk in response.body_iterator])
            try:
                completion_dict = json.loads(response_body)
                chat_completion = ChatCompletion(**completion_dict)
                await self.process_model_usage(request, chat_completion)
            except Exception as e:
                logger.error(f"Error processing model usage: {e}")
            response = Response(content=response_body, headers=dict(response.headers))

        return response

    async def handle_streaming_response(
        self, request: Request, response: StreamingResponse
    ):
        async def streaming_generator():
            try:
                async for chunk in response.body_iterator:
                    data = chunk.decode("utf-8")
                    yield chunk
                    if '"completion_tokens":' in data:
                        completion_dict = json.loads(data.split('data: ')[-1])
                        completion_chunk = ChatCompletionChunk(**completion_dict)
                        await self.process_model_usage(request, completion_chunk)
                        break

                async for chunk in response.body_iterator:
                    yield chunk
            except Exception as e:
                logger.error(f"Error processing streaming response: {e}")

        return StreamingResponse(streaming_generator(), headers=dict(response.headers))

    async def process_model_usage(
        self, request: Request, chat_completion: ChatCompletion | ChatCompletionChunk
    ):
        completion_tokens = chat_completion.usage.completion_tokens
        prompt_tokens = chat_completion.usage.prompt_tokens
        user: User = request.state.user
        model: Model = request.state.model
        fields = {
            "user_id": user.id,
            "model_id": model.id,
            "date": date.today(),
        }
        model_usage = ModelUsage(
            **fields,
            completion_token_count=completion_tokens,
            prompt_token_count=prompt_tokens,
            request_count=1,
            operation="chat_completion",
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
