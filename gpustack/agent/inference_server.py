import asyncio
import logging
import time
from typing import Generator

import llama_cpp
from fastapi import status
from openai.types.chat import ChatCompletionChunk
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.responses import StreamingResponse

from gpustack.schemas.models import ModelInstance


logger = logging.getLogger(__name__)


def time_decorator(func):
    """
    A decorator that logs the execution time of a function.
    """

    if asyncio.iscoroutinefunction(func):

        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            result = await func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"{func.__name__} execution time: {end_time - start_time} s")
            return result

        return async_wrapper
    else:

        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            logger.info(f"{func.__name__} execution time: {end_time - start_time} s")
            return result

        return sync_wrapper


class LlamaInferenceServer:
    @time_decorator
    def __init__(self, model: ModelInstance):
        if model.source != "huggingface":
            raise ValueError("Only huggingface models are supported for now.")

        logger.info(f"Loading model: {model.huggingface_repo_id}")

        self._model_name = model.huggingface_repo_id
        self._model = llama_cpp.Llama.from_pretrained(
            repo_id=model.huggingface_repo_id,
            filename=model.huggingface_filename,
            verbose=False,
            n_gpu_layers=-1,
        )

    @time_decorator
    async def __call__(self, request: Request):
        body = await request.json()
        stream = body.get("stream", False)

        logger.debug(f"Received completion request: {body}")

        try:
            completion_result = self._model.create_chat_completion_openai_v1(**body)

            if stream:
                return StreamingResponse(
                    self.stream_chunks(completion_result),
                    media_type="text/plain",
                )

            logger.debug(f"generated_text: {completion_result}")

            return JSONResponse(completion_result.model_dump())
        except Exception as e:
            logger.error(f"Failed to generate completion: {e}")
            return JSONResponse(
                {"error": str(e)}, status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @staticmethod
    def stream_chunks(completion_result: Generator[ChatCompletionChunk, None, None]):
        for chunk in completion_result:
            yield chunk.model_dump_json() + "\n"
