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

from gpustack.client.generated_clientset import ClientSet
from gpustack.schemas.models import ModelInstance, ModelInstanceUpdate


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
    def __init__(self, clientset: ClientSet, mi: ModelInstance):
        if mi.source != "huggingface":
            raise ValueError("Only huggingface models are supported for now.")

        logger.info(f"Loading model: {mi.huggingface_repo_id}")

        self.hijack_tqdm_progress()

        self._clientset = clientset
        self._model_instance = mi
        self._model_name = mi.huggingface_repo_id
        self._model = llama_cpp.Llama.from_pretrained(
            repo_id=mi.huggingface_repo_id,
            filename=mi.huggingface_filename,
            verbose=False,
            n_gpu_layers=-1,
        )

        try:
            patch_dict = {"download_progress": 100}
            self._update_model_instance(mi.id, **patch_dict)
        except Exception as e:
            logger.error(f"Failed to update model instance: {e}")

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

    def hijack_tqdm_progress(server_self):
        """
        Monkey patch the tqdm progress bar to update the model instance download progress.
        tqdm is used by hf_hub_download under the hood.
        """
        from tqdm import tqdm

        _original_init = tqdm.__init__
        _original_update = tqdm.update

        def _new_init(self: tqdm, *args, **kwargs):
            kwargs["disable"] = False  # enable the progress bar anyway
            _original_init(self, *args, **kwargs)

        def _new_update(self: tqdm, n=1):
            _original_update(self, n)

            try:
                patch_dict = {
                    "download_progress": round(
                        (float(self.n) / float(self.total)) * 100, 2
                    )
                }
                server_self._update_model_instance(
                    server_self._model_instance.id, **patch_dict
                )
            except Exception as e:
                logger.error(f"Failed to update model instance: {e}")

        tqdm.__init__ = _new_init
        tqdm.update = _new_update

    def _update_model_instance(self, id: str, **kwargs):
        mi_public = self._clientset.model_instances.get(id=id)

        mi = ModelInstanceUpdate(**mi_public.model_dump())
        for key, value in kwargs.items():
            setattr(mi, key, value)

        self._clientset.model_instances.update(id=id, model_update=mi)
