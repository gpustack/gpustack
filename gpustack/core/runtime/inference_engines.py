from datetime import datetime
import logging
import time
import uuid
from starlette.requests import Request
from ray import serve
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from starlette.responses import StreamingResponse
from typing import Final

from ...schemas.models import Model
from ...schemas.completion import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    Choice,
    Message,
)


CHAT_COMPLETION_TRUNK: Final = "chat.completion.chunk"
CHAT_COMPLETION: Final = "chat.completion"
ROLE_ASSISTANT: Final = "assistant"
FINISH_REASON_STOP: Final = "stop"

logger = logging.getLogger("ray.serve")


def time_decorator(func):
    """
    A decorator that logs the execution time of a function.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} execution time: {end_time - start_time} s")
        return result

    return wrapper


@serve.deployment()
class TorchInferenceService:

    @time_decorator
    def __init__(self, model: Model):
        if model.source != "huggingface":
            raise ValueError("Only huggingface models are supported for now.")

        logger.info(f"Loading model: {model.name}")

        self.model_name = model.name
        self.model_id = model.huggingface_model_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype="auto", device_map="auto"
        )

    # Provide a OpenAPI compatible chat/completion endpoint for the model.
    async def __call__(self, request: Request):
        body = await request.json()
        messages = body.get("messages", [])
        stream = body.get("stream", False)

        logger.info(f"Received completion request: {messages}")

        chat_id = uuid.uuid4().hex
        created = int(datetime.now().timestamp())

        device = next(self.model.parameters()).device

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)

        generated_ids = self.model.generate(model_inputs.input_ids, max_new_tokens=512)

        def to_chunk(content: str, finished: bool = False):
            created = int(datetime.now().timestamp())
            if finished:
                return ChatCompletionChunk(
                    id=chat_id,
                    object=CHAT_COMPLETION_TRUNK,
                    created=created,
                    model=self.model_name,
                    choices=[
                        Choice(
                            index=0,
                            finish_reason=FINISH_REASON_STOP,
                        )
                    ],
                )
            return ChatCompletionChunk(
                id=chat_id,
                object=CHAT_COMPLETION_TRUNK,
                created=created,
                model=self.model_name,
                choices=[
                    Choice(index=0, delta=Message(content=content, role=ROLE_ASSISTANT))
                ],
            )

        # For streaming
        async def generate_text():
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids):
                # Remove the input_ids part that has already been processed
                output_ids = output_ids[len(input_ids) :]
                # Decode the generated text
                text_output = self.tokenizer.decode(
                    output_ids, skip_special_tokens=True
                )
                yield json.dumps(to_chunk(text_output).model_dump()) + "\n"

            yield json.dumps(to_chunk("", finished=True).model_dump()) + "\n"

        if stream:
            return StreamingResponse(
                generate_text(),
                media_type="text/plain",
            )

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        generated_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        logger.info(f"generated_text: {generated_text}")

        response = ChatCompletionResponse(
            id=chat_id,
            object=CHAT_COMPLETION,
            model=self.model_name,
            created=created,
            choices=[
                Choice(
                    index=0,
                    message=Message(content=generated_text, role=ROLE_ASSISTANT),
                    finish_reason=FINISH_REASON_STOP,
                )
            ],
        )
        return response
