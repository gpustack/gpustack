import sys
from typing import List

from colorama import Fore, Style

from gpustack.api.exceptions import HTTPException
from gpustack.cli.base import BaseCLIClient, CLIConfig, APIRequestError
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from gpustack.schemas.common import PaginatedList
from gpustack.schemas.model_sets import ModelSetPublic, ModelSpec
from gpustack.schemas.models import ModelCreate


def print_completion_result(message):
    # move cursor to the end of previous line
    sys.stdout.write("\033[F\033[1000C")
    print(message)


def print_error(message):
    print(f"{Fore.RED}{message}{Style.RESET_ALL}")


class ChatCLIClient(BaseCLIClient):
    def __init__(self, cfg: CLIConfig) -> None:
        super().__init__(cfg)

        self._prompt = cfg.prompt
        self._history: List[ChatCompletionMessageParam] = []

    def run(self):  # noqa: C901
        try:
            self.ensure_model()
        except HTTPException as e:
            raise Exception(f"Request to server failed: {e}")

        if self._prompt:
            self.chat_completion(self._prompt)
            return

        user_input = None
        while True:
            user_input = input(">")
            if user_input == "\\q" or user_input == "\\quit":
                break
            elif user_input == "\\?" or user_input == "\\h" or user_input == "\\help":
                self._print_help()
                continue
            elif user_input == "\\c" or user_input == "\\clear":
                self._clear_context()
                continue
            elif not user_input.strip():
                continue

            try:
                self.chat_completion(user_input)
            except Exception as e:
                print_error(e)

    def create_model(self):
        data = (
            self._clientset.http_client.get_httpx_client()
            .get(f"{self._clientset.base_url}/v1/model-sets")
            .json()
        )
        model_sets = PaginatedList[ModelSetPublic](**data)
        found_model_set = None
        for model_set in model_sets.items:
            if model_set.name.lower() == self._model_name.lower():
                found_model_set = model_set
                break

        if not found_model_set:
            raise Exception(f"Model {self._model_name} not found.")

        data = (
            self._clientset.http_client.get_httpx_client()
            .get(f"{self._clientset.base_url}/v1/model-sets/{found_model_set.id}/specs")
            .json()
        )
        model_specs = PaginatedList[ModelSpec](**data)
        pick_model_spec = model_specs.items[0]
        for spec in model_specs.items:
            if spec.quantization.lower() == "q4_k_m":
                pick_model_spec = spec
                break

        model_create = ModelCreate(
            **pick_model_spec.model_dump(exclude={"name"}), name=self._model_name
        )
        self._model = self._clientset.models.create(model_create=model_create)

    @staticmethod
    def _print_help():
        print("Commands:")
        print("  \\q or \\quit - Quit the chat")
        print("  \\c or \\clear - Clear chat context in prompt")
        print("  \\h or \\? or \\help - Print this help message")

    def _clear_context(self):
        self._history = []
        print("Chat context cleared.")

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((APIRequestError)),
    )
    def chat_completion(self, prompt: str):
        self._history.append(
            ChatCompletionUserMessageParam(role="user", content=prompt)
        )

        try:
            completion = self._openai_client.chat.completions.create(
                model=self._model_name,
                messages=self._history,
                stream=True,
            )

            result = ""
            for chunk in completion:
                if chunk.choices[0].delta.content:
                    result += chunk.choices[0].delta.content
                    print(chunk.choices[0].delta.content, end="", flush=True)

            self._history.append(
                ChatCompletionAssistantMessageParam(role="assistant", content=result)
            )
            print()
        except Exception as e:
            raise APIRequestError("Error during API request") from e
