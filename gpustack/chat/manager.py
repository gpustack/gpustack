import os
import sys
from typing import List, Optional

from colorama import Fore, Style
from openai import OpenAI
from pydantic_settings import BaseSettings
from tqdm import tqdm

from gpustack.client.generated_clientset import ClientSet
from gpustack.schemas.models import ModelCreate, SourceEnum
from gpustack.server.bus import Event
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)


class ChatConfig(BaseSettings):
    debug: bool = False
    model: str
    prompt: Optional[str] = None
    base_url: str = os.getenv("GPUSTACK_SERVER_URL", "http://127.0.0.1")
    api_key: str = os.getenv("GPUSTACK_API_KEY", "fake")


def parse_arguments(args) -> ChatConfig:
    return ChatConfig(debug=args.debug, model=args.model, prompt=args.prompt)


def print_completion_result(message):
    # move cursor to the end of previous line
    sys.stdout.write("\033[F\033[1000C")
    print(message)


def print_error(message):
    print(f"{Fore.RED}{message}{Style.RESET_ALL}")


class ChatManager:
    def __init__(self, cfg: ChatConfig) -> None:
        self._model = cfg.model
        self._prompt = cfg.prompt
        self._clientset = ClientSet(base_url=cfg.base_url)
        self._openai_client = OpenAI(base_url=f"{cfg.base_url}/v1", api_key=cfg.api_key)
        self._history: List[ChatCompletionMessageParam] = []

    def start(self):
        self._ensure_model()

        if self._prompt:
            self.chat_completion(self._prompt)
            return

        user_input = None
        while True:
            user_input = input(">")
            if user_input == "exit":
                break
            elif user_input.startswith("#"):
                continue
            elif not user_input.strip():
                continue

            try:
                self.chat_completion(user_input)
            except Exception as e:
                print_error(e)

    def _ensure_model(self):
        models = self._clientset.models.list()
        for model in models.items:
            if model.name == self._model:
                return

        self.create_and_watch_model()

    def create_and_watch_model(self):
        model_create = ModelCreate(
            name=self._model,
            source=SourceEnum.OLLAMA_LIBRARY,
            ollama_library_model_name=self._model,
        )
        created_model = self._clientset.models.create(model_create=model_create)

        def stop_when_running(event: Event) -> bool:
            if (
                event.data["id"] == created_model.id
                and event.data["state"] == "Running"
            ):
                return True
            return False

        with tqdm(
            total=100, desc=f"Downloading {self._model} model", leave=False
        ) as pbar:
            current_progress = 0

            def print_progress(event: Event):
                nonlocal current_progress
                if (
                    "download_progress" in event.data
                    and event.data["download_progress"] is not None
                ):
                    increment = event.data["download_progress"] - current_progress
                    if increment > 0:
                        pbar.update(increment)
                        current_progress = event.data["download_progress"]

            self._clientset.model_instances.watch(
                stop_condition=stop_when_running,
                callback=print_progress,
            )

    def chat_completion(self, prompt: str):
        self._history.append(
            ChatCompletionUserMessageParam(role="user", content=prompt)
        )

        completion = self._openai_client.chat.completions.create(
            model=self._model,
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
