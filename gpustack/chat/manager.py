import os
import sys
import threading
import time
from typing import List, Optional
from urllib.parse import urlparse

from colorama import Fore, Style
from openai import OpenAI
from pydantic import model_validator
from pydantic_settings import BaseSettings
from tqdm import tqdm

from gpustack.api.exceptions import HTTPException
from gpustack.client.generated_clientset import ClientSet
from gpustack.schemas.models import (
    ModelCreate,
    ModelInstance,
    ModelInstanceStateEnum,
    SourceEnum,
)
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
    api_key: Optional[str] = os.getenv("GPUSTACK_API_KEY")

    @model_validator(mode="after")
    def check_api_key(self):
        parsed_url = urlparse(self.base_url)
        if not parsed_url.scheme or not parsed_url.hostname:
            raise Exception(f"Invalid server URL: {self.base_url}")

        if parsed_url.hostname not in ["127.0.0.1", "localhost"] and not self.api_key:
            raise Exception("API key is required. Please set GPUSTACK_API_KEY env var.")
        elif parsed_url.hostname in ["127.0.0.1", "localhost"] and not self.api_key:
            self.api_key = "local"


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
        self._model_name = cfg.model
        self._prompt = cfg.prompt
        self._clientset = ClientSet(base_url=cfg.base_url, api_key=cfg.api_key)
        self._openai_client = OpenAI(
            base_url=f"{cfg.base_url}/v1-openai", api_key=cfg.api_key
        )
        self._history: List[ChatCompletionMessageParam] = []

    def start(self):  # noqa: C901
        try:
            self._ensure_model()
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

    @staticmethod
    def _print_help():
        print("Commands:")
        print("  \\q or \\quit - Quit the chat")
        print("  \\c or \\clear - Clear chat context in prompt")
        print("  \\h or \\? or \\help - Print this help message")

    def _clear_context(self):
        self._history = []
        print("Chat context cleared.")

    def _ensure_model(self):
        models = self._clientset.models.list()
        for model in models.items:
            if model.name == self._model_name:
                self._model = model
                break

        if not hasattr(self, "_model"):
            self._create_model()

        self._wait_for_model_ready()

    def _create_model(self):
        model_create = ModelCreate(
            name=self._model_name,
            source=SourceEnum.OLLAMA_LIBRARY,
            ollama_library_model_name=self._model_name,
        )
        created = self._clientset.models.create(model_create=model_create)
        self._model = created

    def _wait_for_model_ready(self):
        if self._model_is_running():
            return

        with tqdm(
            total=0,
            desc=f"Preparing {self._model_name} model...",
            bar_format="{desc}",
            leave=False,
        ) as pbar:
            current_progress = 0

            def print_progress(event: Event):
                nonlocal current_progress
                mi = ModelInstance.model_validate(event.data)
                if mi.download_progress is not None:
                    increment = mi.download_progress - current_progress
                    if increment <= 0:
                        return

                    if pbar.total == 0:
                        pbar.total = 100
                        pbar.bar_format = (
                            "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
                        )
                        pbar.set_description(f"Downloading {self._model_name} model")
                        pbar.reset()

                    pbar.update(increment)
                    current_progress = mi.download_progress

            def refresh_pbar():
                while not stop_event.is_set():
                    pbar.refresh()
                    time.sleep(1)

            stop_event = threading.Event()
            rate_thread = threading.Thread(target=refresh_pbar)
            rate_thread.start()

            try:
                self._clientset.model_instances.watch(
                    stop_condition=self._stop_when_running,
                    callback=print_progress,
                    params={"model_id": self._model.id},
                )
            finally:
                stop_event.set()
                rate_thread.join()

    def _model_is_running(self):
        instances = self._clientset.model_instances.list(
            params={"model_id": self._model.id},
        )
        if (
            instances.items
            and len(instances.items) > 0
            and instances.items[0].state == ModelInstanceStateEnum.RUNNING
        ):
            return True

        return False

    def _stop_when_running(self, event: Event) -> bool:
        if (
            event.data["id"] == self._model.id
            and event.data["state"] == ModelInstanceStateEnum.RUNNING
        ):
            return True
        elif event.data["state"] == ModelInstanceStateEnum.ERROR:
            raise Exception(f"Error running model: {event.data['state_message']}")
        return False

    def chat_completion(self, prompt: str):
        self._history.append(
            ChatCompletionUserMessageParam(role="user", content=prompt)
        )

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
