import os
import threading
import time
from typing import List, Optional, Tuple
from urllib.parse import urlparse
from abc import ABC, abstractmethod

from openai import OpenAI
from pydantic import model_validator
from pydantic_settings import BaseSettings
from tqdm import tqdm

from gpustack.client.generated_clientset import ClientSet
from gpustack.schemas.models import (
    BackendEnum,
    ModelCreate,
    ModelInstance,
    ModelInstanceStateEnum,
    SourceEnum,
)
from gpustack.server.bus import Event
from openai.types.chat import (
    ChatCompletionMessageParam,
)


class CLIConfig(BaseSettings):
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
        return self


class APIRequestError(Exception):
    pass


def parse_arguments(args) -> CLIConfig:
    return CLIConfig(debug=args.debug, model=args.model, prompt=args.prompt)


class BaseCLIClient(ABC):
    def __init__(self, cfg: CLIConfig) -> None:
        self._model_name = cfg.model
        if "hf.co" in cfg.model:
            model_name, repo_id, filename = self.parse_hf_model(cfg.model)
            self._model_name = model_name
            self._hf_repo_id = repo_id
            self._hf_filename = filename

        self._prompt = cfg.prompt
        self._clientset = ClientSet(base_url=cfg.base_url, api_key=cfg.api_key)
        self._openai_client = OpenAI(
            base_url=f"{cfg.base_url}/v1-openai", api_key=cfg.api_key
        )
        self._history: List[ChatCompletionMessageParam] = []

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def create_model(self):
        pass

    def create_hf_model(self):
        model_create = ModelCreate(
            name=self._model_name,
            source=SourceEnum.HUGGING_FACE,
            huggingface_repo_id=self._hf_repo_id,
            huggingface_filename=self._hf_filename,
            cpu_offloading=True,
            distributed_inference_across_workers=True,
            backend=BackendEnum.LLAMA_BOX,
            backend_parameters=["--no-warmup"],
        )
        if "turbo" in self._hf_repo_id and "stable-diffusion" in self._hf_repo_id:
            # A simple hack to make the sugar command reproducible.
            model_create.backend_parameters.extend(
                ["--seed=42", "--image-cfg-scale=1.0"]
            )

        self._model = self._clientset.models.create(model_create=model_create)

    def parse_hf_model(self, model: str) -> Tuple[str, str, str]:
        """
        Parse ollama style hf model like:
        - hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF
        - hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF:Q8_0
        if tag is not provided, it will use Q4_0 as default.

        Returns model_name, repo_id and filename with wildcards
        """
        parts = model.split(":")
        filename = ""
        if len(parts) == 1:
            filename = "Q4_0"
        else:
            filename = parts[1]

        parts = parts[0].split("/")
        if len(parts) != 3:
            raise Exception(f"Invalid model format: {model}")

        repo_id = parts[1] + "/" + parts[2]

        # remove GGUF suffix for the name
        model_name = parts[2].replace("-GGUF", "")

        return model_name, repo_id, f"*{filename}*"

    def ensure_model(self):
        models = self._clientset.models.list()
        for model in models.items:
            if model.name == self._model_name:
                self._model = model
                break

        if not hasattr(self, "_model"):
            if hasattr(self, "_hf_repo_id") and hasattr(self, "_hf_filename"):
                self.create_hf_model()
            else:
                self.create_model()

        self._wait_for_model_ready()

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
            event.data["model_id"] == self._model.id
            and event.data["state"] == ModelInstanceStateEnum.RUNNING
        ):
            return True
        elif event.data["state"] == ModelInstanceStateEnum.ERROR:
            raise Exception(f"Error running model: {event.data['state_message']}")
        return False
