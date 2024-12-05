import base64
from datetime import datetime
import os
import re
import shutil
import subprocess
from typing import Optional

from openai import Stream
from openai.types.images_response import ImagesResponse
from tqdm import tqdm
from tenacity import (
    RetryError,
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from gpustack.api.exceptions import HTTPException
from gpustack.cli.base import BaseCLIClient, CLIConfig, APIRequestError
from gpustack.schemas.images import ImageGenerationChunk
from gpustack.utils import platform


class DrawConfig(CLIConfig):
    show: bool = False
    size: str = "512x512"
    sampler: str = "euler"
    sample_steps: Optional[int] = None
    cfg_scale: Optional[float] = None
    seed: Optional[int] = None
    negative_prompt: Optional[str] = None
    output: Optional[str] = None


def parse_arguments(args) -> DrawConfig:
    return DrawConfig(
        debug=args.debug,
        model=args.model,
        prompt=args.prompt,
        size=args.size,
        sampler=args.sampler,
        sample_steps=args.sample_steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
        output=args.output,
        show=args.show,
    )


class DrawCLIClient(BaseCLIClient):
    def __init__(self, cfg: DrawConfig) -> None:
        super().__init__(cfg)

        self._cfg = cfg

    def run(self):
        try:
            print("")  # prompts may be long, so add a newline
            self.ensure_model()

            with tqdm(
                total=100,
                desc="Generating image",
                leave=False,
            ) as pbar:
                image = self.generate_image_stream(pbar)

            truncated_prompt = self._prompt[:23]
            sanitized_prompt = re.sub(r'[^\w-]', '_', truncated_prompt)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{sanitized_prompt}_{timestamp}.png"
            if self._cfg.output:
                filename = self._cfg.output
            save_image_from_b64(image, filename)
            if self._cfg.show:
                open_image(filename)
        except HTTPException as e:
            raise Exception(f"Request to server failed: {e}")
        except RetryError as e:
            raise e.last_attempt.exception()

    def create_model(self):
        raise Exception(
            f"Model {self._model_name} does not exist, please create it first"
        )

    def generate_image(self):
        """
        Generate image from prompt.
        """
        extra_body = {
            key: value
            for key, value in {
                "sampler": self._cfg.sampler,
                "sample_steps": self._cfg.sample_steps,
                "cfg_scale": self._cfg.cfg_scale,
                "seed": self._cfg.seed,
                "negative_prompt": self._cfg.negative_prompt,
            }.items()
            if value is not None
        }

        response = self._openai_client.images.generate(
            prompt=self._prompt,
            model=self._model_name,
            size=self._cfg.size,
            extra_body=extra_body,
        )
        return [image.b64_json for image in response.data]

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((APIRequestError)),
    )
    def generate_image_stream(self, pbar: tqdm):
        try:
            image = self._openai_client.post(
                "/images/generations",
                body={
                    "model": self._model_name,
                    "prompt": self._prompt,
                    "size": self._cfg.size,
                    "sampler": self._cfg.sampler,
                    "sample_steps": self._cfg.sample_steps,
                    "cfg_scale": self._cfg.cfg_scale,
                    "seed": self._cfg.seed,
                    "negative_prompt": self._cfg.negative_prompt,
                    "stream": True,
                },
                cast_to=ImagesResponse,
                stream=True,
                stream_cls=Stream[ImageGenerationChunk],
            )
        except Exception as e:
            raise APIRequestError(f"Error during API request: {e}")

        current_progress = 0
        for chunk in image:
            if isinstance(chunk, dict):
                chunk = ImageGenerationChunk(**chunk)
            if len(chunk.data) == 0:
                raise Exception("Received invalid data chunk from server")
            image_data = chunk.data[0]
            current_progress = round(image_data.progress, 2)
            pbar.update(current_progress - pbar.n)

            if image_data.b64_json:
                return image_data.b64_json


def save_image_from_b64(b64_data: str, filename: str):
    """
    Save image from base64 data to file.
    """
    image_data = base64.b64decode(b64_data)
    with open(filename, "wb") as f:
        f.write(image_data)
    print(f"Image saved to {filename}")


def open_image(filename: str):
    """
    Open the saved image using the default system viewer.
    """
    try:
        system = platform.system()
        if "windows" in system:
            os.startfile(filename)
        elif "darwin" in system:
            subprocess.run(["open", filename])
        elif "linux" in system:
            if shutil.which("xdg-open"):
                subprocess.run(["xdg-open", filename])
            else:
                raise Exception("xdg-open is required to show images on Linux")
        else:
            raise Exception(f"Unsupported platform: {system}")
    except Exception as e:
        raise Exception(f"Failed to open image: {e}")
