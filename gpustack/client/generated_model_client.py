import json
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel
from gpustack.api.exceptions import raise_if_response_error
from gpustack.server.bus import Event
from gpustack.schemas import *

from .generated_http_client import HTTPClient


class ModelClient:
    def __init__(self, client: HTTPClient):
        self._client = client
        self._url = f"{client._base_url}/v1/models"

    def list(self, params: Dict[str, Any] = None) -> ModelsPublic:
        response = self._client.get_httpx_client().get(self._url, params=params)
        raise_if_response_error(response)

        return ModelsPublic.model_validate(response.json())

    def watch(
        self,
        callback: Callable[[Event], None],
        stop_condition: Optional[Callable[[Event], bool]] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        if params is None:
            params = {}
        params["watch"] = "true"

        if stop_condition is None:
            stop_condition = lambda event: False

        with self._client.get_httpx_client().stream(
            "GET", self._url, params=params, timeout=None
        ) as response:
            raise_if_response_error(response)
            for line in response.iter_lines():
                if line:
                    event_data = json.loads(line)
                    event = Event(**event_data)
                    callback(event)
                    if stop_condition(event):
                        break

    def get(self, id: int) -> ModelPublic:
        response = self._client.get_httpx_client().get(f"{self._url}/{id}")
        raise_if_response_error(response)
        return ModelPublic.model_validate(response.json())

    def create(self, model_create: ModelCreate):
        response = self._client.get_httpx_client().post(
            self._url, json=model_create.model_dump()
        )
        raise_if_response_error(response)
        return ModelPublic.model_validate(response.json())

    def update(self, id: int, model_update: ModelUpdate):
        response = self._client.get_httpx_client().put(
            f"{self._url}/{id}", json=model_update.model_dump()
        )
        raise_if_response_error(response)
        return ModelPublic.model_validate(response.json())

    def delete(self, id: int):
        response = self._client.get_httpx_client().delete(f"{self._url}/{id}")
        raise_if_response_error(response)
