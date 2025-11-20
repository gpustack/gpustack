from gpustack.api.exceptions import raise_if_response_error
from gpustack.schemas.workers import (
    WorkerStatusPublic,
    WorkerCreate,
    WorkerRegistrationPublic,
)
from .generated_http_client import HTTPClient


class WorkerStatusClient:
    def __init__(self, client: HTTPClient):
        self._client = client
        self._url = f"{client._base_url}/v2/worker-status"

    def create(self, model_create: WorkerStatusPublic):
        response = self._client.get_httpx_client().post(
            self._url,
            content=model_create.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        raise_if_response_error(response)
        return None


class WorkerRegistrationClient:
    def __init__(self, client: HTTPClient):
        self._client = client
        self._url = f"{client._base_url}/v2/workers"

    def create(self, model_create: WorkerCreate):
        response = self._client.get_httpx_client().post(
            self._url,
            content=model_create.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        raise_if_response_error(response)
        return WorkerRegistrationPublic.model_validate(response.json())
