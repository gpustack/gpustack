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
        self._url = "/worker-status"

    def create(self, model_create: WorkerStatusPublic):
        response = self._client.get_httpx_client().post(
            self._url,
            content=model_create.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        raise_if_response_error(response)
        return None

    async def create_async(self, model_create: WorkerStatusPublic):
        response = await self._client.get_async_httpx_client().post(
            self._url,
            content=model_create.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        raise_if_response_error(response)
        return None


class WorkerRegistrationClient:
    def __init__(self, client: HTTPClient):
        self._client = client
        self._url = "/workers"

    def create(self, model_create: WorkerCreate):
        response = self._client.get_httpx_client().post(
            self._url,
            content=model_create.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        raise_if_response_error(response)
        return WorkerRegistrationPublic.model_validate(response.json())

    async def create_async(self, model_create: WorkerCreate):
        response = await self._client.get_async_httpx_client().post(
            self._url,
            content=model_create.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        raise_if_response_error(response)
        return WorkerRegistrationPublic.model_validate(response.json())

    async def aclose(self) -> None:
        """Close the underlying HTTP client, releasing pooled connections.

        The registration client is single-use; without an explicit close its
        keep-alive connection is stranded until the process exits, because
        httpcore only reaps dead pooled connections on the next request.
        """
        await self._client.aclose()
