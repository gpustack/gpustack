from gpustack.api.exceptions import raise_if_response_error
from gpustack.schemas.models import (
    ModelInstancePublic,
    ModelInstanceWorkloadUpdate,
    ModelInstanceStopReport,
)

from .generated_model_instance_client import (
    ModelInstanceClient as GeneratedModelInstanceClient,
)


class ModelInstanceClient(GeneratedModelInstanceClient):
    """Worker reporting operations over the generated CRUD client."""

    def get(
        self, id: int, use_cache: bool = True, *, check_configuration: bool = False
    ) -> ModelInstancePublic:
        """Fetch launch authority from the Server when configuration must match."""
        if not check_configuration:
            return super().get(id, use_cache=use_cache)
        response = self._client.get_httpx_client().get(
            f"{self._url}/{id}", params={"check_configuration": True}
        )
        raise_if_response_error(response)
        return ModelInstancePublic.model_validate(response.json())

    def report_workloads(
        self, id: int, report: ModelInstanceWorkloadUpdate
    ) -> ModelInstancePublic:
        response = self._client.get_httpx_client().put(
            f"{self._url}/{id}/workload-report",
            content=report.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        raise_if_response_error(response)
        return ModelInstancePublic.model_validate(response.json())

    def confirm_stop(
        self, id: int, report: ModelInstanceStopReport
    ) -> ModelInstancePublic:
        response = self._client.get_httpx_client().post(
            f"{self._url}/{id}/stop-confirmation",
            content=report.model_dump_json(),
            headers={"Content-Type": "application/json"},
        )
        raise_if_response_error(response)
        return ModelInstancePublic.model_validate(response.json())
