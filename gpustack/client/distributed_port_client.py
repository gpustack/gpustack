from typing import List
from uuid import UUID

from gpustack.api.exceptions import raise_if_response_error
from gpustack.client import ClientSet
from gpustack.schemas.port_reservations import (
    ModelPortClaim,
    PortClaimResult,
    WorkerPortAvailability,
)


class DistributedPortClient:
    """Coordinate worker-local reservations through the authenticated server."""

    def __init__(self, clientset: ClientSet):
        self._http = clientset.http_client.get_httpx_client()

    @staticmethod
    def _path(instance_id: int) -> str:
        return f"/model-instances/{instance_id}/port-reservations"

    def availability(self, instance_id: int) -> List[WorkerPortAvailability]:
        response = self._http.get(self._path(instance_id))
        raise_if_response_error(response)
        return [WorkerPortAvailability.model_validate(item) for item in response.json()]

    def reserve(self, instance_id: int, claim: ModelPortClaim) -> PortClaimResult:
        response = self._http.post(
            self._path(instance_id), json=claim.model_dump(mode="json")
        )
        raise_if_response_error(response)
        return PortClaimResult.model_validate(response.json())

    def release(self, instance_id: int, token: UUID) -> None:
        response = self._http.delete(
            self._path(instance_id), headers={"X-Claim-Token": str(token)}
        )
        # Deleted instances are reaped independently on every participant.
        if response.status_code != 404:
            raise_if_response_error(response)
