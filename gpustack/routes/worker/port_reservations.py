from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Request

from gpustack.api.auth import worker_auth
from gpustack.schemas.models import member_worker_ids
from gpustack.schemas.port_reservations import (
    ModelPortClaim,
    PortClaimResult,
    WorkerPortAvailability,
)
from gpustack.utils import network
from gpustack.worker.port_reservations import model_instance_ports

router = APIRouter(dependencies=[Depends(worker_auth)])


def _instance(request: Request, instance_id: int):
    manager = request.app.state.serve_manager
    instance = manager._clientset.model_instances.get(id=instance_id, use_cache=False)
    if manager._worker_id not in member_worker_ids(instance):
        raise HTTPException(
            status_code=404, detail="Model instance is not assigned here"
        )
    # Tentative allocations must also be reaped if the coordinator disappears
    # before publishing ports or a watch misses the instance's deletion.
    manager._model_instance_by_instance_id.setdefault(instance_id, instance)
    return manager, instance


@router.get("/port-reservations/{instance_id}")
def availability(instance_id: int, request: Request) -> WorkerPortAvailability:
    manager, _ = _instance(request, instance_id)
    port_range = manager._config.service_port_range
    start, end = network.parse_port_range(port_range)
    host = request.app.state.worker_ip_getter()
    generation, claim_version = manager._port_reservations.claim_state(instance_id)
    with manager._port_reservations.allocation(
        manager._clientset, manager._worker_id, model_instance_id=instance_id
    ) as taken:
        busy = set(taken)
    # Claims recheck occupancy under the lock; the advisory socket scan must
    # leave allocations and cleanup free to proceed while it probes the range.
    busy.update(
        port
        for port in range(start, end + 1)
        if port not in taken and not network.is_port_available(port, host)
    )
    return WorkerPortAvailability(
        worker_id=manager._worker_id,
        generation=generation,
        claim_version=claim_version,
        port_range=port_range,
        unavailable_ports=sorted(busy),
        pending_cleanup=taken.pending_cleanup,
    )


@router.post("/port-reservations/{instance_id}")
def reserve(
    instance_id: int, claim: ModelPortClaim, request: Request
) -> PortClaimResult:
    manager, instance = _instance(request, instance_id)
    ports = set(claim.ports)
    generation = claim.generations.get(manager._worker_id)
    if generation is None:
        raise HTTPException(status_code=409, detail="Missing worker port generation")
    if instance.port and ports != model_instance_ports(instance, manager._worker_id):
        raise HTTPException(
            status_code=409, detail="Published model ports cannot change"
        )
    result = manager._port_reservations.reserve_model(
        manager._clientset,
        manager._worker_id,
        instance_id,
        str(claim.token),
        ports,
        manager._config.service_port_range,
        request.app.state.worker_ip_getter(),
        generation=generation,
        replace_version=claim.replaces.get(manager._worker_id),
        recoverable=not bool(instance.port),
    )
    if result.accepted:
        # Reaping can run while the authoritative lookup or lock acquisition
        # is in flight. Keep every accepted claim visible to the next reap.
        manager._model_instance_by_instance_id.setdefault(instance_id, instance)
    return result


@router.delete("/port-reservations/{instance_id}", status_code=204)
def release(
    instance_id: int,
    token: Annotated[UUID, Header(alias="X-Claim-Token")],
    request: Request,
) -> None:
    manager, _ = _instance(request, instance_id)
    manager._port_reservations.release_claim(instance_id, str(token))
