import asyncio
import contextlib
import logging
from typing import Annotated, List, Optional, Set
from uuid import UUID

import aiohttp
from fastapi import APIRouter, Header, Request

from gpustack.api.exceptions import (
    BadRequestException,
    ForbiddenException,
    NotFoundException,
    ServiceUnavailableException,
)
from gpustack.api.tenant import assert_resource_visible
from gpustack.schemas.models import (
    DistributedServerCoordinateModeEnum,
    ModelInstance,
    member_worker_ids,
)
from gpustack.schemas.port_reservations import (
    ModelPortClaim,
    PortClaimResult,
    WorkerPortAvailability,
)
from gpustack.schemas.workers import Worker
from gpustack.server.deps import SessionDep, TenantContextDep
from gpustack.server.worker_request import request_to_worker

logger = logging.getLogger(__name__)
router = APIRouter()
_rollback_tasks: Set[asyncio.Task] = set()


async def _members(session, ctx, instance_id: int) -> List[Worker]:
    instance = await ModelInstance.one_by_id(session, instance_id)
    assert_resource_visible(ctx, instance, not_found_message="Model instance not found")
    caller = ctx.user.worker
    if caller is None or caller.id != instance.worker_id:
        raise ForbiddenException(message="Only the assigned primary may reserve ports")
    if (
        not instance.distributed_servers
        or instance.distributed_servers.mode
        != DistributedServerCoordinateModeEnum.INITIALIZE_LATER
    ):
        raise BadRequestException(message="Instance has no shared worker ports")
    workers = []
    # Claim the primary before touching followers: a stale recovery snapshot
    # must fail at its local ownership fence without replacing an active peer.
    for worker_id in sorted(
        member_worker_ids(instance), key=lambda item: (item != instance.worker_id, item)
    ):
        worker = await Worker.one_by_id(session, worker_id)
        if not worker or worker.cluster_id != instance.cluster_id:
            raise NotFoundException(message="Participating worker not found")
        workers.append(worker)
    # Worker callbacks query this same database. Return the read connection
    # before waiting on RPC so concurrent starts cannot exhaust its pool.
    await session.close()
    return workers


async def _call(
    request: Request,
    worker: Worker,
    instance_id: int,
    method: str,
    claim: Optional[ModelPortClaim] = None,
    token: Optional[UUID] = None,
):
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["X-Claim-Token"] = str(token)
    try:
        response, body = await request_to_worker(
            worker=worker,
            method=method,
            path=f"port-reservations/{instance_id}",
            proxy_client=request.app.state.http_client,
            no_proxy_client=request.app.state.http_client_no_proxy,
            data=claim.model_dump_json().encode() if claim else None,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=15, sock_connect=5),
            raise_on_error=False,
        )
    except (aiohttp.ClientError, TimeoutError) as exc:
        raise ServiceUnavailableException(
            message=f"Cannot coordinate model ports with worker {worker.id}",
        ) from exc
    if method == "DELETE" and response.status == 404:
        return None
    if response.status >= 400:
        raise ServiceUnavailableException(
            message=f"Worker {worker.id} rejected port coordination ({response.status})",
        )
    return body


async def _rollback(request, workers, instance_id: int, token: UUID) -> None:
    results = await asyncio.gather(
        *(
            _call(request, worker, instance_id, "DELETE", token=token)
            for worker in workers
        ),
        return_exceptions=True,
    )
    failed_workers = []
    for worker, result in zip(workers, results):
        if isinstance(result, BaseException):
            failed_workers.append(worker.id)
            logger.warning(
                "Failed to cancel tentative model ports for instance %s on worker %s",
                instance_id,
                worker.id,
            )
    if failed_workers:
        raise ServiceUnavailableException(
            message=f"Tentative port cleanup is pending on workers {failed_workers}"
        )


def _rollback_done(task: asyncio.Task) -> None:
    """Observe cleanup failures even when the request stops awaiting rollback."""
    _rollback_tasks.discard(task)
    try:
        task.result()
    except asyncio.CancelledError:
        logger.warning("Tentative port rollback was cancelled")
    except Exception:
        logger.exception("Tentative port rollback failed")


async def _rollback_claim(
    request: Request, workers: List[Worker], instance_id: int, token: UUID
) -> None:
    # Include the worker whose response was lost: it may have accepted the
    # claim even though the coordinator did not receive its reply.
    task = asyncio.create_task(_rollback(request, workers, instance_id, token))
    _rollback_tasks.add(task)
    task.add_done_callback(_rollback_done)
    await asyncio.shield(task)


@router.get("/{instance_id}/port-reservations")
async def availability(
    instance_id: int, request: Request, session: SessionDep, ctx: TenantContextDep
) -> List[WorkerPortAvailability]:
    workers = await _members(session, ctx, instance_id)
    bodies = await asyncio.gather(
        *(_call(request, worker, instance_id, "GET") for worker in workers)
    )
    snapshots = [WorkerPortAvailability.model_validate_json(body) for body in bodies]
    if [item.worker_id for item in snapshots] != [worker.id for worker in workers]:
        raise ServiceUnavailableException(
            message="Worker port snapshot identity mismatch"
        )
    return snapshots


@router.post("/{instance_id}/port-reservations")
async def reserve(
    instance_id: int,
    claim: ModelPortClaim,
    request: Request,
    session: SessionDep,
    ctx: TenantContextDep,
) -> PortClaimResult:
    workers = await _members(session, ctx, instance_id)
    try:
        for worker in workers:
            body = await _call(request, worker, instance_id, "POST", claim=claim)
            result = PortClaimResult.model_validate_json(body)
            if not result.accepted:
                break
        else:
            return PortClaimResult(accepted=True)
    except BaseException:
        # The callback records cleanup failures. Preserve the original RPC
        # error and its cause, or cancellation, for the caller.
        with contextlib.suppress(Exception):
            await _rollback_claim(request, workers, instance_id, claim.token)
        raise

    try:
        await _rollback_claim(request, workers, instance_id, claim.token)
    except ServiceUnavailableException as exc:
        # A failed rollback must stop retries so the caller retains its claim
        # for cleanup. Include the rejection without reporting a clean release.
        exc.details = {"rejection": result.model_dump(mode="json")}
        raise
    return result


@router.delete("/{instance_id}/port-reservations", status_code=204)
async def release(
    instance_id: int,
    token: Annotated[UUID, Header(alias="X-Claim-Token")],
    request: Request,
    session: SessionDep,
    ctx: TenantContextDep,
) -> None:
    workers = await _members(session, ctx, instance_id)
    # An explicit cleanup must report failed cancellations so the caller can
    # retain its cleanup metadata and retry instead of forgetting remote claims.
    await _rollback(request, workers, instance_id, token)
