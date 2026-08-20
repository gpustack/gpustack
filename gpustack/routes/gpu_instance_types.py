import json
import logging
from typing import (
    Annotated,
    Any,
    AsyncIterator,
    Callable,
    List,
    Literal,
    Optional,
    Tuple,
)

from fastapi import APIRouter, Depends, Query, Request, status
from kubernetes_asyncio import client
from starlette.responses import StreamingResponse

from gpustack.api.exceptions import BadRequestException, NotFoundException
from gpustack.api.tenant import TenantContext, cluster_visibility_conditions
from gpustack.gpu_instances import gateway_client
from gpustack.gpu_instances.cluster_apis import ClusterOps
from gpustack.routes.gpu_instances_helper import (
    assert_cluster_gpu_service,
    build_cluster_ops,
    ensure_writable,
    handle_error,
    watch_event_stream,
)

from gpustack.schemas import (
    Cluster,
    GPUAggregatedInstanceTypePublic,
    GPUAggregatedInstanceTypesPublic,
    GPUInstanceType,
    GPUInstanceTypeCreate,
    GPUInstanceTypeListParams,
    GPUInstanceTypeUpdate,
    GPUInstanceTypePublic,
    GPUInstanceTypesPublic,
)
from gpustack.schemas.clusters import ClusterProvider, is_gpu_service_cluster
from gpustack.schemas.common import Pagination
from gpustack.schemas.gpu_instance_types import DERIVED_FROM_NODE_LABEL
from gpustack.server.db import async_session
from gpustack.server.deps import SessionDep, TenantContextDep

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/aggregated",
    response_model=GPUAggregatedInstanceTypesPublic,
    response_model_exclude_none=True,
)
async def get_gpu_aggregated_instance_types(
    ctx: TenantContextDep,
    watch: bool = False,
):
    # Mirror cluster-list visibility: surface every Kubernetes cluster
    # the caller can see — both the ones they own AND the ones granted
    # via ``cluster_access``. Without the grants path an Org member
    # would see an empty instance-type list even after a platform
    # admin authorised them on a K8s cluster.
    async with async_session() as session:
        clusters = await Cluster.all_by_fields(
            session=session,
            fields={"provider": ClusterProvider.Kubernetes},
            extra_conditions=cluster_visibility_conditions(ctx, Cluster),
        )

    # ...then narrow to the ones registered for GPU Service. A Model Service
    # cluster's operator still derives instance types from its nodes, but they
    # describe capacity committed to model deployment, so offering them here
    # would let a caller launch a GPU Instance against a cluster that was never
    # provisioned to host one. Purpose lives inside the ``k8s_options`` JSON
    # column, so it narrows here rather than in the query's ``fields``.
    # gateway_client list/watch take cluster ids as strings.
    cluster_ids = [str(c.id) for c in clusters if is_gpu_service_cluster(c)]

    if not cluster_ids:
        # No visible GPU Service clusters → return an empty aggregate. The
        # gateway treats an empty cluster filter as "all clusters", so
        # forwarding the empty set would leak the whole fleet to a caller who
        # can see nothing.
        return GPUAggregatedInstanceTypesPublic(items=[])

    if watch:
        # The gateway streams raw Kubernetes watch verbs (ADDED/MODIFIED/
        # DELETED); wrap them into GPUStack events so a caller sees the same
        # event-type contract as every other GPUStack stream.
        return StreamingResponse(
            watch_event_stream(
                _aggregated_instance_type_events(cluster_ids),
                GPUAggregatedInstanceTypePublic.model_validate,
            ),
            media_type="text/event-stream",
        )

    return await gateway_client.list_instance_types(
        clusters=cluster_ids,
        aggregated=True,
    )


@router.get(
    "",
    response_model=GPUInstanceTypesPublic,
    response_model_exclude_none=True,
)
async def get_gpu_instance_types(
    request: Request,
    ctx: TenantContextDep,
    params: GPUInstanceTypeListParams = Depends(),
    name: Optional[str] = None,
    search: Optional[str] = None,
    cluster_id: Optional[int] = None,
    # Annotated rather than ``= Query(None, ...)`` so the default really is None:
    # this handler is also called directly (tests), where a Query object left as
    # the default would read as "a purpose was requested".
    purpose: Annotated[
        Optional[Literal["gpu_service", "model_service"]],
        Query(
            description=(
                "Filter by cluster purpose. ``gpu_service`` keeps only GPU "
                "Service clusters (the Instance Types page); ``model_service`` "
                "keeps only model-deployment clusters. Unset narrows nothing, "
                "which is what the model deploy form's slicing GPU type picker "
                "relies on."
            ),
        ),
    ] = None,
    # Annotated for the same reason as ``purpose`` above.
    source: Annotated[
        Optional[Literal["record", "live"]],
        Query(
            description=(
                "Where to read from. Omitted or ``record`` reads the "
                "control-plane record table (fleet-wide, or one cluster with "
                "``cluster_id``). ``live`` proxies the named cluster's own "
                "``list``, which is the only read that carries the resource "
                "ledger (``status.acceleratorSliced`` / "
                "``status.acceleratorPartitioned``); it requires ``cluster_id``."
            ),
        ),
    ] = None,
):
    """List instance types, from the record table by default or live per cluster.

    The record read is served from ``gpu_instance_types`` instead of proxying a
    live ``list``, so ``cluster_id`` is a filter rather than a scope and a cluster
    with no ready worker yields an empty page by construction — it simply holds no
    rows — rather than surfacing its proxy's 5xx.

    ``source=live`` is the escape hatch for the one field set the table does not
    hold: the record projection deliberately drops the volatile resource ledger
    (see ``GPUInstanceTypeStatusPublic``), which the model deploy form's slicing
    GPU type picker sizes its inputs from, so that caller asks for the cluster's
    live catalog instead. It reads one named cluster and accepts that cluster's
    failures, including a ``503`` when no worker is ready. The cluster hands back
    its catalog whole, so ``name`` / ``search`` are rejected rather than silently
    ignored, while ``page`` / ``perPage`` / ``sort_by`` are simply not applied — the
    synthesized single-page ``pagination`` in the response states that.
    """
    if source == "live":
        # Both rejected before anything is resolved: these are malformed requests,
        # not requests whose answer happens to be empty.
        if cluster_id is None:
            # A live read proxies into exactly one cluster's apiserver, so it has
            # to name one.
            raise BadRequestException(message="cluster_id is required when source=live")
        if name or search:
            # The upstream ``list`` takes no filters, so narrowing cannot be
            # honoured here. Refusing is the point: a caller that asked to narrow
            # and silently received everything is the same failure shape as a
            # parameter read as something it is not. Pagination and sort differ in
            # kind — the response reports the single page it returned, so their
            # non-application is visible rather than invisible.
            raise BadRequestException(
                message=(
                    "name and search are not supported when source=live; "
                    "the cluster returns its catalog whole"
                )
            )

    allowed_ids = await _visible_cluster_ids(ctx, purpose, cluster_id)

    if source == "live":
        ops = await _build_instance_type_ops(request, cluster_id, allowed_ids)
        if ops is not None:
            if params.watch:
                # The cluster's own Kubernetes watch, so the picker sees ledger
                # movement. The source owns and closes ops.
                return StreamingResponse(
                    watch_event_stream(
                        _cluster_instance_type_events(ops),
                        _to_instance_type_public,
                    ),
                    media_type="text/event-stream",
                )

            async with ops, handle_error():
                result = await ops.list_instance_types()
            return _to_instance_types_public(result, params)

        # No cluster to read — a foreign or invisible ``cluster_id``, or a cluster
        # deleted between the two queries. Fall through to the record read rather
        # than answering differently: the answers it already gives for an
        # unreachable cluster set are the ones this path wants too — an empty page,
        # or an empty stream for a watch, and never a 403/404 a caller could probe
        # another tenant's fleet with. (A deleted cluster's rows are cascade-deleted
        # with it, so that case reads empty as well.)

    fields = {"deleted_at": None}
    if name:
        fields["name"] = name
    fuzzy_fields = {"name": search} if search else {}

    if params.watch:
        # Deliberately ahead of the empty-set guard below: that guard is about the
        # query, and this path never runs one. A caller asking for
        # text/event-stream has to get a stream even when it can see nothing — a
        # JSON body would break its reader — and a filter_func over an empty set
        # is exactly an empty stream.
        return StreamingResponse(
            GPUInstanceType.streaming(
                fields=fields,
                fuzzy_fields=fuzzy_fields,
                filter_func=_make_instance_type_visibility_filter(allowed_ids),
            ),
            media_type="text/event-stream",
        )

    if not allowed_ids:
        # Answer directly rather than querying with an empty ``IN`` set: an empty
        # filter that gets reinterpreted as "everything" is the tenant leak shape
        # guarded against in get_gpu_aggregated_instance_types.
        return _empty_instance_types_page(params)

    async with async_session() as session:
        return await GPUInstanceType.paginated_by_query(
            session=session,
            fields=fields,
            fuzzy_fields=fuzzy_fields,
            extra_conditions=[GPUInstanceType.cluster_id.in_(allowed_ids)],
            page=params.page,
            per_page=params.perPage,
            order_by=params.order_by,
        )


@router.post(
    "",
    response_model=GPUInstanceTypePublic,
    response_model_exclude_none=True,
)
async def create_gpu_instance_type(
    request: Request,
    session: SessionDep,
    ctx: TenantContextDep,
    create: GPUInstanceTypeCreate,
    cluster_id: int,
):
    cluster = ensure_writable(
        await Cluster.one_by_id(
            session=session,
            id=cluster_id,
        ),
        ctx,
    )
    assert_cluster_gpu_service(cluster)
    ops = await build_cluster_ops(request, session, cluster)

    spec = create.spec.model_dump(by_alias=True, exclude_none=True)
    async with (
        ops,
        handle_error(
            already_exists_message=(
                f"Instance type {create.name} already exists in cluster {cluster.name}"
            )
        ),
    ):
        # ignore_existed=False: with the idempotent default the create is never
        # attempted and the pre-existing object is read back as a 200 — a
        # "successful" create that created nothing (#6087).
        result = await ops.create_instance_type(create.name, spec, ignore_existed=False)
    return _to_instance_type_public(result)


@router.put(
    "",
    response_model=GPUInstanceTypePublic,
    response_model_exclude_none=True,
)
async def update_gpu_instance_type(
    request: Request,
    session: SessionDep,
    ctx: TenantContextDep,
    update: GPUInstanceTypeUpdate,
    cluster_id: int,
):
    cluster = ensure_writable(
        await Cluster.one_by_id(
            session=session,
            id=cluster_id,
        ),
        ctx,
    )
    assert_cluster_gpu_service(cluster)
    ops = await build_cluster_ops(request, session, cluster)

    spec = update.spec.model_dump(by_alias=True, exclude_none=True)
    async with ops, handle_error():
        result = await ops.update_instance_type(update.name, spec)
    if result is None:
        raise NotFoundException(message=f"Instance type {update.name} not found")
    return _to_instance_type_public(result)


@router.delete("/{name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_gpu_instance_type(
    request: Request,
    session: SessionDep,
    ctx: TenantContextDep,
    name: str,
    cluster_id: int,
):
    cluster = ensure_writable(
        await Cluster.one_by_id(
            session=session,
            id=cluster_id,
        ),
        ctx,
    )
    assert_cluster_gpu_service(cluster)
    ops = await build_cluster_ops(request, session, cluster)

    async with ops, handle_error():
        existed = await ops.delete_instance_type(name)
    if not existed:
        raise NotFoundException(message=f"Instance type {name} not found")


@router.put(
    "/{name}/deactivate",
    response_model=GPUInstanceTypePublic,
    response_model_exclude_none=True,
)
async def deactivate_gpu_instance_type(
    request: Request,
    session: SessionDep,
    ctx: TenantContextDep,
    name: str,
    cluster_id: int,
):
    cluster = ensure_writable(
        await Cluster.one_by_id(
            session=session,
            id=cluster_id,
        ),
        ctx,
    )
    assert_cluster_gpu_service(cluster)
    ops = await build_cluster_ops(request, session, cluster)

    async with ops, handle_error():
        result = await ops.deactivate_instance_type(name)
    if result is None:
        raise NotFoundException(message=f"Instance type {name} not found")
    return _to_instance_type_public(result)


@router.put(
    "/{name}/activate",
    response_model=GPUInstanceTypePublic,
    response_model_exclude_none=True,
)
async def activate_gpu_instance_type(
    request: Request,
    session: SessionDep,
    ctx: TenantContextDep,
    name: str,
    cluster_id: int,
):
    cluster = ensure_writable(
        await Cluster.one_by_id(
            session=session,
            id=cluster_id,
        ),
        ctx,
    )
    assert_cluster_gpu_service(cluster)
    ops = await build_cluster_ops(request, session, cluster)

    async with ops, handle_error():
        result = await ops.activate_instance_type(name)
    if result is None:
        raise NotFoundException(message=f"Instance type {name} not found")
    return _to_instance_type_public(result)


async def _build_instance_type_ops(
    request: Request,
    cluster_id: int,
    allowed_ids: List[int],
) -> Optional[ClusterOps]:
    """Build a client for a live read, or return None when there is no cluster.

    Visibility is decided by ``allowed_ids`` — the very set the record read filters
    on — so the live path cannot reach a cluster the table-backed one hides, and it
    is decided *before* any client is built, so an invisible cluster is never
    contacted. A row that disappeared between the two queries reads as no cluster
    as well: its types are cascade-deleted with it, so there is nothing to serve.
    """
    if cluster_id not in allowed_ids:
        return None

    async with async_session() as session:
        cluster = await Cluster.one_by_id(session=session, id=cluster_id)
        if cluster is None:
            return None
        return await build_cluster_ops(request, session, cluster)


async def _cluster_instance_type_events(
    ops: ClusterOps,
) -> AsyncIterator[Tuple[Optional[str], dict]]:
    """Normalize a cluster's native Kubernetes watch into ``(verb, raw object)``.

    ``ops`` is entered here — not in the consumer — so its client is closed on
    every exit path, including a disconnect that cancels the stream. A watch
    ERROR / unrecoverable ``resourceVersion`` expiry surfaces as an
    ``ApiException`` and ends the source cleanly (logged at WARNING because the
    watch otherwise never terminates) rather than as a data frame.
    """
    async with ops:
        try:
            async for native in ops.watch_instance_types():
                yield native["type"], native["raw_object"]
        except client.exceptions.ApiException as e:
            logger.warning(
                "instance-type watch for cluster %s ended: %s", ops.cluster_id, e
            )


async def _visible_cluster_ids(
    ctx: TenantContext,
    purpose: Optional[str],
    cluster_id: Optional[int],
) -> List[int]:
    """Resolve the clusters whose recorded instance types the caller may read.

    The record table is cluster-scoped and carries no ``owner_principal_id``, so
    the allowed cluster set *is* the authorization boundary, and it is resolved
    before anything is queried. ``cluster_visibility_conditions`` mirrors the
    cluster list — clusters the caller owns AND the ones granted via
    ``cluster_access`` — so an Org member authorised on a K8s cluster sees its
    types. Kubernetes only: no other provider has an InstanceType catalog for the
    controller to project.
    """
    fields: dict = {"provider": ClusterProvider.Kubernetes}
    if cluster_id is not None:
        # Narrow in SQL rather than loading every visible cluster to keep one.
        # This filters rather than trusts, because the visibility conditions
        # still apply: a cluster_id outside the allowed set comes back as no
        # rows, which yields an empty page and not a 403 — so a caller cannot
        # probe for the existence of another tenant's cluster.
        fields["id"] = cluster_id

    async with async_session() as session:
        clusters = await Cluster.all_by_fields(
            session=session,
            fields=fields,
            extra_conditions=cluster_visibility_conditions(ctx, Cluster),
        )

    if purpose is not None:
        # An opt-in semantic filter, never an authorization boundary — a Model
        # Service cluster's types are no secret from a caller who can see the
        # cluster. This route has two consumers wanting opposite halves of the
        # fleet: the Instance Types page (GPU Service) and the model deploy form's
        # slicing GPU type picker, which targets Model Service clusters by
        # definition. Narrowing unconditionally would break that picker, so an
        # omitted purpose narrows nothing.
        want_gpu_service = purpose == "gpu_service"
        clusters = [
            c for c in clusters if is_gpu_service_cluster(c) == want_gpu_service
        ]

    return [c.id for c in clusters]


def _make_instance_type_visibility_filter(
    allowed_ids: List[int],
) -> Callable[[Any], bool]:
    """Row-level twin of the read's ``cluster_id IN (...)``, for the watch stream.

    Serves the same purpose as ``_make_worker_visibility_filter``: the SQL
    ``extra_conditions`` never reach bus events, so without this the stream would
    leak rows the REST read hides. ``getattr`` rather than plain attribute access
    because a DELETED event can carry an id-only dict when the change detector held
    no object to enrich it with — and an AttributeError there is swallowed by
    ``streaming``, silently ending the whole stream. A payload naming no cluster
    cannot be attributed to an allowed one, so it is dropped.

    It does NOT share that helper's self-healing property, and the difference
    matters. ``_make_worker_visibility_filter`` re-derives visibility from each
    row's own ``owner_principal_id``, so it tracks grants as they change. This one
    closes over a snapshot of the ``clusters`` table taken when the stream opened,
    because the table it filters carries no owner of its own. So for the life of one
    stream: a cluster registered or granted afterwards yields no events, and a
    ``cluster_access`` grant *revoked* afterwards keeps yielding them, while the
    REST read correctly reflects both immediately. The client reconnects on
    navigation, which is what bounds the window.
    """
    allowed = frozenset(allowed_ids)

    def _visible(row: Any) -> bool:
        return getattr(row, "cluster_id", None) in allowed

    return _visible


async def _aggregated_instance_type_events(
    cluster_ids: list[str],
) -> AsyncIterator[Tuple[Optional[str], dict]]:
    """Normalize the gateway's aggregated InstanceType watch into ``(verb, object)``.

    The gateway re-frames each ``manager.WorkerEvent`` as ``<json>\\n\\n`` (see
    gateway_client._stream); the JSON carries a Kubernetes watch ``type`` and an
    already-aggregated ``object`` (name / spec / aggregated status). Malformed
    lines are dropped, mirroring controllers._on_downstream_event.
    """
    async for line in gateway_client.watch_instance_types(
        clusters=cluster_ids,
        aggregated=True,
    ):
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            logger.warning("discarding malformed aggregated event: %r", line)
            continue
        yield event.get("type"), event.get("object") or {}


def _to_instance_type_public(item: dict) -> GPUInstanceTypePublic:
    """Map a raw ``worker.gpustack.ai/v1`` InstanceType dict into the public
    schema, hoisting ``metadata.name`` to ``name`` and the operator's
    derived-from-node marker out of ``metadata.labels``. A freshly-created CR
    may lack a reconciled status, so an empty status maps to ``{}`` (every
    ``GPUInstanceTypeStatus`` field is Optional)."""
    metadata = item.get("metadata") or {}
    labels = metadata.get("labels") or {}
    return GPUInstanceTypePublic(
        name=metadata.get("name"),
        spec=item.get("spec") or {},
        status=item.get("status") or {},
        derived_from_node=labels.get(DERIVED_FROM_NODE_LABEL) == "true",
    )


def _empty_instance_types_page(
    params: GPUInstanceTypeListParams,
) -> GPUInstanceTypesPublic:
    """The empty answer, shared by both sources so "nothing here" has one shape.

    ``totalPage`` is 0 because there is no page to fetch, and ``page`` /
    ``perPage`` echo what was asked for. A live read whose catalog is empty returns
    this too: a client must not be able to tell the two sources apart by the shape
    of an empty response.
    """
    return GPUInstanceTypesPublic(
        items=[],
        pagination=Pagination(
            page=params.page,
            perPage=params.perPage,
            total=0,
            totalPage=0,
        ),
    )


def _to_instance_types_public(
    result: dict,
    params: GPUInstanceTypeListParams,
) -> GPUInstanceTypesPublic:
    """Wrap one cluster's live ``list`` payload as a single-page paginated list.

    The upstream ``list`` is not paginated — the cluster hands back its whole
    catalog — so the pagination block states exactly that, and the caller's
    ``page`` / ``perPage`` / ``sort_by`` are not applied: there is no second page
    to ask for. The table-backed read is the paginated one.
    """
    items = [_to_instance_type_public(i) for i in result.get("items", [])]
    if not items:
        return _empty_instance_types_page(params)
    return GPUInstanceTypesPublic(
        items=items,
        pagination=Pagination(
            page=1,
            perPage=len(items),
            total=len(items),
            totalPage=1,
        ),
    )
