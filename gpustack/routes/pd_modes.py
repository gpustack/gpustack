import math
from typing import List, Optional

from fastapi import APIRouter

from gpustack.api.tenant import assert_cluster_visible
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.common import PaginatedList, Pagination
from gpustack.schemas.pd_mode_resolution import PDModeResolution
from gpustack.schemas.pd_modes import PDMode
from gpustack.server.cluster_accelerators import cluster_vendors
from gpustack.server.pd_mode_catalog import get_pd_modes
from gpustack.server.pd_mode_resolver import resolve_pd_mode
from gpustack.server.deps import ListParamsDep, SessionDep, TenantContextDep

router = APIRouter()


@router.get("", response_model=PaginatedList[PDMode])
async def list_pd_modes(
    params: ListParamsDep,
    search: Optional[str] = None,
):
    """The catalog behind the deployment form's single PD-mode dropdown.
    Read-only and bundled with the release, like the cache-provider
    catalog."""
    modes: List[PDMode] = get_pd_modes()
    if search:
        search = search.strip().lower()
        modes = [
            mode
            for mode in modes
            if search in mode.name.lower()
            or (mode.display_name and search in mode.display_name.lower())
        ]

    count = len(modes)

    if params.page < 1 or params.perPage < 1:
        # Return all items.
        pagination = Pagination(
            page=1,
            perPage=count,
            total=count,
            totalPage=1,
        )
        return PaginatedList[PDMode](items=modes, pagination=pagination)

    # Paginate results.
    total_page = math.ceil(count / params.perPage)

    start_index = (params.page - 1) * params.perPage
    end_index = start_index + params.perPage

    paginated_items = modes[start_index:end_index]

    pagination = Pagination(
        page=params.page,
        perPage=params.perPage,
        total=count,
        totalPage=total_page,
    )

    return PaginatedList[PDMode](items=paginated_items, pagination=pagination)


@router.get("/resolve", response_model=PDModeResolution)
async def resolve(
    session: SessionDep,
    ctx: TenantContextDep,
    cluster_id: Optional[int] = None,
    backend: Optional[str] = None,
    vendor: Optional[str] = None,
):
    """Which recipe this deployment gets, and why every other one is out.

    Declared before ``/{name}`` on purpose: FastAPI matches in order, so the
    path parameter would otherwise swallow ``/resolve``.

    The judgement lives here rather than in the client because the deciding
    fact -- which accelerators the cluster's ready workers report -- is not in
    the deploy form. What the client has is the cluster's ``provider``
    (Docker / Kubernetes), which is the infrastructure provider, not the
    accelerator vendor.
    """
    if cluster_id is not None:
        # The catalog itself carries no tenant data, but the accelerators a
        # cluster reports do: answering for an id the caller cannot see would
        # let any Org member probe another Org's hardware. Checked before the
        # query rather than filtered inside it, so an invisible cluster reads
        # as absent rather than as one with no accelerators — the latter would
        # come back as a resolution failure and leak its existence anyway.
        cluster = await Cluster.one_by_id(session, cluster_id)
        assert_cluster_visible(
            ctx, cluster, not_found_message=f"cluster {cluster_id} not found"
        )
    vendors = await cluster_vendors(session, cluster_id)
    return resolve_pd_mode(backend, vendors, vendor=vendor)
