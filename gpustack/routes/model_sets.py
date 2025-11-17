import math
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, Query
from packaging.version import Version
from packaging.specifiers import SpecifierSet

from gpustack.routes.models import NotFoundException
from gpustack.schemas.common import PaginatedList, Pagination
from gpustack.schemas.gpu_devices import GPUDevice
from gpustack.server.catalog import (
    ModelSet,
    ModelSetPublic,
    ModelSpec,
    get_model_sets,
    get_model_set_specs,
)
from gpustack.server.deps import ListParamsDep, SessionDep

router = APIRouter()


@router.get("", response_model=PaginatedList[ModelSetPublic])
async def get_model_sets(
    params: ListParamsDep,
    search: str = None,
    categories: Optional[List[str]] = Query(None, description="Filter by categories."),
    model_sets: List[ModelSet] = Depends(get_model_sets),
):
    if search:
        model_sets = [
            model for model in model_sets if search.lower() in model.name.lower()
        ]

    if categories:
        model_sets = [
            model
            for model in model_sets
            if model.categories is not None
            and any(category in model.categories for category in categories)
        ]

    count = len(model_sets)
    total_page = math.ceil(count / params.perPage)

    start_index = (params.page - 1) * params.perPage
    end_index = start_index + params.perPage

    paginated_items = model_sets[start_index:end_index]

    pagination = Pagination(
        page=params.page,
        perPage=params.perPage,
        total=count,
        totalPage=total_page,
    )

    return PaginatedList[ModelSetPublic](items=paginated_items, pagination=pagination)


@router.get("/{id}/specs", response_model=PaginatedList[ModelSpec])
async def get_model_specs(
    session: SessionDep,
    id: int,
    params: ListParamsDep,
    cluster_id: Optional[int] = Query(
        None, description="Filter specs compatible with the given cluster ID."
    ),
    model_set_specs: Dict[int, List[ModelSpec]] = Depends(get_model_set_specs),
):

    specs = model_set_specs.get(id, [])
    if not specs:
        raise NotFoundException(message="Model set not found")

    fields = {}
    if cluster_id:
        fields["cluster_id"] = cluster_id
    gpus = await GPUDevice.all_by_fields(session, fields)
    specs = filter_specs_by_gpu(gpus or [], specs)

    count = len(specs)
    total_page = math.ceil(count / params.perPage)
    pagination = Pagination(
        page=params.page,
        perPage=params.perPage,
        total=count,
        totalPage=total_page,
    )

    return PaginatedList[ModelSpec](items=specs, pagination=pagination)


def filter_specs_by_gpu(
    gpus: List[GPUDevice], specs: List[ModelSpec]
) -> List[ModelSpec]:
    """Filter model specs based on the GPUs available."""

    # Matched specs mapping by mode (standard, throughput, latency, etc.).
    filtered = {}

    gpu_vendors = {gpu.vendor.lower() for gpu in gpus}

    for spec in specs:
        if spec.mode in filtered:
            continue

        gf = spec.gpu_filters
        if gf is None:
            filtered[spec.mode] = spec
            continue

        # GPU Vendor match
        vendor_ok = True
        if gf.vendor:
            wanted = {v.lower() for v in gf.vendor}
            vendor_ok = not wanted.isdisjoint(gpu_vendors)

        # Compute capability (pip-style version matching)
        cc_ok = True
        if gf.compute_capability:
            cc_ok = any(
                match_compute_capability(gf.compute_capability, gpu.compute_capability)
                for gpu in gpus
            )

        if vendor_ok and cc_ok:
            filtered[spec.mode] = spec

    return list(filtered.values())


def match_compute_capability(filter_str: Optional[str], gpu_cc: Optional[str]) -> bool:
    """Check if the GPU compute capability matches the given filter string.

    Args:
        filter_str (Optional[str]): The pip-style version specifier string.
        gpu_cc (Optional[str]): The GPU compute capability version string.

    Returns:
        bool: True if the GPU compute capability matches the filter, False otherwise.
    """
    if not filter_str:
        return True

    if not gpu_cc:
        return False

    try:
        spec_set = SpecifierSet(filter_str)
        cc_version = Version(gpu_cc)
        return cc_version in spec_set
    except Exception:
        return False
