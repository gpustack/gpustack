import math
from typing import List, Optional
from fastapi import APIRouter, Query
from gpustack.api.exceptions import NotFoundException
from gpustack.schemas.common import PaginatedList, Pagination
from gpustack.schemas.gpu_devices import GPUDevice
from gpustack.server.catalog import (
    ModelSetPublic,
    get_model_sets as list_catalog_model_sets,
    get_model_set_specs as get_catalog_model_set_specs,
)
from gpustack.schemas.catalog_source import (
    BUILTIN_CATALOG_SOURCE_NAME,
    CUSTOM_CATALOG_SOURCE_NAME,
    CatalogSource,
    normalize_catalog_yaml,
)
from gpustack.server.deps import ListParamsDep, SessionDep
from gpustack.server.sources.routes import SourceConfigSpec
from gpustack.utils.search import rank_matches
from gpustack.schemas.model_sets import ModelSpec
from gpustack.utils.model_catalog import filter_specs_by_gpu

router = APIRouter()

# The catalog's source binding for the shared engine. Configured through
# ``routes/ota_sources.py``, which mounts one endpoint family for all three kinds.
CATALOG_SOURCE_SPEC = SourceConfigSpec(
    source_cls=CatalogSource,
    normalize=normalize_catalog_yaml,
    builtin_name=BUILTIN_CATALOG_SOURCE_NAME,
    custom_name=CUSTOM_CATALOG_SOURCE_NAME,
)


@router.get("", response_model=PaginatedList[ModelSetPublic])
async def get_model_sets(
    session: SessionDep,
    params: ListParamsDep,
    search: str = None,
    categories: Optional[List[str]] = Query(None, description="Filter by categories."),
):
    model_sets = await list_catalog_model_sets(session)

    if search:
        # Matching ignores separators, which admits loose matches; ranking by
        # relevance is what keeps the model the user typed above them, since
        # the curated (order, release_date) sequence knows nothing about the
        # query.
        model_sets = rank_matches(model_sets, search, key=lambda model: model.name)

    if categories:
        model_sets = [
            model
            for model in model_sets
            if model.categories is not None
            and any(category in model.categories for category in categories)
        ]

    count = len(model_sets)

    if params.page < 1 or params.perPage < 1:
        # Return all items.
        pagination = Pagination(
            page=1,
            perPage=count,
            total=count,
            totalPage=1,
        )
        return PaginatedList[ModelSetPublic](items=model_sets, pagination=pagination)

    # Paginate results.
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
):
    specs = await get_catalog_model_set_specs(session, id)
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
