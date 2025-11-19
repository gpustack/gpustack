import math
from typing import List, Optional
from fastapi import APIRouter, Depends, Query

from gpustack.schemas.common import PaginatedList, Pagination
from gpustack.server.catalog import (
    DraftModel,
    get_catalog_draft_models,
)
from gpustack.server.deps import ListParamsDep

router = APIRouter()


@router.get("", response_model=PaginatedList[DraftModel])
async def get_draft_models(
    params: ListParamsDep,
    search: str = None,
    algorithm: Optional[str] = Query(None, description="Filter by algorithm."),
    draft_models: List[DraftModel] = Depends(get_catalog_draft_models),
):
    if search:
        search = search.strip().lower()
        draft_models = [model for model in draft_models if search in model.name.lower()]

    if algorithm:
        draft_models = [
            model
            for model in draft_models
            if model.algorithm is not None and model.algorithm == algorithm
        ]

    count = len(draft_models)

    if params.page < 1 or params.perPage < 1:
        # Return all items.
        pagination = Pagination(
            page=1,
            perPage=count,
            total=count,
            totalPage=1,
        )
        return PaginatedList[DraftModel](items=draft_models, pagination=pagination)

    # Paginate results.
    total_page = math.ceil(count / params.perPage)

    start_index = (params.page - 1) * params.perPage
    end_index = start_index + params.perPage

    paginated_items = draft_models[start_index:end_index]

    pagination = Pagination(
        page=params.page,
        perPage=params.perPage,
        total=count,
        totalPage=total_page,
    )

    return PaginatedList[DraftModel](items=paginated_items, pagination=pagination)
