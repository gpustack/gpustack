import math
from typing import Optional
from fastapi import APIRouter, Query

from gpustack.schemas.common import PaginatedList, Pagination
from gpustack.server.catalog import (
    DraftModel,
    get_catalog_draft_models,
)
from gpustack.server.deps import ListParamsDep, SessionDep
from gpustack.utils.search import rank_matches

router = APIRouter()


@router.get("", response_model=PaginatedList[DraftModel])
async def get_draft_models(
    session: SessionDep,
    params: ListParamsDep,
    search: str = None,
    algorithm: Optional[str] = Query(None, description="Filter by algorithm."),
):
    draft_models = await get_catalog_draft_models(session)

    if search:
        # Same catalog, same directory page in the UI as /model-sets, so the
        # same matching: `qwen3 235b eagle3` finds the model whose name is
        # spelled Qwen3-235B-A22B-EAGLE3.
        draft_models = rank_matches(draft_models, search, key=lambda model: model.name)

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
