import math
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, Query

from gpustack.routes.models import NotFoundException
from gpustack.schemas.common import PaginatedList, Pagination
from gpustack.server.catalog import (
    ModelRecipe,
    ModelSet,
    ModelSetPublic,
    ModelSpec,
    get_model_sets,
    get_model_set_recipes,
)
from gpustack.server.deps import ListParamsDep

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
    id: int,
    params: ListParamsDep,
    model_set_recipes: Dict[int, List[ModelRecipe]] = Depends(get_model_set_recipes),
):

    recipes = model_set_recipes.get(id, [])
    if not recipes:
        raise NotFoundException(message="Model set not found")

    specs = model_recipes_to_specs(recipes)
    count = len(specs)
    total_page = math.ceil(count / params.perPage)
    pagination = Pagination(
        page=params.page,
        perPage=params.perPage,
        total=count,
        totalPage=total_page,
    )

    return PaginatedList[ModelSpec](items=specs, pagination=pagination)


def model_recipes_to_specs(recipes: List[ModelRecipe]) -> List[ModelSpec]:
    return [ModelSpec(**recipe.model_dump()) for recipe in recipes]
