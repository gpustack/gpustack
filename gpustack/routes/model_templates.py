import math
from fastapi import APIRouter, Depends

from gpustack.schemas.common import PaginatedList, Pagination
from gpustack.server.catalog import ModelTemplate, get_model_catalog
from gpustack.server.deps import ListParamsDep

router = APIRouter()


@router.get("", response_model=PaginatedList[ModelTemplate])
async def get_model_templates(
    params: ListParamsDep, search: str = None, model_catalog=Depends(get_model_catalog)
):
    model_templates = []
    if search:
        for model_template in model_catalog:
            if search.lower() in model_template.name.lower():
                model_templates.append(model_template)
    else:
        model_templates = model_catalog

    count = len(model_templates)
    total_page = math.ceil(count / params.perPage)
    pagination = Pagination(
        page=params.page,
        perPage=params.perPage,
        total=count,
        totalPage=total_page,
    )

    return PaginatedList[ModelTemplate](items=model_templates, pagination=pagination)
