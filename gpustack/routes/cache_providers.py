import math
from typing import List, Optional

from fastapi import APIRouter

from gpustack.api.exceptions import NotFoundException
from gpustack.schemas.cache_providers import CacheProvider
from gpustack.schemas.common import PaginatedList, Pagination
from gpustack.server.cache_provider_catalog import (
    get_cache_provider,
    get_cache_providers,
)
from gpustack.server.deps import ListParamsDep

router = APIRouter()


@router.get("", response_model=PaginatedList[CacheProvider])
async def list_cache_providers(
    params: ListParamsDep,
    search: Optional[str] = None,
):
    providers: List[CacheProvider] = get_cache_providers()
    if search:
        search = search.strip().lower()
        providers = [
            provider
            for provider in providers
            if search in provider.name.lower()
            or (provider.display_name and search in provider.display_name.lower())
        ]

    count = len(providers)

    if params.page < 1 or params.perPage < 1:
        # Return all items.
        pagination = Pagination(
            page=1,
            perPage=count,
            total=count,
            totalPage=1,
        )
        return PaginatedList[CacheProvider](items=providers, pagination=pagination)

    # Paginate results.
    total_page = math.ceil(count / params.perPage)

    start_index = (params.page - 1) * params.perPage
    end_index = start_index + params.perPage

    paginated_items = providers[start_index:end_index]

    pagination = Pagination(
        page=params.page,
        perPage=params.perPage,
        total=count,
        totalPage=total_page,
    )

    return PaginatedList[CacheProvider](items=paginated_items, pagination=pagination)


@router.get("/{name}", response_model=CacheProvider)
async def get_cache_provider_by_name(name: str):
    provider = get_cache_provider(name)
    if provider is None:
        raise NotFoundException(message=f"Cache provider '{name}' not found")
    return provider
