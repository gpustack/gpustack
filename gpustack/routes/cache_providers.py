import math
from typing import Dict, List, Optional, Set, Tuple

from fastapi import APIRouter
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.exceptions import BadRequestException, NotFoundException
from gpustack.schemas.cache_provider_source import (
    BUILTIN_CACHE_PROVIDER_SOURCE_NAME,
    CUSTOM_CACHE_PROVIDER_SOURCE_NAME,
    CacheProviderSource,
    normalize_cache_provider_yaml,
)
from gpustack.schemas.runner_source import (
    RunnerOverrideEntry,
    merged_runners,
)
from gpustack.schemas.cache_providers import (
    CUSTOM_VERSION,
    CacheProvider,
    localized_values,
)
from gpustack.schemas.cache_services import CacheService
from gpustack.schemas.common import PaginatedList, Pagination
from gpustack.schemas.source import SourceContent, SourceTypeEnum
from gpustack.server.cache_provider_catalog import (
    get_cache_provider,
    get_cache_providers,
    providers_from_documents,
)
from gpustack.server.deps import ListParamsDep, SessionDep
from gpustack.server.sources.routes import SourceConfigSpec

router = APIRouter()


@router.get("", response_model=PaginatedList[CacheProvider])
async def list_cache_providers(
    session: SessionDep,
    params: ListParamsDep,
    search: Optional[str] = None,
):
    providers: List[CacheProvider] = await get_cache_providers(session)
    if search:
        search = search.strip().lower()
        # Every translation of the display name is searchable, not just
        # the fallback: the term a user types is in the locale they are
        # reading the catalog in.
        providers = [
            provider
            for provider in providers
            if search in provider.name.lower()
            or any(
                search in name.lower()
                for name in localized_values(provider.display_name)
            )
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
async def get_cache_provider_by_name(name: str, session: SessionDep):
    provider = await get_cache_provider(session, name)
    if provider is None:
        raise NotFoundException(message=f"Cache provider '{name}' not found")
    return provider


async def _services_by_pinned_provider(
    session: AsyncSession,
) -> Dict[Tuple[str, Optional[str]], Set[str]]:
    """What every cache service pins, as ``(provider, version) -> service
    names``.

    The version is kept verbatim, because the three it can hold mean different
    things: a declared version must still be declared, the reserved "custom"
    identifier needs the provider to still opt into user-supplied images, and
    ``None`` is a service that never named one — it runs whatever the provider
    defaults to, so any document declaring versions satisfies it.
    """
    pinned: Dict[Tuple[str, Optional[str]], Set[str]] = {}
    for service in await CacheService.all(session):
        key = (service.provider_name.lower(), service.provider_version or None)
        pinned.setdefault(key, set()).add(service.name)
    return pinned


async def _reject_taking_away_a_provider_in_use(
    session: AsyncSession, proposed: List[SourceContent]
) -> None:
    """Refuse a source write that would take a running cache service's provider
    — or the version it pins — away.

    A configured document replaces the catalog outright, so the common mistake
    is writing one that simply forgets a provider: every service on it would
    then have nothing to launch, and the failure would surface at the next
    instance recreation rather than at the write that caused it.

    Every offending pin is named in one message: an admin whose document is
    missing three providers should not have to submit three times to learn all
    three.
    """
    # The catalog judged here is the one the write would materialize, derived
    # release lines and all: a version a service pins may come from the runner
    # images rather than from any document, and a check blind to those would
    # read every such pin as taken away.
    runners = merged_runners(await RunnerOverrideEntry.all(session))
    catalog = {
        provider.name.lower(): provider
        for provider in providers_from_documents(
            [source.content for source in proposed], runners
        )
    }
    blocked: List[str] = []
    for (name, version), services in sorted(
        (await _services_by_pinned_provider(session)).items()
    ):
        used_by = f"(used by: {', '.join(sorted(services))})"
        provider = catalog.get(name)
        if provider is None:
            blocked.append(f"cache provider '{name}' {used_by}")
        elif version == CUSTOM_VERSION:
            # This one names its own image, which only a provider opting into
            # user-supplied images can launch. Withdrawing the opt-in leaves
            # those services with nothing to run, the same way dropping a
            # declared version does.
            if not provider.custom_version:
                blocked.append(
                    f"the custom version of cache provider '{name}' {used_by}"
                )
        elif version is None:
            # Never named one: it runs whatever the provider defaults to, so a
            # document declaring any version satisfies it and one declaring
            # none does not.
            if not provider.versions:
                blocked.append(
                    f"the default version of cache provider '{name}' {used_by}"
                )
        elif version not in (provider.versions or {}):
            blocked.append(f"version '{version}' of cache provider '{name}' {used_by}")
    if blocked:
        raise BadRequestException(
            message="Cannot remove cache providers that are currently being "
            f"used by cache services: {'; '.join(blocked)}",
        )


# The cache-provider catalog source, exposed through ``routes/ota_sources.py``.
# A document pasted in or one fetched from an address of the admin's own, the
# same two an admin configures anywhere else. What this kind has no use for is a
# schedule: nothing publishes this catalog, so the refresh round (which walks
# ``OFFICIAL_KINDS``) never reaches it, and a URL source is re-read when it is
# reloaded. ``CacheProviderSourceController`` seeds the BUILTIN row, and every
# server's refresher installs what the rows serve.
CACHE_PROVIDER_SOURCE_SPEC = SourceConfigSpec(
    source_cls=CacheProviderSource,
    normalize=normalize_cache_provider_yaml,
    builtin_name=BUILTIN_CACHE_PROVIDER_SOURCE_NAME,
    custom_name=CUSTOM_CACHE_PROVIDER_SOURCE_NAME,
    allowed_types=(SourceTypeEnum.FILE, SourceTypeEnum.URL),
    pre_write_check=_reject_taking_away_a_provider_in_use,
)
