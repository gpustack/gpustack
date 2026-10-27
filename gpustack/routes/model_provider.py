import asyncio
import httpx
import logging
import hashlib
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
from sqlalchemy.orm import selectinload
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from gpustack import envs
from gpustack.utils import network
from gpustack.schemas.model_provider import (
    ANTHROPIC_API_VERSION,
    MaskedAPIToken,
    ModelProvider,
    ModelProviderCreate,
    ModelProviderUpdate,
    ModelProviderPublic,
    ModelProvidersPublic,
    ModelProviderListParams,
    ProviderModelsInput,
    ModelProviderTypeEnum,
    TypesafeConfig,
    TestDecisionModelInput,
    TestProviderModelInput,
    TestProviderModelResult,
    ProviderModel,
    OpenAIConfig,
    V1_MODELS_URI,
)
from gpustack.routes.plugins.decision_service.providers import (
    DECISION_PROVIDER_TYPES,
    is_decision_config,
)
from gpustack.routes.plugins.decision_service.plugin import (
    decision_referencing_route_ids,
)
from gpustack.routes.plugins.capability_policy import CapabilityPolicy
from gpustack.schemas.models import CategoryEnum
from gpustack.schemas.model_routes import ModelRoute, ModelRouteTarget
from gpustack.api.streaming import tenant_streaming
from gpustack.api.exceptions import (
    AlreadyExistsException,
    BadRequestException,
    InternalServerErrorException,
    NotFoundException,
    InvalidException,
)
from gpustack.api.tenant import (
    assert_resource_visible,
    tenant_list_conditions,
)
from gpustack.schemas.principals import platform_principal_id
from gpustack.server.db import async_session
from gpustack.server.deps import SessionDep, TenantContextDep
from openai.types import Model as OAIModel
from openai.pagination import SyncPage

router = APIRouter()
logger = logging.getLogger(__name__)


def _is_provider_default_endpoint(config, endpoint: Optional[str]) -> bool:
    """Whether ``endpoint`` is exactly the config class's own hosted default.

    The hosted defaults (`_public_endpoint` + `_default_schema`) are constants
    this server ships, not values the caller chose, so they are outside what
    the egress gate protects against and must keep working out of the box
    once the gate is on -- an operator should not have to allowlist
    api.typesafe.ai to test a decision config that never named a custom
    endpoint. The match is on the full origin (scheme, host, port): a
    caller-supplied URL on the same host but a different scheme or port is
    not the default and stays subject to the gate.
    """
    default_host = getattr(config, "_public_endpoint", None)
    if not default_host or not endpoint:
        return False
    default_scheme = getattr(config, "_default_schema", "https") or "https"
    parsed = urlparse(endpoint)
    if parsed.scheme != default_scheme:
        return False
    if (parsed.hostname or "").lower() != default_host.lower():
        return False
    default_port = {"http": 80, "https": 443}[default_scheme]
    return parsed.port is None or parsed.port == default_port


async def _assert_provider_egress_allowed(
    endpoint: Optional[str], proxy_url: Optional[str] = None
) -> None:
    """Reject caller-supplied URLs the deployment has gated off.

    ``PROVIDER_TEST_EGRESS_ALLOWLIST`` is the gate: once non-empty it is the
    whole rule -- a base URL (and, when the request carries one, the proxy
    URL) that does not use an allowed scheme, or whose host matches no name
    entry and resolves to any address outside the CIDR entries, is refused
    before any connection is opened. A base URL that equals the config
    class's hosted default is not caller-supplied and is passed in as None
    by the callers. Errors name the hostname only: the URL itself may embed
    credentials in its userinfo, which has no place in a response.

    Resolving the host here and letting httpx resolve it again is still a
    DNS-rebinding window -- the address checked is not the address dialed.
    Closing that would mean pinning the verified address into the connection
    (at the cost of TLS SNI/certificate handling), which this gate does not
    attempt; it raises the bar for probing, it is not an airlock. Proxies
    configured through the environment are the operator's own and are not
    checked.
    """
    allowlist = envs.PROVIDER_TEST_EGRESS_ALLOWLIST
    if not allowlist:
        return
    targets = [
        ("provider base URL", endpoint, ("http", "https")),
        ("proxy URL", proxy_url, ("http", "https", "socks5", "socks5h")),
    ]
    for kind, url, schemes in targets:
        if url is None:
            continue
        parsed = urlparse(url)
        if parsed.scheme not in schemes or not parsed.hostname:
            raise InvalidException(
                message=(f"{kind} is empty or not a URL of an allowed scheme")
            )
        hostname = parsed.hostname
        # httpx normalizes the host through IDNA before dialing (faß.example
        # becomes xn--fa-hia.example), while getaddrinfo would encode it
        # differently (fass.example) -- validate the name the client will
        # actually dial, so the two never disagree. The raw host comes from
        # re-parsing the full URL: rebuilding it from parsed.hostname would
        # drop the brackets of an IPv6 literal and fail to parse.
        hostname = httpx.URL(url).raw_host.decode("ascii")
        disallowed = await asyncio.to_thread(
            network.egress_disallowed_address, hostname, allowlist
        )
        if disallowed == network.EGRESS_HOST_UNRESOLVED:
            raise InvalidException(
                message=(
                    f"{kind} host {hostname!r} could not be resolved for verification "
                    "against GPUSTACK_PROVIDER_TEST_EGRESS_ALLOWLIST"
                )
            )
        if disallowed is not None:
            raise InvalidException(
                message=(
                    f"{kind} host {hostname!r} resolves to {disallowed}, which is "
                    "outside GPUSTACK_PROVIDER_TEST_EGRESS_ALLOWLIST"
                )
            )


@router.get("", response_model=ModelProvidersPublic, response_model_exclude_none=True)
async def get_model_providers(
    ctx: TenantContextDep,
    params: ModelProviderListParams = Depends(),
    name: str = None,
    search: str = None,
):
    fuzzy_fields = {}
    if search:
        fuzzy_fields = {"name": search}

    fields = {'deleted_at': None}
    if name:
        fields = {"name": name}

    extra_conditions = list(tenant_list_conditions(ctx, ModelProvider))

    if params.watch:
        return StreamingResponse(
            tenant_streaming(
                ModelProvider,
                ctx,
                fields=fields,
                fuzzy_fields=fuzzy_fields,
            ),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        provider_list = await ModelProvider.paginated_by_query(
            session=session,
            fields=fields,
            fuzzy_fields=fuzzy_fields,
            extra_conditions=extra_conditions,
            page=params.page,
            per_page=params.perPage,
            order_by=params.order_by,
        )
        provider_list.items = [
            ModelProvider._convert_to_public_class(provider)
            for provider in provider_list.items
        ]

    return provider_list


def validate_provider(provider: Union[ModelProviderCreate, ModelProviderUpdate]):
    if provider.config is not None and len(provider.config.model_extra or {}) > 0:
        raise InvalidException(
            message=f"fields {', '.join(provider.config.model_extra.keys())} are not allowed in {provider.config.type.value} config"
        )
    try:
        provider.config.check_required_fields()
    except ValueError as e:
        raise InvalidException(message=f"{e}")

    # Decision services fail over their keys inside the wasm plugin, not
    # through ai-proxy's failover config, so they never need an llm model
    # for the multi-token requirement.
    if len(provider.api_tokens) > 1 and not is_decision_config(provider.config):
        llm_model = next(
            (model for model in provider.models or [] if model.category == "llm"),
            None,
        )
        if not llm_model:
            raise InvalidException(
                message="At least one llm model is required when api_tokens has more than 1 token for failover"
            )
    if len(provider.models or []) == 0:
        raise InvalidException(message="At least one model is required for a provider")

    if isinstance(provider.config, OpenAIConfig) and provider.config.openaiCustomUrl:
        parsed_url = urlparse(provider.config.openaiCustomUrl.rstrip("/"))
        if parsed_url.path == "":
            raise InvalidException(
                message=f"openaiCustomUrl {provider.config.openaiCustomUrl} is invalid, it must include a path, e.g. http://my-openai.com/v1"
            )


def parse_api_tokens(
    existing_tokens: List[str], api_tokens: List[MaskedAPIToken]
) -> List[str]:
    target_tokens = []
    hashed_token_dict = {
        hashlib.sha256(token.encode()).hexdigest(): token for token in existing_tokens
    }
    for index, api_token in enumerate(api_tokens):
        token_value = api_token.input
        if api_token.hash is not None:
            token_value = hashed_token_dict.get(api_token.hash)
        if not token_value or not token_value.strip():
            raise InvalidException(
                message=f"API token at index {index} is invalid, empty, or does not match any existing token"
            )
        target_tokens.append(token_value)
    return target_tokens


@router.post(
    "",
    response_model=ModelProviderPublic,
    response_model_exclude_none=True,
)
async def create_model_provider(
    session: SessionDep, ctx: TenantContextDep, input: ModelProviderCreate
):
    # Every provider belongs to one Org. Admin in "All" mode (no
    # current principal) falls back to the platform Org so the row has
    # a concrete owner — provider does not carry a NULL-owner Global
    # notion (unlike Instance Template / Inference Backend).
    target_org_id = ctx.current_principal_id or platform_principal_id()

    # Provider names are unique within their owning Org.
    existing = await ModelProvider.one_by_fields(
        session,
        {
            'deleted_at': None,
            "name": input.name,
            "owner_principal_id": target_org_id,
        },
    )
    if existing:
        raise AlreadyExistsException(
            message=f"Model provider with name '{input.name}' already exists."
        )
    validate_provider(input)
    input_dict = input.model_dump(exclude={"api_tokens", "clone_from_id"})
    existing_tokens = []
    if input.clone_from_id is not None:
        clone_from = await ModelProvider.one_by_id(
            session=session,
            id=input.clone_from_id,
        )
        if not clone_from or clone_from.deleted_at is not None:
            raise NotFoundException(
                message=f"provider {input.clone_from_id} to clone from not found"
            )
        assert_resource_visible(
            ctx,
            clone_from,
            not_found_message=f"provider {input.clone_from_id} to clone from not found",
        )
        existing_tokens = clone_from.api_tokens or []
    input_dict["api_tokens"] = parse_api_tokens(
        existing_tokens=existing_tokens, api_tokens=input.api_tokens
    )
    input_dict["owner_principal_id"] = target_org_id
    try:
        created = await ModelProvider.create(session=session, source=input_dict)
        return ModelProvider._convert_to_public_class(created)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to create provider {input.name}: {e}"
        )


@router.get(
    "/{id}", response_model=ModelProviderPublic, response_model_exclude_none=True
)
async def get_model_provider(session: SessionDep, ctx: TenantContextDep, id: int):
    provider = await ModelProvider.one_by_id(session=session, id=id)
    assert_resource_visible(
        ctx,
        provider,
        not_found_message=f"provider {id} not found",
    )
    return ModelProvider._convert_to_public_class(provider)


def deleted_model_names(
    existing_models: List[ProviderModel],
    input_models: List[ProviderModel],
) -> List[str]:
    input_model_names = {model.name for model in input_models}
    deleted_names = [
        model.name for model in existing_models if model.name not in input_model_names
    ]
    return deleted_names


@router.put(
    "/{id}",
    response_model=ModelProviderPublic,
    response_model_exclude_none=True,
)
async def update_model_provider(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    input: ModelProviderUpdate,
):
    provider = await ModelProvider.one_by_id(session=session, id=id)
    assert_resource_visible(
        ctx,
        provider,
        not_found_message=f"provider {id} not found",
    )
    validate_provider(input)
    # Rename check: if the caller is changing ``name``, make sure the
    # new name doesn't collide with another provider under the same
    # owner. The composite UNIQUE on the table would catch this at
    # commit time, but we surface a friendly 409 first.
    if input.name != provider.name:
        clash = await ModelProvider.one_by_fields(
            session,
            {
                'deleted_at': None,
                "name": input.name,
                "owner_principal_id": provider.owner_principal_id,
            },
        )
        if clash and clash.id != provider.id:
            raise AlreadyExistsException(
                message=f"Model provider with name '{input.name}' already exists."
            )
    deleted_models = deleted_model_names(provider.models or [], input.models or [])
    # An ordinary provider already serving inference targets cannot become a
    # decision service: the targets would survive the type flip (their
    # validation only runs on target writes) while the provider drops out
    # of the ai-proxy catalogue -- routes pointing at nothing.
    if is_decision_config(input.config) and not is_decision_config(provider.config):
        existing_targets = await ModelRouteTarget.all_by_field(
            session, "provider_id", id
        )
        if existing_targets:
            raise InvalidException(
                message=(
                    f"provider {provider.name} is referenced by "
                    f"{len(existing_targets)} inference route target(s); it "
                    "cannot change to a decision-service type "
                    "(gpustack-lb-typesafe) while those targets exist. "
                    "Remove the targets first."
                )
            )
    try:
        input_dict = input.model_dump(exclude={"api_tokens"})
        if input.api_tokens is not None:
            input_dict["api_tokens"] = parse_api_tokens(
                existing_tokens=provider.api_tokens or [],
                api_tokens=input.api_tokens,
            )
        await provider.update(
            session=session, source=input_dict, auto_commit=len(deleted_models) == 0
        )
        if len(deleted_models) > 0:
            routes = await ModelRouteTarget.all_by_fields(
                session=session,
                fields={"provider_id": id},
                extra_conditions=[
                    ModelRouteTarget.overridden_model_name.in_(deleted_models)
                ],
            )
            for route in routes:
                await route.delete(session=session, auto_commit=False)
            await session.commit()
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update provider {id}: {e}"
        )
    updated_provider = await ModelProvider.one_by_id(session=session, id=id)
    return ModelProvider._convert_to_public_class(updated_provider)


@router.delete(
    "/{id}",
)
async def delete_model_provider(session: SessionDep, ctx: TenantContextDep, id: int):
    existing = await ModelProvider.one_by_id(
        session=session,
        id=id,
        for_update=True,
        options=[selectinload(ModelProvider.model_route_targets)],
    )
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"provider {id} not found")
    assert_resource_visible(
        ctx,
        existing,
        not_found_message=f"provider {id} not found",
    )
    # A decision-service provider pinned by a route's decision policy would
    # leave the route rendering a dangling providerId after deletion. The
    # row lock above serializes this check with the route-write path:
    # `_validate_provider_id` takes the same lock on the provider row, so a
    # concurrent route write cannot validate and store a new providerId
    # between the reference scan and the delete commit.
    if is_decision_config(existing.config):
        policies = await CapabilityPolicy.all_by_fields(
            # deleted_at: None matches every other policy read
            # (`policy_for_route`, `capability_sections`): a soft-deleted
            # row keeps its old providerId but no longer pins anything.
            session,
            fields={"capability": "decision-service", "deleted_at": None},
        )
        route_ids = decision_referencing_route_ids(policies, id)
        if route_ids:
            # CapabilityPolicy rows carry no owner column, so the route
            # lookup itself is scoped to the caller: a stale or foreign
            # policy can neither leak another Org's route names nor block
            # this delete. References the caller cannot see render inert
            # (the plugin strips an unresolvable providerId), so deleting
            # past them is safe.
            routes = await ModelRoute.all_by_fields(
                session,
                extra_conditions=[
                    ModelRoute.id.in_(route_ids),
                    *tenant_list_conditions(ctx, ModelRoute),
                ],
            )
            names = sorted(route.name for route in routes)
            if names:
                raise BadRequestException(
                    message=(
                        "Decision service provider is in use by route(s): "
                        + ", ".join(names)
                    )
                )
    try:
        await existing.delete(session=session)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to delete provider {id}: {e}"
        )


def get_model_name(model: Dict[str, Any]) -> Optional[str]:
    return model.get("id", model.get("name", None))


categories_to_infer = [
    CategoryEnum.IMAGE,
    CategoryEnum.EMBEDDING,
    CategoryEnum.RERANKER,
]

category_values = {e.value for e in CategoryEnum}


def determine_model_category(
    provider_type: ModelProviderTypeEnum,
    model: Dict[str, Any],
) -> List[str]:
    if provider_type in DECISION_PROVIDER_TYPES:
        # Decision-engine versions (jev-latest, ...), not servable models:
        # the one category the UI filters the decision-model dropdowns by.
        return ["decision"]
    if provider_type == ModelProviderTypeEnum.DOUBAO:
        domain: str = model.get("domain", "").lower()
        if domain in category_values:
            return [domain]
    model_id: str = get_model_name(model) or ""
    model_name = model_id.rsplit("/", 1)[-1]

    for category_enum in categories_to_infer:
        if category_enum.value in model_name:
            return [category_enum.value]

    return [CategoryEnum.LLM.value]


class CustomOAIModel(OAIModel):
    categories: Optional[List[str]] = None


def _model_list(response: httpx.Response) -> List[Dict[str, Any]]:
    """The model array of a model-list response.

    A path that is not a model list often answers 200 anyway -- a gateway's UI,
    an error object, a bare array -- so a body that cannot be read as one counts
    as a failed candidate rather than a 500. Both the decode and the shape raise
    ``ValueError``, leaving the caller one thing to catch.

    OpenAI-compatible servers answer ``{"data": [...]}``; the Jev decision
    services answer ``{"models": [...]}`` (jevcompat SPEC) -- both are model
    lists, so both are accepted. Item shape differences do not matter here:
    ``get_model_name`` already reads ``id`` or ``name``.
    """
    content = response.json()
    if not isinstance(content, dict):
        raise ValueError(
            "expected a JSON object with a data or models array, got "
            f"{type(content).__name__}"
        )
    data = content.get("data")
    if isinstance(data, list):
        # An empty data array is an empty model list, not a missing key --
        # the models fallback must not turn it into someone else's list.
        return data
    models = content.get("models")
    if isinstance(models, list):
        return models
    raise ValueError(
        "expected a data or models array, got "
        f"{type(data if data is not None else models).__name__}"
    )


async def _first_model_list(
    client: httpx.AsyncClient,
    uris: List[str],
    headers: Dict[str, str],
    provider_type: ModelProviderTypeEnum,
) -> List[Dict[str, Any]]:
    """The model list from the first candidate path that has one.

    The first failure is the one reported, because that is the path the provider
    config points at -- any candidate after it is a guess at a different way of
    mounting the same API. A transport failure ends the walk instead of
    contributing to it: that is about the host, and another path on the same host
    cannot do better.
    """
    failure: Optional[Exception] = None
    failure_detail: Optional[str] = None
    for uri in uris:
        try:
            response = await client.get(url=uri, headers=headers, timeout=30)
            response.raise_for_status()
            return _model_list(response)
        except (httpx.HTTPStatusError, ValueError) as exc:
            detail = (
                f"{exc.response.status_code} {exc.response.text}"
                if isinstance(exc, httpx.HTTPStatusError)
                else f"{exc}"
            )
            if failure is None:
                failure, failure_detail = exc, detail
            logger.debug(
                f"Failed to get models from {provider_type} at {uri}: {detail}"
            )
        except httpx.RequestError as exc:
            raise InternalServerErrorException(
                message=f"Network error: {exc.__class__.__name__}: {exc}"
            ) from exc
    raise InvalidException(
        message=f"Failed to get models from {provider_type}: {failure_detail}"
    ) from failure


@router.post(
    "/get-models",
)
async def get_models_from_provider(
    input: ProviderModelsInput,
):
    if input.api_token is None or input.config is None:
        raise InvalidException(
            message="api_token and config are required to fetch models from provider"
        )

    result = SyncPage[CustomOAIModel](data=[], object="list")
    try:
        input.config.check_required_fields()
    except ValueError as e:
        logger.error(f"{e}")
        raise InvalidException(message=f"{e}")
    base_url, model_uri = input.config.get_model_url()
    if not base_url or not model_uri:
        logger.warning(
            f"provider type {input.config.type} not supported for fetching models"
        )
        return result
    await _assert_provider_egress_allowed(
        None if _is_provider_default_endpoint(input.config, base_url) else base_url,
        input.proxy_url,
    )

    model_uris = [model_uri]
    async with httpx.AsyncClient(
        base_url=base_url,
        proxy=input.proxy_url,
        trust_env=True,
    ) as client:
        headers = {}
        if input.config.type == ModelProviderTypeEnum.CLAUDE:
            headers["X-API-Key"] = input.api_token
            headers["anthropic-version"] = (
                getattr(input.config, "claudeVersion", None) or ANTHROPIC_API_VERSION
            )
            # An Anthropic-compatible endpoint behind a base path may mount its
            # whole API under that prefix, or only /v1/messages while the model
            # list stays at the root -- both shapes exist in the wild and the
            # config cannot tell them apart. The path the config derives is tried
            # first and the bare one only as a fallback, so the common case (they
            # are the same path, or the first one answers) is one request.
            if model_uri != V1_MODELS_URI:
                model_uris.append(V1_MODELS_URI)
        else:
            headers["Authorization"] = f"Bearer {input.api_token}"
        data = await _first_model_list(client, model_uris, headers, input.config.type)
    fallback_created = int(datetime.now(timezone.utc).timestamp())
    for item in data:
        if input.config.type == ModelProviderTypeEnum.DOUBAO:
            status = item.get("status", None)
            if status is not None:
                continue
        model_id = get_model_name(item)
        if not model_id:
            continue
        categories = determine_model_category(input.config.type, item)
        model = CustomOAIModel(
            id=model_id,
            created=item.get("created") or fallback_created,
            object=item.get("object") or "model",
            owned_by=item.get("owned_by") or input.config.type.value,
            categories=categories,
        )
        result.data.append(model)
    return result


@router.post(
    "/{id}/get-models",
)
async def get_models_from_specific_provider(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    input: ProviderModelsInput,
):
    provider = await ModelProvider.one_by_id(session=session, id=id)
    if not provider or provider.deleted_at is not None:
        raise NotFoundException(message=f"provider {id} not found")
    assert_resource_visible(
        ctx,
        provider,
        not_found_message=f"provider {id} not found",
    )
    if provider.api_tokens is None or len(provider.api_tokens) == 0:
        raise InvalidException(
            message=f"provider {provider.name} id: {id} has no API tokens configured"
        )
    proxy_url = (
        input.proxy_url if 'proxy_url' in input.model_fields_set else provider.proxy_url
    )
    return await get_models_from_provider(
        ProviderModelsInput(
            api_token=input.api_token or provider.api_tokens[0],
            config=input.config or provider.config,
            proxy_url=proxy_url,
        )
    )


def _get_model_output_token_dict(model_name: str) -> Dict[str, Any]:
    name = model_name.lower().rsplit("/", 1)[-1]
    max_token_key = (
        "max_completion_tokens"
        if name.startswith(("gpt-5", "o1", "o3", "o4"))
        else "max_tokens"
    )
    return {max_token_key: 16}


@router.post(
    "/test-model",
    response_model=TestProviderModelResult,
    response_model_exclude_none=True,
)
async def try_model_with_provider(
    input: TestProviderModelInput,
):
    if input.api_token is None or input.config is None:
        raise InvalidException(
            message="api_token and config are required to fetch models from provider"
        )

    endpoint, completion_url = input.config.get_chat_url()
    if not endpoint or not completion_url:
        raise InvalidException(
            message=f"provider type {input.config.type} does not support testing model accessibility"
        )
    await _assert_provider_egress_allowed(
        None if _is_provider_default_endpoint(input.config, endpoint) else endpoint,
        input.proxy_url,
    )
    max_output_token_dict = _get_model_output_token_dict(input.model_name)
    data = {
        "model": input.model_name,
        "messages": [{"role": "user", "content": "Ping"}],
        **max_output_token_dict,
    }
    async with httpx.AsyncClient(
        base_url=f"{endpoint}",
        proxy=input.proxy_url,
        trust_env=True,
    ) as client:
        headers = {}
        if input.config.type == ModelProviderTypeEnum.CLAUDE:
            headers["X-API-Key"] = input.api_token
            headers["anthropic-version"] = (
                getattr(input.config, "claudeVersion", None) or ANTHROPIC_API_VERSION
            )
        else:
            headers["Authorization"] = f"Bearer {input.api_token}"
        for attempt in range(2):
            try:
                response = await client.post(
                    url=completion_url, json=data, headers=headers, timeout=60
                )
                response.raise_for_status()
                return TestProviderModelResult(
                    model_name=input.model_name,
                    accessible=True,
                )
            except httpx.HTTPStatusError as exc:
                if (
                    attempt == 0
                    and _is_thinking_restricted_qwen_rejection(input.config.type, exc)
                    and "enable_thinking" not in data
                ):
                    # A thinking-only Qwen model (e.g. qwen3.7-max) rejects the
                    # parameterless ping -- DashScope defaults enable_thinking to
                    # False while the model requires True. Retry once with it
                    # explicit instead of failing the provider test.
                    data = {**data, "enable_thinking": True}
                    continue
                return TestProviderModelResult(
                    model_name=input.model_name,
                    accessible=False,
                    error_message=f"Provider API error: {exc.response.status_code} {exc.response.text}",
                )
            except httpx.RequestError as exc:
                raise InternalServerErrorException(
                    message=f"Network error: {exc.__class__.__name__}: {exc}"
                )


def _is_thinking_restricted_qwen_rejection(
    provider_type: ModelProviderTypeEnum, exc: httpx.HTTPStatusError
) -> bool:
    """Whether a failed ping should be retried with ``enable_thinking`` set.

    Only the one rejection DashScope words as a thinking-mode restriction
    qualifies, and only for the Qwen provider -- anything else (bad token,
    quota, a non-thinking model) answers the same on a retry, so the first
    error is reported as-is.
    """
    return (
        provider_type == ModelProviderTypeEnum.QWEN
        and exc.response.status_code == 400
        and "enable_thinking" in exc.response.text
    )


# The decision-service ping question: two synthetic candidates, so the
# service exercises the same model_selection path a route's request would
# take (criteria < 2 would be skipped by the plugin as not worth an opinion,
# and the same shape is what the service itself reasons over).
_DECISION_TEST_QUESTION = {
    "type": "choice",
    "instructions": (
        "This is a connectivity test sent by GPUStack. Pick either candidate."
    ),
    "criteria": {
        "candidate-a": "A synthetic candidate used only for testing.",
        "candidate-b": "Another synthetic candidate used only for testing.",
    },
}


async def _try_decision_model(
    config: TypesafeConfig,
    api_token: Optional[str],
    model_name: Optional[str],
    proxy_url: Optional[str],
) -> TestProviderModelResult:
    """Ping a Jev decision service through its real decision path.

    POSTs the service's own ``/v1/systemone`` endpoint a minimal
    model_selection question — the same call the gateway plugin makes per
    request — so a pass certifies endpoint, token and the decision path at
    once, strictly more than the ``/v1/models`` fetch the provider form
    uses. Any 2xx counts as a pass: the verdict body is the plugin's to
    consume, not ours. The request shape is the jevcompat wire contract:
    ``model`` (the decision-engine alias, omitted when neither the caller
    nor the provider config names one), ``state``, and the question under
    ``questions.model_selection``. The endpoint is the provider's custom
    base url or the TypeSafe hosted default.
    """
    endpoint, decision_url = config.get_chat_url()
    await _assert_provider_egress_allowed(
        None if _is_provider_default_endpoint(config, endpoint) else endpoint,
        proxy_url,
    )
    alias = model_name or config.model
    data: Dict[str, Any] = {
        "state": "GPUStack decision-service connectivity test.",
        "questions": {"model_selection": _DECISION_TEST_QUESTION},
    }
    if alias:
        data["model"] = alias
    headers = {}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    # Never logs the token: the body is what a 400 "Invalid request" comes
    # down to, and the URL carries no secret either.
    logger.info(
        "testing decision service %s: POST %s body=%s",
        config.type.value,
        decision_url,
        data,
    )
    async with httpx.AsyncClient(
        base_url=f"{endpoint}",
        proxy=proxy_url,
        trust_env=True,
    ) as client:
        try:
            response = await client.post(
                url=decision_url, json=data, headers=headers, timeout=30
            )
            response.raise_for_status()
            # Reachable is not the same as usable: a proxy's HTML error
            # page answers 200 on some deployments, and a JSON body that is
            # an error envelope is a failed decision, not a verdict. A
            # JSON object without an error key is the minimum shape every
            # jevcompat verdict shares.
            try:
                verdict = response.json()
            except ValueError:
                return TestProviderModelResult(
                    model_name=model_name or config.model or "",
                    accessible=False,
                    error_message=(
                        "Decision service answered 2xx with a non-JSON "
                        f"body: {response.text[:200]!r}"
                    ),
                )
            if not isinstance(verdict, dict):
                return TestProviderModelResult(
                    model_name=model_name or config.model or "",
                    accessible=False,
                    error_message=(
                        "Decision service answered 2xx without a JSON "
                        f"object: {str(verdict)[:200]}"
                    ),
                )
            if "error" in verdict:
                return TestProviderModelResult(
                    model_name=model_name or config.model or "",
                    accessible=False,
                    error_message=(
                        "Decision service answered 2xx with an error "
                        f"envelope: {str(verdict)[:200]}"
                    ),
                )
            # The official ChoiceAnswer contract nests the verdict under
            # answers.<question> with choice, confidence and probabilities
            # all required -- anything else is reachable but not usable.
            answers = verdict.get("answers")
            answer = (
                answers.get("model_selection") if isinstance(answers, dict) else None
            )
            if not isinstance(answer, dict):
                return TestProviderModelResult(
                    model_name=model_name or config.model or "",
                    accessible=False,
                    error_message=(
                        "Decision service answered 2xx without a "
                        "model_selection answer: " + str(verdict)[:200]
                    ),
                )
            choice = answer.get("choice")
            if not isinstance(choice, str) or not choice.strip():
                return TestProviderModelResult(
                    model_name=model_name or config.model or "",
                    accessible=False,
                    error_message=(
                        "Decision service answered 2xx without a usable "
                        f"verdict (no choice): {str(verdict)[:200]}"
                    ),
                )
            confidence = answer.get("confidence")
            probabilities = answer.get("probabilities")
            if (
                not isinstance(confidence, (int, float))
                or not isinstance(probabilities, dict)
                or not probabilities
            ):
                return TestProviderModelResult(
                    model_name=model_name or config.model or "",
                    accessible=False,
                    error_message=(
                        "Decision service verdict is missing required "
                        "confidence/probabilities: " + str(verdict)[:200]
                    ),
                )
            return TestProviderModelResult(
                model_name=model_name or config.model or "",
                accessible=True,
            )
        except httpx.HTTPStatusError as exc:
            return TestProviderModelResult(
                model_name=model_name or config.model or "",
                accessible=False,
                # truncated: an HTML error page from a proxy in front of
                # the service can be arbitrarily large
                error_message=(
                    "Decision service error: "
                    f"{exc.response.status_code} {exc.response.text[:500]}"
                ),
            )
        except httpx.RequestError as exc:
            raise InternalServerErrorException(
                message=f"Network error: {exc.__class__.__name__}: {exc}"
            )


@router.post(
    "/test-decision-model",
    response_model=TestProviderModelResult,
    response_model_exclude_none=True,
)
async def try_decision_model_with_provider(
    input: TestDecisionModelInput,
):
    if input.config is None:
        raise InvalidException(message="config is required to test a decision service")
    if not is_decision_config(input.config):
        raise InvalidException(
            message=(
                f"provider type {input.config.type} is not a decision service "
                "(gpustack-lb-typesafe type)"
            )
        )
    return await _try_decision_model(
        input.config, input.api_token, input.model_name, input.proxy_url
    )


@router.post(
    "/{id}/test-decision-model",
    response_model=TestProviderModelResult,
    response_model_exclude_none=True,
)
async def try_decision_model_with_specific_provider(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    input: TestDecisionModelInput,
):
    provider = await ModelProvider.one_by_id(session=session, id=id)
    if not provider or provider.deleted_at is not None:
        raise NotFoundException(message=f"provider {id} not found")
    assert_resource_visible(
        ctx,
        provider,
        not_found_message=f"provider {id} not found",
    )
    if not is_decision_config(provider.config):
        raise InvalidException(
            message=(
                f"provider {provider.name} is not a decision service "
                "(gpustack-lb-typesafe type)"
            )
        )
    config = provider.config
    if input.config is not None:
        # an override that is not a decision config is a caller error, not
        # something to silently swap back for the stored config
        if not is_decision_config(input.config):
            raise InvalidException(
                message=(
                    f"provider type {input.config.type} is not a decision "
                    "service (gpustack-lb-typesafe type)"
                )
            )
        config = input.config
    return await _try_decision_model(
        config,
        (
            input.api_token
            if input.api_token is not None
            else (provider.api_tokens[0] if provider.api_tokens else None)
        ),
        input.model_name,
        (
            input.proxy_url
            if "proxy_url" in input.model_fields_set
            else provider.proxy_url
        ),
    )


@router.post(
    "/{id}/test-model",
    response_model=TestProviderModelResult,
    response_model_exclude_none=True,
)
async def try_model_with_specific_provider(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    input: TestProviderModelInput,
):
    provider = await ModelProvider.one_by_id(session=session, id=id)
    if not provider or provider.deleted_at is not None:
        raise NotFoundException(message=f"provider {id} not found")
    assert_resource_visible(
        ctx,
        provider,
        not_found_message=f"provider {id} not found",
    )
    if provider.api_tokens is None or len(provider.api_tokens) == 0:
        raise InvalidException(
            message=f"provider {provider.name} id: {id} has no API tokens configured"
        )
    proxy_url = (
        input.proxy_url if 'proxy_url' in input.model_fields_set else provider.proxy_url
    )
    return await try_model_with_provider(
        TestProviderModelInput(
            api_token=input.api_token or provider.api_tokens[0],
            config=input.config or provider.config,
            proxy_url=proxy_url,
            model_name=input.model_name,
        )
    )
