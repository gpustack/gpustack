import httpx
import logging
import hashlib
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
from sqlalchemy.orm import selectinload
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
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
    TestProviderModelInput,
    TestProviderModelResult,
    ProviderModel,
    OpenAIConfig,
    V1_MODELS_URI,
)
from gpustack.schemas.models import CategoryEnum
from gpustack.schemas.model_routes import ModelRouteTarget
from gpustack.api.exceptions import (
    AlreadyExistsException,
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
            ModelProvider.streaming(fields=fields, fuzzy_fields=fuzzy_fields),
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

    if len(provider.api_tokens) > 1:
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
        if not clone_from:
            raise NotFoundException(
                message=f"provider {input.clone_from_id} to clone from not found"
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
        options=[selectinload(ModelProvider.model_route_targets)],
    )
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"provider {id} not found")
    assert_resource_visible(
        ctx,
        existing,
        not_found_message=f"provider {id} not found",
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
    """The ``data`` array of a model-list response.

    A path that is not a model list often answers 200 anyway -- a gateway's UI,
    an error object, a bare array -- so a body that cannot be read as one counts
    as a failed candidate rather than a 500. Both the decode and the shape raise
    ``ValueError``, leaving the caller one thing to catch.
    """
    content = response.json()
    if not isinstance(content, dict):
        raise ValueError(
            f"expected a JSON object with a data array, got {type(content).__name__}"
        )
    return content.get("data") or []


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
            return TestProviderModelResult(
                model_name=input.model_name,
                accessible=False,
                error_message=f"Provider API error: {exc.response.status_code} {exc.response.text}",
            )
        except httpx.RequestError as exc:
            raise InternalServerErrorException(
                message=f"Network error: {exc.__class__.__name__}: {exc}"
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
