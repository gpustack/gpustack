import httpx
import logging
import hashlib
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone
from sqlalchemy.orm import selectinload
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from gpustack.schemas.model_provider import (
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
)
from gpustack.schemas.models import CategoryEnum
from gpustack.schemas.model_routes import ModelRouteTarget
from gpustack.api.exceptions import (
    AlreadyExistsException,
    InternalServerErrorException,
    NotFoundException,
    InvalidException,
)
from gpustack.server.deps import SessionDep
from openai.types import Model as OAIModel
from openai.pagination import SyncPage

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("", response_model=ModelProvidersPublic, response_model_exclude_none=True)
async def get_model_providers(
    session: SessionDep,
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

    if params.watch:
        return StreamingResponse(
            ModelProvider.streaming(fields=fields, fuzzy_fields=fuzzy_fields),
            media_type="text/event-stream",
        )

    provider_list = await ModelProvider.paginated_by_query(
        session=session,
        fields=fields,
        fuzzy_fields=fuzzy_fields,
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


@router.post("", response_model=ModelProviderPublic, response_model_exclude_none=True)
async def create_model_provider(session: SessionDep, input: ModelProviderCreate):
    existing = await ModelProvider.one_by_fields(
        session,
        {'deleted_at': None, "name": input.name},
    )
    if existing:
        raise AlreadyExistsException(message=f"provider {input.name} already exists")
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
async def get_model_provider(session: SessionDep, id: int):
    provider = await ModelProvider.one_by_id(session=session, id=id)
    if not provider:
        raise NotFoundException(message=f"provider {id} not found")
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
    "/{id}", response_model=ModelProviderPublic, response_model_exclude_none=True
)
async def update_model_provider(
    session: SessionDep, id: int, input: ModelProviderUpdate
):
    provider = await ModelProvider.one_by_id(session=session, id=id)
    if not provider:
        raise NotFoundException(message=f"provider {id} not found")
    validate_provider(input)
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
                    ModelRouteTarget.provider_model_name.in_(deleted_models)
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


@router.delete("/{id}")
async def delete_model_provider(session: SessionDep, id: int):
    existing = await ModelProvider.one_by_id(
        session=session,
        id=id,
        options=[selectinload(ModelProvider.model_route_targets)],
    )
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"provider {id} not found")
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


@router.post("/get-models")
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
    data = []
    async with httpx.AsyncClient(
        base_url=base_url,
        proxy=input.proxy_url,
        trust_env=True,
    ) as client:
        headers = {}
        if input.config.type == ModelProviderTypeEnum.CLAUDE:
            headers["X-API-Key"] = input.api_token
        else:
            headers["Authorization"] = f"Bearer {input.api_token}"
        try:
            response = await client.get(url=model_uri, headers=headers, timeout=30)
            response.raise_for_status()
            content = response.json()
            data: List[Dict[str, Any]] = content.get("data") or []
        except httpx.HTTPStatusError as exc:
            raise InvalidException(
                message=f"Failed to get models from {input.config.type}: {exc.response.status_code} {exc.response.text}"
            )
        except httpx.RequestError as exc:
            raise InternalServerErrorException(
                message=f"Network error: {exc.__class__.__name__}: {exc}"
            )
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


@router.post("/{id}/get-models")
async def get_models_from_specific_provider(
    session: SessionDep,
    id: int,
    input: ProviderModelsInput,
):
    provider = await ModelProvider.one_by_id(session=session, id=id)
    if not provider or provider.deleted_at is not None:
        raise NotFoundException(message=f"provider {id} not found")
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

    if input.config.type not in [
        ModelProviderTypeEnum.QWEN,
        ModelProviderTypeEnum.DEEPSEEK,
        ModelProviderTypeEnum.DOUBAO,
        ModelProviderTypeEnum.CLAUDE,
        ModelProviderTypeEnum.OPENAI,
    ]:
        raise InvalidException(
            message=f"provider type {input.config.type} not supported for testing model accessibility"
        )
    endpoint = input.config.get_base_url()
    prefix = ""
    completion_url = "chat/completions"
    max_output_token_dict = {"max_tokens": 16}
    if input.config.type == ModelProviderTypeEnum.DOUBAO:
        prefix = "api/v3/"
    elif input.config.type == ModelProviderTypeEnum.QWEN:
        prefix = "compatible-mode/v1/"
    elif input.config.type == ModelProviderTypeEnum.CLAUDE:
        completion_url = "v1/messages"
    data = {
        "model": input.model_name,
        "messages": [{"role": "user", "content": "Ping"}],
        **max_output_token_dict,
    }
    if input.config.type == ModelProviderTypeEnum.QWEN:
        data["enable_thinking"] = False
    async with httpx.AsyncClient(
        base_url=f"{endpoint}/{prefix}",
        proxy=input.proxy_url,
        trust_env=True,
    ) as client:
        headers = {}
        if input.config.type == ModelProviderTypeEnum.CLAUDE:
            headers["X-API-Key"] = input.api_token
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
    id: int,
    input: TestProviderModelInput,
):
    provider = await ModelProvider.one_by_id(session=session, id=id)
    if not provider or provider.deleted_at is not None:
        raise NotFoundException(message=f"provider {id} not found")
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
