import httpx
from datetime import datetime, timezone
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from gpustack.schemas.model_provider import (
    ModelProvider,
    ModelProviderCreate,
    ModelProviderUpdate,
    ModelProviderPublic,
    ModelProvidersPublic,
    ModelProviderListParams,
    ProviderModelsInput,
    ModelProviderTypeEnum,
    TestModelForExistingProviderInput,
    TestProviderModelInput,
    TestProviderModelResult,
)
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

    return await ModelProvider.paginated_by_query(
        session=session,
        fields=fields,
        fuzzy_fields=fuzzy_fields,
        page=params.page,
        per_page=params.perPage,
        order_by=params.order_by,
    )


@router.post("", response_model=ModelProviderPublic, response_model_exclude_none=True)
async def create_model_provider(session: SessionDep, input: ModelProviderCreate):
    existing = await ModelProvider.one_by_fields(
        session,
        {'deleted_at': None, "name": input.name},
    )
    if existing:
        raise AlreadyExistsException(message=f"provider {input.name} already exists")
    if input.config is not None and len(input.config.model_extra or {}) > 0:
        raise InvalidException(
            message=f"fields {', '.join(input.config.model_extra.keys())} are not allowed in {input.config.type.value} config"
        )
    try:
        return await ModelProvider.create(session=session, source=input)
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
    return provider


@router.put(
    "/{id}", response_model=ModelProviderPublic, response_model_exclude_none=True
)
async def update_model_provider(
    session: SessionDep, id: int, input: ModelProviderUpdate
):
    provider = await ModelProvider.one_by_id(session=session, id=id)
    if not provider:
        raise NotFoundException(message=f"provider {id} not found")
    try:
        await provider.update(session=session, source=input)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update provider {id}: {e}"
        )
    return await ModelProvider.one_by_id(session=session, id=id)


@router.delete("/{id}")
async def delete_model_provider(session: SessionDep, id: int):
    existing = await ModelProvider.one_by_id(session=session, id=id)
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"provider {id} not found")
    try:
        await existing.delete(session=session)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to delete provider {id}: {e}"
        )


@router.post("/get-models")
async def get_models_from_provider(
    input: ProviderModelsInput,
):
    if input.config.type not in [
        ModelProviderTypeEnum.QWEN,
        ModelProviderTypeEnum.DOUBAO,
        ModelProviderTypeEnum.DEEPSEEK,
        ModelProviderTypeEnum.OPENAI,
        ModelProviderTypeEnum.CLAUDE,
    ]:
        raise InvalidException(
            message=f"provider type {input.config.type} not supported for fetching models"
        )
    endpoint = input.config.get_base_url()
    prefix = ""
    if input.config.type == ModelProviderTypeEnum.DOUBAO:
        prefix = "api/v3/"
    elif input.config.type == ModelProviderTypeEnum.QWEN:
        prefix = "compatible-mode/v1/"
    data = None
    async with httpx.AsyncClient(base_url=f"{endpoint}/{prefix}") as client:
        headers = {}
        if input.config.type == ModelProviderTypeEnum.CLAUDE:
            headers["X-API-Key"] = input.api_token
        else:
            headers["Authorization"] = f"Bearer {input.api_token}"
        try:
            response = await client.get(url="models", headers=headers, timeout=30)
            response.raise_for_status()
            content = response.json()
            data = content.get("data") or []
        except httpx.HTTPStatusError as exc:
            raise InvalidException(
                message=f"Failed to get models from {input.config.type}: {exc.response.status_code} {exc.response.text}"
            )
        except httpx.RequestError as exc:
            raise InternalServerErrorException(
                message=f"Network error: {exc.__class__.__name__}: {exc}"
            )
    fallback_created = int(datetime.now(timezone.utc).timestamp())
    result = SyncPage[OAIModel](data=[], object="list")
    for item in data:
        if input.config.type == ModelProviderTypeEnum.DOUBAO:
            status = item.get("status", None)
            if status is not None:
                continue
        model = OAIModel(
            id=item.get("id"),
            created=item.get("created") or fallback_created,
            object=item.get("object") or "model",
            owned_by=item.get("owned_by") or input.config.type.value,
        )
        result.data.append(model)
    return result


@router.get("/{id}/models")
async def get_models_from_specific_provider(
    session: SessionDep,
    id: int,
):
    provider = await ModelProvider.one_by_id(session=session, id=id)
    if not provider or provider.deleted_at is not None:
        raise NotFoundException(message=f"provider {id} not found")
    if provider.api_tokens is None or len(provider.api_tokens) == 0:
        raise InvalidException(
            message=f"provider {provider.name} id: {id} has no API tokens configured"
        )
    return await get_models_from_provider(
        ProviderModelsInput(
            api_token=provider.api_tokens[0],
            config=provider.config,
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
    async with httpx.AsyncClient(base_url=f"{endpoint}/{prefix}") as client:
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
    input: TestModelForExistingProviderInput,
):
    provider = await ModelProvider.one_by_id(session=session, id=id)
    if not provider or provider.deleted_at is not None:
        raise NotFoundException(message=f"provider {id} not found")
    if provider.api_tokens is None or len(provider.api_tokens) == 0:
        raise InvalidException(
            message=f"provider {provider.name} id: {id} has no API tokens configured"
        )
    return await try_model_with_provider(
        TestProviderModelInput(
            api_token=provider.api_tokens[0],
            config=provider.config,
            model_name=input.model_name,
        )
    )
