from urllib.parse import urljoin
from functools import partial
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from gpustack.api.exceptions import (
    AlreadyExistsException,
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.server.deps import SessionDep, EngineDep
from gpustack.schemas.clusters import (
    CloudCredentialCreate,
    CloudCredentialListParams,
    CloudCredentialPublic,
    CloudCredentialsPublic,
    CloudCredentialUpdate,
    CloudCredential,
    ClusterProvider,
)
from gpustack.cloud_providers.common import factory
from gpustack.routes.proxy import proxy_to

router = APIRouter()


@router.get("", response_model=CloudCredentialsPublic)
async def list(
    engine: EngineDep,
    session: SessionDep,
    params: CloudCredentialListParams = Depends(),
    name: str = None,
    search: str = None,
):
    fuzzy_fields = {}
    if search:
        fuzzy_fields = {"name": search}

    fields = {"deleted_at": None}
    if name:
        fields = {"name": name}

    if params.watch:
        return StreamingResponse(
            CloudCredential.streaming(engine, fields=fields, fuzzy_fields=fuzzy_fields),
            media_type="text/event-stream",
        )

    return await CloudCredential.paginated_by_query(
        session=session,
        fields=fields,
        fuzzy_fields=fuzzy_fields,
        page=params.page,
        per_page=params.perPage,
        order_by=params.order_by,
    )


@router.get("/{id}", response_model=CloudCredentialPublic)
async def get(session: SessionDep, id: int):
    existing = await CloudCredential.one_by_id(session, id)
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"cloud credential {id} not found")

    return existing


@router.post("", response_model=CloudCredentialPublic)
async def create(session: SessionDep, input: CloudCredentialCreate):
    existing = await CloudCredential.one_by_fields(
        session,
        {"deleted_at": None, "name": input.name},
    )
    if existing:
        raise AlreadyExistsException(
            message=f"cloud credential {input.name} already exists"
        )

    try:
        return await CloudCredential.create(session, input)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to create cloud credential: {e}"
        )


@router.put("/{id}", response_model=CloudCredentialPublic)
async def update(session: SessionDep, id: int, input: CloudCredentialUpdate):
    existing = await CloudCredential.one_by_id(session, id)
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"cloud credential {id} not found")

    try:
        await CloudCredential.update(existing, session=session, source=input)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update cloud credential: {e}"
        )

    return await CloudCredential.one_by_id(session, id)


@router.delete("/{id}")
async def delete(session: SessionDep, id: int):
    existing = await CloudCredential.one_by_id(session, id)
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"cloud credential {id} not found")

    try:
        await existing.delete(session=session)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to delete cloud credential: {e}"
        )


@router.api_route("/{id}/provider-proxy/{path:path}", methods=["GET"])
async def proxy_cluster_provider_api(
    request: Request, session: SessionDep, id: int, path: str
):
    """
    To support other provider in the future, use api_route instead of get.
    """

    credential = await CloudCredential.one_by_id(session=session, id=id)
    if not credential:
        raise NotFoundException(message=f"Credential {id} not found")
    if credential.provider in [ClusterProvider.Docker, ClusterProvider.Kubernetes]:
        raise NotFoundException(message=f"Provider {credential.provider} not supported")
    provider = factory.get(credential.provider, None)
    if provider is None:
        raise NotFoundException(message=f"Provider {credential.provider} not found")
    url = urljoin(provider[0].get_api_endpoint(), path)
    if request.query_params:
        url = f"{url}?{str(request.query_params)}"
    options = {
        **(credential.options or {}),
    }
    header_modifier = partial(
        provider[0].process_header, credential.key, credential.secret, options
    )
    response = await proxy_to(request, url, header_modifier)
    if response.status_code in [401, 403, 404]:
        original_status = response.status_code
        response.status_code = 400
        response.headers.append("X-GPUStack-Original-Status", str(original_status))
    return response
