from urllib.parse import urljoin, urlparse
from functools import partial
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from gpustack.api.exceptions import (
    AlreadyExistsException,
    BadRequestException,
    InternalServerErrorException,
    NotFoundException,
)
from gpustack.api.tenant import (
    assert_org_owned_writable,
    assert_resource_visible,
    tenant_list_conditions,
    validate_owner_principal,
)
from gpustack.server.db import async_session
from gpustack.server.deps import SessionDep, TenantContextDep
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
from gpustack.schemas.principals import platform_principal_id

router = APIRouter()


@router.get("", response_model=CloudCredentialsPublic)
async def list(
    ctx: TenantContextDep,
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
            CloudCredential.streaming(fields=fields, fuzzy_fields=fuzzy_fields),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        extra_conditions = tenant_list_conditions(ctx, CloudCredential)
        return await CloudCredential.paginated_by_query(
            session=session,
            fields=fields,
            fuzzy_fields=fuzzy_fields,
            extra_conditions=extra_conditions,
            page=params.page,
            per_page=params.perPage,
            order_by=params.order_by,
        )


@router.get("/{id}", response_model=CloudCredentialPublic)
async def get(session: SessionDep, ctx: TenantContextDep, id: int):
    existing = await CloudCredential.one_by_id(session, id)
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"cloud credential {id} not found")
    assert_resource_visible(
        ctx,
        existing,
        not_found_message=f"cloud credential {id} not found",
    )
    return existing


@router.post("", response_model=CloudCredentialPublic)
async def create(
    session: SessionDep, ctx: TenantContextDep, input: CloudCredentialCreate
):
    # Mirror cluster-create: every credential has an owner Org. Fill in
    # ctx.current_principal_id, or PLATFORM_ORG for admin in "All" mode.
    if input.owner_principal_id is None:
        input.owner_principal_id = ctx.current_principal_id or platform_principal_id()
    validate_owner_principal(
        input.owner_principal_id, ctx, resource_label="cloud credential"
    )
    # Names are unique within their owning Org.
    existing = await CloudCredential.one_by_fields(
        session,
        {
            "deleted_at": None,
            "name": input.name,
            "owner_principal_id": input.owner_principal_id,
        },
    )
    if existing:
        raise AlreadyExistsException(
            message=f"Cloud credential with name '{input.name}' already exists."
        )

    try:
        return await CloudCredential.create(session, input)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to create cloud credential: {e}"
        )


@router.put("/{id}", response_model=CloudCredentialPublic)
async def update(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    input: CloudCredentialUpdate,
):
    existing = await CloudCredential.one_by_id(session, id)
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"cloud credential {id} not found")
    assert_org_owned_writable(ctx, existing, resource_label="cloud credential")

    try:
        await CloudCredential.update(existing, session=session, source=input)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update cloud credential: {e}"
        )

    return await CloudCredential.one_by_id(session, id)


@router.delete("/{id}")
async def delete(session: SessionDep, ctx: TenantContextDep, id: int):
    existing = await CloudCredential.one_by_id(session, id)
    if not existing or existing.deleted_at is not None:
        raise NotFoundException(message=f"cloud credential {id} not found")
    assert_org_owned_writable(ctx, existing, resource_label="cloud credential")

    try:
        await existing.delete(session=session)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to delete cloud credential: {e}"
        )


@router.api_route("/{id}/provider-proxy/{path:path}", methods=["GET"])
async def proxy_cluster_provider_api(
    request: Request, session: SessionDep, ctx: TenantContextDep, id: int, path: str
):
    """
    To support other provider in the future, use api_route instead of get.
    """

    credential = await CloudCredential.one_by_id(session=session, id=id)
    if not credential:
        raise NotFoundException(message=f"Credential {id} not found")
    # Proxying via the credential's secret bridges into the cloud
    # provider's API; treat as a "use" / read-class permission, gated
    # the same way as a visibility check.
    assert_resource_visible(
        ctx,
        credential,
        not_found_message=f"Credential {id} not found",
    )
    if credential.provider in [ClusterProvider.Docker, ClusterProvider.Kubernetes]:
        raise NotFoundException(message=f"Provider {credential.provider} not supported")
    provider = factory.get(credential.provider, None)
    if provider is None:
        raise NotFoundException(message=f"Provider {credential.provider} not found")
    endpoint = provider[0].get_api_endpoint()
    # The request carries the credential's auth header, so it must stay on
    # the provider host. Require a strictly relative path first: an
    # absolute, scheme-relative, or backslash-bearing path can be parsed
    # differently by urlparse and by the HTTP client (parser confusion),
    # so rejecting it up front is more reliable than only inspecting the
    # joined URL.
    parsed_path = urlparse(path)
    if (
        "\\" in path
        or path.startswith("//")
        or parsed_path.scheme
        or parsed_path.netloc
    ):
        raise BadRequestException(message="Invalid provider API path")
    url = urljoin(endpoint, path)
    # Defense in depth: the joined URL must still resolve to the provider host.
    endpoint_parts = urlparse(endpoint)
    url_parts = urlparse(url)
    if (url_parts.scheme, url_parts.netloc) != (
        endpoint_parts.scheme,
        endpoint_parts.netloc,
    ):
        raise BadRequestException(message="Invalid provider API path")
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
