"""Organization management — platform admin only.

"Organizations" in the API surface are ORG-kind ``Principal`` rows.
This file is the CRUD adapter that maps the legacy Organization shape
on/off the unified principals table.
"""

from typing import Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlmodel import select

from gpustack.api.exceptions import (
    AlreadyExistsException,
    ConflictException,
    ForbiddenException,
    InternalServerErrorException,
    InvalidException,
    NotFoundException,
)
from gpustack.schemas.organizations import (
    OrganizationCreate,
    OrganizationListParams,
    OrganizationPublic,
    OrganizationUpdate,
    OrganizationsPublic,
    validate_org_input,
)
from gpustack.schemas.principals import (
    OrgRole,
    Principal,
    PrincipalType,
    platform_principal_id,
)
from gpustack.server.deps import SessionDep, TenantContextDep

router = APIRouter()
directory_router = APIRouter()


def _to_public(p: Principal) -> OrganizationPublic:
    return OrganizationPublic.from_principal(p)


@router.get("", response_model=OrganizationsPublic)
async def get_organizations(
    session: SessionDep,
    params: OrganizationListParams = Depends(),
    search: Optional[str] = None,
):
    fuzzy_fields = {}
    if search:
        fuzzy_fields = {"name": search, "slug": search}

    fields = {"deleted_at": None, "kind": PrincipalType.ORG}

    if params.watch:
        return StreamingResponse(
            Principal.streaming(fields=fields, fuzzy_fields=fuzzy_fields),
            media_type="text/event-stream",
        )

    page = await Principal.paginated_by_query(
        session=session,
        fields=fields,
        fuzzy_fields=fuzzy_fields,
        page=params.page,
        per_page=params.perPage,
        order_by=params.order_by,
    )
    page.items = [_to_public(p) for p in page.items]
    return page


@router.get("/{id}", response_model=OrganizationPublic)
async def get_organization(session: SessionDep, id: int):
    org = await Principal.one_by_id(session, id)
    if not org or org.deleted_at is not None or org.kind != PrincipalType.ORG:
        raise NotFoundException(message="Organization not found")
    return _to_public(org)


@router.post("", response_model=OrganizationPublic)
async def create_organization(session: SessionDep, org_in: OrganizationCreate):
    # Block reserved names ("Personal" / "Global") and slug patterns
    # ("user-N") on the input side. Validation lives in the route, not
    # the schema, so the same model can serialize already-existing
    # auto-created USER-principals without rejecting them.
    try:
        validate_org_input(name=org_in.name, slug=org_in.slug)
    except ValueError as e:
        raise InvalidException(message=str(e))

    existing = await Principal.one_by_fields(
        session, {"slug": org_in.slug, "deleted_at": None}
    )
    if existing:
        raise AlreadyExistsException(
            message=f"Organization with slug '{org_in.slug}' already exists"
        )

    try:
        to_create = Principal(
            kind=PrincipalType.ORG,
            name=org_in.name,
            slug=org_in.slug,
            description=org_in.description,
        )
        created = await Principal.create(session, to_create)
        return _to_public(created)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to create organization: {e}"
        )


@router.put("/{id}", response_model=OrganizationPublic)
async def update_organization(session: SessionDep, id: int, org_in: OrganizationUpdate):
    org = await Principal.one_by_id(session, id)
    if not org or org.deleted_at is not None or org.kind != PrincipalType.ORG:
        raise NotFoundException(message="Organization not found")

    try:
        validate_org_input(name=org_in.name)
    except ValueError as e:
        raise InvalidException(message=str(e))

    try:
        await org.update(session, org_in.model_dump(exclude_unset=True))
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to update organization: {e}"
        )
    return _to_public(org)


@router.delete("/{id}")
async def delete_organization(session: SessionDep, id: int):
    org = await Principal.one_by_id(session, id)
    if not org or org.deleted_at is not None or org.kind != PrincipalType.ORG:
        raise NotFoundException(message="Organization not found")
    if org.id == platform_principal_id():
        raise ConflictException(
            message="The built-in platform organization cannot be deleted"
        )

    # Block delete when any tenant-owned resource still references this org.
    # FK CASCADE would silently destroy users' resources; surfacing the
    # conflict lets the operator decide.
    blockers = await _has_resources(session, id)
    if blockers:
        raise ConflictException(
            message=(
                "Organization still owns resources: "
                f"{', '.join(blockers)}. Remove them before deleting."
            )
        )

    try:
        await org.delete(session)
    except Exception as e:
        raise InternalServerErrorException(
            message=f"Failed to delete organization: {e}"
        )


async def _has_resources(session, owner_principal_id: int) -> list[str]:
    """Return resource types that still belong to this principal.

    Cover every tenant-scoped resource (anything carrying an
    ``owner_principal_id``) so an admin who deletes an Org can't
    silently orphan or destroy clusters, worker pools, cloud
    credentials, user groups, benchmarks, or backend overrides. The
    check matches the spirit of FK CASCADE — but surfaces the
    conflict so the operator can decide what to do.
    """
    from gpustack.schemas.api_keys import ApiKey
    from gpustack.schemas.benchmark import Benchmark
    from gpustack.schemas.clusters import Cluster, CloudCredential, WorkerPool
    from gpustack.schemas.inference_backend import InferenceBackend
    from gpustack.schemas.model_routes import ModelRoute
    from gpustack.schemas.models import Model, ModelInstance

    blockers: list[str] = []
    for resource_cls, label in (
        (ApiKey, "api_keys"),
        (Model, "models"),
        (ModelInstance, "model_instances"),
        (ModelRoute, "model_routes"),
        (Cluster, "clusters"),
        (WorkerPool, "worker_pools"),
        (CloudCredential, "cloud_credentials"),
        (Benchmark, "benchmarks"),
        (InferenceBackend, "inference_backends"),
    ):
        stmt = (
            select(resource_cls.id)
            .where(
                resource_cls.owner_principal_id == owner_principal_id,
                resource_cls.deleted_at.is_(None),
            )
            .limit(1)
        )
        if (await session.exec(stmt)).first() is not None:
            blockers.append(label)

    # Group memberships into this Org — Group-principals are peer-level
    # now, but if any are still joined to this Org via an active
    # ``principal_memberships`` row, surface that to the operator before
    # the Org is deleted. The grant rows themselves CASCADE on Org
    # delete, but the Groups don't, so block to make the operator
    # explicit.
    from gpustack.schemas.principals import PrincipalMembership

    group_member_stmt = (
        select(PrincipalMembership.id)
        .join(Principal, Principal.id == PrincipalMembership.member_principal_id)
        .where(
            PrincipalMembership.parent_principal_id == owner_principal_id,
            PrincipalMembership.deleted_at.is_(None),
            Principal.kind == PrincipalType.GROUP,
            Principal.deleted_at.is_(None),
        )
        .limit(1)
    )
    if (await session.exec(group_member_stmt)).first() is not None:
        blockers.append("user_groups")

    return blockers


# ---- Org directory (slim, non-admin-readable) -------------------------------
#
# The full ``/organizations`` CRUD surface is admin-only. But picker UIs
# need to enumerate Orgs even when the caller is an Org owner — e.g.
# the cluster-detail "Grant access" form, where the cluster's owner Org
# owner picks a target Org to delegate cluster access to.
#
# Mirror the ``/user-directory`` pattern: a slim, paginated, search-
# friendly endpoint that returns the same ``OrganizationsPublic`` shape
# but is gated to platform admin OR the caller-in-owner-role-of-an-Org.
@directory_router.get("/organization-directory", response_model=OrganizationsPublic)
async def list_organization_directory(
    session: SessionDep,
    ctx: TenantContextDep,
    page: int = 1,
    perPage: int = 30,
    search: Optional[str] = None,
):
    if not ctx.is_platform_admin and ctx.org_role != OrgRole.OWNER:
        raise ForbiddenException(message="Insufficient permission")
    fuzzy_fields = {}
    if search:
        fuzzy_fields = {"name": search, "slug": search}
    fields = {"deleted_at": None, "kind": PrincipalType.ORG}
    page_data = await Principal.paginated_by_query(
        session=session,
        fields=fields,
        fuzzy_fields=fuzzy_fields,
        page=page,
        per_page=perPage,
    )
    page_data.items = [_to_public(p) for p in page_data.items]
    return page_data
