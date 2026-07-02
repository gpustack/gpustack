"""GPU instance HTTP routes.

``status.phase`` is the source of truth; the controller reconciles it
against the worker-side ``Instance`` CR. State machine:

    None ──► Pending ──► NotReady ──► Ready
                                       │
                              /stop ───┤
                                       ▼
            Stopping ─(operator reports phase=Stopped)─► Stopped
                                                          │
                                                    /start│
                                                          ▼
                                                       Starting
                                                          │
                                          ────────────────┘
                                          ▼ (re-enters Pending → … → Ready)

    *Failed (initial-create failures; terminal until):
        - /delete ─► Deleting (cleanup)

    /delete works from **any** phase and is sticky: the controller's
    ``_write_status`` refuses to overwrite a row that already reads
    Deleting, so an in-flight Stopping/Starting reconcile cannot
    resurrect it.

Route → target phase:
    DELETE /{id}        → Deleting   (any phase)
    PUT    /{id}/stop   → Stopping   (rejected from in-flight / terminal phases)
    PUT    /{id}/start  → Starting   (only from Stopped)
"""

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import APIRouter, Depends
from starlette.responses import StreamingResponse

from gpustack.api.exceptions import (
    AlreadyExistsException,
    InternalServerErrorException,
    InvalidException,
    NotFoundException,
)
from gpustack.api.tenant import (
    bypass_tenant_filter,
    TenantContext,
    assert_org_owned_writable,
    validate_owner_principal,
)
from gpustack.gpu_instances import validate_k8s_object_name
from gpustack.routes.gpu_instance_persistent_volumes import resolve_pv_type_for_ctx

from gpustack.schemas import (
    GPUInstance,
    GPUInstancePersistentVolume,
    GPUInstanceUpdate,
    GPUInstancePublic,
    GPUInstanceListParams,
    GPUInstancesPublic,
    GPUInstanceCreate,
)
from gpustack.schemas.gpu_instances import (
    FAILED_PHASES,
    INTERRUPTED_PHASES,
    TRANSITIONING_PHASES,
    GPUInstancePhase,
    GPUInstanceStatus,
)
from gpustack.schemas.principals import platform_principal_id
from gpustack.server.db import async_session
from gpustack.server.deps import SessionDep, TenantContextDep
from sqlmodel.ext.asyncio.session import AsyncSession

router = APIRouter()


@router.get("", response_model=GPUInstancesPublic)
async def get_gpu_instances(
    ctx: TenantContextDep,
    params: GPUInstanceListParams = Depends(),
    search: Optional[str] = None,
):
    owner_principal_id = ctx.current_principal_id or platform_principal_id()
    if bypass_tenant_filter(ctx):
        owner_principal_id = None

    fields: dict = {}
    if owner_principal_id is not None:
        fields["owner_principal_id"] = owner_principal_id

    fuzzy_fields: dict = {}
    if search:
        fuzzy_fields["name"] = search

    if params.watch:
        return StreamingResponse(
            GPUInstance.streaming(
                fields=fields,
                fuzzy_fields=fuzzy_fields,
            ),
            media_type="text/event-stream",
        )

    async with async_session() as session:
        return await GPUInstance.paginated_by_query(
            session=session,
            fields=fields,
            fuzzy_fields=fuzzy_fields,
            order_by=params.order_by,
            page=params.page,
            per_page=params.perPage,
        )


@router.get("/{id}", response_model=GPUInstancePublic)
async def get_gpu_instance(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
):
    return ensure_visible(
        await GPUInstance.one_by_id(
            session=session,
            id=id,
        ),
        ctx,
    )


@router.post("", response_model=GPUInstancePublic)
async def create_gpu_instance(
    session: SessionDep,
    ctx: TenantContextDep,
    create_obj: GPUInstanceCreate,
):
    if create_obj.owner_principal_id is None:
        create_obj.owner_principal_id = (
            ctx.current_principal_id or platform_principal_id()
        )
    validate_owner_principal(
        create_obj.owner_principal_id,
        ctx,
        resource_label="GPU instance",
        allow_member=True,
    )

    persistent_volume_id = await _validate_create_obj(session, ctx, create_obj)

    existed = await GPUInstance.exist_by_fields(
        session=session,
        fields={
            "owner_principal_id": create_obj.owner_principal_id,
            "name": create_obj.name,
        },
    )
    if existed:
        raise AlreadyExistsException(
            message=(f"GPU instance with name '{create_obj.name}' already exists."),
        )

    source = _build_create_source(create_obj, ctx.user.id, persistent_volume_id)
    async with handle_error(
        message="Failed to create GPU instance",
    ):
        return await GPUInstance.create(
            session=session,
            source=source,
        )


@router.put("/{id}", response_model=GPUInstancePublic)
async def update_gpu_instance(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
    update_obj: GPUInstanceUpdate,
):
    ret = ensure_writable(
        await GPUInstance.one_by_id(
            session=session,
            id=id,
        ),
        ctx,
    )

    source = _build_update_source(update_obj, ret)
    if not source:
        return ret

    async with handle_error(
        message="Failed to update GPU instance",
    ):
        await ret.update(
            session=session,
            source=source,
        )
        return ret


@router.delete("/{id}", status_code=202, response_model=GPUInstancePublic)
async def delete_gpu_instance(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
):
    """Mark the instance for deletion (allowed from any phase)."""
    return await _transition_to_phase(
        session,
        ctx,
        id,
        action="delete",
        target_phase=GPUInstancePhase.DELETING,
        fail_message="Failed to delete GPU instance",
    )


@router.put("/{id}/stop", status_code=202, response_model=GPUInstancePublic)
async def stop_gpu_instance(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
):
    """Mark the instance for stopping."""
    return await _transition_to_phase(
        session,
        ctx,
        id,
        action="stop",
        target_phase=GPUInstancePhase.STOPPING,
        fail_message="Failed to stop GPU instance",
    )


@router.put("/{id}/start", status_code=202, response_model=GPUInstancePublic)
async def start_gpu_instance(
    session: SessionDep,
    ctx: TenantContextDep,
    id: int,
):
    """Mark the instance for starting."""
    return await _transition_to_phase(
        session,
        ctx,
        id,
        action="start",
        target_phase=GPUInstancePhase.STARTING,
        fail_message="Failed to start GPU instance",
    )


def ensure_visible(obj, ctx: TenantContext):
    if obj and is_visible(obj, ctx):
        return obj
    raise NotFoundException(message="GPU instance not found")


def ensure_writable(obj, ctx: TenantContext):
    if obj is None:
        raise NotFoundException(message="GPU instance not found")
    assert_org_owned_writable(
        ctx, obj, resource_label="GPU instance", allow_member=True
    )
    return obj


def is_visible(obj, ctx: TenantContext) -> bool:
    if bypass_tenant_filter(ctx):
        return True
    return ctx.current_principal_id == obj.owner_principal_id


@asynccontextmanager
async def handle_error(message: str):
    try:
        yield
    except Exception as e:
        raise InternalServerErrorException(
            message=message,
        ) from e


async def _validate_create_obj(
    session: AsyncSession,
    ctx: TenantContext,
    create_obj: GPUInstanceCreate,
) -> Optional[int]:
    """Enforce create-time invariants that aren't expressible in the schema:

    - ``spec.volume`` is required, and exactly one of ``ephemeral``,
      ``persistent``, ``persistentTemplate`` must be set.
    - If ``spec.volume.persistent`` is set, it must reference an existing PV by name,
      then resolved PV's id is returned.
      So the caller can stamp it onto the new GPU instance's ``persistent_volume_id`` FK.
    - If ``spec.volume.persistentTemplate`` is set,
      its name must not yet exist and its type must reference an existing PV type by name.
      The template is then provisioned inline,
      and the new PV's id is returned for stamping onto the GPU instance's FK.

    Return the resolved ``persistent_volume_id`` (``None`` for ephemeral volumes).

    The returned id is what the caller stamps onto
    :class:`GPUInstance.persistent_volume_id`. Three branches mirror
    ``spec.volume``:

    - ``ephemeral``: no FK; returns ``None``.
    - ``persistent``: existing PV by ``(owner, name)``; returns its id.
    - ``persistentTemplate``: provisions a new PV inline (after
      resolving its ``spec.type`` to a PV type id) and returns the new
      row's id.
    """

    validate_k8s_object_name(create_obj.name)

    volume = create_obj.spec.volume

    if volume is None:
        raise InvalidException(
            message=(
                "spec.volume is required: set exactly one of "
                "ephemeral, persistent, persistentTemplate"
            ),
        )
    volume_choices_set = [
        name
        for name, value in (
            ("ephemeral", volume.ephemeral),
            ("persistent", volume.persistent),
            ("persistentTemplate", volume.persistent_template),
        )
        if value is not None
    ]
    if len(volume_choices_set) != 1:
        raise InvalidException(
            message=(
                "spec.volume must specify exactly one of "
                "ephemeral, persistent, persistentTemplate; "
                f"got {volume_choices_set or 'none'}"
            ),
        )

    if volume.persistent is not None:
        pv = await GPUInstancePersistentVolume.first_by_fields(
            session=session,
            fields={
                "owner_principal_id": create_obj.owner_principal_id,
                "name": volume.persistent.name,
            },
        )
        if pv is None:
            raise InvalidException(
                message=(
                    f"GPU instance persistent volume "
                    f"'{volume.persistent.name}' not found"
                ),
            )
        return pv.id

    if (template := volume.persistent_template) is not None:
        existing = await GPUInstancePersistentVolume.first_by_fields(
            session=session,
            fields={
                "owner_principal_id": create_obj.owner_principal_id,
                "name": template.name,
            },
        )
        if existing is not None:
            raise InvalidException(
                message=(
                    f"GPU instance persistent volume '{template.name}' "
                    "already exists; use spec.volume.persistent to reference it"
                ),
            )

        pvt = await resolve_pv_type_for_ctx(
            session,
            ctx,
            owner_principal_id=create_obj.owner_principal_id,
            name=template.spec.type_,
        )
        if pvt is None:
            raise InvalidException(
                message=(
                    f"GPU instance persistent volume type "
                    f"'{template.spec.type_}' not found"
                ),
            )

        pv = GPUInstancePersistentVolume(
            name=template.name,
            spec=template.spec,
            owner_principal_id=create_obj.owner_principal_id,
            creator_id=ctx.user.id,
            persistent_volume_type_id=pvt.id,
        )
        created = await GPUInstancePersistentVolume.create(
            session=session,
            source=pv,
            auto_commit=False,
        )
        return created.id

    return None


def _build_create_source(
    create_obj: GPUInstanceCreate,
    creator_id: int,
    persistent_volume_id: Optional[int],
) -> dict:
    source: dict = create_obj.model_dump()
    source["creator_id"] = creator_id
    if persistent_volume_id is not None:
        source["persistent_volume_id"] = persistent_volume_id
    return source


def _build_update_source(
    update_obj: GPUInstanceUpdate, existing_obj: GPUInstance
) -> dict:
    """Filter ``update_obj`` down to the fields that may actually change.

    Only ``display_name``, ``description``, and ``spec.sshPublicKeys`` are
    mutable post-create. ``spec`` on the request is a
    :class:`GPUInstanceSpecUpdate` (a narrow subset of the row's
    :class:`GPUInstanceSpec`); assigning it directly would clobber every
    other field on the row's spec, so we merge ``sshPublicKeys`` onto the
    existing spec and keep the row's type intact.
    """
    source: dict = {}

    fields_set = update_obj.model_fields_set
    if "display_name" in fields_set:
        source["display_name"] = update_obj.display_name
    if "description" in fields_set:
        source["description"] = update_obj.description

    if (
        "spec" in fields_set
        and update_obj.spec is not None
        and "ssh_public_keys" in update_obj.spec.model_fields_set
    ):
        merged_spec = existing_obj.spec.model_copy(
            update={"ssh_public_keys": update_obj.spec.ssh_public_keys or []},
        )
        source["spec"] = merged_spec

    return source


def _build_update_phase_source(existing_obj: GPUInstance, phase: str) -> dict:
    """Stamp ``phase`` onto status, clearing ``phase_message`` so the new phase
    starts clean."""
    base = existing_obj.status or GPUInstanceStatus()
    return {
        "status": base.model_copy(
            update={
                "phase": phase,
                "phase_message": None,
            },
        ),
    }


# Source-phase gates for each lifecycle action. /delete has no gate (allowed
# from any phase, including ``None`` pre-create). The phase categories these
# compose from (``TRANSITIONING_PHASES`` / ``INTERRUPTED_PHASES`` /
# ``FAILED_PHASES``) are owned by the schema.
#
# Stop is allowed only from a settled, running phase: disallow it while the
# instance is transitioning, interrupted, or failed (``None`` pre-create is
# rejected at the call site).
_STOP_DISALLOWED_FROM = TRANSITIONING_PHASES | INTERRUPTED_PHASES | FAILED_PHASES
# Start resumes an interrupted (Stopped) instance.
_START_ALLOWED_FROM = INTERRUPTED_PHASES


async def _transition_to_phase(
    session: AsyncSession,
    ctx: TenantContext,
    id: int,
    *,
    action: str,
    target_phase: str,
    fail_message: str,
) -> GPUInstance:
    """Load + ownership-check the row, gate the transition by ``action``,
    stamp the target phase, and persist. Returns the updated row."""
    ret = ensure_writable(
        await GPUInstance.one_by_id(
            session=session,
            id=id,
        ),
        ctx,
    )

    current_phase = ret.status.phase if ret.status else None
    if action == "stop":
        if current_phase is None or current_phase in _STOP_DISALLOWED_FROM:
            raise InvalidException(
                message=(
                    f"GPU instance cannot be stopped from "
                    f"{current_phase or 'pending creation'} phase"
                ),
            )
    elif action == "start":
        if current_phase not in _START_ALLOWED_FROM:
            raise InvalidException(
                message=(
                    f"GPU instance cannot be started from "
                    f"{current_phase or 'pending creation'} phase"
                ),
            )

    source = _build_update_phase_source(ret, target_phase)

    async with handle_error(message=fail_message):
        await ret.update(
            session=session,
            source=source,
        )
        return ret
