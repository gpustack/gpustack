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
from gpustack.routes.models import assert_cluster_belongs_to_org

from gpustack.schemas import (
    GPUInstance,
    GPUInstancePersistentVolume,
    GPUInstanceType,
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
    GPUInstanceSpec,
    GPUInstanceStatus,
    GPUInstanceVolume,
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
    if create_obj.cluster_id is None:
        raise InvalidException(message="cluster_id is required")

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

    # A GPU instance runs on infrastructure owned by its Org, so the chosen
    # cluster must be visible to the caller and owned by the instance's Org.
    await assert_cluster_belongs_to_org(
        ctx, session, create_obj.cluster_id, create_obj.owner_principal_id
    )

    persistent_volume_id = await _validate_create_obj(session, ctx, create_obj)

    type_snapshot = await _resolve_type_snapshot(
        session,
        cluster_id=create_obj.cluster_id,
        type_name=create_obj.spec.type_,
    )

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

    source = _build_create_source(
        create_obj, ctx.user.id, persistent_volume_id, type_snapshot
    )
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

    source = await _build_update_source(session, ctx, update_obj, ret)
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


def ensure_visible(obj: GPUInstance, ctx: TenantContext) -> GPUInstance:
    if obj and is_visible(obj, ctx):
        return obj
    raise NotFoundException(message="GPU instance not found")


def ensure_writable(obj: GPUInstance, ctx: TenantContext) -> GPUInstance:
    if obj is None:
        raise NotFoundException(message="GPU instance not found")
    assert_org_owned_writable(
        ctx, obj, resource_label="GPU instance", allow_member=True
    )
    return obj


def is_visible(obj: GPUInstance, ctx: TenantContext) -> bool:
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
    """Enforce create-time invariants that aren't expressible in the schema.

    Validates the (create-only) k8s object name, then resolves ``spec.volume``
    to a ``persistent_volume_id`` via :func:`_resolve_volume` (shared with the
    stopped-instance spec update). Returns the resolved id (``None`` for
    ephemeral volumes) for the caller to stamp onto
    :class:`GPUInstance.persistent_volume_id`.
    """

    validate_k8s_object_name(create_obj.name)

    return await _resolve_volume(
        session,
        ctx,
        owner_principal_id=create_obj.owner_principal_id,
        volume=create_obj.spec.volume,
    )


async def _resolve_volume(
    session: AsyncSession,
    ctx: TenantContext,
    *,
    owner_principal_id: int,
    volume: Optional[GPUInstanceVolume],
) -> Optional[int]:
    """Validate a ``spec.volume`` selection and resolve it to a
    ``persistent_volume_id``. Shared by create and the stopped-instance spec
    update.

    - ``spec.volume`` is required, and exactly one of ``ephemeral``,
      ``persistent``, ``persistentTemplate`` must be set.
    - ``ephemeral``: no FK; returns ``None``.
    - ``persistent``: existing PV by ``(owner, name)``; returns its id.
    - ``persistentTemplate``: its name must not yet exist and its type must
      reference an existing PV type by name; the template is provisioned inline
      (``auto_commit=False``, committed by the caller's row write) and the new
      PV's id is returned.
    """

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
                "owner_principal_id": owner_principal_id,
                "name": volume.persistent.name,
            },
        )
        if pv is None or pv.is_deleting():
            raise InvalidException(
                message=(
                    f"GPU instance persistent volume "
                    f"'{volume.persistent.name}' not found or is being deleted"
                ),
            )
        return pv.id

    if (template := volume.persistent_template) is not None:
        existing = await GPUInstancePersistentVolume.first_by_fields(
            session=session,
            fields={
                "owner_principal_id": owner_principal_id,
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
            owner_principal_id=owner_principal_id,
            name=template.spec.type_,
        )
        if pvt is None or pvt.is_deleting():
            raise InvalidException(
                message=(
                    f"GPU instance persistent volume type "
                    f"'{template.spec.type_}' not found or is being deleted"
                ),
            )

        pv = GPUInstancePersistentVolume(
            name=template.name,
            spec=template.spec,
            owner_principal_id=owner_principal_id,
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


async def _resolve_type_snapshot(
    session: AsyncSession,
    *,
    cluster_id: Optional[int],
    type_name: str,
) -> str:
    """Resolve ``spec.type_`` to the persisted instance type's snapshot.

    Looks up the active (non-soft-deleted) :class:`GPUInstanceType` for the
    target cluster and returns its ``snapshot`` to stamp onto the instance. A
    missing or soft-deleted type means the instance references a type the
    cluster does not offer, so it is a client error
    (:class:`InvalidException`) rather than an instance created against a
    non-existent type.
    """
    row = await GPUInstanceType.first_by_fields(
        session=session,
        fields={
            "cluster_id": cluster_id,
            "name": type_name,
            "deleted_at": None,
        },
    )
    if row is None:
        raise InvalidException(
            message=(
                f"GPU instance type '{type_name}' not found in " f"cluster {cluster_id}"
            ),
        )
    return row.snapshot


def _build_create_source(
    create_obj: GPUInstanceCreate,
    creator_id: int,
    persistent_volume_id: Optional[int],
    type_snapshot: str,
) -> dict:
    source: dict = create_obj.model_dump()
    source["creator_id"] = creator_id
    source["type_snapshot"] = type_snapshot
    if persistent_volume_id is not None:
        source["persistent_volume_id"] = persistent_volume_id
    return source


async def _build_update_source(
    session: AsyncSession,
    ctx: TenantContext,
    update_obj: GPUInstanceUpdate,
    existing_obj: GPUInstance,
) -> dict:
    """Filter ``update_obj`` down to the fields that may actually change.

    ``display_name`` / ``description`` are metadata and may change from any
    phase. ``spec`` is a full replacement carrying every field except the
    instance ``name``; the client always PUTs the whole desired spec, and what
    it changed decides the gate:

    - ``spec.sshPublicKeys`` alone may change from any phase (keys are resynced
      to the worker live), so an edit whose only diff is the SSH key list is
      accepted while the instance is running.
    - any other field change requires the instance to be Stopped; such an edit
      takes effect on the next ``/start`` when the controller re-applies the
      spec to the worker CR.

    When ``spec.volume`` changes (only possible while Stopped) it is re-resolved
    to a ``persistent_volume_id`` FK (same rules as create); an unchanged volume
    keeps the existing FK. A volume swap only re-points the FK; it never deletes
    the previously bound persistent volume (that could destroy user data), so an
    auto-provisioned template PV left behind by a swap must be removed
    explicitly via the PV delete endpoint.
    """
    source: dict = {}

    fields_set = update_obj.model_fields_set
    if "display_name" in fields_set:
        source["display_name"] = update_obj.display_name
    if "description" in fields_set:
        source["description"] = update_obj.description

    if "spec" in fields_set and update_obj.spec is not None:
        new_spec = update_obj.spec
        if not existing_obj.is_stopped():
            # Only the SSH key list is editable while running; reject an edit
            # that touches any other field until the instance is stopped.
            if _spec_changes_beyond_ssh(new_spec, existing_obj.spec):
                raise InvalidException(
                    message=(
                        "Only spec.sshPublicKeys can be updated while the GPU "
                        "instance is running; stop it to change any other "
                        "configuration"
                    ),
                )
            # SSH-only edit: the volume is unchanged, so the FK stays put.
            source["spec"] = new_spec
            return source

        if new_spec.volume == existing_obj.spec.volume:
            persistent_volume_id = existing_obj.persistent_volume_id
        else:
            persistent_volume_id = await _resolve_volume(
                session,
                ctx,
                owner_principal_id=existing_obj.owner_principal_id,
                volume=new_spec.volume,
            )
        source["spec"] = new_spec
        source["persistent_volume_id"] = persistent_volume_id
        # Re-stamp the type snapshot ONLY when the referenced type changed. An
        # edit to any other field (image, resources, volume) keeps the original
        # pinned snapshot, so an unrelated edit neither rebinds the instance's
        # recorded type nor fails when that type has since been removed. The
        # cluster is immutable post-create; a now-missing new type is rejected,
        # same as at create.
        if new_spec.type_ != existing_obj.spec.type_:
            source["type_snapshot"] = await _resolve_type_snapshot(
                session,
                cluster_id=existing_obj.cluster_id,
                type_name=new_spec.type_,
            )

    return source


def _spec_changes_beyond_ssh(
    new_spec: GPUInstanceSpec, current_spec: GPUInstanceSpec
) -> bool:
    """Whether ``new_spec`` differs from ``current_spec`` in any field other than
    ``ssh_public_keys`` — the one field editable while the instance runs."""
    ignore_ssh = {"ssh_public_keys": None}
    return new_spec.model_copy(update=ignore_ssh) != current_spec.model_copy(
        update=ignore_ssh
    )


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
