import logging
from typing import Set

from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack import envs
from gpustack.api.exceptions import BadRequestException
from gpustack.schemas.models import (
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
)
from gpustack.schemas.links import ModelRoutePrincipalLink
from gpustack.schemas.model_routes import (
    AccessPolicyEnum,
    ModelRoute,
    ModelRouteTarget,
    TargetStateEnum,
)
from gpustack.schemas.principals import platform_principal_id
from gpustack.server.services import ModelRouteService
from gpustack.utils.lora_model_source import (
    lora_route_name_for,
    normalized_lora_list,
)
from gpustack.utils.network import is_offline

logger = logging.getLogger(__name__)


async def create_lora_model_routes(
    session: AsyncSession,
    model: Model,
    *,
    access_policy,
    generic_proxy: bool,
) -> None:
    """Ensure one ModelRoute + ModelRouteTarget per entry in model.lora_list.

    Idempotent: skips entries whose route already exists for this base model.
    Raises BadRequestException if the desired route name is already taken by a
    route belonging to a different base model.

    Caller commits the session; all writes use auto_commit=False.
    """
    entries = normalized_lora_list(model)
    if not entries:
        return

    existing = await ModelRoute.all_by_fields(
        session,
        fields={"created_model_id": model.id, "deleted_at": None},
    )
    existing_names = {r.name for r in existing}

    # Org-scoped models default to ALLOWED_PRINCIPALS; their routes are
    # made visible to the owning Org via a principal grant (see
    # ``create_model``). A LoRA child route is a separate ACL subject, so
    # it needs its own grant to match the base model's visibility — auto-
    # grant the owning (non-platform) Org on each newly created route.
    # Platform-owned or non-ALLOWED_PRINCIPALS routes get nothing here.
    owner_org_id = model.owner_principal_id
    grant_owner_org = (
        access_policy == AccessPolicyEnum.ALLOWED_PRINCIPALS
        and owner_org_id is not None
        and owner_org_id != platform_principal_id()
    )

    for entry in entries:
        route_name = lora_route_name_for(model.name, entry.lora_name)
        if route_name in existing_names:
            continue

        conflict = await ModelRoute.one_by_field(session, "name", route_name)
        if conflict and conflict.created_model_id != model.id:
            raise BadRequestException(
                message=(
                    f"LoRA route name {route_name!r} conflicts with existing "
                    f"model route id={conflict.id}; choose a different "
                    f"lora_name on base model {model.name!r}."
                )
            )

        model_route = ModelRoute(
            name=route_name,
            description=f"LoRA adapter {entry.lora_name!r} for base model {model.name}",
            categories=model.categories or [],
            meta={
                "lora_name": entry.lora_name,
                "lora_repo_name": entry.lora_repo_name,
                "source": entry.source,
            },
            generic_proxy=generic_proxy,
            access_policy=access_policy,
            created_model_id=model.id,
            targets=1,
            ready_targets=0,
        )
        model_route = await ModelRoute.create(session, model_route, auto_commit=False)
        await session.flush()

        if grant_owner_org:
            session.add(
                ModelRoutePrincipalLink(
                    route_id=model_route.id,
                    principal_id=owner_org_id,
                )
            )

        model_route_target = ModelRouteTarget(
            name=f"{route_name}-deployment",
            route_name=model_route.name,
            route_id=model_route.id,
            model_route=model_route,
            model=model,
            model_id=model.id,
            weight=100,
            state=TargetStateEnum.UNAVAILABLE,
            overridden_model_name=route_name,
        )
        await ModelRouteTarget.create(session, model_route_target, auto_commit=False)
        logger.info(
            f"Created LoRA model route {route_name!r} for base model {model.name!r}"
        )


def is_lora_list_stale(model: Model) -> bool:
    """True if any RUNNING instance's `mounted_loras` differs from
    `model.lora_list`. Requires `model.instances` to be eager-loaded.
    """
    desired = {
        lora_route_name_for(model.name, entry.lora_name)
        for entry in normalized_lora_list(model)
    }
    for instance in getattr(model, "instances", None) or []:
        if instance.state != ModelInstanceStateEnum.RUNNING:
            continue
        mounted = {
            mount.lora_name
            for mount in (instance.mounted_loras or [])
            if mount.lora_name
        }
        if mounted != desired:
            return True
    return False


async def cleanup_orphan_lora_routes(session: AsyncSession, model: Model) -> bool:
    """Remove LoRA ModelRoutes no longer in model.lora_list, unless
    still mounted on a non-stale RUNNING instance. Caller commits.
    """
    desired_names: Set[str] = {
        lora_route_name_for(model.name, entry.lora_name)
        for entry in normalized_lora_list(model)
    }

    existing = await ModelRoute.all_by_fields(
        session,
        fields={"created_model_id": model.id, "deleted_at": None},
    )
    # LoRA child routes are named `<base>:<lora>`; the primary auto-route
    # for the Model itself has no `:` and must not be reaped here.
    orphans = [r for r in existing if ":" in r.name and r.name not in desired_names]
    if not orphans:
        return False

    instances = await ModelInstance.all_by_fields(
        session,
        fields={"model_id": model.id, "deleted_at": None},
        extra_conditions=[ModelInstance.state == ModelInstanceStateEnum.RUNNING],
    )
    mounted_names: Set[str] = set()
    for instance in instances:
        if not instance.mounted_loras:
            continue
        stale, _ = is_offline(
            instance.updated_at, envs.MODEL_INSTANCE_RESCHEDULE_GRACE_PERIOD
        )
        if stale:
            logger.debug(
                f"Ignoring stale RUNNING instance {instance.name} "
                f"(updated_at={instance.updated_at}) for mounted LoRA protection"
            )
            continue
        for mounted in instance.mounted_loras:
            if mounted.lora_name:
                mounted_names.add(mounted.lora_name)

    any_deleted = False
    service = ModelRouteService(session)
    for route in orphans:
        if route.name in mounted_names:
            logger.debug(
                f"Skip cleanup of LoRA route {route.name!r}: still mounted on a running instance"
            )
            continue
        await service.delete(route, auto_commit=False)
        any_deleted = True
        logger.info(
            f"Removed orphan LoRA model route {route.name!r} for base model "
            f"{model.name!r} (no longer in lora_list and not mounted)"
        )
    return any_deleted
