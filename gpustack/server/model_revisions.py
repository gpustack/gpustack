"""Deployment configuration snapshots, history reads and transactional writes."""

from typing import Any, Dict, Optional

from sqlalchemy import delete
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.api.tenant import TenantContext, assert_resource_visible
from gpustack.schemas.models import Model, ModelBase
from gpustack.schemas.model_revisions import ModelRevision

# An explicit projection keeps new runtime fields out of saved configuration.
REVISION_FIELDS = (
    "source",
    "huggingface_repo_id",
    "huggingface_filename",
    "model_scope_model_id",
    "model_scope_file_path",
    "local_path",
    "replicas",
    "categories",
    "placement_strategy",
    "cpu_offloading",
    "distributed_inference_across_workers",
    "worker_selector",
    "gpu_selector",
    "gpu_type_selector",
    "backend",
    "backend_version",
    "backend_parameters",
    "image_name",
    "run_command",
    "native_anthropic_api",
    "generic_proxy",
    "env",
    "restart_on_error",
    "extended_kv_cache",
    "speculative_config",
    "scaling_schedule",
    "lora_list",
    "roles",
    "disaggregation",
    "gather",
)


def deployment_spec(model: ModelBase) -> Dict[str, Any]:
    """Return detached, normalized user configuration without identity or status."""
    data = model.model_dump(mode="json", include=set(REVISION_FIELDS))
    schedule = data.get("scaling_schedule")
    if schedule and schedule.get("enabled"):
        data["replicas"] = schedule["baseline_replicas"]
    _normalize_spec(data, model.name)
    return data


def _normalize_spec(
    data: Dict[str, Any], model_name: str, inherited: bool = False
) -> None:
    for field in ("worker_selector", "env", "backend_parameters", "lora_list", "roles"):
        if not inherited and field in data and not data[field]:
            data[field] = None
    for adapter in data.get("lora_list") or []:
        adapter.pop("path", None)
        adapter.pop("model_file_id", None)
        adapter["lora_name"] = adapter["lora_name"].removeprefix(f"{model_name}:")
    for role in data.get("roles") or []:
        _normalize_spec(role, model_name, inherited=True)


async def lock_model(session: AsyncSession, ctx: TenantContext, model_id: int) -> Model:
    """Serialize history changes and refresh rows already loaded by an import plan."""
    statement = (
        select(Model)
        .where(Model.id == model_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    model = (await session.exec(statement)).one_or_none()
    assert_resource_visible(ctx, model, not_found_message="Model not found")
    return model


async def prepare_history_read(
    session: AsyncSession, ctx: TenantContext, model_id: int
) -> None:
    """Authorize a history read and commit a baseline only when history is absent."""
    model = (
        await session.exec(
            select(Model)
            .where(Model.id == model_id)
            .execution_options(populate_existing=True)
        )
    ).one_or_none()
    assert_resource_visible(ctx, model, not_found_message="Model not found")
    revision_id = (
        await session.exec(
            select(ModelRevision.id).where(ModelRevision.model_id == model_id).limit(1)
        )
    ).first()
    if revision_id is not None:
        return

    # Refresh and recheck under the parent lock: another request may have
    # initialized history or changed the configuration since the ordinary read.
    model = await lock_model(session, ctx, model_id)
    await ensure_baseline(session, model)
    # Release the initialization lock before reading the response. Ending the
    # transaction also replaces a MySQL repeatable-read snapshot with no history.
    await session.commit()


async def latest_revision(
    session: AsyncSession, model_id: int
) -> Optional[ModelRevision]:
    # Locking reads also see the latest committed history under MySQL's
    # repeatable-read isolation after waiting for the parent row lock.
    statement = (
        select(ModelRevision)
        .where(ModelRevision.model_id == model_id)
        .order_by(ModelRevision.revision.desc())
        .limit(1)
        .with_for_update()
    )
    return (await session.exec(statement)).first()


async def ensure_baseline(session: AsyncSession, model: Model) -> ModelRevision:
    """Save the pre-edit configuration once; callers hold the parent row lock."""
    latest = await latest_revision(session, model.id)
    if latest is None:
        latest = await append_revision(session, model, 1)
    return latest


async def append_revision(
    session: AsyncSession,
    model: Model,
    revision: int,
    created_by: Optional[int] = None,
) -> ModelRevision:
    item = ModelRevision(
        model_id=model.id,
        revision=revision,
        spec=deployment_spec(model),
        created_by=created_by,
    )
    session.add(item)
    await session.flush()
    return item


async def record_update(
    session: AsyncSession,
    model: Model,
    before: Dict[str, Any],
    latest: ModelRevision,
    created_by: Optional[int] = None,
) -> None:
    """Compare the live before/after configuration, excluding background writes."""
    if deployment_spec(model) != before:
        await append_revision(session, model, latest.revision + 1, created_by)
    await prune_revisions(session, model)


async def prune_revisions(session: AsyncSession, model: Model) -> None:
    """Retain the latest configuration plus the requested number of older ones."""
    # Offset N selects the (N + 1)th newest revision. Keep that boundary and
    # delete only older entries, retaining N older revisions plus the latest.
    statement = (
        select(ModelRevision.revision)
        .where(ModelRevision.model_id == model.id)
        .order_by(ModelRevision.revision.desc())
        .offset(model.revision_history_limit)
        .limit(1)
        .with_for_update()
    )
    boundary = (await session.exec(statement)).first()
    if boundary is not None:
        await session.execute(
            delete(ModelRevision).where(
                ModelRevision.model_id == model.id, ModelRevision.revision < boundary
            )
        )
