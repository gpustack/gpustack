from typing import Optional
from datetime import date

from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.model_provider import ModelProvider
from gpustack.schemas.models import Model
from gpustack.schemas.usage import USAGE_GRANULARITY_MONTH
from gpustack.schemas.users import User


def format_usage_user_label(user_name: Optional[str]) -> str:
    return user_name or "Unknown User"


def format_usage_route_label(route_name: Optional[str]) -> str:
    return route_name or "Untracked"


def format_usage_api_key_label(
    user_name: Optional[str] = None,
    api_key_name: Optional[str] = None,
) -> str:
    parts = [p for p in [user_name, api_key_name] if p]
    return " / ".join(parts) or "-"


def format_usage_date_label(value: date, granularity: str) -> str:
    if granularity == USAGE_GRANULARITY_MONTH:
        return value.strftime("%Y-%m")
    return value.isoformat()


def build_model_usage_snapshot(
    model: Model,
    cluster_name: Optional[str] = None,
    user: Optional[User] = None,
    api_key: Optional[ApiKey] = None,
    provider: Optional[ModelProvider] = None,
    model_route_id: Optional[int] = None,
    model_route_name: Optional[str] = None,
) -> dict:
    """Build a usage snapshot dict capturing the model identity at request time.

    Records model ID, name, and cluster; optionally includes user and API key
    fields when the caller supplies them. The snapshot is stored alongside usage
    records so usage can be attributed even after models or keys are deleted.

    ``cluster_name`` is resolved from ``model.cluster`` when not passed
    explicitly.

    Shared-snapshot contract: the returned dict is splatted (``**snapshot``)
    into BOTH ``ModelUsage`` and ``ModelUsageDetails`` constructors. Every
    key emitted here MUST therefore be a valid column on both tables,
    otherwise the rollup write or the details write will fail at runtime.
    Fields specific to one table only (e.g. ``cluster_id`` / ``started_at``
    / ``completed_at`` on details) must be passed via dedicated kwargs at
    the call site, NOT added here.

    ``model_route_id`` / ``model_route_name`` are kept as separate scalar
    kwargs (instead of a ``ModelRoute`` object) because the live route row
    may already be gone by flush time — the caller resolves the name from
    a pre-fetched ``route_name_by_id`` map and passes ``None`` when the
    route was deleted, preserving the id for audit while signalling the
    name is unrecoverable.
    """
    if cluster_name is None:
        cluster = getattr(model, "cluster", None)
        cluster_name = None if cluster is None else cluster.name

    snapshot = {
        "model_id": model.id,
        "model_name": model.name,
        "cluster_name": cluster_name,
        # Usage rows inherit the model's tenant scope so dashboard/filtering
        # by Org doesn't need to re-join models.
        "owner_principal_id": getattr(model, "owner_principal_id", None),
    }
    if user is not None:
        snapshot.update(
            {
                "user_id": user.id,
                "user_name": user.slug,
            }
        )
    if provider is not None:
        provider_type = getattr(getattr(provider, "config", None), "type", None)
        if provider_type is not None and hasattr(provider_type, "value"):
            provider_type = provider_type.value
        snapshot.update(
            {
                "provider_id": provider.id,
                "provider_name": provider.name,
                "provider_type": provider_type,
            }
        )
    if api_key is not None:
        snapshot.update(
            {
                "api_key_id": api_key.id,
                "api_key_name": api_key.name,
                "access_key": api_key.access_key,
                "api_key_is_custom": api_key.is_custom,
            }
        )
    if model_route_id is not None:
        snapshot["model_route_id"] = model_route_id
        snapshot["model_route_name"] = model_route_name
    return snapshot
