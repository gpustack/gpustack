from typing import Optional

from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.model_provider import ModelProvider
from gpustack.schemas.models import Model
from gpustack.schemas.users import User
from gpustack.utils.api_keys import get_masked_api_key_value


def format_model_snapshot_label(
    model_name: str,
    cluster_name: Optional[str] = None,
) -> str:
    """Return a human-readable display label for a model usage snapshot.

    Format: ``"<cluster_name> / <model_name>"`` when a cluster is given,
    otherwise just ``"<model_name>"``.
    """
    if cluster_name:
        return f"{cluster_name} / {model_name}"
    return model_name


def format_usage_model_label(
    model_name: Optional[str],
    cluster_name: Optional[str] = None,
    provider_name: Optional[str] = None,
) -> str:
    """Return the display label for usage records grouped by model.

    Provider-routed usage is identified by provider + model name. Directly
    deployed usage is identified by cluster + model name.
    """
    if provider_name and model_name:
        parts = [provider_name, model_name]
        if cluster_name:
            parts.insert(0, cluster_name)
        return " / ".join(parts)
    if model_name:
        return format_model_snapshot_label(model_name, cluster_name)
    return "Unknown Model"


def format_usage_user_label(user_name: Optional[str]) -> str:
    return user_name or "Unknown User"


def format_usage_api_key_label(
    user_name: Optional[str] = None,
    api_key_name: Optional[str] = None,
    access_key: Optional[str] = None,
    api_key_is_custom: Optional[bool] = None,
) -> str:
    parts = []
    if user_name:
        parts.append(user_name)
    if api_key_name:
        parts.append(api_key_name)
    if access_key:
        parts.append(get_masked_api_key_value(access_key, api_key_is_custom))
    else:
        parts.append("Unknown API Key")
    return " / ".join(parts) if parts else "Unknown API Key"


def build_model_usage_snapshot(
    model: Model,
    cluster_name: Optional[str] = None,
    user: Optional[User] = None,
    api_key: Optional[ApiKey] = None,
    provider: Optional[ModelProvider] = None,
) -> dict:
    """Build a usage snapshot dict capturing the model identity at request time.

    Records model ID, name, and cluster; optionally includes user and API key
    fields when the caller supplies them. The snapshot is stored alongside usage
    records so usage can be attributed even after models or keys are deleted.

    ``cluster_name`` is resolved from ``model.cluster`` when not passed
    explicitly.
    """
    if cluster_name is None:
        cluster = getattr(model, "cluster", None)
        cluster_name = None if cluster is None else cluster.name

    snapshot = {
        "model_id": model.id,
        "model_name": model.name,
        "cluster_name": cluster_name,
    }
    if user is not None:
        snapshot.update(
            {
                "user_id": user.id,
                "user_name": user.username,
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
    return snapshot
