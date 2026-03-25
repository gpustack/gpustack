from typing import Optional

from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.models import Model
from gpustack.schemas.users import User


def format_model_snapshot_key(
    model_name: str,
    cluster_name: Optional[str] = None,
) -> str:
    """Return a dict-key string for a model usage snapshot.

    Format: ``"<cluster_name>/<model_name>"`` when a cluster is given,
    otherwise just ``"<model_name>"``.
    """
    if cluster_name:
        return f"{cluster_name}/{model_name}"
    return model_name


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


def build_model_usage_snapshot(
    model: Model,
    cluster_name: Optional[str] = None,
    user: Optional[User] = None,
    api_key: Optional[ApiKey] = None,
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
