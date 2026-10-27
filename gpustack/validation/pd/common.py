"""Resolve submitted PD fields without changing sparse-update semantics."""

from typing import Any, Optional

from gpustack.schemas.models import Model, ModelSpecBase


def model_field(model: ModelSpecBase, stored: Optional[Model], name: str) -> Any:
    """Read a submitted field, falling back to storage only when omitted.

    Args:
        model: The submitted deployment specification.
        stored: The stored deployment supplying omitted fields.
        name: The field to read.

    Returns:
        The effective field value, including an explicitly submitted null.
    """
    value = getattr(model, name, None)
    if value is not None:
        return value
    if stored is not None and name not in getattr(model, "model_fields_set", set()):
        return getattr(stored, name, None)
    return value


def effective_model_spec(
    model: ModelSpecBase, stored: Optional[Model]
) -> ModelSpecBase:
    """Build a read-only validation view without adding fields to the patch.

    Args:
        model: The submitted deployment specification.
        stored: The stored deployment supplying omitted update fields.

    Returns:
        The effective specification, retaining explicitly submitted nulls.
    """
    if stored is None:
        return model
    return model.model_copy(
        update={
            name: model_field(model, stored, name) for name in type(model).model_fields
        }
    )
