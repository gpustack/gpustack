"""Shared Hybrid-scope backend definition merging."""

from typing import Optional

from gpustack.schemas.inference_backend import InferenceBackend, VersionConfigDict

_IDENTITY_FIELDS = {
    'backend_name',
    'owner_principal_id',
    'version_configs',
    'id',
    'created_at',
    'updated_at',
    'deleted_at',
}


def merge_backend_definitions(
    platform: Optional[InferenceBackend], owner: Optional[InferenceBackend]
) -> Optional[InferenceBackend]:
    """Merge one owner's overrides over Platform defaults without mutating rows."""
    if owner is None:
        return platform
    if platform is None:
        return owner
    # A pydantic copy of a table model shares its SQLAlchemy InstanceState.
    # Revalidation creates a transient object that cannot dirty the source row.
    merged = InferenceBackend.model_validate(owner.model_dump())
    for name in type(owner).model_fields:
        if name not in _IDENTITY_FIELDS and getattr(merged, name) is None:
            setattr(merged, name, getattr(platform, name))
    # Shared catalogs use union semantics: an Org's disabled override must not
    # hide a Platform-enabled backend or its shared versions.
    merged.enabled = bool(owner.enabled) or bool(platform.enabled)
    merged.is_built_in = owner.is_built_in or platform.is_built_in
    merged.version_configs = VersionConfigDict(
        root={
            **(platform.version_configs.root if platform.version_configs else {}),
            **(owner.version_configs.root if owner.version_configs else {}),
        }
    )
    return merged
