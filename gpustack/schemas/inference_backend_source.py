import logging
from typing import Dict, List, Optional, Set, Tuple

import yaml
from pydantic import ValidationError
from sqlalchemy.orm.attributes import flag_modified
from sqlmodel import Field as SQLField
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.mixins import BaseModelMixin
from gpustack.schemas.inference_backend import (
    InferenceBackend,
    InferenceBackendBase,
    VersionConfig,
    VersionConfigDict,
    is_built_in_backend,
)
from gpustack.schemas.models import BackendSourceEnum
from gpustack.schemas.source import (
    SourceContent,
    SourceMixin,
    SourceTypeEnum,
    validate_icon,
)

logger = logging.getLogger(__name__)

# Name of the BUILTIN source seeded by InferenceBackendController from the
# packaged community-inference-backends.yaml.
BUILTIN_BACKEND_SOURCE_NAME = "builtin"

# The single custom InferenceBackendSource an admin configures via
# ``/v2/ota-sources/community-backend`` (many rows, so more sources later need no
# schema change).
CUSTOM_BACKEND_SOURCE_NAME = "custom"


class InferenceBackendSource(SourceMixin, BaseModelMixin, table=True):
    """A source of community inference backends; the leader reconciles all
    enabled rows into the Platform-NULL community rows of ``inference_backends``
    (see ``reconcile_backend``), smart-merging so user modifications survive a
    hot update. ``content`` is normalized YAML validated by
    ``normalize_backend_yaml``.
    """

    __tablename__ = "inference_backend_sources"
    id: Optional[int] = SQLField(default=None, primary_key=True)


# What ``_upsert_community_backend`` writes into an ``inference_backends`` row.
_MATERIALIZED_KEYS = (
    "backend_name",
    "version_configs",
    "default_version",
    "default_backend_param",
    "default_run_command",
    "default_entrypoint",
    "health_check_path",
    "description",
    "icon",
    "default_env",
    "parameter_format",
    "common_parameters",
)

# What a document may carry. Wider than what is materialized: the packaged
# document is dumped from the read API, so it brings two fields the write side
# drops on purpose — ``is_built_in`` (a community card must not be able to claim
# built-in status) and ``framework_index_map`` (the read route derives it per
# request). Recognized here, or the scan below reports GPUStack's own document
# as published for a newer GPUStack.
_RECOGNIZED_KEYS = _MATERIALIZED_KEYS + ("is_built_in", "framework_index_map")


def _normalized_version_configs(raw_version_configs: dict) -> VersionConfigDict:
    """The ``version_configs`` the materialization writes: every version a source
    carries is predefined, so its framework info moves into
    ``built_in_frameworks`` and ``custom_framework`` is cleared.

    Shared with ``_reject_unmodelable_card`` so the strict pre-check refuses
    exactly the cards the reconcile skips. Copies rather than mutating — that
    check runs on the dict about to be re-serialized as stored content.
    """
    normalized: Dict[str, VersionConfig] = {}
    for version, ver_config in raw_version_configs.items():
        frameworks = None
        if "built_in_frameworks" in ver_config:
            frameworks = ver_config["built_in_frameworks"]
        elif ver_config.get("custom_framework"):
            frameworks = [ver_config["custom_framework"]]
        normalized[version] = VersionConfig(
            **{
                **ver_config,
                "built_in_frameworks": (
                    (frameworks if isinstance(frameworks, list) else [frameworks])
                    if frameworks
                    else []
                ),
                "custom_framework": None,
            }
        )
    return VersionConfigDict(root=normalized)


class UnmodelableCard(ValueError):
    """A card this version cannot model. Its own type because pydantic's
    ``ValidationError`` is a ``ValueError`` too, and the write path may raise one —
    ``reconcile_backend`` must not mistake that for a card it can skip.
    """


def _reject_unmodelable_card(config: dict, name: str) -> None:
    """Whether the materialization can model this card, as one error type."""
    backend_data = {key: config[key] for key in _MATERIALIZED_KEYS if key in config}
    try:
        if backend_data.get("version_configs"):
            backend_data["version_configs"] = _normalized_version_configs(
                backend_data["version_configs"]
            )
        InferenceBackendBase.model_validate(backend_data)
    except (ValidationError, TypeError) as e:
        raise UnmodelableCard(f"backend '{name}' is unreadable: {e}")


def _parse_backend_yaml(raw: Optional[str], strict: bool = False) -> List[dict]:
    """Parse and validate community-backend YAML, returning the backend config
    dicts. Raises ``ValueError`` on any problem (→ HTTP 400, source not stored).

    Built-in engine names are reserved: ``_upsert_community_backend`` resolves by
    ``backend_name`` alone, so a document naming one would take over that engine's
    row and re-stamp it COMMUNITY. Refused here, whole document at a time.

    A field this version does not know is kept in the stored content (so an
    upgrade picks it up) and warned about once per document rather than rejected.
    ``strict`` raises on it, and on a card the materialization would skip.
    """
    try:
        data = yaml.safe_load(raw or "")
    except yaml.YAMLError as e:
        raise ValueError(f"content is not valid YAML: {e}")
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError("content must be a YAML list of backend configs")
    unknown_fields: Set[str] = set()
    for index, config in enumerate(data):
        if not isinstance(config, dict):
            raise ValueError(f"entry #{index} must be a mapping")
        # ``str``: a YAML key can be an int, and the report below sorts them.
        unknown_fields.update(str(key) for key in config if key not in _RECOGNIZED_KEYS)
        name = config.get("backend_name")
        if not name:
            raise ValueError(f"entry #{index} is missing backend_name")
        if is_built_in_backend(name):
            raise ValueError(
                f"backend '{name}' is a built-in engine and cannot be defined by "
                "a community backend document"
            )
        version_configs = config.get("version_configs")
        if not version_configs or not isinstance(version_configs, dict):
            raise ValueError(f"backend '{name}' has no version_configs")
        for version, ver_config in version_configs.items():
            if not isinstance(ver_config, dict) or not ver_config.get("image_name"):
                raise ValueError(
                    f"backend '{name}' version '{version}' is missing image_name"
                )
            unknown_fields.update(
                str(key) for key in ver_config if key not in VersionConfig.model_fields
            )
        if strict:
            _reject_unmodelable_card(config, name)
    if unknown_fields:
        listed = ", ".join(sorted(unknown_fields))
        if strict:
            raise ValueError(f"unknown community backend field(s): {listed}")
        logger.warning(
            f"Ignoring community backend field(s) this version does not know: "
            f"{listed}. The document was published for a newer GPUStack."
        )
    return data


def normalize_backend_yaml(raw: Optional[str], strict: bool = False) -> str:
    """Validate raw community-backend YAML and return the canonical text stored
    in a source's ``content`` (the ``normalize`` for ``COMMUNITY_BACKEND_SPEC``).
    Raises ``ValueError`` on malformed input; otherwise re-serializes to a stable form.

    ``strict`` names what would be dropped instead of dropping it — for the
    configuration API only (see ``SourceConfigSpec.normalize``).
    """
    configs = _parse_backend_yaml(raw, strict)
    for config in configs:
        if config.get("icon"):
            config["icon"] = validate_icon(config["icon"])
    return yaml.safe_dump(configs, sort_keys=True, allow_unicode=True)


async def _upsert_community_backend(
    session: AsyncSession,
    config: dict,
    source_name: str,
    source_type: SourceTypeEnum,
) -> None:
    """Create or update a single Platform-NULL community backend, smart-merging
    to preserve user customizations: ``enabled``, custom versions
    (``built_in_frameworks is None``) and ``default_env`` (source unions the
    user's). The source stamp is always refreshed to the writer.

    Raises ``UnmodelableCard`` before touching the session for a config this
    version cannot model — how ``reconcile_backend`` isolates one card.

    Uncommitted: ``reconcile_backend`` commits the whole materialization at once.
    """
    backend_name = config.get("backend_name")
    if not backend_name:
        return

    # Before any session work: a table model skips validation, so an unmodelable
    # value would be written and only raise on the next read.
    _reject_unmodelable_card(config, backend_name)

    backend_data = {k: config[k] for k in _MATERIALIZED_KEYS if k in config}
    backend_data["backend_source"] = BackendSourceEnum.COMMUNITY
    backend_data["enabled"] = False
    backend_data["source_name"] = source_name
    backend_data["source_type"] = source_type

    if backend_data.get("version_configs"):
        backend_data["version_configs"] = _normalized_version_configs(
            backend_data["version_configs"]
        )

    existing = await InferenceBackend.one_by_fields(
        session, {"backend_name": backend_name, "owner_principal_id": None}
    )
    if not existing:
        await InferenceBackend.create(
            session, InferenceBackend(**backend_data), auto_commit=False
        )
        return

    # Merge version_configs: source versions win on a shared key, the user's
    # custom versions (absent from the source) are preserved.
    if backend_data.get("version_configs"):
        source_versions = backend_data["version_configs"].root
        existing_versions = (
            existing.version_configs.root if existing.version_configs else {}
        )
        merged_versions = dict(source_versions)
        for version, ver_config in existing_versions.items():
            if (
                ver_config.built_in_frameworks is None
                and version not in source_versions
            ):
                merged_versions[version] = ver_config
        backend_data["version_configs"] = VersionConfigDict(root=merged_versions)

    # Preserve a user-enabled backend; merge user-added default_env.
    if existing.enabled:
        backend_data["enabled"] = True
    if existing.default_env:
        if backend_data.get("default_env"):
            merged_env = dict(existing.default_env)
            merged_env.update(backend_data["default_env"])
            backend_data["default_env"] = merged_env
        else:
            backend_data["default_env"] = existing.default_env

    await existing.update(session, backend_data, auto_commit=False)


def _merge_backend_sources(
    sources: List[SourceContent],
) -> Tuple[Dict[str, dict], Dict[str, SourceContent]]:
    """Merge backend configs from the (stable-ordered) sources by
    ``backend_name``: a later source wins the card, ``version_configs`` union
    (later wins a shared version). Shared by ``reconcile_backend`` and
    ``compute_disappearing_backend_versions`` so the two never drift.

    An unparseable source is skipped, not fatal to the kind — the others still
    serve. All of them unreadable raises instead: an empty union is what a document
    deliberately publishing nothing looks like, and the caller would delete or
    downgrade every community backend over it.
    """
    merged: Dict[str, dict] = {}
    origin: Dict[str, SourceContent] = {}
    skipped: List[str] = []
    for source in sources:
        try:
            configs = _parse_backend_yaml(source.content)
        except ValueError as e:
            logger.error(f"Skipping unreadable backend source {source.name}: {e}")
            skipped.append(source.name)
            continue
        for config in configs:
            name = config["backend_name"]
            # Stamp per-version source origin so a later source overriding a
            # shared version carries its own source into the merged card.
            for ver_config in (config.get("version_configs") or {}).values():
                ver_config["source_name"] = source.name
                ver_config["source_type"] = source.source_type
            previous = merged.get(name)
            if previous is None:
                merged[name] = config
            else:
                union_versions = {
                    **(previous.get("version_configs") or {}),
                    **(config.get("version_configs") or {}),
                }
                combined = {**previous, **config, "version_configs": union_versions}
                merged[name] = combined
            origin[name] = source
    if skipped and len(skipped) == len(sources):
        raise ValueError(f"no backend source could be read: {', '.join(skipped)}")
    return merged, origin


def compute_disappearing_backend_versions(
    sources: List[SourceContent],
    current_versions: Set[Tuple[str, str]],
) -> Set[Tuple[str, str]]:
    """Recompute the merged ``(backend_name, version)`` union from the proposed
    enabled sources (via the same merge ``reconcile_backend`` uses) and return
    the subset of ``current_versions`` that would disappear. ``current_versions``
    is the caller's snapshot before the change.
    """
    merged, _ = _merge_backend_sources(sources)
    proposed_versions: Set[Tuple[str, str]] = {
        (name, version)
        for name, config in merged.items()
        for version in (config.get("version_configs") or {})
    }
    return current_versions - proposed_versions


def _carries_user_state(backend: InferenceBackend) -> bool:
    """Whether the cleanup must keep a dropped backend rather than delete it:
    it is enabled (something may be serving on it), or it carries versions the
    user added by hand (``built_in_frameworks is None``).

    The hand-added versions need naming here because the source-owned in-use
    check (``_source_owned_versions``) deliberately skips them — correct for a
    kept row, whose versions the downgrade preserves, but a deleted row takes
    them away with no check having looked.
    """
    if backend.enabled:
        return True
    version_configs = backend.version_configs.root if backend.version_configs else {}
    return any(
        version_config.built_in_frameworks is None
        for version_config in version_configs.values()
    )


async def reconcile_backend(
    session: AsyncSession, sources: List[SourceContent]
) -> None:
    """Materialize the Platform-NULL community backends from all enabled sources.
    Configs merge by ``backend_name`` (stable order, later wins the card,
    versions union); each is upserted preserving user state, then backends no
    longer in any source are downgraded to custom (if they carry user state) or
    deleted. Org-private and built-in engine rows are never touched.

    Upserts and cleanup land in one transaction, as in the other two reconciles: a
    part-way failure must not leave community backends half-materialized. Nothing
    readable raises before any write instead of deleting all of them, and a single
    card this version cannot model costs itself rather than the round.
    """
    merged, origin = _merge_backend_sources(sources)

    for name, config in merged.items():
        try:
            await _upsert_community_backend(
                session, config, origin[name].name, origin[name].source_type
            )
        except UnmodelableCard as e:
            # Raised before the session is touched. A write failure has already
            # rolled the round back and must not be continued past, so it keeps its
            # own type. The card stays in ``union_names``, so its stored row survives.
            logger.error(f"Skipping unusable community backend {name}: {e}")

    union_names = set(merged.keys())
    for backend in await InferenceBackend.all(session):
        if (
            backend.backend_source != BackendSourceEnum.COMMUNITY
            or backend.owner_principal_id is not None
            or backend.backend_name in union_names
        ):
            continue
        if _carries_user_state(backend):
            # Convert built-in-framework versions to custom so the user's
            # deployments keep working after the source drops the backend.
            converted_versions = {}
            if backend.version_configs and backend.version_configs.root:
                for version, ver_config in backend.version_configs.root.items():
                    config_data = ver_config.model_dump()
                    if config_data.get("built_in_frameworks"):
                        config_data["custom_framework"] = config_data[
                            "built_in_frameworks"
                        ][0]
                        config_data["built_in_frameworks"] = None
                    converted_versions[version] = VersionConfig(**config_data)
            flag_modified(backend, "version_configs")
            await backend.update(
                session,
                {
                    "backend_source": BackendSourceEnum.CUSTOM,
                    "enabled": False,
                    "version_configs": VersionConfigDict(root=converted_versions),
                },
                auto_commit=False,
            )
        else:
            await backend.delete(session, auto_commit=False)

    await session.commit()
    # ``delete(auto_commit=False)`` skips the ``cached_all`` invalidation (as in
    # ``services.py``); the upserts above already invalidate on flush.
    await InferenceBackend._invalidate_cached_all()
