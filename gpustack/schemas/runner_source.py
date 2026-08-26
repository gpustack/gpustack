import dataclasses
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

from sqlalchemy import Column, Text
from sqlmodel import SQLModel, Field as SQLField
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack_runner import (
    BackendRunners,
    Runners,
    ServiceRunners,
    list_runners,
)
from gpustack_runner.runner import (
    Runner,
    build_backend_runners,
    build_service_runners,
)

from gpustack.mixins import BaseModelMixin
from .inference_backend import (
    built_in_backend_names_by_service,
    deployed_backend_versions,
)
from .source import SourceContent, SourceMixin, SourceTypeEnum
from .common import PaginatedList

logger = logging.getLogger(__name__)

# Natural key across the packaged catalog and the official layer; two rows
# sharing it describe the same slot, official wins (see ``merged_runners``).
_KEY_FIELDS: Tuple[str, ...] = (
    "backend",
    "backend_version",
    "backend_variant",
    "service",
    "service_version",
    "platform",
)

# All packaged ``Runner`` fields in a stable order, to serialize an override row
# back to canonical JSON for a source's ``content``.
_RUNNER_FIELDS: Tuple[str, ...] = (
    "backend",
    "backend_version",
    "original_backend_version",
    "backend_variant",
    "service",
    "service_version",
    "platform",
    "docker_image",
    "deprecated",
)

# Exact-match filter keys the merge helpers accept (the ``list_runners`` subset
# the three consumers use); ``*_prefix`` filters are rejected, not ignored.
_FILTER_KEYS = frozenset(_KEY_FIELDS)


class RunnerOverrideEntryBase(SQLModel):
    """A single per-platform runner override row, aligned field-for-field
    with the packaged ``gpustack_runner.Runner`` dataclass."""

    backend: str = SQLField(index=True)
    backend_version: str = SQLField(default="")
    original_backend_version: str = SQLField(default="")
    backend_variant: str = SQLField(default="")
    service: str = SQLField(index=True)
    service_version: str = SQLField(default="")
    platform: str = SQLField(default="")
    docker_image: str = SQLField(sa_column=Column(Text, nullable=False))
    deprecated: bool = SQLField(default=False)
    # Which source produced this row (last writer wins a key), same shape as
    # ``CatalogModelEntry``. Workers merge overrides without a DB session, so the
    # replace-or-layer decision has to travel with the entries themselves.
    source_name: str = SQLField(default="")
    source_type: SourceTypeEnum = SQLField(default=SourceTypeEnum.BUILTIN)
    # Reserved for future multi-tenancy; excluded from serialized responses.
    owner_principal_id: Optional[int] = SQLField(default=None, exclude=True)


class RunnerOverrideEntry(RunnerOverrideEntryBase, BaseModelMixin, table=True):
    __tablename__ = "runner_override_entries"
    id: Optional[int] = SQLField(default=None, primary_key=True)


class RunnerOverrideEntryPublic(RunnerOverrideEntryBase):
    id: Optional[int]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]


RunnerOverrideEntriesPublic = PaginatedList[RunnerOverrideEntryPublic]


# The admin's row of the runner source table. Runner has no BUILTIN row (its
# baseline is the in-code ``list_runners()`` catalog); OFFICIAL uses the shared
# ``OFFICIAL_SOURCE_NAME``.
CUSTOM_RUNNER_SOURCE_NAME = "custom"


class InferenceRunnerSource(SourceMixin, BaseModelMixin, table=True):
    """A source of runner overrides; the leader reconciles all enabled rows into
    ``RunnerOverrideEntry``. FILE holds an inline JSON array, URL fetches one;
    ``content`` is validated by ``normalize_runner_json``.
    """

    __tablename__ = "inference_runner_sources"
    id: Optional[int] = SQLField(default=None, primary_key=True)


def _runner_key(obj) -> Tuple:
    """Natural key for either a packaged ``Runner`` or a ``RunnerOverrideEntry``."""
    return tuple(getattr(obj, field) for field in _KEY_FIELDS)


def _to_runner(entry: RunnerOverrideEntry) -> Runner:
    """Convert an override row into a packaged ``Runner`` (lossless 1:1)."""
    return Runner(
        backend=entry.backend,
        backend_version=entry.backend_version,
        original_backend_version=entry.original_backend_version,
        backend_variant=entry.backend_variant,
        service=entry.service,
        service_version=entry.service_version,
        platform=entry.platform,
        docker_image=entry.docker_image,
        deprecated=entry.deprecated,
    )


def _entry_from_runner(runner: Runner) -> RunnerOverrideEntry:
    """Inverse of ``_to_runner``: a packaged ``Runner`` → an override row."""
    return RunnerOverrideEntry(
        backend=runner.backend,
        backend_version=runner.backend_version,
        original_backend_version=runner.original_backend_version,
        backend_variant=runner.backend_variant,
        service=runner.service,
        service_version=runner.service_version,
        platform=runner.platform,
        docker_image=runner.docker_image,
        deprecated=runner.deprecated,
    )


def _filter_overrides(runners: Runners, **filters) -> Runners:
    """Apply the same exact-match filtering ``list_runners`` would, so
    overrides for an unrelated backend/service don't leak into a scoped query."""
    with_deprecated = filters.pop("with_deprecated", True)
    if with_deprecated is None:
        with_deprecated = True

    prefix_keys = [key for key in filters if key.endswith("_prefix")]
    if prefix_keys:
        raise ValueError(f"prefix filters are unsupported for overrides: {prefix_keys}")

    result: Runners = []
    for runner in runners:
        if not with_deprecated and runner.deprecated:
            continue
        if all(
            getattr(runner, key) == expected
            for key, expected in filters.items()
            if expected is not None and key in _FILTER_KEYS
        ):
            result.append(runner)
    return result


def merged_runners(overrides: List[RunnerOverrideEntry], **filters) -> Runners:
    """The runner catalog these override rows come to.

    - Whole-content replacement, as ``order_source_contents`` does it for the other
      two kinds: the rows stand in for the packaged catalog rather than layering
      over it per key, which is what lets a coordinate be *withdrawn*.
    - Decided here rather than in ``order_source_contents`` because runner's
      baseline is in-code, not a BUILTIN source row.
    - Decided on the *unfiltered* rows: a scoped query the document has nothing for
      comes back empty instead of falling back to images it replaced.
    """
    if overrides:
        return _filter_overrides(
            [_to_runner(override) for override in overrides], **filters
        )
    return list_runners(**filters)  # process-wide @lru_cache, static baseline


def merged_backend_runners(
    overrides: List[RunnerOverrideEntry], **filters
) -> BackendRunners:
    """Backend-grouped view of the merged runners (reuses the package grouping)."""
    return build_backend_runners(merged_runners(overrides, **filters))


def merged_service_runners(
    overrides: List[RunnerOverrideEntry], **filters
) -> ServiceRunners:
    """Service-grouped view of the merged runners (reuses the package grouping)."""
    return build_service_runners(merged_runners(overrides, **filters))


# Fields a runner-override entry must carry non-empty; the rest
# (backend_version / original_backend_version / backend_variant) default to "".
_REQUIRED_ENTRY_FIELDS: Tuple[str, ...] = (
    "backend",
    "service",
    "service_version",
    "platform",
    "docker_image",
)


def _parse_runner_json(raw: Optional[str]) -> List[RunnerOverrideEntry]:
    """Parse and validate runner-override JSON against the packaged ``Runner``
    schema, returning unsaved ``RunnerOverrideEntry`` rows. Missing required
    fields are rejected by entry index; raises ``ValueError``.

    A field this version does not know is dropped rather than rejected: every
    cluster reads the same published document, so rejecting the whole thing over
    one added key would freeze runner updates on every installation older than
    the one that added it.
    """
    try:
        raw_entries = json.loads(raw or "")
    except (TypeError, json.JSONDecodeError) as e:
        raise ValueError(f"content is not valid JSON: {e}")
    if not isinstance(raw_entries, list):
        raise ValueError("content must be a JSON array of runner entries")

    known_fields = {field.name for field in dataclasses.fields(Runner)}
    # Collected across the whole document: one added key lands on every entry,
    # and a published catalog carries hundreds of them.
    unknown_fields: Set[str] = set()
    entries: List[RunnerOverrideEntry] = []
    for index, item in enumerate(raw_entries):
        if not isinstance(item, dict):
            raise ValueError(f"entry #{index} must be a JSON object")
        unknown_fields.update(key for key in item if key not in known_fields)
        missing = [key for key in _REQUIRED_ENTRY_FIELDS if not item.get(key)]
        if missing:
            raise ValueError(
                f"entry #{index} is missing required field(s): {', '.join(missing)}"
            )
        # Fill the optional fields so the packaged Runner validates, then convert
        # 1:1 to an override row.
        runner = Runner(
            backend=item["backend"],
            backend_version=item.get("backend_version", ""),
            original_backend_version=item.get("original_backend_version", ""),
            backend_variant=item.get("backend_variant", ""),
            service=item["service"],
            service_version=item["service_version"],
            platform=item["platform"],
            docker_image=item["docker_image"],
            deprecated=bool(item.get("deprecated", False)),
        )
        entries.append(_entry_from_runner(runner))
    if unknown_fields:
        logger.warning(
            f"Ignoring runner field(s) this version does not know: "
            f"{', '.join(sorted(unknown_fields))}. The document was published "
            f"for a newer GPUStack."
        )
    return entries


def normalize_runner_json(raw: Optional[str]) -> str:
    """Validate raw runner JSON and return the canonical text stored in a
    source's ``content`` (the ``normalize`` the probe applies). Raises
    ``ValueError`` on malformed input; otherwise re-serializes to a stable form.
    """
    entries = _parse_runner_json(raw)
    return json.dumps(
        [_entry_to_dict(entry) for entry in entries], ensure_ascii=False, sort_keys=True
    )


def _entry_to_dict(entry: RunnerOverrideEntry) -> Dict:
    """Canonical dict form of an override row (the packaged Runner fields)."""
    return {field: getattr(entry, field) for field in _RUNNER_FIELDS}


def _entries_from_sources(sources: List[SourceContent]) -> List[RunnerOverrideEntry]:
    """The override rows the ordered sources produce: parsed, stamped with their
    source, merged by natural key (later source wins).

    Shared by the reconcile and the pre-write check, so both answer "what does
    this set of sources come to" identically — a check modelling the merge
    differently would pass writes the reconcile then makes unschedulable.

    An unparseable source is skipped, not fatal to the kind — the others still
    serve. All of them unreadable raises instead: an empty result is what a
    document deliberately carrying nothing looks like, and the caller would clear
    the override table over it.
    """
    merged: Dict[Tuple, RunnerOverrideEntry] = {}
    skipped: List[str] = []
    for source in sources:
        try:
            entries = _parse_runner_json(source.content)
        except ValueError as e:
            logger.error(f"Skipping unreadable runner source {source.name}: {e}")
            skipped.append(source.name)
            continue
        for entry in entries:
            entry.source_name = source.name
            entry.source_type = source.source_type
            merged[_runner_key(entry)] = entry
    if skipped and len(skipped) == len(sources):
        raise ValueError(f"no runner source could be read: {', '.join(skipped)}")
    return list(merged.values())


def runner_versions(overrides: List[RunnerOverrideEntry]) -> Set[Tuple[str, str]]:
    """The ``(service, service_version)`` coordinates these overrides leave
    available, resolved exactly as ``merged_runners`` resolves them — the set a
    deployment has to find itself in (``service`` is a built-in backend's name,
    ``service_version`` the version a model pins).

    Pass the stored rows for what is available now, or ``_entries_from_sources``
    of a proposed write for what it would leave; the latter is how the pre-write
    check asks before the write lands.
    """
    return {
        (runner.service, runner.service_version) for runner in merged_runners(overrides)
    }


def proposed_runner_versions(sources: List[SourceContent]) -> Set[Tuple[str, str]]:
    """``runner_versions`` for a write that has not landed yet."""
    return runner_versions(_entries_from_sources(sources))


async def _pinned_coordinates(
    session: AsyncSession,
) -> Dict[Tuple[str, str], List[str]]:
    """The ``(service, service_version)`` coordinates a live deployment pins, to
    the models pinning them: ``deployed_backend_versions`` across the same name
    bridge the pre-write check crosses (a model says ``vLLM``, a runner
    ``vllm``)."""
    service_by_backend = {
        backend_name: service
        for service, backend_name in built_in_backend_names_by_service().items()
    }
    return {
        (service_by_backend[backend_name], version): models
        for (backend_name, version), models in (
            await deployed_backend_versions(session)
        ).items()
        if backend_name in service_by_backend
    }


async def reconcile_runner_overrides(
    session: AsyncSession, sources: List[SourceContent]
) -> None:
    """Rewrite ``RunnerOverrideEntry`` from the ordered sources: each is parsed and
    merged by natural key (later source wins), then the table is replaced
    atomically. Empty input clears it, falling the cluster back to the packaged
    catalog.

    Coordinates a live deployment pins survive a document that dropped them, so
    an upstream mistake cannot leave a pinned model with no image; the rest of
    the document still applies, and the survivor's rows go on the first reconcile
    after its deployment does. Whole-group per coordinate: one
    ``(service, service_version)`` spans a row per platform / variant, and half of
    one resolves for some hardware and fails for the rest. Never on empty input,
    where the baseline taking over is complete and a single surviving row would
    stop ``merged_runners`` falling back to it at all.

    Nothing readable raises before any write, rather than making every model
    pinned to a remote-only version unschedulable.
    """
    entries = _entries_from_sources(sources)
    existing_rows = await RunnerOverrideEntry.all(session)
    # Off the rows, not ``runner_versions``: that falls back to the package on an
    # empty table, so every packaged coordinate would read as dropped.
    proposed = {(entry.service, entry.service_version) for entry in entries}
    dropped = {(row.service, row.service_version) for row in existing_rows} - proposed
    survivors: Dict[Tuple[str, str], List[str]] = {}
    if entries and dropped:
        pinned = await _pinned_coordinates(session)
        survivors = {
            coordinate: models
            for coordinate, models in pinned.items()
            if coordinate in dropped
        }
        for (service, version), models in sorted(survivors.items()):
            logger.warning(
                f"Runner source no longer carries {service} {version}, kept because "
                f"it is deployed by: {', '.join(sorted(models))}. Migrate those "
                f"models off it; the coordinate goes away once they are gone."
            )
    # Delete + insert in one transaction so a partial failure never empties the
    # table. Nothing subscribes to RunnerOverrideEntry, so no watch event.
    for existing in existing_rows:
        if (existing.service, existing.service_version) in survivors:
            continue
        await existing.delete(session, auto_commit=False)
    for entry in entries:
        await RunnerOverrideEntry.create(session, entry, auto_commit=False)
    await session.commit()
