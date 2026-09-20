"""The cache-provider catalog as a content source.

The catalog an installation serves is a packaged asset plus whatever assets
installed plugins contribute. This module lets an admin replace it with a
document of their own: one row holds that document, another holds the packaged
baseline, and the serving catalog is derived from the two
(``server/cache_provider_catalog.py``).

``normalize_cache_provider_yaml`` is the validator the source layer calls before
storing anything. It is deliberately stricter than the loader: a declaration the
loader skips with a log line is, in a document the admin just pasted, a mistake
they want named.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set, Type, Union, get_args, get_origin

import yaml
from pydantic import BaseModel, ValidationError
from sqlalchemy import JSON, Column, UniqueConstraint
from sqlmodel import SQLModel, Field as SQLField
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.mixins import BaseModelMixin

from .cache_providers import (
    CacheProvider,
    validate_injection_templates,
    with_runner_versions,
    validate_localized_text,
    validate_template_filters,
)
from .runner_source import RunnerOverrideEntry, merged_runners
from .source import SourceContent, SourceMixin, SourceTypeEnum

logger = logging.getLogger(__name__)

# The packaged baseline: the bundled asset merged with every plugin's, seeded by
# the leader on start and offered in the UI as the document to edit from.
BUILTIN_CACHE_PROVIDER_SOURCE_NAME = "builtin"

# The admin's own document, configured through ``/v2/ota-sources/cache-provider``.
CUSTOM_CACHE_PROVIDER_SOURCE_NAME = "custom"


class CacheProviderSource(SourceMixin, BaseModelMixin, table=True):
    """A source of cache-provider catalog content: a document pasted in, or one
    fetched from an address of the admin's own.

    Nothing publishes this catalog, so there is no official slot to follow and
    no cadence — a URL source is read when it is written and again when it is
    reloaded, and ``auto_update_hours`` stays 0."""

    __tablename__ = "cache_provider_sources"
    id: Optional[int] = SQLField(default=None, primary_key=True)


class CacheProviderEntryBase(SQLModel):
    """One materialized provider declaration. The declaration itself is stored
    as a JSON ``payload``; only what a query needs is promoted to a column."""

    name: str = SQLField(index=True)
    # Catalog order, which is the order the cards appear in — the document's own
    # order, and the reason this is a column rather than an ORDER BY id: a
    # document that reorders its declarations keeps the ids it already has.
    position: int = SQLField(default=0)
    payload: Dict[str, Any] = SQLField(sa_column=Column(JSON), default_factory=dict)
    # Which source produced this record (last writer wins a name). The packaged
    # baseline carries it too; the UI shows no badge for a built-in one.
    source_name: str = SQLField(default="")
    source_type: SourceTypeEnum = SQLField(default=SourceTypeEnum.BUILTIN)


class CacheProviderEntry(CacheProviderEntryBase, BaseModelMixin, table=True):
    __tablename__ = "cache_provider_entries"
    # ``name`` is the upsert key; named explicitly so the constraint the
    # migration creates is droppable by name on every dialect.
    __table_args__ = (UniqueConstraint("name", name="uix_cache_provider_entries_name"),)
    id: Optional[int] = SQLField(default=None, primary_key=True)


def _model_for(annotation: Any) -> Optional[Type[BaseModel]]:
    """The pydantic model a field's annotation carries, through the wrappers a
    declaration uses: ``Optional[X]``, ``List[X]``, ``Dict[str, X]``. ``None``
    when the annotation holds no model, or more than one (a union of models has
    no single set of field names to check against)."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    origin = get_origin(annotation)
    if origin is None:
        return None
    models = [
        model
        for model in (_model_for(arg) for arg in get_args(annotation))
        if model is not None
    ]
    # A union of two models (e.g. a port declared as a name or a mapping) has no
    # unambiguous field set; checking against either one would report the other's
    # keys as unknown.
    if origin is Union and len(models) > 1:
        return None
    return models[0] if len(models) == 1 else None


def _unknown_keys(raw: Any, model: Type[BaseModel], path: str = "") -> Set[str]:
    """Keys the declaration carries that ``model`` has no field for, reported as
    dotted paths, walking into nested declarations.

    A typo does not fail validation — an unknown key is simply ignored, and a
    component whose ``run_command`` was spelled ``run_cmd`` launches with no
    command at all. Naming the key is the only way that mistake is visible.
    """
    if not isinstance(raw, dict):
        return set()
    unknown = {f"{path}{key}" for key in raw if str(key) not in model.model_fields}
    for key, value in raw.items():
        field = model.model_fields.get(str(key))
        if field is None:
            continue
        nested = _model_for(field.annotation)
        if nested is None:
            # Nothing with field names under here: a mapping of strings (env
            # vars, the files an injection writes) carries keys the document
            # invents, not fields to check.
            continue
        for label, item in _nested_declarations(key, value, field.annotation):
            unknown |= _unknown_keys(item, nested, f"{path}{label}.")
    return unknown


def _nested_declarations(key: Any, value: Any, annotation: Any):
    """(label, mapping) pairs to check under ``key``, read off the annotation
    rather than the value.

    Which of the three shapes a field holds — one declaration, a list of them,
    or a mapping of them keyed by a name the document chooses — is what the
    declaration says it is. Guessing from the value cannot tell a mapping of
    declarations from a declaration whose every field happens to be a mapping,
    and an injection carrying only env, files and a transfer config is exactly
    that.
    """
    origin = get_origin(annotation)
    if origin is Union:
        # Optional[X], and "a name or a mapping" alike: the shape is whichever
        # member carries the model.
        for arg in get_args(annotation):
            if arg is type(None):
                continue
            nested = _nested_declarations(key, value, arg)
            if nested:
                return nested
        return []
    if origin in (list, set, tuple):
        if not isinstance(value, list):
            return []
        return [(f"{key}[{index}]", item) for index, item in enumerate(value)]
    if origin is dict:
        if not isinstance(value, dict):
            return []
        return [(f"{key}.{name}", item) for name, item in value.items()]
    return [(str(key), value)] if isinstance(value, dict) else []


_DERIVED_VERSION_KEYS = (
    "versions",
    "default_version",
    "default_image",
    "default_runtime_images",
)


def _reject_mixed_version_sources(name: Any, entry: Any) -> None:
    """A declaration naming ``runner_dependency`` may not also name versions.

    Judged on the document rather than on the model, because it is a rule about
    what an author may write: once the derivation has run, the provider
    legitimately carries both. And it is the only form of the rule that can be
    checked here — the runner catalog the derivation reads is not in hand when a
    document is validated, so "the declared default is among the derived
    versions" is not a question this layer can answer.

    On the values, not on the keys: what a derived provider dumps back carries
    these keys holding nothing, so a document downloaded and saved unchanged
    would be refused by a check that read presence alone.
    """
    if not isinstance(entry, dict) or not entry.get("runner_dependency"):
        return
    declared = [key for key in _DERIVED_VERSION_KEYS if entry.get(key)]
    if declared:
        raise ValueError(
            f"cache provider {name} declares runner_dependency "
            f"'{entry['runner_dependency']}', so its versions come from the "
            f"runner catalog; remove {', '.join(declared)}"
        )


def load_cache_providers_document(
    raw: Optional[str], strict: bool = False
) -> List[CacheProvider]:
    """Parse a catalog document into declarations.

    A declaration this version cannot read is dropped so the rest still serves;
    ``strict`` raises on anything it would drop, which is what the configuration
    API wants — the admin owns the text and can fix it.
    """
    try:
        data = yaml.safe_load(raw or "")
    except yaml.YAMLError as e:
        raise ValueError(f"content is not valid YAML: {e}")
    if data is None:
        data = []
    if not isinstance(data, list):
        raise ValueError("content must be a YAML list of cache provider declarations")

    providers: List[CacheProvider] = []
    unknown_fields: Set[str] = set()
    for index, entry in enumerate(data):
        # An entry that is not a mapping has no name to report; its position is
        # what identifies it in the document.
        name = entry.get("name") if isinstance(entry, dict) else f"#{index}"
        try:
            _reject_mixed_version_sources(name, entry)
            provider = CacheProvider(**entry)
        except (ValidationError, TypeError, ValueError, AttributeError) as e:
            if strict:
                raise ValueError(f"cache provider {name} is unreadable: {e}")
            logger.error(f"Skipping malformed cache provider {name}: {e}")
            continue
        # A provider violating the injection placeholder or the localized-text
        # contract fails silently at runtime (literal placeholders corrupting
        # connector config, secrets riding into instance snapshots, text that
        # renders for no locale), so it never reaches the catalog.
        violations = validate_injection_templates(provider)
        violations += validate_localized_text(provider)
        violations += validate_template_filters(provider)
        if violations:
            listed = "; ".join(violations)
            if strict:
                raise ValueError(f"cache provider {name} is invalid: {listed}")
            logger.error(f"Skipping cache provider {provider.name}: {listed}")
            continue
        unknown_fields |= _unknown_keys(entry, CacheProvider, f"{name}.")
        providers.append(provider)

    # One added key lands on every declaration, so report the whole set once.
    if unknown_fields:
        listed = ", ".join(sorted(unknown_fields))
        if strict:
            raise ValueError(f"unknown cache provider field(s): {listed}")
        logger.warning(
            f"Ignoring cache provider field(s) this version does not know: "
            f"{listed}. The document was written for a different GPUStack."
        )

    # Nothing surviving is not an empty catalog; the caller would serve none.
    if data and not providers:
        raise ValueError("none of the declarations this document carries is readable")
    return providers


def normalize_cache_provider_yaml(raw: Optional[str], strict: bool = False) -> str:
    """Validate a catalog document and return the canonical text stored in a
    source's ``content`` (the ``normalize`` for ``CACHE_PROVIDER_SOURCE_SPEC``).

    Declaration order is preserved: it is the order the cards appear in.
    """
    return dump_cache_providers(load_cache_providers_document(raw, strict))


def merge_cache_providers(
    providers: List[CacheProvider], loaded: List[CacheProvider]
) -> List[CacheProvider]:
    """Fold a newly read document into the catalog, a same-named declaration
    replacing the one already there — in its place, so the catalog's order is
    the order a user sees the cards in."""
    # A document naming one provider twice keeps its last declaration, the way a
    # later document replaces an earlier one's: one name, one card.
    by_name = {provider.name.lower(): provider for provider in loaded}
    merged = [by_name.pop(p.name.lower(), p) for p in providers]
    merged.extend(
        by_name.pop(p.name.lower()) for p in loaded if p.name.lower() in by_name
    )
    return merged


def build_cache_provider_entries(
    sources: List[SourceContent],
    runners: Optional[List[Any]] = None,
) -> List[CacheProviderEntry]:
    """The catalog the ordered sources produce, as rows to materialize.

    Each row is stamped with the source that produced it, which is what the UI
    reads to tell a declaration of the admin's from a packaged one.

    A provider reading its release line off ``runners`` has it filled in here,
    so the row holds the images that will actually be pulled — the table is what
    serves, and a reader of it never has to know a version was derived.
    """
    merged: List[CacheProvider] = []
    origin: Dict[str, SourceContent] = {}
    for source in sources:
        try:
            loaded = load_cache_providers_document(source.content)
        except ValueError as e:
            # A document is validated when it is written, so one that will not
            # parse here was corrupted after the fact. It costs its own
            # declarations and not the catalog: raising would leave the table
            # serving whatever it last held, with no way to notice.
            logger.error(
                f"Skipping unreadable cache provider source {source.name}: {e}"
            )
            continue
        for provider in loaded:
            origin[provider.name.lower()] = source
        merged = merge_cache_providers(merged, loaded)

    merged = with_runner_versions(merged, runners)

    entries: List[CacheProviderEntry] = []
    for position, provider in enumerate(merged):
        source = origin[provider.name.lower()]
        entries.append(
            CacheProviderEntry(
                name=provider.name,
                position=position,
                payload=provider.model_dump(mode="json"),
                source_name=source.name,
                source_type=source.source_type,
            )
        )
    return entries


async def reconcile_cache_providers(
    session: AsyncSession, sources: List[SourceContent]
) -> None:
    """Full-rewrite ``CacheProviderEntry`` from the ordered sources, upserting by
    name so a row's ``id`` stays stable; vanished names are deleted, and empty
    input clears the table.

    Nothing readable raises before any write, so the table keeps serving
    whatever it held.

    The runner catalog is read here rather than passed in: a provider deriving
    its release line from it produces different rows as images come and go, so
    the two have to be read together or the table would hold images from one
    moment and declarations from another.
    """
    runners = merged_runners(await RunnerOverrideEntry.all(session))
    # Off the loop: parsing and validating a catalog runs into tens of ms, and a
    # request handler awaiting the database during a reconcile would wait it out.
    entries = await asyncio.to_thread(build_cache_provider_entries, sources, runners)
    desired = {entry.name.lower(): entry for entry in entries}
    existing = {row.name.lower(): row for row in await CacheProviderEntry.all(session)}

    for key, entry in desired.items():
        row = existing.get(key)
        if row is None:
            await CacheProviderEntry.create(session, entry, auto_commit=False)
            continue
        await row.update(
            session,
            {
                "name": entry.name,
                "position": entry.position,
                "payload": entry.payload,
                "source_name": entry.source_name,
                "source_type": entry.source_type,
            },
            auto_commit=False,
        )

    for key, row in existing.items():
        if key not in desired:
            await row.delete(session, auto_commit=False)

    await session.commit()


def dump_cache_providers(providers: List[CacheProvider]) -> str:
    """Serialize declarations back to the document form an admin edits.

    ``exclude_none``: a field left out of a declaration round-trips to None
    either way, so writing it back as an explicit null only adds noise to a
    document meant to be read and edited by hand.

    Keys keep the order the model declares them in rather than being sorted:
    a declaration read alphabetically opens on ``attach_locality`` and buries
    ``name`` in the middle, and this document is meant to be edited by hand.
    """
    documents: List[Dict[str, Any]] = [
        provider.model_dump(mode="json", exclude_none=True) for provider in providers
    ]
    return yaml.safe_dump(documents, sort_keys=False, allow_unicode=True)
