import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import ValidationError
from sqlalchemy import JSON, Column, UniqueConstraint
from sqlmodel import SQLModel, Field as SQLField
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.mixins import BaseModelMixin
from .model_sets import Catalog, DraftModel, ModelSet, ModelSpec
from .source import SourceContent, SourceMixin, SourceTypeEnum, validate_icon

logger = logging.getLogger(__name__)

# The single BUILTIN CatalogSource seeded from the packaged model-catalog.yaml;
# content refreshed on every leader start, the user's enabled toggle preserved.
BUILTIN_CATALOG_SOURCE_NAME = "builtin"

# The single custom CatalogSource an admin configures via ``/v2/ota-sources/catalog``
# (the table holds many rows, so multiple sources later need no schema change).
CUSTOM_CATALOG_SOURCE_NAME = "custom"

# ``CatalogModelEntry.kind`` discriminates the two record types sharing the
# table: model sets (carry specs) and speculative-decoding draft models.
KIND_MODEL_SET = "model_set"
KIND_DRAFT = "draft"


class CatalogSource(SourceMixin, BaseModelMixin, table=True):
    """A source of model-catalog content (FILE or URL); the leader reconciles all
    enabled rows into ``CatalogModelEntry``. ``normalize_catalog_yaml`` validates
    and canonicalizes the stored text.
    """

    __tablename__ = "catalog_sources"
    id: Optional[int] = SQLField(default=None, primary_key=True)


class CatalogModelEntryBase(SQLModel):
    """A single materialized catalog record (a model set or draft model). The
    full object is stored as a JSON ``payload``; only the query keys and
    source-origin columns are promoted to real columns.
    """

    kind: str = SQLField(index=True)
    name: str = SQLField(index=True)
    payload: Dict[str, Any] = SQLField(sa_column=Column(JSON), default_factory=dict)
    # Source-origin display: which source produced this record (last writer wins
    # a key). BUILTIN records carry it too; the UI shows no badge for built-in.
    source_name: str = SQLField(default="")
    source_type: SourceTypeEnum = SQLField(default=SourceTypeEnum.BUILTIN)
    # Reserved for future multi-tenancy; excluded from serialized responses.
    owner_principal_id: Optional[int] = SQLField(default=None, exclude=True)


class CatalogModelEntry(CatalogModelEntryBase, BaseModelMixin, table=True):
    __tablename__ = "catalog_model_entries"
    # ``(kind, name)`` is the upsert key; named explicitly so the constraint the
    # migration creates is droppable by name on every dialect.
    __table_args__ = (
        UniqueConstraint("kind", "name", name="uix_catalog_model_entries_kind_name"),
    )
    id: Optional[int] = SQLField(default=None, primary_key=True)


def _load_catalog(raw: Optional[str]) -> Catalog:
    """Parse and validate a catalog document via the ``Catalog`` schema. Missing
    ``model_sets``/``draft_models`` default to empty; raises ``ValueError``
    (→ HTTP 400) on malformed YAML or an invalid catalog.
    """
    try:
        data = yaml.safe_load(raw or "")
    except yaml.YAMLError as e:
        raise ValueError(f"content is not valid YAML: {e}")
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(
            "content must be a YAML mapping with model_sets / draft_models"
        )
    data.setdefault("model_sets", [])
    data.setdefault("draft_models", [])
    try:
        return Catalog(**data)
    except ValidationError as e:
        raise ValueError(f"invalid catalog: {e}")


def normalize_catalog_yaml(raw: Optional[str]) -> str:
    """Validate raw catalog YAML and return the canonical text stored in a
    source's ``content`` (the ``normalize`` for ``CATALOG_SOURCE_SPEC``). Raises
    ``ValueError`` on malformed input; otherwise re-serializes to a stable form.
    """
    catalog = _load_catalog(raw)
    for model_set in catalog.model_sets:
        model_set.icon = validate_icon(model_set.icon)
    # ``exclude_none``: a field left out of the document round-trips to None
    # either way, so dumping it back as an explicit null adds nothing a reader or
    # a reconcile can use — and it is 37% of the stored text.
    return yaml.safe_dump(
        catalog.model_dump(mode="json", exclude_none=True),
        sort_keys=True,
        allow_unicode=True,
    )


def _spec_identity(spec: ModelSpec) -> Tuple:
    """Full identity of a spec within a model set, for cross-source dedup."""
    gpu_filters = spec.gpu_filters.model_dump() if spec.gpu_filters else None
    return (
        spec.model_source_key,
        spec.mode,
        spec.quantization,
        json.dumps(gpu_filters, sort_keys=True),
    )


def _dedup_specs(specs: List[ModelSpec]) -> List[ModelSpec]:
    """Dedup specs by full identity, keeping the last occurrence (later source
    wins on a shared spec)."""
    by_identity: Dict[Tuple, ModelSpec] = {}
    for spec in specs:
        by_identity[_spec_identity(spec)] = spec
    return list(by_identity.values())


def _model_set_entry(
    existing: Optional[CatalogModelEntry], model_set: ModelSet, source: SourceContent
) -> CatalogModelEntry:
    """Build (or merge into) the entry for a model set: same-named sets union
    their specs (deduped by identity); metadata and source stamp from the last
    writer.
    """
    specs = model_set.specs
    if existing is not None:
        previous = ModelSet(**existing.payload)
        specs = _dedup_specs(previous.specs + model_set.specs)
    merged = model_set.model_copy(update={"specs": specs})
    return CatalogModelEntry(
        kind=KIND_MODEL_SET,
        name=model_set.name,
        payload=merged.model_dump(mode="json"),
        source_name=source.name,
        source_type=source.source_type,
    )


def _draft_entry(draft: DraftModel, source: SourceContent) -> CatalogModelEntry:
    return CatalogModelEntry(
        kind=KIND_DRAFT,
        name=draft.name,
        payload=draft.model_dump(mode="json"),
        source_name=source.name,
        source_type=source.source_type,
    )


def build_catalog_entries(sources: List[SourceContent]) -> List[CatalogModelEntry]:
    """Merge the ordered sources into the desired ``CatalogModelEntry`` set. Pure
    function keyed by ``(kind, name)`` (later source wins; model sets union their
    specs), split from ``reconcile_catalog`` to be testable without a session.

    An unparseable source is skipped, not fatal to the kind — the others still
    serve. All of them unreadable raises instead: an empty result is what a
    document deliberately carrying nothing looks like, and the caller would clear
    the materialized table over it.
    """
    merged: Dict[Tuple[str, str], CatalogModelEntry] = {}
    skipped: List[str] = []
    for source in sources:
        try:
            catalog = _load_catalog(source.content)
        except ValueError as e:
            logger.error(f"Skipping unreadable catalog source {source.name}: {e}")
            skipped.append(source.name)
            continue
        for model_set in catalog.model_sets:
            key = (KIND_MODEL_SET, model_set.name)
            merged[key] = _model_set_entry(merged.get(key), model_set, source)
        for draft in catalog.draft_models:
            merged[(KIND_DRAFT, draft.name)] = _draft_entry(draft, source)
    if skipped and len(skipped) == len(sources):
        raise ValueError(f"no catalog source could be read: {', '.join(skipped)}")
    return list(merged.values())


async def reconcile_catalog(
    session: AsyncSession, sources: List[SourceContent]
) -> None:
    """Full-rewrite ``CatalogModelEntry`` from the ordered sources, upserting by
    ``(kind, name)`` so an entry's ``id`` stays stable (``/model-sets/{id}/specs``
    depends on it); vanished keys are deleted, empty input clears the table.

    Nothing readable raises before any write, so the table keeps serving.
    """
    # Off the loop: parsing every source's YAML runs into hundreds of ms, and a
    # request handler awaiting the database during a reconcile would wait it out.
    entries = await asyncio.to_thread(build_catalog_entries, sources)
    desired = {(entry.kind, entry.name): entry for entry in entries}
    existing = {
        (row.kind, row.name): row for row in await CatalogModelEntry.all(session)
    }

    for key, entry in desired.items():
        row = existing.get(key)
        if row is not None:
            await row.update(
                session,
                {
                    "payload": entry.payload,
                    "source_name": entry.source_name,
                    "source_type": entry.source_type,
                },
                auto_commit=False,
            )
        else:
            await CatalogModelEntry.create(session, entry, auto_commit=False)

    for key, row in existing.items():
        if key not in desired:
            await row.delete(session, auto_commit=False)

    await session.commit()
