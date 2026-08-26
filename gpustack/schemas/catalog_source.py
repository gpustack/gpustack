import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import yaml
from sqlalchemy import JSON, Column, UniqueConstraint
from sqlmodel import SQLModel, Field as SQLField
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.mixins import BaseModelMixin
from .models import Model
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


def _load_model_set(raw: Any) -> Optional[ModelSet]:
    """One model set, with the specs this version cannot read dropped. ``None``
    when the card itself is unreadable, or when every spec it carried was.

    Dropped per spec rather than per card: a newly published quantization should
    cost its own row, not take the deployable specs of the same model with it. A
    card that never carried any spec is kept — that is a legitimate document.
    """
    name = raw.get("name") if isinstance(raw, dict) else None
    try:
        raw_specs = raw["specs"] if isinstance(raw, dict) and "specs" in raw else []
        specs: List[ModelSpec] = []
        for raw_spec in raw_specs:
            try:
                specs.append(ModelSpec(**raw_spec))
            except Exception as e:
                logger.warning(f"Skipping an unreadable spec of model set {name}: {e}")
        if raw_specs and not specs:
            logger.warning(f"Skipping model set {name}: none of its specs is readable")
            return None
        return ModelSet(**{**raw, "specs": specs})
    except Exception as e:
        logger.warning(f"Skipping unreadable model set {name}: {e}")
        return None


def _load_catalog(raw: Optional[str]) -> Catalog:
    """Parse a catalog document. Missing ``model_sets``/``draft_models`` default
    to empty; raises ``ValueError`` (→ HTTP 400) on malformed YAML or a document
    whose shape is wrong.

    A record this version cannot read is dropped, not fatal to the document:
    every cluster reads the same published catalog, so one model set using a
    field or an enum value added after this release must not stop the rest of it
    from serving. Structure still raises — ``model_sets: not-a-list`` is a broken
    document rather than a newer one, and reading it as "no model sets" would
    clear the materialized table.
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
    raw_model_sets = data.get("model_sets") or []
    raw_draft_models = data.get("draft_models") or []
    if not isinstance(raw_model_sets, list) or not isinstance(raw_draft_models, list):
        raise ValueError("model_sets and draft_models must be lists")

    # A ``.``-prefixed key hosts a YAML anchor the document references below
    # (the packaged catalog pins backend versions that way), not a field.
    unknown_fields = {
        key
        for key in set(data) - {"model_sets", "draft_models"}
        if not key.startswith(".")
    }
    if unknown_fields:
        logger.warning(
            f"Ignoring catalog field(s) this version does not know: "
            f"{', '.join(sorted(unknown_fields))}. The document was published "
            f"for a newer GPUStack."
        )

    draft_models: List[DraftModel] = []
    for raw_draft in raw_draft_models:
        try:
            draft_models.append(DraftModel(**raw_draft))
        except Exception as e:
            name = raw_draft.get("name") if isinstance(raw_draft, dict) else None
            logger.warning(f"Skipping unreadable draft model {name}: {e}")

    model_sets = [_load_model_set(item) for item in raw_model_sets]
    return Catalog(
        model_sets=[model_set for model_set in model_sets if model_set is not None],
        draft_models=draft_models,
    )


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


async def _pinned_draft_names(session: AsyncSession) -> Dict[str, List[str]]:
    """Every draft model name a live deployment pins, to the models pinning it. A
    model stores the name, not a resolved source, so the catalog entry is what
    turns it into a repository (``get_draft_model_source``)."""
    pinned: Dict[str, List[str]] = {}
    for model in await Model.all(session):
        speculative_config = model.speculative_config
        if model.replicas > 0 and speculative_config and speculative_config.draft_model:
            pinned.setdefault(speculative_config.draft_model, []).append(model.name)
    return pinned


async def reconcile_catalog(
    session: AsyncSession, sources: List[SourceContent]
) -> None:
    """Full-rewrite ``CatalogModelEntry`` from the ordered sources, upserting by
    ``(kind, name)`` so an entry's ``id`` stays stable (``/model-sets/{id}/specs``
    depends on it); vanished keys are deleted, empty input clears the table.

    A draft model a live deployment pins is kept even when no source carries it
    any more, and released on the first reconcile after that deployment goes.
    Losing one does not fail loudly: ``get_draft_model_source`` reads the name as
    a repository id instead, so it becomes a 404 naming a repository nobody
    published, with the catalog never mentioned. ``model_sets`` needs no guard —
    ``set_default_spec`` only fills fields a model left empty.

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

    vanished = [key for key in existing if key not in desired]
    pinned = (
        await _pinned_draft_names(session)
        if any(kind == KIND_DRAFT for kind, _ in vanished)
        else {}
    )
    for key in vanished:
        kind, name = key
        if kind == KIND_DRAFT and name in pinned:
            logger.warning(
                f"No catalog source carries draft model {name} any more, kept "
                f"because it is deployed by: {', '.join(sorted(pinned[name]))}. "
                f"Migrate those models off it; it goes away once they are gone."
            )
            continue
        await existing[key].delete(session, auto_commit=False)

    await session.commit()
