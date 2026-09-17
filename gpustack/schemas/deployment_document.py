"""Deployment documents: the YAML a set of model deployments is exported to
and imported from.

The document holds one entry per YAML document, separated by ``---``. An entry
is a ``ModelCreate`` body minus the server-derived and environment-bound
fields, so a document exported here imports unchanged into the same cluster.
Only the schema and the dump live here; the routes do the visibility checks
and the transaction.
"""

from datetime import datetime, timezone
from enum import Enum
from types import UnionType
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Set,
    Type,
    Union,
    get_args,
    get_origin,
)

import yaml
from pydantic import BaseModel, NonNegativeInt, ValidationError

from gpustack import __version__
from .models import Model, ModelCreate, ModelPublic
from .source import unknown_keys

# Fields the server derives or that bind a row to one environment. Dropped on
# export and re-derived on import, so a document never carries them.
SERVER_MANAGED_FIELDS = frozenset(
    {
        "id",
        "created_at",
        "updated_at",
        "deleted_at",
        "ready_replicas",
        "meta",
        "cluster_id",
        "owner_principal_id",
        "access_policy",
    }
)

# LoraListEntry fields populated only once an adapter is mounted on an instance.
LORA_RUNTIME_FIELDS = frozenset({"path", "model_file_id"})

# ``name`` first so a reader can tell entries apart at a glance; the rest keep
# the schema's declaration order (ModelSource fields, then the deployment ones).
ENTRY_FIELDS: List[str] = ["name"] + [
    field
    for field in ModelCreate.model_fields
    if field != "name" and field not in SERVER_MANAGED_FIELDS
]

# What an overwrite writes onto the existing row. Everything outside this
# list is left alone: a ModelCreate dump also carries owner_principal_id and
# access_policy at their defaults, and writing those would re-home the row to
# the platform Org and drop the grants scoping it to its own.
# ``enable_model_route`` is not a column -- it settles routes instead.
#
# Derived, so a new ModelSpecBase field joins both the export and the
# overwrite set on its own. ``test_entry_fields_are_pinned`` holds the list to
# a golden copy so that stays a deliberate act.
OVERWRITABLE_FIELDS: List[str] = [
    field for field in ENTRY_FIELDS if field != "enable_model_route"
]


class DeploymentExportRequest(BaseModel):
    ids: Optional[List[int]] = None
    """Deployments to export; omitted means every deployment the caller can see."""
    cluster_id: Optional[int] = None
    """Narrow the export to one cluster."""


def _bare_lora_entry(entry: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    lora_name = entry.get("lora_name") or ""
    if lora_name.startswith(prefix):
        entry["lora_name"] = lora_name[len(prefix) :]
    return {
        key: value for key, value in entry.items() if key not in LORA_RUNTIME_FIELDS
    }


def _entry_projection(
    data: Dict[str, Any], name: str, enable_model_route: bool
) -> Dict[str, Any]:
    """Project a dumped model onto the document's fields, in declaration order
    with ``None`` values left out. LoRA names come out as the bare short names
    clients see; import re-adds the stored ``<base>:`` prefix."""
    data["enable_model_route"] = enable_model_route
    if data.get("lora_list"):
        prefix = f"{name}:"
        data["lora_list"] = [
            _bare_lora_entry(entry, prefix) for entry in data["lora_list"]
        ]
    return {field: data[field] for field in ENTRY_FIELDS if field in data}


def deployment_entry(model: Model, enable_model_route: bool) -> Dict[str, Any]:
    """One document entry for a stored deployment."""
    return _entry_projection(
        model.model_dump(mode="json", exclude_none=True), model.name, enable_model_route
    )


def entry_document_form(entry: ModelCreate) -> Dict[str, Any]:
    """An import entry in the shape :func:`deployment_entry` renders a row in,
    so a diff between the two speaks the document's vocabulary rather than the
    ORM's. Call it before the create checks run: they normalize LoRA names and
    the replica count in place, and the user is owed a diff against what they
    wrote.
    """
    return _entry_projection(
        entry.model_dump(mode="json", exclude_none=True),
        entry.name,
        bool(entry.enable_model_route),
    )


def _unset(value: Any) -> bool:
    """Whether a projected field says nothing is configured.

    The two sides spell that differently. A column stored as NULL is left out
    of the export, while re-reading the same document materializes the field's
    ``ModelCreate`` default — an empty ``worker_selector`` against a missing
    one, an empty ``lora_list`` against no key at all. Both mean unset, so
    neither may read as a change.
    """
    return value is None or value == {} or value == []


def diff_entries(
    current: Dict[str, Any], desired: Dict[str, Any]
) -> List["DeploymentChange"]:
    """Fields that differ between two projections, in document order.

    Both sides leave unset fields out, so a field present on one side only
    diffs against ``None`` — which is exactly what deleting it from the
    document means.
    """
    changes: List["DeploymentChange"] = []
    for field in ENTRY_FIELDS:
        before, after = current.get(field), desired.get(field)
        if before == after or (_unset(before) and _unset(after)):
            continue
        changes.append(DeploymentChange(field=field, current=before, desired=after))
    return changes


def dump_deployments(
    models: Iterable[Model],
    route_backed_ids: Set[int],
    exported_at: Optional[datetime] = None,
) -> str:
    """Render ``models`` as a deployment document.

    ``route_backed_ids`` are the model ids that have a model route created
    alongside them — what ``enable_model_route`` re-creates on import. Apart
    from the timestamp in the header comment the output is a pure function of
    the rows, so a re-export diffs cleanly.
    """
    if exported_at is None:
        exported_at = datetime.now(timezone.utc)
    header = (
        f"# Exported from GPUStack v{__version__} "
        f"at {exported_at:%Y-%m-%dT%H:%M:%SZ}\n"
    )
    entries = [
        deployment_entry(model, model.id in route_backed_ids) for model in models
    ]
    # One YAML document per deployment, so each starts at the left margin and
    # `---` marks where the next begins. A deployment can then be read, copied
    # or hand-edited on its own, without re-indenting it out of a list.
    return header + yaml.safe_dump_all(entries, sort_keys=False, allow_unicode=True)


class DeploymentImportRequest(BaseModel):
    content: str
    """The document text; the client reads the file and uploads it as a string."""
    cluster_id: int
    """Cluster every entry is created in; the document itself carries none."""
    dry_run: bool = False
    """Plan only, write nothing."""
    overwrite: List[str] = []
    """Deployments the caller agreed to replace, by name. An entry that would
    overwrite one not named here is refused rather than applied.

    Consent is to the *set of rows*, not to the diff that was on screen: a
    name here still authorizes whatever the document says by the time the
    write runs. A name that turns out not to need overwriting is ignored
    rather than rejected -- an entry can go from ``update`` to ``unchanged``
    between the plan and the write, and that is not the caller's mistake.
    ``replica_overrides`` is stricter because an unmatched name there would
    silently fail to apply a value the caller asked for."""
    replica_overrides: Dict[str, NonNegativeInt] = {}
    """Replica counts to use instead of the document's, by deployment name, so
    one document suits environments of different sizes without being edited."""


class DeploymentActionEnum(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    UNCHANGED = "unchanged"


class DeploymentChange(BaseModel):
    field: str
    """The document's own field name. Clients map it to a display label —
    the UI wording lives with the rest of their translations, not here."""
    current: Any = None
    desired: Any = None


class DeploymentPlanEntry(BaseModel):
    index: int
    name: Optional[str] = None
    action: Optional[DeploymentActionEnum] = None
    """None when the entry did not parse, leaving nothing to plan."""
    desired: Dict[str, Any] = {}
    """The entry as the document describes it, in the document's own fields —
    what a client renders the preview row from. Empty when it did not parse."""
    current: Dict[str, Any] = {}
    """The deployment this entry would replace, projected onto the same fields
    in the same order, so the two render as a diff of one document against
    another. Empty when no deployment of that name exists yet."""
    raw: Dict[str, Any] = {}
    """The entry as the file spells it, populated only when it failed to
    validate — exactly when ``desired`` is empty. An invalid entry is the one
    a user most needs to edit, and without this there would be nothing to put
    in front of them to edit."""
    changes: List[DeploymentChange] = []
    """Populated for an overwrite; what replacing the row would alter."""
    errors: List[str] = []
    """Unlabelled: the entry they belong to is named right here."""


class DeploymentImportResult(BaseModel):
    dry_run: bool
    valid: bool
    """No entry has errors. What gates the client's confirm button."""
    entries: List[DeploymentPlanEntry]
    items: List[ModelPublic] = []
    """The rows as written, in document order; empty on a dry run."""


def entry_label(index: int, name: Optional[str]) -> str:
    """How an entry is named in an error message."""
    return f"deployment[{index}] ({name})" if name else f"deployment[{index}]"


class LoadedEntry(NamedTuple):
    """One item of the document, whether or not it parsed."""

    index: int
    name: Optional[str]
    """The raw ``name``, kept even when the entry failed to validate."""
    entry: Optional[ModelCreate]
    """None when the entry did not parse."""
    errors: List[str]
    raw: Dict[str, Any] = {}
    """The item as the file spells it, before validation."""

    @property
    def label(self) -> str:
        return entry_label(self.index, self.name)


def _validation_error_line(error: Dict[str, Any]) -> str:
    location = ".".join(str(part) for part in error["loc"])
    return f"{location}: {error['msg']}" if location else error["msg"]


def _nested_model(annotation: Any) -> Optional[Type[BaseModel]]:
    """The pydantic model a field holds, seen through Optional / List wrappers."""
    if isinstance(annotation, type):
        return annotation if issubclass(annotation, BaseModel) else None
    if get_origin(annotation) in (Union, UnionType, list):
        for argument in get_args(annotation):
            nested = _nested_model(argument)
            if nested is not None:
                return nested
    return None


def _unknown_paths(
    raw: Dict[str, Any], model: Type[BaseModel], prefix: str = ""
) -> List[str]:
    """Keys ``model`` has no field for, at every depth, as ``a.b[0].c`` paths.

    Pydantic ignores extra keys inside nested models too, so a typo in
    ``gpu_selector`` or a new ``lora_list`` option would otherwise vanish.
    """
    paths = [f"{prefix}{key}" for key in unknown_keys(raw, model)]
    for name, field in model.model_fields.items():
        nested = _nested_model(field.annotation)
        value = raw.get(name)
        if nested is None or value is None:
            continue
        items = value if isinstance(value, list) else [value]
        for position, item in enumerate(items):
            if isinstance(item, dict):
                index = f"[{position}]" if isinstance(value, list) else ""
                paths += _unknown_paths(item, nested, f"{prefix}{name}{index}.")
    return paths


def _lora_runtime_paths(raw: Dict[str, Any]) -> List[str]:
    lora_list = raw.get("lora_list")
    if not isinstance(lora_list, list):
        return []
    return [
        f"lora_list[{position}].{key}"
        for position, item in enumerate(lora_list)
        if isinstance(item, dict)
        for key in sorted(LORA_RUNTIME_FIELDS & {str(key) for key in item})
    ]


def _document_entries(documents: List[Any]) -> List[Any]:
    """The deployments a parsed file carries, in either spelling.

    Export writes one deployment per YAML document. A single document holding
    a list is how it used to be written, and still reads: a file exported
    before the change stays importable, as does a hand-written list.
    """
    if len(documents) == 1 and isinstance(documents[0], list):
        return documents[0]
    # A trailing `---`, or a leading one before the first deployment, parses as
    # an empty document rather than an entry with nothing in it.
    entries = [document for document in documents if document is not None]
    if not entries or not all(isinstance(entry, dict) for entry in entries):
        raise ValueError(
            "the document must hold one deployment per '---' section, "
            "or a single list of deployments"
        )
    return entries


def load_deployments(text: str) -> List[LoadedEntry]:
    """Parse a document strictly, one ``LoadedEntry`` per deployment in order.

    A file that is not valid YAML, or that holds something other than
    deployments, raises ``ValueError`` at once — there is no plan to show for
    it. Entry problems are collected on the entry they belong to instead,
    alongside the entries that did parse, so the route can run its own checks
    on those and report everything in one pass. Unknown and server-managed
    fields are errors at every depth, not silently dropped: a document from a
    newer GPUStack must not lose configuration on the way in.
    """
    try:
        documents = list(yaml.safe_load_all(text))
    except yaml.YAMLError as e:
        raise ValueError(f"the document is not valid YAML: {e}")
    raw_entries = _document_entries(documents)

    loaded: List[LoadedEntry] = []
    first_index_by_name: Dict[str, int] = {}
    for index, raw in enumerate(raw_entries):
        name = raw.get("name") if isinstance(raw, dict) else None
        errors: List[str] = []
        if not isinstance(raw, dict):
            loaded.append(LoadedEntry(index, None, None, ["must be a mapping"]))
            continue
        managed = sorted(str(key) for key in raw if key in SERVER_MANAGED_FIELDS)
        managed += _lora_runtime_paths(raw)
        if managed:
            errors.append(
                f"server-managed field(s) are not allowed: {', '.join(managed)}"
            )
        unknown = sorted(
            path
            for path in _unknown_paths(raw, ModelCreate)
            if path not in SERVER_MANAGED_FIELDS
        )
        if unknown:
            errors.append(f"unknown field(s): {', '.join(unknown)}")
        try:
            entry = ModelCreate.model_validate(raw)
        except ValidationError as e:
            errors.extend(_validation_error_line(error) for error in e.errors())
            loaded.append(LoadedEntry(index, name, None, errors, raw))
            continue
        if entry.name in first_index_by_name:
            errors.append(
                "duplicate name, already used by "
                f"deployment[{first_index_by_name[entry.name]}]"
            )
            loaded.append(LoadedEntry(index, entry.name, None, errors, raw))
            continue
        first_index_by_name[entry.name] = index
        loaded.append(LoadedEntry(index, entry.name, entry, errors))
    return loaded
