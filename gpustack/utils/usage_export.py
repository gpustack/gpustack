"""Column definitions and row flattening for usage exports.

The exported schema is defined HERE, on the server, and versioned with the
API — not derived from whatever columns the table happens to render. An export
file is consumed by customer reconciliation scripts; if its shape tracked the
UI's column list, every visual tweak would silently break them.

Two rules the layout follows:

* Entity dimensions expand to ``<dim>_id`` / ``<dim>_name`` / ``<dim>_deleted``
  instead of one pre-formatted label. ``_id`` is the only stable join key —
  names are resolved live and change when an entity is renamed — and deletion
  is a boolean column rather than a ``[Deleted.12]`` suffix glued onto the
  name, which is a UI convention and a parsing hazard.
* Metrics are written as raw numbers, never formatted, so they stay
  summable in a spreadsheet.
* The header row is a fixed English title (``Model``, ``Input Tokens``) — a
  readable label for whoever opens the file, but the SAME label in every
  language. Localizing it would make the file's shape depend on the viewer's
  language setting, which is precisely what a reconciliation script cannot
  tolerate. The machine keys behind those titles are exposed separately via
  ``export_column_keys`` so the preview can render the identical column set.
"""

from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, NamedTuple, Sequence, Tuple

from gpustack.schemas.usage import (
    USAGE_GRANULARITY_DAY,
    USAGE_GRANULARITY_HOUR,
    USAGE_GROUP_BY_API_KEY,
    USAGE_GROUP_BY_DATE,
    USAGE_GROUP_BY_ORGANIZATION,
    USAGE_GROUP_BY_ROUTE,
    USAGE_GROUP_BY_USER,
    USAGE_METRIC_API_KEYS_USED,
    USAGE_METRIC_API_REQUESTS,
    USAGE_METRIC_AVG_TOKENS_PER_REQUEST,
    USAGE_METRIC_INPUT_CACHED_TOKENS,
    USAGE_METRIC_INPUT_TOKENS,
    USAGE_METRIC_LAST_ACTIVE,
    USAGE_METRIC_MODELS_CALLED,
    USAGE_METRIC_OUTPUT_TOKENS,
    USAGE_METRIC_TOTAL_TOKENS,
    UsageBreakdownItem,
)

# Rows that carry no consumer principal are un-attributed direct traffic
# (cookie-authed calls with no API key, e.g. the built-in Playground). They are
# real usage, so an Organization export must show them as their own bucket —
# dropping them would make the file's totals disagree with the page's, and
# leaving the name blank would read as missing data.
UNTRACKED_ORGANIZATION_NAME = "Untracked"

# Header titles. Stable across languages (never localized — see the export
# docs) but written for a human opening the file, not as raw field names.
#
# Note the vocabulary: the product calls these Models. ``route`` is the
# internal name for the same thing and has never appeared in the UI, so it
# has no business appearing in a file handed to a customer.
EXPORT_COLUMN_TITLES: Dict[str, str] = {
    "date": "Date",
    "user_id": "User ID",
    "user_name": "User",
    "user_deleted": "User Deleted",
    "api_key_id": "API Key ID",
    "api_key_name": "API Key",
    "api_key_deleted": "API Key Deleted",
    "route_id": "Model ID",
    "route_name": "Model",
    "route_deleted": "Model Deleted",
    "organization_id": "Organization ID",
    "organization_name": "Organization",
    "organization_kind": "Organization Type",
    "organization_deleted": "Organization Deleted",
    USAGE_METRIC_INPUT_TOKENS: "Input Tokens",
    USAGE_METRIC_INPUT_CACHED_TOKENS: "Input Tokens Cached",
    USAGE_METRIC_OUTPUT_TOKENS: "Output Tokens",
    USAGE_METRIC_TOTAL_TOKENS: "Total Tokens",
    USAGE_METRIC_API_REQUESTS: "API Requests",
    USAGE_METRIC_AVG_TOKENS_PER_REQUEST: "Avg Tokens / Request",
    USAGE_METRIC_MODELS_CALLED: "Models Called",
    USAGE_METRIC_API_KEYS_USED: "API Keys Used",
    USAGE_METRIC_LAST_ACTIVE: "Last Active",
    # Resource dimensions and metrics.
    "instance_id": "Instance ID",
    "instance_name": "Instance",
    "instance_deleted": "Instance Deleted",
    "volume_id": "Volume ID",
    "volume_name": "Volume",
    "volume_deleted": "Volume Deleted",
    "instance_type_name": "Instance Type",
    "storage_type_name": "Storage Type",
    "resource_type_name": "Resource Type",
    "owner_id": "Owner ID",
    "owner_name": "Owner",
    "owner_deleted": "Owner Deleted",
    "gpu_hours": "GPU Hours",
    "instance_hours": "Instance Hours",
    "gb_days": "GB Days",
    "gb_hours": "GB Hours",
    # One server metric (``metrics.resources`` — distinct resources still alive)
    # under the two names the product actually uses. "Resources" was neither,
    # so a reader could not line the column up with the page it came from.
    "active_instances": "Active Instances",
    "active_volumes": "Active Volumes",
    "active_users": "Active Users",
}


def export_column_title(key: str) -> str:
    """Header text for a column key, falling back to the key itself."""
    return EXPORT_COLUMN_TITLES.get(key, key)


# group_by → the id / name attribute names on ``UsageIdentity``.
_ENTITY_DIMENSION_FIELDS = {
    USAGE_GROUP_BY_USER: ("user_id", "user_name"),
    USAGE_GROUP_BY_API_KEY: ("api_key_id", "api_key_name"),
    USAGE_GROUP_BY_ROUTE: ("route_id", "route_name"),
    USAGE_GROUP_BY_ORGANIZATION: ("organization_id", "organization_name"),
}

# Emitted for every sheet, after the dimension columns.
_METRIC_KEYS = [
    USAGE_METRIC_INPUT_TOKENS,
    USAGE_METRIC_INPUT_CACHED_TOKENS,
    USAGE_METRIC_OUTPUT_TOKENS,
    USAGE_METRIC_TOTAL_TOKENS,
    USAGE_METRIC_API_REQUESTS,
    USAGE_METRIC_AVG_TOKENS_PER_REQUEST,
    USAGE_METRIC_LAST_ACTIVE,
]

# Only meaningful when the sheet groups by exactly this dimension — the
# breakdown handler populates them for single-dimension groupings alone.
_SINGLE_GROUP_EXTRA_KEYS = {
    USAGE_GROUP_BY_USER: [USAGE_METRIC_MODELS_CALLED, USAGE_METRIC_API_KEYS_USED],
    USAGE_GROUP_BY_API_KEY: [USAGE_METRIC_MODELS_CALLED],
    USAGE_GROUP_BY_ROUTE: [USAGE_METRIC_MODELS_CALLED, USAGE_METRIC_API_KEYS_USED],
    USAGE_GROUP_BY_ORGANIZATION: [
        USAGE_METRIC_MODELS_CALLED,
        USAGE_METRIC_API_KEYS_USED,
    ],
}


def _column_keys(group_by: Sequence[str]) -> List[str]:
    keys: List[str] = []
    if USAGE_GROUP_BY_DATE in group_by:
        keys.append("date")
    # Dimension order follows the Usage page: Organization (broadest) first,
    # then Model, User, API Key. It is fixed here rather than taken from the
    # request's ``group_by`` so that two exports of the same dimensions always
    # produce the same column layout, whatever order the caller listed them
    # in — a file whose columns move around is not a stable schema.
    for dimension in (
        USAGE_GROUP_BY_ORGANIZATION,
        USAGE_GROUP_BY_ROUTE,
        USAGE_GROUP_BY_USER,
        USAGE_GROUP_BY_API_KEY,
    ):
        if dimension not in group_by:
            continue
        id_field, name_field = _ENTITY_DIMENSION_FIELDS[dimension]
        keys.extend([id_field, name_field])
        if dimension == USAGE_GROUP_BY_ORGANIZATION:
            keys.append("organization_kind")
        keys.append(f"{dimension}_deleted")
    keys.extend(_METRIC_KEYS)
    if len(group_by) == 1:
        keys.extend(_SINGLE_GROUP_EXTRA_KEYS.get(group_by[0], []))
    return keys


def build_export_columns(group_by: Sequence[str]) -> List[str]:
    """The header row for one sheet, in file order."""
    return [export_column_title(key) for key in _column_keys(group_by)]


def export_column_keys(group_by: Sequence[str]) -> List[str]:
    """The machine keys behind that header row, in the same order.

    The preview renders from these while the file shows the titles, so both
    describe the same columns without the client re-deriving the list.
    """
    return _column_keys(group_by)


def _dimension_values(dimension, id_field: str, name_field: str) -> Dict[str, Any]:
    """Flatten one dimension into its id / name / deleted values.

    A row can reference no entity at all — usage with no API key, for
    instance. Those cells are left empty rather than filled in: the UI's
    ``-`` placeholder is a rendering convention, and ``deleted = False`` for
    something that never existed asserts a fact about nothing. Both mislead a
    reader of the file.
    """
    empty = {id_field: None, name_field: None, "deleted": None}
    if dimension is None:
        return empty
    identity = dimension.identity
    current = identity.current if identity else None
    value = identity.value if identity else None
    entity_id = getattr(current, id_field, None) if current else None
    snapshot_name = getattr(value, name_field, None) if value else None
    if entity_id is None and not snapshot_name:
        return empty
    return {
        id_field: entity_id,
        # Prefer the resolved label so the file matches the screen; fall back
        # to the raw snapshot when there is no label.
        name_field: dimension.label or snapshot_name,
        "deleted": bool(dimension.deleted),
    }


def export_row(item: UsageBreakdownItem, group_by: Sequence[str]) -> List[Any]:
    """Flatten one breakdown item into a row matching ``build_export_columns``.

    Takes the same :class:`UsageBreakdownItem` the JSON endpoint returns, so a
    number can never differ between what the table shows and what the file
    contains.
    """
    values: Dict[str, Any] = {}
    if USAGE_GROUP_BY_DATE in group_by:
        values["date"] = item.date.value if item.date else None

    dimension_by_key = {
        USAGE_GROUP_BY_ORGANIZATION: item.organization,
        USAGE_GROUP_BY_USER: item.user,
        USAGE_GROUP_BY_ROUTE: item.route,
        USAGE_GROUP_BY_API_KEY: item.api_key,
    }
    for dimension_key, dimension in dimension_by_key.items():
        if dimension_key not in group_by:
            continue
        id_field, name_field = _ENTITY_DIMENSION_FIELDS[dimension_key]
        flattened = _dimension_values(dimension, id_field, name_field)
        values[id_field] = flattened[id_field]
        values[name_field] = flattened[name_field]
        values[f"{dimension_key}_deleted"] = flattened["deleted"]
        if dimension_key == USAGE_GROUP_BY_ORGANIZATION:
            identity = dimension.identity if dimension else None
            values["organization_kind"] = (
                getattr(identity.value, "organization_kind", None) if identity else None
            )
            # An Untracked bucket has no identity at all; name it explicitly
            # so the row doesn't read as a data gap.
            if identity is None:
                values[name_field] = UNTRACKED_ORGANIZATION_NAME

    for key in _METRIC_KEYS:
        values[key] = getattr(item, key, None)
    if len(group_by) == 1:
        for key in _SINGLE_GROUP_EXTRA_KEYS.get(group_by[0], []):
            values[key] = getattr(item, key, None)

    return [values.get(key) for key in _column_keys(group_by)]


# ---------------------------------------------------------------------------
# Resource usage (GPU instances / storage)
# ---------------------------------------------------------------------------

# Resource breakdown items are dicts, not typed models, and their dimension is
# carried generically as ``id`` / ``key`` / ``deleted`` whatever the grouping.
# The exported column names spell the dimension out (``instance_id``,
# ``volume_name``, …) so a file is self-describing without the request that
# produced it.
_RESOURCE_DIMENSION_LABELS = {
    "instance": "instance",
    "volume": "volume",
    "user": "user",
    "organization": "organization",
    "instance_type": "instance_type",
    "type": "storage_type",
    "resource_type": "resource_type",
}


def _resource_dimension_keys(dimension: str) -> List[str]:
    label = _RESOURCE_DIMENSION_LABELS.get(dimension, dimension)
    # Only entity dimensions have an id and a deletion state; a bucket like
    # instance_type is just a name.
    if dimension in ("instance", "volume", "user", "organization"):
        keys = [f"{label}_id", f"{label}_name"]
        if dimension == "organization":
            keys.append("organization_kind")
        keys.append(f"{label}_deleted")
        return keys
    return [f"{label}_name"]


# Metrics only the Storage tab meters. The sheet's metric set is what tells the
# two resource exports apart — both tabs share every column-building helper, and
# ``group_by`` can't say which tab it came from (``["user"]`` is a valid sheet on
# either). Used solely to name the ``resources`` column after what it counts.
_STORAGE_METRIC_KEYS = frozenset({"gb_days", "gb_hours"})


def _resources_column_key(metric_keys: Sequence[str]) -> str:
    """Column key for ``metrics.resources`` on this sheet.

    The server counts one thing — distinct resources in the group that still
    exist — but the product names it per tab: Active Instances on GPU
    Instances, Active Volumes on Storage. The file follows the page, so a
    column can be matched to the screen it was exported from.
    """
    if _STORAGE_METRIC_KEYS.intersection(metric_keys):
        return "active_volumes"
    return "active_instances"


def resource_export_column_keys(
    group_by: Sequence[str],
    metric_keys: Sequence[str],
) -> List[str]:
    keys: List[str] = []
    if USAGE_GROUP_BY_DATE in group_by:
        keys.append("date")
    for dimension in group_by:
        if dimension == USAGE_GROUP_BY_DATE:
            continue
        keys.extend(_resource_dimension_keys(dimension))
    # Per-resource rows carry their owner; coarser groupings omit it rather
    # than pair one user's id with another's name.
    if any(dimension in ("instance", "volume") for dimension in group_by):
        keys.extend(["owner_id", "owner_name", "owner_deleted"])
    keys.extend(
        list(metric_keys)
        + [_resources_column_key(metric_keys), "active_users", "last_active"]
    )

    return keys


class _ResourceExportShape(NamedTuple):
    """The part of a sheet's layout that does not vary from row to row."""

    column_keys: Tuple[str, ...]
    resources_key: str
    carries_owner: bool


@lru_cache(maxsize=64)
def _resource_export_shape(
    group_by: Tuple[str, ...], metric_keys: Tuple[str, ...]
) -> _ResourceExportShape:
    """Resolve a sheet's fixed layout once per (group_by, metric_keys).

    Every value here is a function of the sheet definition alone, yet
    ``resource_export_row`` used to recompute all of it for every row: the
    column-key list, the name of the ``resources`` column, and the scan for
    whether the grouping carries an owner. On a 100k-row export that is 100k
    identical answers — cheap individually, but it is the per-row call count
    that a loaded server multiplies.

    Cached rather than passed in as an argument so every caller benefits
    without threading a context object through the row loop. The key space is
    tiny and fixed (a handful of tab / grouping combinations), so the cache
    cannot grow with traffic.
    """
    return _ResourceExportShape(
        column_keys=tuple(resource_export_column_keys(group_by, metric_keys)),
        resources_key=_resources_column_key(metric_keys),
        carries_owner=any(
            dimension in ("instance", "volume") for dimension in group_by
        ),
    )


def _export_day(value: Any) -> Any:
    """Coerce an instant to its calendar day for an export cell.

    The token export's Date and Last Active are plain days, because
    ``model_usages`` is a daily rollup. ``metered_usage`` is hourly, so the
    resource rows arrive as datetimes — written raw they put
    "2026-08-02 00:00:00" in a Date column and pinned Last Active to whichever
    hour bucket happened to be last. One report should not carry two date
    formats, and the extra precision is an artifact of the storage layer, not
    something the user asked for.

    Export-only: the JSON API keeps the instant, which the trend chart needs.
    """
    return value.date() if isinstance(value, datetime) else value


def _export_bucket(value: Any, granularity: str) -> Any:
    """The Date cell for one bucket, at the precision that bucket has.

    Day/week/month buckets are calendar days and get :func:`_export_day`. An
    HOUR bucket is not: collapsing it would put the same date on all 24 rows
    of a volume's day, leaving a file whose rows differ only in their numbers
    and cannot be told apart or re-sorted. The whole point of asking for hour
    granularity is that the hour is the identity of the row.

    ``Last Active`` stays a calendar day either way — it is a "when was this
    last used" summary for a reader, not the bucket key.
    """
    if granularity == USAGE_GRANULARITY_HOUR:
        return value
    return _export_day(value)


def resource_export_row(
    item: Dict[str, Any],
    group_by: Sequence[str],
    metric_keys: Sequence[str],
    granularity: str = USAGE_GRANULARITY_DAY,
) -> List[Any]:
    """Flatten one enriched resource breakdown item into an export row."""
    shape = _resource_export_shape(tuple(group_by), tuple(metric_keys))
    values: Dict[str, Any] = {}
    if USAGE_GROUP_BY_DATE in group_by:
        values["date"] = _export_bucket(item.get("date"), granularity)

    for dimension in group_by:
        if dimension == USAGE_GROUP_BY_DATE:
            continue
        label = _RESOURCE_DIMENSION_LABELS.get(dimension, dimension)
        if dimension in ("instance", "volume", "user", "organization"):
            values[f"{label}_id"] = item.get("id")
            values[f"{label}_name"] = (
                item.get("key")
                if item.get("id") is not None or item.get("key")
                else UNTRACKED_ORGANIZATION_NAME
            )
            if dimension == "organization":
                values["organization_kind"] = item.get("organization_kind")
                # A NULL consumer principal is un-attributed direct traffic —
                # name it rather than emit a blank that reads as missing data.
                if item.get("id") is None:
                    values[f"{label}_name"] = UNTRACKED_ORGANIZATION_NAME
            values[f"{label}_deleted"] = bool(item.get("deleted"))
        else:
            values[f"{label}_name"] = item.get("key")

    if shape.carries_owner:
        values["owner_id"] = item.get("creator_id")
        values["owner_name"] = item.get("creator_name")
        values["owner_deleted"] = bool(item.get("creator_deleted"))

    metrics = item.get("metrics") or {}
    for key in metric_keys:
        values[key] = metrics.get(key)
    values["active_users"] = metrics.get("active_users")
    # The breakdown item keeps the server's generic ``resources``; the column it
    # lands in is named after the resource kind (see _resources_column_key).
    values[shape.resources_key] = metrics.get("resources")
    values["last_active"] = _export_day(metrics.get("last_active"))

    return [values.get(key) for key in shape.column_keys]


def build_resource_export_columns(
    group_by: Sequence[str],
    metric_keys: Sequence[str],
) -> List[str]:
    """The header row for one resource sheet, in file order."""
    return [
        export_column_title(key)
        for key in resource_export_column_keys(group_by, metric_keys)
    ]
