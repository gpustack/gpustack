from datetime import date as Date
from typing import List, Optional

from pydantic import (
    BaseModel,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from gpustack.api.exceptions import InvalidException
from gpustack.schemas.common import Pagination

USAGE_METRIC_INPUT_TOKENS = "input_tokens"
USAGE_METRIC_OUTPUT_TOKENS = "output_tokens"
USAGE_METRIC_INPUT_CACHED_TOKENS = "input_cached_tokens"
USAGE_METRIC_TOTAL_TOKENS = "total_tokens"
USAGE_METRIC_API_REQUESTS = "api_requests"
USAGE_METRIC_MODELS_CALLED = "models_called"
USAGE_METRIC_API_KEYS_USED = "api_keys_used"
USAGE_METRIC_AVG_TOKENS_PER_REQUEST = "avg_tokens_per_request"
USAGE_METRIC_LAST_ACTIVE = "last_active"
USAGE_METRIC_DATE = "date"

USAGE_GROUP_BY_DATE = "date"
USAGE_GROUP_BY_USER = "user"
USAGE_GROUP_BY_API_KEY = "api_key"
USAGE_GROUP_BY_ROUTE = "route"
# Groups by the consumer principal (``consumer_principal_id`` — the
# API-key owner Org). Reserved for the platform-wide "All" view (admin
# in cross-org context); enforced in the route handler.
USAGE_GROUP_BY_ORGANIZATION = "organization"

USAGE_GRANULARITY_DAY = "day"
USAGE_GRANULARITY_WEEK = "week"
USAGE_GRANULARITY_MONTH = "month"
# Resource usage only — ``metered_usage`` is an hourly rollup, while token
# usage (``model_usages``) is daily and has no hour to offer. Deliberately
# absent from ``USAGE_GRANULARITIES`` below, which is what the token requests
# validate against. It lives here rather than in the route module so the
# export writer can ask "does this bucket carry a time of day?" without
# importing routes (routes already import utils).
USAGE_GRANULARITY_HOUR = "hour"

USAGE_SORT_ASC = "asc"
USAGE_SORT_DESC = "desc"

# Usage view scope. ``self`` filters to the caller's own rows
# (``user_id = self``); ``all`` filters to the current Org's rows
# (``owner_principal_id = current_principal_id``), or — for platform admin in
# cross-org context — to every Org. ``all`` is reserved for admin /
# Org owner / manager; others are forced to ``self``.
USAGE_SCOPE_SELF = "self"
USAGE_SCOPE_ALL = "all"
USAGE_SCOPES = {USAGE_SCOPE_SELF, USAGE_SCOPE_ALL}

USAGE_GROUP_BYS = {
    USAGE_GROUP_BY_DATE,
    USAGE_GROUP_BY_USER,
    USAGE_GROUP_BY_API_KEY,
    USAGE_GROUP_BY_ROUTE,
    USAGE_GROUP_BY_ORGANIZATION,
}
USAGE_GRANULARITIES = {
    USAGE_GRANULARITY_DAY,
    USAGE_GRANULARITY_WEEK,
    USAGE_GRANULARITY_MONTH,
}
USAGE_SORTABLE_FIELDS = {
    USAGE_METRIC_INPUT_TOKENS,
    USAGE_METRIC_OUTPUT_TOKENS,
    USAGE_METRIC_INPUT_CACHED_TOKENS,
    USAGE_METRIC_TOTAL_TOKENS,
    USAGE_METRIC_API_REQUESTS,
    USAGE_METRIC_AVG_TOKENS_PER_REQUEST,
    USAGE_METRIC_MODELS_CALLED,
    USAGE_METRIC_API_KEYS_USED,
    USAGE_METRIC_LAST_ACTIVE,
    USAGE_METRIC_DATE,
}


class UsageOption(BaseModel):
    key: str
    label: str


class UsageIdentityValue(BaseModel):
    user_name: Optional[str] = None
    api_key_name: Optional[str] = None
    access_key: Optional[str] = None
    api_key_is_custom: Optional[bool] = None
    route_name: Optional[str] = None
    # ``organization_name`` — consumer Org display name (snapshotted on
    # model_usages at ingest, with a live fallback for pre-upgrade rows);
    # ``organization_kind`` — the consumer principal's kind (``org`` / ``user``
    # / ``group``) so the client can tag a personal (USER) row; ``group_name``
    # — user-group display name (filter-only dimension).
    organization_name: Optional[str] = None
    organization_kind: Optional[str] = None
    group_name: Optional[str] = None


class UsageIdentityCurrent(BaseModel):
    user_id: Optional[int] = None
    api_key_id: Optional[int] = None
    route_id: Optional[int] = None
    organization_id: Optional[int] = None
    group_id: Optional[int] = None


class UsageIdentity(BaseModel):
    value: UsageIdentityValue
    current: Optional[UsageIdentityCurrent] = None


class UsageFilterItem(BaseModel):
    identity: UsageIdentity


class UsageFilterOption(UsageFilterItem):
    label: str
    deleted: bool


class UsageFilters(BaseModel):
    users: List[UsageFilterOption] = Field(default_factory=list)
    api_keys: List[UsageFilterOption] = Field(default_factory=list)
    routes: List[UsageFilterOption] = Field(default_factory=list)
    # Platform-wide "All" view only. ``organizations`` — consumer Orgs
    # with usage; ``user_groups`` — groups whose members can be filtered on.
    organizations: List[UsageFilterOption] = Field(default_factory=list)
    user_groups: List[UsageFilterOption] = Field(default_factory=list)


class UsageMetaResponse(BaseModel):
    metrics: List[UsageOption]
    granularities: List[UsageOption]
    group_bys: List[UsageOption]
    filters: UsageFilters


class UsageFilterRequest(BaseModel):
    users: List[UsageFilterItem] = Field(default_factory=list)
    api_keys: List[UsageFilterItem] = Field(default_factory=list)
    routes: List[UsageFilterItem] = Field(default_factory=list)
    # Consumer-Org filter (``consumer_principal_id``) and user-group
    # filter (expanded to the group's direct USER members). Both are
    # platform-admin-only; enforced in the route handler.
    organizations: List[UsageFilterItem] = Field(default_factory=list)
    user_groups: List[UsageFilterItem] = Field(default_factory=list)


class UsageBaseRequest(BaseModel):
    start_date: Date
    end_date: Date
    filters: UsageFilterRequest = Field(default_factory=UsageFilterRequest)
    # See USAGE_SCOPE_* constants. Defaults to "all" so that managers /
    # admins who omit the parameter get the org-wide view; the endpoint
    # downgrades to "self" automatically when the caller has no
    # managerial role (and rejects the request if they explicitly
    # asked for "all").
    scope: str = USAGE_SCOPE_ALL

    @field_validator("end_date")
    @classmethod
    def validate_date_range(cls, value: Date, info) -> Date:
        start_date = info.data.get("start_date")
        if start_date and value < start_date:
            raise ValueError("end_date must be on or after start_date")
        return value

    @field_validator("scope")
    @classmethod
    def validate_scope(cls, value: str) -> str:
        if value not in USAGE_SCOPES:
            raise ValueError(f"Unsupported scope: {value}")
        return value


def _validate_group_by_values(value: List[str]) -> List[str]:
    """Shared by the breakdown request and every export sheet, so a grouping
    that the list endpoint rejects can't sneak in through the export."""
    if not value:
        raise ValueError("group_by must not be empty")
    unsupported = [item for item in value if item not in USAGE_GROUP_BYS]
    if unsupported:
        raise ValueError(f"Unsupported group_by: {', '.join(unsupported)}")
    if len(value) != len(set(value)):
        raise ValueError("group_by must not contain duplicate values")
    return value


class UsageBreakdownRequest(UsageBaseRequest):
    group_by: List[str]
    granularity: Optional[str] = None
    sort_by: Optional[str] = f"-{USAGE_METRIC_TOTAL_TOKENS}"
    page: int = 1
    perPage: int = 20

    @field_validator("group_by")
    @classmethod
    def validate_group_by(cls, value: List[str]) -> List[str]:
        return _validate_group_by_values(value)

    @field_validator("granularity")
    @classmethod
    def validate_granularity(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in USAGE_GRANULARITIES:
            raise ValueError(f"Unsupported granularity: {value}")
        return value

    @field_validator("sort_by")
    @classmethod
    def validate_sort_by(cls, value: Optional[str]) -> Optional[str]:
        if not value:
            return value
        for field in value.split(","):
            field = field.strip()
            if not field:
                continue
            field_name = field[1:] if field.startswith("-") else field
            if field_name not in USAGE_SORTABLE_FIELDS:
                raise InvalidException(
                    f"Field '{field_name}' is not sortable. "
                    f"Allowed fields: {', '.join(sorted(USAGE_SORTABLE_FIELDS))}"
                )
        return value

    @field_validator("perPage")
    @classmethod
    def validate_per_page(cls, value: int) -> int:
        # Generous ceiling (abuse cap only). Kept at 10000 for backward compat
        # with older UIs that fetch the whole set via perPage=10000; new callers
        # use page=-1. Lowering it would 400 cached old UIs on upgrade.
        if value < 1 or value > 10000:
            raise ValueError("perPage must be between 1 and 10000")
        return value

    @field_validator("page")
    @classmethod
    def validate_page(cls, value: int) -> int:
        # ``-1`` is the no-pagination sentinel (return all buckets); otherwise a
        # positive page. Reject 0 and any other negative so a stray value can't
        # slip through as "no pagination" and get echoed back as a bogus
        # ``pagination.page`` (e.g. "page -42 of 1").
        if value != -1 and value < 1:
            raise ValueError("page must be a positive number or -1 (no pagination)")
        return value

    @computed_field
    @property
    def order_by(self) -> List[tuple[str, str]]:
        if not self.sort_by:
            return []
        order_by = []
        for field in self.sort_by.split(","):
            field = field.strip()
            if not field:
                continue
            if field.startswith("-"):
                order_by.append((field[1:], USAGE_SORT_DESC))
            else:
                order_by.append((field, USAGE_SORT_ASC))
        return order_by


class UsageSummary(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    input_cached_tokens: int = 0
    total_tokens: int = 0
    api_requests: int = 0
    models_called: int = 0


class UsageBreakdownDimension(BaseModel):
    identity: Optional[UsageIdentity] = None
    label: str
    deleted: bool


class UsageBreakdownDateDimension(BaseModel):
    value: Date
    label: str
    deleted: bool = False


class UsageBreakdownItem(BaseModel):
    date: Optional[UsageBreakdownDateDimension] = None
    user: Optional[UsageBreakdownDimension] = None
    api_key: Optional[UsageBreakdownDimension] = None
    route: Optional[UsageBreakdownDimension] = None
    organization: Optional[UsageBreakdownDimension] = None
    input_tokens: int = 0
    output_tokens: int = 0
    input_cached_tokens: int = 0
    total_tokens: int = 0
    api_requests: int = 0
    avg_tokens_per_request: float = 0
    models_called: Optional[int] = None
    api_keys_used: Optional[int] = None
    last_active: Optional[Date] = None


class UsageBreakdownResponse(BaseModel):
    summary: UsageSummary
    group_by: List[str]
    granularity: Optional[str] = None
    pagination: Pagination
    items: List[UsageBreakdownItem]


USAGE_EXPORT_FORMAT_CSV = "csv"
USAGE_EXPORT_FORMAT_XLSX = "xlsx"
USAGE_EXPORT_FORMATS = {USAGE_EXPORT_FORMAT_CSV, USAGE_EXPORT_FORMAT_XLSX}

USAGE_EXPORT_SPLIT_AUTO = "auto"
USAGE_EXPORT_SPLITS = {USAGE_EXPORT_SPLIT_AUTO}


class UsageExportSheet(BaseModel):
    """One logical table in an export.

    A sheet is an INDEPENDENT breakdown query — not one more dimension on a
    shared query. That distinction is the whole reason this type exists:
    ``group_by`` is a compound grouping (group by A *and* B), so the number of
    tables can never be inferred from its length. The Tokens tab's chart
    export is a single table with a four-element ``group_by``, while its table
    export is three tables whose ``group_by`` is one element each.
    """

    # Stable machine key. Drives the CSV member name (``by_<key>.csv``) and the
    # xlsx worksheet name when ``name`` is absent. Downstream scripts match on
    # this, never on ``name``, which is localized and free to change.
    key: str
    group_by: List[str]
    # Optional localized display name, used only as the xlsx worksheet title.
    name: Optional[str] = None
    # Optional per-sheet sort override; different dimensions sort differently
    # (a date series by date, a top-N table by tokens).
    sort_by: Optional[str] = None

    @field_validator("key")
    @classmethod
    def validate_key(cls, value: str) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError("sheet key must not be empty")
        # Lands in a zip member path / worksheet name — keep it inert.
        if not all(ch.isalnum() or ch in "_-" for ch in value):
            raise ValueError("sheet key may only contain letters, digits, '_' and '-'")
        return value

    @field_validator("group_by")
    @classmethod
    def validate_group_by(cls, value: List[str]) -> List[str]:
        # Structure only. Which dimension names are legal is endpoint-specific
        # — the token export accepts route / api_key, the GPU export accepts
        # instance_type / instance — so the vocabulary check belongs to the
        # route, not to this shared type.
        if not value:
            raise ValueError("group_by must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("group_by must not contain duplicate values")
        return value


class UsageExportShape(BaseModel):
    """The file-shape knobs every export payload carries, validated once.

    The token and resource exports produce the same kinds of file from the
    same three decisions — one table or several, which format, whether an
    over-large result may be split — and the checks behind them are not
    cosmetic: two sheets sharing a key overwrite each other's row counts and
    parts, and land in the archive as two members with one name. Defining them
    here is what stops one endpoint from being validated less than the other.

    Subclasses redeclare ``group_by`` to validate it, since which dimension
    names are legal is endpoint-specific.
    """

    # xlsx by default: that is what these exports have always produced. The
    # server falls back to CSV only when the result cannot fit a worksheet —
    # see ``USAGE_EXPORT_FORMAT_*``.
    format: str = USAGE_EXPORT_FORMAT_XLSX
    # ``"auto"`` slices an over-large result into files inside one archive
    # instead of rejecting it.
    split: Optional[str] = None
    # Exactly one of ``sheets`` and ``group_by``. ``group_by`` is the
    # single-table form and keeps the payload identical to ``/breakdown``; it
    # is declared here as well as in the subclasses so the shape check below
    # holds for any model that inherits this.
    sheets: Optional[List[UsageExportSheet]] = None
    group_by: Optional[List[str]] = None

    @field_validator("format")
    @classmethod
    def validate_format(cls, value: str) -> str:
        if value not in USAGE_EXPORT_FORMATS:
            raise ValueError(
                f"Unsupported format: {value}. "
                f"Expected one of: {', '.join(sorted(USAGE_EXPORT_FORMATS))}"
            )
        return value

    @field_validator("split")
    @classmethod
    def validate_split(cls, value: Optional[str]) -> Optional[str]:
        # An unrecognized value must not read as "no split": the caller asked
        # for a way around the row limit and would otherwise get a plain
        # over-limit refusal with nothing pointing at the word it sent.
        if value is not None and value not in USAGE_EXPORT_SPLITS:
            raise ValueError(f"Unsupported split: {value}")
        return value

    @field_validator("sheets")
    @classmethod
    def validate_sheets(
        cls, value: Optional[List[UsageExportSheet]]
    ) -> Optional[List[UsageExportSheet]]:
        if value is None:
            return value
        if not value:
            raise ValueError("sheets must not be empty")
        keys = [sheet.key for sheet in value]
        if len(keys) != len(set(keys)):
            raise ValueError("sheets must not contain duplicate keys")
        return value

    @model_validator(mode="after")
    def validate_exclusive_shape(self) -> "UsageExportShape":
        if bool(self.group_by) == bool(self.sheets):
            raise ValueError(
                "exactly one of 'group_by' (single table) or 'sheets' "
                "(multiple tables) must be provided"
            )
        return self

    def default_sheet_sort_by(self) -> Optional[str]:
        """Sort the single-table form gives its one sheet.

        ``None`` unless the payload has a request-level sort to hand down.
        """
        return None

    def resolved_sheets(self) -> List[UsageExportSheet]:
        """Normalize both request shapes to a list of sheets.

        The single-table form becomes one sheet whose key is derived from its
        grouping, so everything downstream — naming, per-sheet row limits,
        estimates — has exactly one shape to handle.
        """
        if self.sheets:
            return self.sheets
        group_by = self.group_by or []
        return [
            UsageExportSheet(
                key="_".join(group_by) or "usage",
                group_by=group_by,
                sort_by=self.default_sheet_sort_by(),
            )
        ]


class UsageExportRequest(UsageBaseRequest, UsageExportShape):
    """``/breakdown/export`` and ``/breakdown/export/estimate`` payload.

    Shares every filter with :class:`UsageBreakdownRequest` so the exported
    rows and the on-screen rows come from the same predicate. The date range,
    ``filters`` and ``scope`` live ONLY at the top level and cannot be
    overridden per sheet: one file must have one set of conditions, or the
    tables inside it can't be reconciled with each other.
    """

    granularity: Optional[str] = None
    # No default: the ROW ORDER OF A FILE is the server's to define, and it
    # depends on the grouping — a date-grouped sheet comes out in date order,
    # anything else by its headline metric. A fixed default here would order a
    # date-grouped export by tokens and shuffle its Date column. An explicit
    # value still wins.
    sort_by: Optional[str] = None

    group_by: Optional[List[str]] = None

    @field_validator("group_by")
    @classmethod
    def validate_group_by(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return value
        return _validate_group_by_values(value)

    @field_validator("granularity")
    @classmethod
    def validate_granularity(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in USAGE_GRANULARITIES:
            raise ValueError(f"Unsupported granularity: {value}")
        return value

    def default_sheet_sort_by(self) -> Optional[str]:
        return self.sort_by

    def to_breakdown_request(self, sheet: UsageExportSheet) -> UsageBreakdownRequest:
        """Build the equivalent breakdown request for one sheet.

        ``page=-1`` is the no-pagination sentinel: an export is by definition
        the whole result set. Going through ``UsageBreakdownRequest`` (rather
        than passing loose arguments around) means the export inherits the
        same field validation the list endpoint enforces.
        """
        return UsageBreakdownRequest(
            start_date=self.start_date,
            end_date=self.end_date,
            filters=self.filters,
            scope=self.scope,
            group_by=sheet.group_by,
            granularity=self.granularity,
            sort_by=sheet.sort_by or self.sort_by,
            page=-1,
        )


class UsageExportColumn(BaseModel):
    """One exported column, as the preview needs to know it.

    ``key`` is what the client reads a value by; ``title`` is the header the
    file will carry. Shipping both means the preview renders exactly the
    columns the file will have without the client re-deriving the list — the
    drift this whole design exists to prevent.
    """

    key: str
    title: str


class UsageExportSheetEstimate(BaseModel):
    key: str
    name: Optional[str] = None
    total: int = 0
    columns: List[UsageExportColumn] = Field(default_factory=list)
    # False when the caller may not run this sheet (e.g. the Organization
    # breakdown outside the platform-wide view). Reported per sheet rather
    # than failing the whole estimate so the UI can grey out just that table.
    available: bool = True
    reason: Optional[str] = None


class UsageExportEstimateResponse(BaseModel):
    sheets: List[UsageExportSheetEstimate]
    total: int = 0
    # Effective ceilings, echoed so the UI doesn't hardcode them and support
    # can see the deployment's values in a bug report.
    soft_limit: int = 0
    hard_limit: int = 0
    # The format the export will actually produce. Differs from the requested
    # one when the result cannot fit an Excel worksheet, so the dialog can say
    # so before the click rather than letting a .csv arrive unannounced.
    effective_format: str = USAGE_EXPORT_FORMAT_XLSX
    # Days that would fit under ``hard_limit`` at the current row density.
    # Computed here — by the same helper the over-limit error uses — so the
    # advice shown before the click and the advice shown after a rejection
    # can't contradict each other.
    suggested_max_days: Optional[int] = None
    # Same remedies the over-limit error carries, so the dialog can offer them
    # as actions BEFORE the user commits instead of only after a failure.
    # Empty when the export fits.
    suggestions: List[dict] = Field(default_factory=list)
    # Files a ``split: "auto"`` export would produce. ``None`` when it fits.
    split_parts: Optional[int] = None
    # The verdicts themselves, not just the numbers to compare.
    #
    # ``total`` is the SUM over sheets — the row count of the whole file, which
    # is what the dialog should show. The limits, though, are enforced PER
    # SHEET (each is its own query and its own worksheet). A client comparing
    # ``total`` against ``hard_limit`` therefore refuses four 30k-row tables
    # that the export endpoint would happily produce. Deciding here, off the
    # same number the export path checks, is the only way the advice before the
    # click cannot contradict the behaviour after it.
    exceeds_soft_limit: bool = False
    exceeds_hard_limit: bool = False
