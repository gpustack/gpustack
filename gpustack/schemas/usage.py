from datetime import date
from typing import List, Optional

from pydantic import BaseModel, Field, computed_field, field_validator

from gpustack.api.exceptions import InvalidException
from gpustack.schemas.common import Pagination


USAGE_METRIC_INPUT_TOKENS = "input_tokens"
USAGE_METRIC_OUTPUT_TOKENS = "output_tokens"
USAGE_METRIC_TOTAL_TOKENS = "total_tokens"
USAGE_METRIC_API_REQUESTS = "api_requests"
USAGE_METRIC_MODELS_CALLED = "models_called"
USAGE_METRIC_API_KEYS_USED = "api_keys_used"
USAGE_METRIC_AVG_TOKENS_PER_REQUEST = "avg_tokens_per_request"
USAGE_METRIC_LAST_ACTIVE = "last_active"

USAGE_GROUP_BY_MODEL = "model"
USAGE_GROUP_BY_USER = "user"
USAGE_GROUP_BY_API_KEY = "api_key"

USAGE_SCOPE_ALL = "all"
USAGE_SCOPE_SELF = "self"

USAGE_GRANULARITY_DAY = "day"
USAGE_GRANULARITY_WEEK = "week"
USAGE_GRANULARITY_MONTH = "month"

USAGE_SORT_ASC = "asc"
USAGE_SORT_DESC = "desc"

USAGE_METRICS = {
    USAGE_METRIC_INPUT_TOKENS,
    USAGE_METRIC_OUTPUT_TOKENS,
    USAGE_METRIC_TOTAL_TOKENS,
    USAGE_METRIC_API_REQUESTS,
}
USAGE_GROUP_BYS = {
    USAGE_GROUP_BY_MODEL,
    USAGE_GROUP_BY_USER,
    USAGE_GROUP_BY_API_KEY,
}
USAGE_SCOPES = {USAGE_SCOPE_ALL, USAGE_SCOPE_SELF}
USAGE_GRANULARITIES = {
    USAGE_GRANULARITY_DAY,
    USAGE_GRANULARITY_WEEK,
    USAGE_GRANULARITY_MONTH,
}
USAGE_SORTABLE_FIELDS = {
    USAGE_METRIC_INPUT_TOKENS,
    USAGE_METRIC_OUTPUT_TOKENS,
    USAGE_METRIC_TOTAL_TOKENS,
    USAGE_METRIC_API_REQUESTS,
    USAGE_METRIC_AVG_TOKENS_PER_REQUEST,
    USAGE_METRIC_MODELS_CALLED,
    USAGE_METRIC_API_KEYS_USED,
    USAGE_METRIC_LAST_ACTIVE,
}


class UsageOption(BaseModel):
    key: str
    label: str


class UsageIdentityValue(BaseModel):
    cluster_name: Optional[str] = None
    model_name: Optional[str] = None
    user_name: Optional[str] = None
    api_key_name: Optional[str] = None
    access_key: Optional[str] = None
    api_key_is_custom: Optional[bool] = None


class UsageIdentityCurrent(BaseModel):
    model_id: Optional[int] = None
    user_id: Optional[int] = None
    api_key_id: Optional[int] = None


class UsageIdentity(BaseModel):
    value: UsageIdentityValue
    current: Optional[UsageIdentityCurrent] = None


class UsageFilterItem(BaseModel):
    identity: UsageIdentity


class UsageFilterOption(UsageFilterItem):
    label: str
    deleted: bool


class UsageFilters(BaseModel):
    models: List[UsageFilterOption] = Field(default_factory=list)
    users: List[UsageFilterOption] = Field(default_factory=list)
    api_keys: List[UsageFilterOption] = Field(default_factory=list)


class UsageMetaResponse(BaseModel):
    scopes: List[UsageOption] = Field(default_factory=list)
    metrics: List[UsageOption]
    granularities: List[UsageOption]
    group_bys: List[UsageOption]
    filters: UsageFilters


class UsageFilterRequest(BaseModel):
    models: List[UsageFilterItem] = Field(default_factory=list)
    users: List[UsageFilterItem] = Field(default_factory=list)
    api_keys: List[UsageFilterItem] = Field(default_factory=list)


class UsageBaseRequest(BaseModel):
    start_date: date
    end_date: date
    scope: str = USAGE_SCOPE_SELF
    filters: UsageFilterRequest = Field(default_factory=UsageFilterRequest)

    @field_validator("scope")
    @classmethod
    def validate_scope(cls, value: str) -> str:
        if value not in USAGE_SCOPES:
            raise ValueError(f"Unsupported scope: {value}")
        return value

    @field_validator("end_date")
    @classmethod
    def validate_date_range(cls, value: date, info) -> date:
        start_date = info.data.get("start_date")
        if start_date and value < start_date:
            raise ValueError("end_date must be on or after start_date")
        return value


class UsageTimeSeriesRequest(UsageBaseRequest):
    metric: str
    group_by: str
    granularity: str = USAGE_GRANULARITY_DAY

    @field_validator("metric")
    @classmethod
    def validate_metric(cls, value: str) -> str:
        if value not in USAGE_METRICS:
            raise ValueError(f"Unsupported metric: {value}")
        return value

    @field_validator("group_by")
    @classmethod
    def validate_group_by(cls, value: str) -> str:
        if value not in USAGE_GROUP_BYS:
            raise ValueError(f"Unsupported group_by: {value}")
        return value

    @field_validator("granularity")
    @classmethod
    def validate_granularity(cls, value: str) -> str:
        if value not in USAGE_GRANULARITIES:
            raise ValueError(f"Unsupported granularity: {value}")
        return value


class UsageBreakdownRequest(UsageBaseRequest):
    group_by: str
    sort_by: Optional[str] = f"-{USAGE_METRIC_TOTAL_TOKENS}"
    page: int = 1
    perPage: int = 20

    @field_validator("group_by")
    @classmethod
    def validate_group_by(cls, value: str) -> str:
        if value not in USAGE_GROUP_BYS:
            raise ValueError(f"Unsupported group_by: {value}")
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

    @field_validator("page", "perPage")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value < 1:
            raise ValueError("page and perPage must be positive")
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
    total_tokens: int = 0
    api_requests: int = 0
    models_called: int = 0


class UsageTimelinePoint(BaseModel):
    date: date
    value: int


class UsageSeries(BaseModel):
    identity: UsageIdentity
    label: str
    deleted: bool
    timeline: List[UsageTimelinePoint]


class UsageTimeSeriesResponse(BaseModel):
    summary: UsageSummary
    metric: str
    group_by: str
    granularity: str
    series: List[UsageSeries]


class UsageBreakdownItem(BaseModel):
    identity: UsageIdentity
    label: str
    deleted: bool
    cluster_name: Optional[str] = None
    model_name: Optional[str] = None
    user_name: Optional[str] = None
    api_key_name: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    api_requests: int = 0
    avg_tokens_per_request: float = 0
    models_called: Optional[int] = None
    api_keys_used: Optional[int] = None
    last_active: Optional[date] = None


class UsageBreakdownResponse(BaseModel):
    group_by: str
    pagination: Pagination
    items: List[UsageBreakdownItem]
