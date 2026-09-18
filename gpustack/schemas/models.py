from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Union
from croniter import croniter
from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    field_validator,
    model_validator,
)
from sqlalchemy import JSON, Column, ForeignKey, Integer, UniqueConstraint
from sqlalchemy import false as sa_false
from sqlalchemy.orm import selectinload
from sqlmodel import Field, Relationship, SQLModel, Text, select

from gpustack.schemas.common import (
    ListParams,
    PaginatedList,
    UTCDateTime,
    pydantic_column_type,
)
from gpustack.mixins import BaseModelMixin
from gpustack.schemas.links import (
    ModelInstanceDraftModelFileLink,
    ModelInstanceModelFileLink,
)
from gpustack.utils.command import find_parameter, find_bool_parameter
from gpustack.schemas.model_routes import (
    ModelRoute,
    ModelRouteTarget,
    AccessPolicyEnum,
)
from gpustack.schemas.principals import _platform_principal_id
from gpustack.schemas.cache_services import CacheConfigSnapshot

if TYPE_CHECKING:
    from gpustack.schemas.model_files import ModelFile
    from gpustack.schemas.clusters import Cluster

# Models


class SourceEnum(str, Enum):
    HUGGING_FACE = "huggingface"
    MODEL_SCOPE = "model_scope"
    LOCAL_PATH = "local_path"


class CategoryEnum(str, Enum):
    LLM = "llm"
    EMBEDDING = "embedding"
    IMAGE = "image"
    RERANKER = "reranker"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    UNKNOWN = "unknown"


class PlacementStrategyEnum(str, Enum):
    SPREAD = "spread"
    BINPACK = "binpack"


class BackendEnum(str, Enum):
    VLLM = "vLLM"
    VOX_BOX = "VoxBox"
    ASCEND_MINDIE = "MindIE"
    SGLANG = "SGLang"
    CUSTOM = "Custom"


class BackendSourceEnum(str, Enum):
    CUSTOM = "custom"
    BUILT_IN = "built_in"
    COMMUNITY = "community"


class SpeculativeAlgorithmEnum(str, Enum):
    EAGLE3 = "eagle3"
    MTP = "mtp"
    NGRAM = "ngram"


class GPUSelector(BaseModel):
    # format of each element: "worker_name:device:gpu_index", example: "worker1:cuda:0"
    gpu_ids: Optional[List[str]] = None
    gpus_per_replica: Optional[int] = None


class GPUTypeSelector(BaseModel):
    """
    Selects a sliced GPU from a gpustack-operator InstanceType pool.

    Field names mirror ``GPUInstanceResources`` / the operator's
    InstanceResources conventions.

    Mutually exclusive with manual GPU selection: ``gpu_selector.gpu_ids`` must
    be empty, since the card is chosen by the operator's device plugin, not by
    index. A ``gpu_selector`` is otherwise allowed — this implies exactly one
    card per worker per replica, so ``gpus_per_replica`` is constrained to 1
    rather than rejected.
    """

    type: str
    """
    Name of the operator InstanceType (pool) to schedule onto.
    """

    accelerator_sliced_memory_percentage: Optional[int] = Field(
        default=None, ge=0, le=100
    )
    """
    Per-card VRAM budget requested on a sliced InstanceType, as a percentage.
    Required (in [1,100]) for a sliced request; 0 is valid only together with
    a 0/unset cores percentage and means a whole-card exclusive request.
    """

    accelerator_sliced_cores_percentage: Optional[int] = Field(
        default=None, ge=0, le=100
    )
    """
    Per-card compute budget requested on a sliced InstanceType, as a
    percentage in [1,100]; an independent dimension from memory. Defaults to
    100 when unset on a sliced request (operator webhook defaulting rule). 0
    is valid only together with a 0/unset memory percentage (whole-card
    exclusive).
    """

    accelerator_partitioned_profile: Optional[str] = None
    """
    Hardware partition profile requested on a partition-offering InstanceType,
    e.g. "1g.5gb". Mutually exclusive with non-zero slice percentages:
    hardware partitioning and software slicing cannot both apply to one card.
    """

    @model_validator(mode="after")
    def normalize_slice_percentages(self):
        if self.accelerator_partitioned_profile:
            # Slicing percentages don't apply to hardware partitioning; their
            # exclusivity with a profile is enforced by route validation.
            return self

        memory = self.accelerator_sliced_memory_percentage
        cores = self.accelerator_sliced_cores_percentage
        memory_sliced = memory is not None and memory > 0
        cores_sliced = cores is not None and cores > 0

        if not memory_sliced and not cores_sliced:
            # Whole-card exclusive mode: valid only as both-0 (or both-unset);
            # normalize unset to 0.
            self.accelerator_sliced_memory_percentage = 0
            self.accelerator_sliced_cores_percentage = 0
            return self

        if not memory_sliced:
            # Covers both "cores set, memory unset" and the mixed
            # "memory 0, cores non-zero" case: memory is required (and
            # non-zero) for any sliced request.
            raise ValueError(
                "accelerator_sliced_memory_percentage is required in the "
                "range 1-100 for a sliced request; 0 is only valid when both "
                "percentages are 0 (whole-card exclusive)"
            )
        if cores is not None and not cores_sliced:
            raise ValueError(
                "accelerator_sliced_cores_percentage must be in the range "
                "1-100; 0 is only valid when both percentages are 0 "
                "(whole-card exclusive)"
            )
        if cores is None:
            # Mirror the operator webhook: cores defaults to 100 when unset.
            self.accelerator_sliced_cores_percentage = 100
        return self


class LoraListEntry(BaseModel):
    """
    One LoRA adapter configured on a base Model (download + runtime + optional route).
    """

    lora_name: str = Field(..., min_length=1)
    """Fully-qualified LoRA id in the form "<base_model_name>:<suffix>". The API
    strips the prefix on the way out (see ModelPublic._strip_lora_prefix), so
    clients only ever see/enter the bare short name."""

    lora_repo_name: Optional[str] = None
    """HuggingFace repo id, ModelScope model id, or absolute filesystem path
    (used as a fallback when source=local_path and local_path is empty)."""

    source: str = SourceEnum.HUGGING_FACE.value
    huggingface_filename: Optional[str] = None
    model_scope_file_path: Optional[str] = None
    local_path: Optional[str] = None

    # Runtime fields populated when mounted on an instance.
    path: Optional[str] = None
    """Resolved filesystem path when mounted on an instance."""
    model_file_id: Optional[int] = None
    """ID of the ModelFile record backing this adapter."""


class KVCacheModeEnum(str, Enum):
    LOCAL = "local"
    SHARED = "shared"

    def __str__(self):
        return self.value


class ExtendedKVCacheConfig(BaseModel):
    enabled: bool = False
    """ Enable extended KV cache for the model."""

    mode: Optional[KVCacheModeEnum] = KVCacheModeEnum.LOCAL
    """ "local": per-instance cache offloaded to CPU memory. "shared": attach to a shared cache service. Absent means local. """

    cache_service_id: Optional[int] = None
    """ ID of the CacheService to attach to. Required when mode is "shared". """

    ram_ratio: Optional[float] = 1.2
    """ RAM-to-VRAM ratio for KV cache. For example, 2.0 means the RAM is twice the size of the VRAM. """

    ram_size: Optional[int] = None
    """ Maximum size of the KV cache to be stored in local CPU memory (unit: GiB). Overrides ram_ratio if both are set. """

    chunk_size: Optional[int] = None
    """ Size for each KV cache chunk (unit: number of tokens). """

    def is_shared(self) -> bool:
        return bool(self.enabled and self.mode == KVCacheModeEnum.SHARED)

    def is_local(self) -> bool:
        return bool(self.enabled and not self.is_shared())


# A window may span at most a year. Longer values are meaningless for a
# recurring schedule and overflow the timedelta used to compute the window end.
MAX_SCALING_WINDOW_SECONDS = 366 * 24 * 3600


def _assert_satisfiable_cron(expr: str) -> None:
    """Reject cron expressions that parse but can never fire.

    ``croniter.is_valid`` accepts impossible dates such as ``0 0 30 2 *``
    (February 30th); the window would simply never open while the scheduler
    logged an evaluation failure on every tick. Resolve an occurrence to prove
    the expression is reachable.
    """
    try:
        croniter(expr).get_next(datetime)
    except Exception as e:
        raise ValueError(f"Invalid cron expression: {expr!r} ({e})")


class ScalingScheduleRule(BaseModel):
    """
    One scheduled-scaling window (GCP scaling-schedule / KEDA Cron scaler
    semantics). ``start_cron`` fires the window open; the window stays open for
    ``duration_seconds``. While ``now`` falls inside the window the model's
    replicas is driven to this rule's ``replicas``. Outside every rule's window
    the model falls back to the schedule's ``baseline_replicas``. Multiple rules
    cover multiple windows (e.g. day / night). A start + duration model (rather
    than start + end) expresses windows that cross midnight / span whole days
    (e.g. a weekend) without wrap-around ambiguity.
    """

    start_cron: str = ""
    """Cron marking the window start, e.g. "0 8 * * *" (every day at 08:00)."""
    duration_seconds: Optional[int] = Field(
        default=None, gt=0, le=MAX_SCALING_WINDOW_SECONDS
    )
    """How long the window stays open after ``start_cron`` fires, in seconds.
    Capped at a year: the window end is computed as a ``timedelta``, which
    overflows (and would surface as a 500) for astronomically large values."""
    replicas: int = Field(ge=0)
    """Desired replica count while ``now`` is inside this window."""
    name: Optional[str] = None
    """Optional human-readable label, e.g. "daytime"."""

    @field_validator("start_cron")
    @classmethod
    def validate_cron(cls, v: str) -> str:
        # Empty is allowed for a not-yet-filled rule (e.g. a disabled schedule
        # or a freshly added row). Enabled schedules require a non-empty cron
        # for every rule — that check lives in ScalingSchedule below.
        if v:
            _assert_satisfiable_cron(v)
        return v


class ScalingSchedule(BaseModel):
    """Scheduled scaling configuration attached to a Model."""

    enabled: bool = False
    """Whether scheduled scaling drives this model's replicas."""
    baseline_replicas: Optional[int] = Field(default=None, ge=0)
    """Replica count when ``now`` is outside every rule window. Required while
    the schedule is enabled — together with ``rules`` it is the sole input to
    the effective replica count, and the model's ``replicas`` field becomes a
    scheduler-driven value rather than a user setting."""
    rules: List[ScalingScheduleRule] = Field(default_factory=list)
    """Window rules. Order does not matter: when windows overlap the one that
    started most recently wins, and windows sharing a start instant resolve to
    the largest replica count."""

    @model_validator(mode="after")
    def validate_schedule(self):
        # Only a live schedule is held to the "every rule must have valid
        # crons" bar. A disabled schedule may carry incomplete rows (they are
        # ignored at runtime), so don't 422 on them — this also keeps the
        # real-time preview from rejecting in-progress edits.
        if not self.enabled:
            return self

        if self.baseline_replicas is None:
            raise ValueError(
                "baseline_replicas is required when scaling schedule is enabled."
            )

        if not self.rules:
            raise ValueError(
                "At least one rule is required when scaling schedule is enabled."
            )

        # Field validators already proved every non-empty cron is satisfiable and
        # every duration positive; a live schedule additionally requires both to
        # be present on every rule.
        for rule in self.rules:
            if not rule.start_cron:
                raise ValueError(
                    "start_cron is required for every rule when scaling schedule "
                    "is enabled."
                )
            if not rule.duration_seconds:
                raise ValueError(
                    "duration_seconds is required for every rule when scaling "
                    "schedule is enabled."
                )
        return self


class ModelSource(BaseModel):
    source: SourceEnum
    huggingface_repo_id: Optional[str] = None
    huggingface_filename: Optional[str] = None
    model_scope_model_id: Optional[str] = None
    model_scope_file_path: Optional[str] = None
    local_path: Optional[str] = None

    @property
    def model_source_key(self) -> str:
        """Returns a unique identifier for the model, independent of quantization."""
        if self.source == SourceEnum.HUGGING_FACE:
            return self.huggingface_repo_id or ""
        elif self.source == SourceEnum.MODEL_SCOPE:
            return self.model_scope_model_id or ""
        elif self.source == SourceEnum.LOCAL_PATH:
            return self.local_path or ""
        return ""

    @property
    def readable_source(self) -> str:
        values = []
        if self.source == SourceEnum.HUGGING_FACE:
            values.extend([self.huggingface_repo_id, self.huggingface_filename])
        elif self.source == SourceEnum.MODEL_SCOPE:
            values.extend([self.model_scope_model_id, self.model_scope_file_path])
        elif self.source == SourceEnum.LOCAL_PATH:
            values.extend([self.local_path])

        return "/".join([value for value in values if value is not None])

    @property
    def model_source_index(self) -> str:
        values = []
        if self.source == SourceEnum.HUGGING_FACE:
            values.extend([self.huggingface_repo_id, self.huggingface_filename])
        elif self.source == SourceEnum.MODEL_SCOPE:
            values.extend(
                [self.source, self.model_scope_model_id, self.model_scope_file_path]
            )
        elif self.source == SourceEnum.LOCAL_PATH:
            values.extend([self.local_path])

        # Filter out None values and join
        filtered_values = [v for v in values if v is not None]
        source_string = "/".join(filtered_values)
        return hashlib.sha256(source_string.encode()).hexdigest()

    @model_validator(mode="after")
    def check_huggingface_fields(self):
        if self.source == SourceEnum.HUGGING_FACE:
            if not self.huggingface_repo_id:
                raise ValueError(
                    "huggingface_repo_id must be provided "
                    "when source is 'huggingface'"
                )

        if self.source == SourceEnum.MODEL_SCOPE:
            if not self.model_scope_model_id:
                raise ValueError(
                    "model_scope_model_id must be provided when source is 'model_scope'"
                )

        if self.source == SourceEnum.LOCAL_PATH:
            if not self.local_path:
                raise ValueError(
                    "local_path must be provided when source is 'local_path'"
                )
        return self

    model_config = ConfigDict(protected_namespaces=())


class SpeculativeConfig(BaseModel):
    """Configuration for speculative decoding."""

    enabled: bool = False
    """Whether speculative decoding is enabled."""
    algorithm: Optional[SpeculativeAlgorithmEnum] = None
    """The algorithm to use for speculative decoding."""
    draft_model: Optional[str] = None
    """The draft model to use for speculative decoding.

    It can be a draft model name from the model catalog, a local path or a model ID from the main model source."""
    num_draft_tokens: Optional[int] = None
    """The number of draft tokens."""
    # For ngram only
    ngram_min_match_length: Optional[int] = None
    """Minimum length of the n-gram to match."""
    ngram_max_match_length: Optional[int] = None
    """Maximum length of the n-gram to match."""


class ModelSpecBase(SQLModel, ModelSource):
    name: str = Field(index=True)
    description: Optional[str] = Field(
        sa_type=Text,
        nullable=True,
        default=None,
    )
    meta: Optional[Dict[str, Any]] = Field(sa_type=JSON, default={})

    replicas: int = Field(default=1, ge=0)
    ready_replicas: int = Field(default=0, ge=0)
    categories: List[str] = Field(sa_type=JSON, default=[])
    placement_strategy: PlacementStrategyEnum = PlacementStrategyEnum.SPREAD
    cpu_offloading: Optional[bool] = None
    distributed_inference_across_workers: Optional[bool] = None
    worker_selector: Optional[Dict[str, str]] = Field(sa_type=JSON, default={})
    gpu_selector: Optional[GPUSelector] = Field(
        sa_type=pydantic_column_type(GPUSelector), default=None
    )
    gpu_type_selector: Optional[GPUTypeSelector] = Field(
        sa_type=pydantic_column_type(GPUTypeSelector), default=None
    )

    backend: Optional[str] = None
    backend_version: Optional[str] = None
    backend_parameters: Optional[List[str]] = Field(sa_type=JSON, default=None)
    image_name: Optional[str] = None
    run_command: Optional[str] = Field(sa_type=Text, default=None)
    # Whether this deployment's inference server implements the Anthropic
    # Messages API itself, letting the gateway forward an inbound /v1/messages
    # untouched instead of translating it to /v1/chat/completions. False, the
    # pre-existing behavior, still serves /v1/messages -- by translating.
    #
    # A statement about the server, not about the gateway: what the operator
    # knows is whether their image is a recent enough vLLM, not what ai-proxy
    # does with that fact.
    #
    # Declared on the deployment rather than derived from its inference backend
    # because the answer belongs to the running image, and the image is settled
    # per instance (``ModelInstance.gpu_type`` picks it): one deployment can
    # spread over workers of different accelerators whose images need not agree
    # -- vllm-ascend against vllm-openai. A single ai-proxy provider entry
    # covers the whole deployment, so no per-image source can answer for it.
    #
    # Not nullable: with NULL and False meaning the same thing there would be
    # two spellings of "no" and nothing to tell a caller which to send.
    native_anthropic_api: bool = Field(
        default=False,
        nullable=False,
        sa_column_kwargs={"server_default": sa_false()},
    )

    env: Optional[Dict[str, str]] = Field(sa_type=JSON, default=None)
    restart_on_error: Optional[bool] = True
    distributable: Optional[bool] = False

    # Extended KV Cache configuration. Maps to LMCache in vLLM, and to SGLang's native HiCache (LMCache in shared mode).
    extended_kv_cache: Optional[ExtendedKVCacheConfig] = Field(
        sa_type=pydantic_column_type(ExtendedKVCacheConfig), default=None
    )

    speculative_config: Optional[SpeculativeConfig] = Field(
        sa_type=pydantic_column_type(SpeculativeConfig), default=None
    )

    # Scheduled scaling: drives `replicas` on a cron timetable.
    scaling_schedule: Optional[ScalingSchedule] = Field(
        sa_type=pydantic_column_type(ScalingSchedule), default=None
    )

    # Enable generic proxy for model, the control of generic proxy
    # is migrated to ModelAccess. Keeping this field for backward compatibility
    generic_proxy: Optional[bool] = Field(default=False)

    lora_list: Optional[List[LoraListEntry]] = Field(
        default=None,
        sa_column=Column(pydantic_column_type(List[LoraListEntry]), nullable=True),
    )

    @model_validator(mode="after")
    def set_defaults(self):
        backend = get_backend(self)
        if self.distributed_inference_across_workers is None:
            self.distributed_inference_across_workers = (
                True
                if backend
                in [BackendEnum.VLLM, BackendEnum.ASCEND_MINDIE, BackendEnum.SGLANG]
                else False
            )
        return self


class ModelBase(ModelSpecBase):
    cluster_id: Optional[int] = Field(default=None, foreign_key="clusters.id")
    owner_principal_id: int = Field(
        default_factory=_platform_principal_id,
        sa_column=Column(
            Integer,
            ForeignKey("principals.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )
    # Deprecated field, kept for backward compatibility
    access_policy: AccessPolicyEnum = Field(default=AccessPolicyEnum.AUTHED)


class Model(ModelBase, BaseModelMixin, table=True):
    __tablename__ = 'models'
    __table_args__ = (
        # Model names are unique within their owning Org — two Orgs
        # can each have a "qwen3-0.6b" without colliding.
        UniqueConstraint(
            'owner_principal_id', 'name', name='uix_models_name_per_owner'
        ),
    )
    id: Optional[int] = Field(default=None, primary_key=True)

    instances: list["ModelInstance"] = Relationship(
        sa_relationship_kwargs={"cascade": "delete", "lazy": "noload"},
        back_populates="model",
    )

    cluster: "Cluster" = Relationship(
        back_populates="cluster_models",
        sa_relationship_kwargs={"lazy": "noload"},
    )

    model_route_targets: List["ModelRouteTarget"] = Relationship(
        back_populates="model",
        sa_relationship_kwargs={
            "lazy": "noload",
            "overlaps": "models",
            "cascade": "delete",
        },
    )

    model_routes: List["ModelRoute"] = Relationship(
        back_populates="models",
        link_model=ModelRouteTarget,
        sa_relationship_kwargs={
            "lazy": "noload",
            "overlaps": "model,model_route_targets,route_targets,model_route",
        },
    )


class ModelListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "name",
        "source",
        "cluster_id",
        "replicas",
        "ready_replicas",
        "created_at",
        "updated_at",
    ]


class ModelCreate(ModelBase):
    enable_model_route: Optional[bool] = Field(default=None)


class ModelUpdate(ModelBase):
    pass


class ModelPublic(
    ModelBase,
):
    id: int
    created_at: datetime
    updated_at: datetime
    # Populated only by the detail endpoint; None on list responses.
    has_stale_lora_instances: Optional[bool] = None

    @field_serializer("lora_list")
    def _strip_lora_prefix(self, lora_list, _info):
        """Hide the internal "<base>:" prefix; clients only see the short name."""
        if not lora_list:
            return lora_list
        prefix = f"{self.name}:"
        out = []
        for entry in lora_list:
            data = entry.model_dump() if isinstance(entry, BaseModel) else dict(entry)
            name = data.get("lora_name") or ""
            if name.startswith(prefix):
                data["lora_name"] = name[len(prefix) :]
            out.append(data)
        return out


ModelsPublic = PaginatedList[ModelPublic]


# Model Instances


class ModelInstanceStateEnum(str, Enum):
    r"""
    Enum for Model Instance State

    Transitions:

       |- - - - - Scheduler - - - - |- - ServeManager - -|- - - - Controller - - - -|- ServeManager -|
       |                            |                    |                          |                |
    PENDING ---> ANALYZING ---> SCHEDULED ---> INITIALIZING ---> DOWNLOADING ---> STARTING ---> RUNNING
                     |            ^  |               |                |               |          ^
                     |            |  |               |                |               |          |(Worker ready)
                     |------------|--|---------------|----------------|---------------|----------|
                     \____________|_____________________________________________________________/|
                                  |                  ERROR                                       |(Worker unreachable)
                                  └--------------------┘                                         v
                                    (Restart on Error)                                       UNREACHABLE
    """

    INITIALIZING = "initializing"
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    SCHEDULED = "scheduled"
    ERROR = "error"
    DOWNLOADING = "downloading"
    ANALYZING = "analyzing"
    UNREACHABLE = "unreachable"

    def __str__(self):
        return self.value


class ComputedResourceClaim(BaseModel):
    is_unified_memory: Optional[bool] = False
    offload_layers: Optional[int] = None
    total_layers: Optional[int] = None
    ram: Optional[int] = Field(default=None)  # in bytes
    vram: Optional[Dict[int, int]] = Field(default=None)  # in bytes
    tensor_split: Optional[List[int]] = Field(default=None)
    vram_utilization: Optional[float] = Field(default=None)


class ModelInstanceSubordinateWorker(BaseModel):
    worker_id: Optional[int] = None
    worker_name: Optional[str] = None
    worker_ip: Optional[str] = None
    worker_ifname: Optional[str] = None
    total_gpus: Optional[int] = None
    gpu_type: Optional[str] = None
    gpu_indexes: Optional[List[int]] = Field(sa_column=Column(JSON), default=[])
    gpu_addresses: Optional[List[str]] = Field(sa_column=Column(JSON), default=[])
    computed_resource_claim: Optional[ComputedResourceClaim] = Field(
        sa_column=Column(pydantic_column_type(ComputedResourceClaim)), default=None
    )
    # - For model file preparation
    download_progress: Optional[float] = None
    # - For model instance serving preparation
    pid: Optional[int] = None
    ports: Optional[List[int]] = Field(sa_column=Column(JSON), default=[])
    arguments: Optional[List[str]] = Field(sa_column=Column(JSON), default=[])
    state: ModelInstanceStateEnum = ModelInstanceStateEnum.PENDING
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    # Same write-once-per-streak contract as ModelInstanceBase, for the
    # subordinate's own failure. This is where a multi-node root cause lives:
    # the main worker only ever reports a summary built from a subordinate's
    # state_message, so once the restart clears that, the true first failure is
    # gone and only cascading peer errors remain (issue #6019). Embedded in the
    # distributed_servers JSON column, so no schema migration. Plain fields: this
    # model is not a table, so an sa_column here would be inert decoration (the
    # neighbouring state_message still carries one).
    first_failure_message: Optional[str] = None
    first_failure_at: Optional[datetime] = None


class DistributedServerCoordinateModeEnum(Enum):
    # DELEGATED means that the subordinate workers' coordinate is by-pass to other framework.
    DELEGATED = "delegated"
    # INITIALIZE_LATER means that the subordinate workers' coordinate is handled by GPUStack,
    # all subordinate workers belong to one model instance SHOULD start after the main worker initializes.
    # For example, Ascend MindIE/vLLM/SGLang instances need to start their subordinate workers after the main worker initializes.
    INITIALIZE_LATER = "initialize_later"
    # RUN_FIRST means that the subordinate workers' coordinate is handled by GPUStack,
    # all subordinate workers belong to one model instance MUST get ready before the main worker starts.
    RUN_FIRST = "run_first"


class DistributedServers(BaseModel):
    # Indicates how the distributed servers coordinate with the main worker.
    mode: DistributedServerCoordinateModeEnum = (
        DistributedServerCoordinateModeEnum.DELEGATED
    )
    # Indicates if subordinate workers should download model files.
    download_model_files: Optional[bool] = True
    subordinate_workers: Optional[List[ModelInstanceSubordinateWorker]] = Field(
        sa_column=Column(JSON), default=[]
    )
    model_config = ConfigDict(from_attributes=True)


@dataclass
class ModelInstanceDeploymentMetadata:
    """
    Metadata for model instance deployment.
    """

    name: str
    """
    Name for model instance deployment.
    """
    distributed: bool = False
    """
    Whether the model instance is deployed in distributed mode.
    """
    distributed_leader: bool = False
    """
    Whether the model instance is the leader in distributed mode.
    """
    distributed_follower: bool = False
    """
    Whether the model instance is a follower in distributed mode.
    """
    distributed_follower_index: Optional[int] = None
    """
    Index of the follower in distributed mode.
    It is None for leader or non-distributed mode.
    """


class ModelInstanceBase(SQLModel, ModelSource):
    name: str = Field(index=True, unique=True)
    worker_id: Optional[int] = None
    worker_name: Optional[str] = None
    worker_advertise_address: Optional[str] = None
    worker_ip: Optional[str] = None
    worker_ifname: Optional[str] = None
    pid: Optional[int] = None
    # FIXME: Migrate to ports.
    port: Optional[int] = None
    ports: Optional[List[int]] = Field(sa_column=Column(JSON), default=[])
    download_progress: Optional[float] = None
    resolved_path: Optional[str] = None
    draft_model_source: Optional[ModelSource] = Field(
        sa_column=Column(pydantic_column_type(ModelSource)), default=None
    )
    draft_model_download_progress: Optional[float] = None
    draft_model_resolved_path: Optional[str] = None
    restart_count: Optional[int] = 0
    last_restart_time: Optional[datetime] = Field(
        sa_column=Column(UTCDateTime), default=None
    )
    state: ModelInstanceStateEnum = ModelInstanceStateEnum.PENDING
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    # The failure that started the current unhealthy streak. Restarting clears
    # state_message, so without this the recovery destroys the reason for the
    # failure it recovered from, and a crash loop leaves only the latest
    # secondary error behind (issue #6019). Written once per streak and cleared
    # when the instance reaches RUNNING again, so it describes the streak in
    # progress rather than some unrelated failure months earlier.
    first_failure_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    first_failure_at: Optional[datetime] = Field(
        sa_column=Column(UTCDateTime, nullable=True), default=None
    )
    computed_resource_claim: Optional[ComputedResourceClaim] = Field(
        sa_column=Column(pydantic_column_type(ComputedResourceClaim)), default=None
    )
    cache_config: Optional[CacheConfigSnapshot] = Field(
        sa_column=Column(pydantic_column_type(CacheConfigSnapshot)), default=None
    )
    """Resolved shared-cache connection info; None for local/disabled KV cache."""
    gpu_type: Optional[str] = None
    gpu_indexes: Optional[List[int]] = Field(sa_column=Column(JSON), default=[])
    gpu_addresses: Optional[List[str]] = Field(sa_column=Column(JSON), default=[])

    model_id: int = Field(default=None, foreign_key="models.id")
    model_name: str

    backend: Optional[str] = None
    backend_version: Optional[str] = None
    api_detected_backend_version: Optional[str] = None
    injected_backend_parameters: Optional[List[str]] = Field(
        sa_column=Column(JSON), default=None
    )

    distributed_servers: Optional[DistributedServers] = Field(
        sa_column=Column(pydantic_column_type(DistributedServers)), default=None
    )
    # The "model_id" field conflicts with the protected namespace "model_" in Pydantic.
    # Disable it given that it's not a real issue for this particular field.
    model_config = ConfigDict(protected_namespaces=())

    cluster_id: Optional[int] = Field(default=None, foreign_key="clusters.id")
    owner_principal_id: int = Field(
        default_factory=_platform_principal_id,
        sa_column=Column(
            Integer,
            ForeignKey("principals.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )

    mounted_loras: Optional[List[LoraListEntry]] = Field(
        default=None,
        sa_column=Column(pydantic_column_type(List[LoraListEntry]), nullable=True),
    )

    @property
    def spans_workers(self) -> bool:
        """Whether this instance is actually placed across several
        workers (subordinate workers assigned at scheduling) — the
        placement fact, as opposed to the model's
        distributed_inference_across_workers permission flag."""
        dservers = self.distributed_servers
        return bool(dservers and dservers.subordinate_workers)

    def get_deployment_metadata(
        self,
        worker_id: int,
    ) -> Optional[ModelInstanceDeploymentMetadata]:
        """
        Get the deployment metadata for the model instance.

        Args:
            worker_id:
                The ID of the worker to get the deployment metadata for.

        Returns:
            The deployment metadata,
            or None if the model instance is not handling by the given `worker_id` worker.
        """

        dservers = self.distributed_servers
        subworkers = (
            dservers.subordinate_workers
            if dservers and dservers.subordinate_workers
            else []
        )

        name = self.name
        distributed = bool(subworkers)
        distributed_leader = distributed and self.worker_id == worker_id
        distributed_follower = distributed and not distributed_leader
        distributed_follower_index = None
        if distributed_follower:
            for idx, subworker in enumerate(subworkers):
                if subworker.worker_id == worker_id:
                    distributed_follower_index = idx
                    break
            if distributed_follower_index is not None:
                # Mutate the name to include the follower index,
                # so that each follower has a unique name.
                name += f"-f{distributed_follower_index}"

        if self.worker_id != worker_id and distributed_follower_index is None:
            # This model instance is not handling by the given worker.
            return None

        return ModelInstanceDeploymentMetadata(
            name=name,
            distributed=distributed,
            distributed_leader=distributed_leader,
            distributed_follower=distributed_follower,
            distributed_follower_index=distributed_follower_index,
        )


class ModelInstance(ModelInstanceBase, BaseModelMixin, table=True):
    __tablename__ = 'model_instances'
    id: Optional[int] = Field(default=None, primary_key=True)

    model: Optional[Model] = Relationship(
        back_populates="instances",
        sa_relationship_kwargs={"lazy": "noload"},
    )

    model_files: List["ModelFile"] = Relationship(
        back_populates="instances",
        link_model=ModelInstanceModelFileLink,
        sa_relationship_kwargs={"lazy": "noload"},
    )

    draft_model_files: List["ModelFile"] = Relationship(
        back_populates="draft_instances",
        link_model=ModelInstanceDraftModelFileLink,
        sa_relationship_kwargs={"lazy": "noload"},
    )

    cluster: "Cluster" = Relationship(
        back_populates="cluster_model_instances",
        sa_relationship_kwargs={"lazy": "noload"},
    )

    @classmethod
    async def one_by_id_with_model_files(
        cls,
        session,
        instance_id: int,
        populate_existing: bool = True,
    ) -> Optional["ModelInstance"]:
        """Load a model instance with primary/LoRA + draft model_files and model spec eagerly loaded."""
        stmt = (
            select(cls)
            .where(cls.id == instance_id)
            .options(
                selectinload(cls.model_files),
                selectinload(cls.draft_model_files),
                selectinload(cls.model),
            )
        )
        if populate_existing:
            stmt = stmt.execution_options(populate_existing=True)
        return (await session.exec(stmt)).first()

    # overwrite the hash to use in uniquequeue
    def __hash__(self):
        return self.id


class ModelInstanceCreate(ModelInstanceBase):
    pass


class ModelInstanceUpdate(ModelInstanceBase):
    pass


class ModelInstancePublic(
    ModelInstanceBase,
):
    id: int
    created_at: datetime
    updated_at: datetime


ModelInstancesPublic = PaginatedList[ModelInstancePublic]


class ModelInstanceLogWorker(BaseModel):
    id: int
    name: str


class ModelInstanceLogRestartEntry(BaseModel):
    """One main serve log session on disk, with optional UX label time."""

    previous: bool = False
    started_at: Optional[datetime] = Field(
        default=None,
        description=(
            "Approximate start time from the main log file metadata "
            "(birthtime if available, else mtime), UTC."
        ),
    )
    containers: List[str] = Field(
        default_factory=list,
        description=(
            "Available container names for this restart. "
            "'default' is the main workload container; others are sidecars "
            "(e.g., ['default', 'ray-head'])."
        ),
    )


class ModelInstanceLogWorkerOption(BaseModel):
    """Per-worker result for GET /model-instances/{id}/log-options (one node on disk)."""

    worker_id: int
    name: str = ""
    restarts: List[ModelInstanceLogRestartEntry] = Field(default_factory=list)
    error: Optional[str] = Field(
        default=None,
        description="If set, log options could not be fetched from this worker.",
    )


class ServeLogOptionsResponse(BaseModel):
    """Worker GET /serveLogOptions JSON; also validates that payload when the server proxies."""

    restarts: List[ModelInstanceLogRestartEntry] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _legacy_restart_counts(cls, data: Any) -> Any:
        """Old workers only sent restart_counts; expand to restarts when `restarts` is absent."""
        if not isinstance(data, dict):
            return data
        if "restarts" in data:
            return data
        raw = data.get("restart_counts")
        if not isinstance(raw, list):
            return {**data, "restarts": []}
        counts: List[int] = []
        for x in raw:
            try:
                counts.append(int(x))
            except (TypeError, ValueError):
                continue
        counts.sort(reverse=True)
        # Map the highest restart_count to previous=False (current),
        # the second highest to previous=True.
        entries = []
        for i, c in enumerate(counts):
            entries.append({"previous": i > 0, "started_at": None})
        return {**data, "restarts": entries}


class ModelInstanceLogOptions(BaseModel):
    """Server GET /model-instances/{id}/log-options: per-worker serve log distribution."""

    main_worker_id: Optional[int] = Field(
        default=None,
        description="same as model instance worker_id.",
    )
    workers: List[ModelInstanceLogWorkerOption] = Field(
        default_factory=list,
        description=(
            "Ordered list: main worker first, then subordinate workers. "
            "Each entry reflects that worker's local serve logs."
        ),
    )


def is_gguf_model(model: Union[Model, ModelSource]):
    """
    Check if the model is a GGUF model.
    Args:
        model: Model to check.
    """
    return (
        (
            model.source == SourceEnum.HUGGING_FACE
            and model.huggingface_filename
            and model.huggingface_filename.endswith(".gguf")
        )
        or (
            model.source == SourceEnum.MODEL_SCOPE
            and model.model_scope_file_path
            and model.model_scope_file_path.endswith(".gguf")
        )
        or (
            model.source == SourceEnum.LOCAL_PATH
            and model.local_path
            and model.local_path.endswith(".gguf")
        )
    )


def is_audio_model(model: Model):
    """
    Check if the model is a STT or TTS model.
    Args:
        model: Model to check.
    """
    if model.backend == BackendEnum.VOX_BOX:
        return True

    if model.categories:
        return (
            'speech_to_text' in model.categories or 'text_to_speech' in model.categories
        )

    return False


def is_llm_model(model: Model):
    """
    Check if the model is an LLM model.
    Args:
        model: Model to check.
    """
    return not model.categories or CategoryEnum.LLM in model.categories


def is_omni_model(model: Model) -> bool:
    """
    Check if the model is an omni model (Image or Audio category).
    Args:
        model: Model to check.
    """

    if model.backend == BackendEnum.VLLM and find_bool_parameter(
        model.backend_parameters, ["omni"]
    ):
        return True

    OMNI_CATEGORIES = (
        CategoryEnum.IMAGE,
        CategoryEnum.TEXT_TO_SPEECH,
    )
    return any(cat in model.categories for cat in OMNI_CATEGORIES)


def is_image_model(model: Model):
    """
    Check if the model is an image model.
    Args:
        model: Model to check.
    """
    return "image" in model.categories


def is_embedding_model(model: Model):
    """
    Check if the model is an embedding model.
    Args:
        model: Model to check.
    """
    return "embedding" in model.categories


def is_reranker_model(model: Model):
    """
    Check if the model is a reranker model.
    Args:
        model: Model to check.
    """
    return "reranker" in model.categories


def get_backend(model: Model) -> str:
    if model.backend:
        return model.backend

    if is_gguf_model(model):
        return BackendEnum.CUSTOM

    return BackendEnum.VLLM


def get_mmproj_filename(model: Union[Model, ModelSource]) -> Optional[str]:
    """
    Get the mmproj filename for the model. If the mmproj is not provided in the model's
    backend parameters, it will try to find the default mmproj file.
    """
    if not is_gguf_model(model):
        return None

    if hasattr(model, "backend_parameters"):
        mmproj = find_parameter(model.backend_parameters, ["mmproj"])
        if mmproj and Path(mmproj).name == mmproj:
            return mmproj

    return "*mmproj*.gguf"
