import copy
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import hashlib
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Union,
)
from croniter import croniter
from pydantic import (
    BaseModel,
    ConfigDict,
    field_serializer,
    field_validator,
    model_validator,
)
from sqlalchemy import (
    JSON,
    Column,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
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

# The enum lives with the cluster because the cluster is where the default is
# set; a model only overrides it. Runtime import is safe in this direction —
# `clusters` imports `models` under TYPE_CHECKING only.
from gpustack.schemas.clusters import GatherStrategyEnum

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


# Prefill/decode disaggregation. A Model with `roles` set is a *group*: one
# pool, one router, one generation at a time.


class RoleNameEnum(str, Enum):
    """Role names the API accepts.

    The data model allows any name — the API validation layer is what limits
    it to these three.
    """

    PREFILL = "prefill"
    DECODE = "decode"
    ROUTER = "router"

    def __str__(self):
        return self.value


class PortBand(BaseModel):
    """A contiguous run of ports, not a single point.

    Some KV connectors derive several ports from one base — NIXL's side
    channel takes one per tensor-parallel rank — so a port declaration has to
    carry its width as well as its base.
    """

    base: int
    count: int = 1


class RoleResources(BaseModel):
    """What a role's container asks for besides accelerators.

    Only meaningful for a role that holds no weights — today the router — and
    that is why it is not a Model-level field with the usual inherit-when-None
    rule: prefill and decode get these numbers from sizing, and a hand-typed
    value there would only fight the estimate.
    """

    cpu: Optional[float] = None
    """Cores. Rendered into the container's requests *and* limits on
    Kubernetes (so the Pod lands in the Guaranteed QoS class) and into
    ``cpu_shares`` on Docker, which makes it a weight rather than a cap there.

    It does **not** take part in placement. The scheduler's allocatable
    view has two dimensions, RAM and VRAM, and CPU is not one of them, so
    nothing subtracts this from a worker before choosing it. Declaring it
    still buys cgroup enforcement and kubelet admission; "the scheduler will
    place the router according to this number" is not true yet and must not
    be implied in the UI.
    """
    memory: Optional[int] = None
    """Bytes. Unlike `cpu` this one *is* consumed by placement: it becomes the
    role's ``ComputedResourceClaim.ram``, which the allocatable view already
    tracks and subtracts."""


# A managed router is a proxy: it forwards requests and loads no weights, so
# its footprint is a fixed floor rather than something to estimate. The memory
# figure is deliberately the one the CPU-only claim already hardcoded, so
# turning it into a declared default changes no placement decision.
ROUTER_DEFAULT_CPU = 2.0
ROUTER_DEFAULT_MEMORY = 2 * 1024**3


class RoleSpec(BaseModel):
    """One role of a multi-role deployment.

    Every deployment field left as ``None`` inherits the ``Model``-level field
    of the same name; giving it a value overrides it. The override surface is
    deliberately the *whole* of ``backend_parameters`` and ``env`` rather than
    a PD-specific subset: on Ascend, prefill and decode differ in nearly every
    performance-related parameter, down to ``HCCL_CONNECT_TIMEOUT`` (120 vs
    1200) and ``HCCL_BUFFSIZE`` (2560 vs 1024). Any narrower surface runs out
    immediately.

    Note that the inherit-when-None rule is not applied here: the projection
    onto an effective per-role Model happens on the read path, and its result
    is deliberately never persisted so that one intent has one source of
    truth.
    """

    name: str
    replicas: int = Field(default=1, ge=1)
    """The x and y of xPyD, and the only scaling truth for a group.

    Not wanting a role means removing it, not setting this to zero — a zero
    would leave `dependencies` pointing at a role that never appears.
    """

    backend: Optional[str] = None
    backend_version: Optional[str] = None
    image_name: Optional[str] = None
    run_command: Optional[str] = None
    backend_parameters: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    gpu_selector: Optional[GPUSelector] = None
    worker_selector: Optional[Dict[str, str]] = None
    gpu_type_selector: Optional[GPUTypeSelector] = None
    """The only entry point for a heterogeneous group, and the precondition
    for gang admission."""
    extended_kv_cache: Optional[ExtendedKVCacheConfig] = None
    speculative_config: Optional[SpeculativeConfig] = None
    """Overridable per role because prefill and decode need *different*
    values, not because one of them should switch it off.

    The first reading of this was backwards: prefill does not decode, so a
    draft model looked like pure waste there. What the NIXL handshake actually
    hashes is the model — `model`, `num_hidden_layers`, `num_kv_heads`,
    `head_size` — and for MTP-style speculation the draft head is *part of the
    model*. A prefill that does not load it therefore produces a different
    structure and fails the compatibility check. Upstream's own recipes
    (vllm-ascend's DeepSeek-V4-Flash and GLM5 tutorials) say the same thing in
    numbers: prefill runs `num_speculative_tokens: 1` and decode runs 3 or
    more. The 1 is not prefill speculating; it is prefill loading the same
    shape.

    So the model-level value cannot serve both, and neither can switching it
    off on one side. Absent still inherits the model's, which is right for a
    non-MTP draft model where prefill genuinely gains nothing — the field
    makes the split possible, it does not force it.

    No migration: `roles` is already a JSON column, so a new field on this
    model is a schema change only.
    """

    lora_list: Optional[List[LoraListEntry]] = None
    """Declared so it can be *refused*, which is the only thing admission does
    with it today.

    It is here because its absence was invisible. `RoleSpec` takes pydantic's
    default `extra="ignore"` — the same leniency that lets an old row's
    `cpu_only` be read straight past — so a deployment that put its adapters on
    one role got a 200, a read-back with no `lora_list` anywhere, and not one
    word about where they went. A field that exists and is rejected says what
    happened; a field that does not exist says nothing.

    Nothing consumes it: `validate_roles` refuses any role that sets it. Role
    names would have to reach the engine through the group's router, and the
    router's member table is indexed by served-model name — the same wall that
    makes LoRA and disaggregation refuse each other outright (see
    `_reject_lora_under_disaggregation`). Implementing per-role adapters before
    that is settled would be building on it.

    It is nonetheless a real override field rather than a role-own one, so the
    projection carries it: the day the refusal lifts, `role_effective_model`
    already hands each member its own list and there is no second mechanism to
    add.
    """

    dependencies: Optional[List[str]] = None
    """Roles that must be ready before this one starts. Must not cycle."""
    # No `cpu_only`. It was a boolean standing in for a quantity: the
    # scheduler needs a VRAM claim, and the flag only decided whether to go ask
    # `estimate_model_vram` — which sizes the MODEL'S WEIGHTS and is therefore
    # the wrong number for any role that does not load them. So the `False`
    # branch had no correct implementation for a router: the only way to say
    # "give this router a card" booked it at the whole model, and there was no
    # per-role way to override that (`GPUSTACK_MODEL_VRAM_CLAIM` is
    # model-level, so setting it would mis-size prefill and decode too).
    #
    # Deleting it makes "a router takes no accelerator" a property of the role
    # rather than a checkbox someone had to remember, which matters because
    # `role_takes_no_accelerator` gates three separate things — the VRAM claim,
    # whether the CONTAINER asks for devices, and gang membership — each with
    # its own recorded incident from getting it wrong.
    #
    # Old rows and old clients still carrying it are read straight past:
    # `RoleSpec` takes pydantic's default `extra="ignore"`, and the value they
    # carried (`False` on every group the UI ever produced) now means what it
    # already meant in practice.
    #
    # A GPU-bearing role that is not prefill or decode comes back as a new role
    # NAME, not as a flag on the router.
    resources: Optional[RoleResources] = None
    """CPU and memory for a role that claims no accelerator — the router.

    Role-own rather than an override, because there is no Model-level field to
    inherit from: `ram_size` / `ram_ratio` feed the *VRAM* estimate, not a
    container's memory request. Left as `None` the router still gets
    `ROUTER_DEFAULT_CPU` / `ROUTER_DEFAULT_MEMORY`, so the field only exists
    to move off that floor.

    Refused on prefill and decode at admission: their footprint is what sizing
    computes from the weights and the parallelism, and a second, hand-written
    source for the same number is a way to disagree with it silently.
    """


class PDModeEnum(str, Enum):
    """A disaggregation recipe: engine plus KV connector.

    These values must match the entry names in ``pd-modes.yaml`` verbatim —
    the catalog is looked up by them, so a mismatch is a silent miss. The
    loader asserts the two sets are equal at start-up.
    """

    VLLM_NIXL = "vllm-nixl"
    SGLANG_MOONCAKE = "sglang-mooncake"
    SGLANG_NIXL = "sglang-nixl"
    VLLM_ASCEND_MOONCAKE = "vllm-ascend-mooncake"
    CUSTOM = "custom"
    """The user supplies every connection-state parameter themselves. Also the
    only way to mix engines across roles, since a recipe injects one engine's
    connector config into every role."""

    def __str__(self):
        return self.value


# Which engines may be disaggregated at all, whatever the mode.
#
# A narrower question than a recipe's own `backends`, and asked of a different
# thing: that says which engine a *recipe* can be injected into, and
# `custom` answers "any" because it injects nothing. This one says which
# engines the *feature* applies to, and `custom` is subject to it like every
# other mode -- writing the connection parameters yourself does not give an
# engine a KV cache to disaggregate.
#
# VoxBox runs speech models, where there is no prompt KV to hand across, so
# prefill/decode does not name anything it does. MindIE has disaggregation of
# its own on Ascend, and is excluded as a product decision rather than a
# technical one: GPUStack ships no recipe for it, so the only way in would be
# `custom` with every connection parameter hand-written, and a path that
# reaches a running group only if the user already knows the engine's
# disaggregation protocol is one this form should not offer.
PD_BACKENDS: List[str] = [
    BackendEnum.VLLM.value,
    BackendEnum.SGLANG.value,
    # A user-supplied engine image, which is how a BYO engine runs PD here.
    BackendEnum.CUSTOM.value,
]


class GatherSpec(BaseModel):
    """How tightly this deployment's members should sit together.

    The only source of the requirement — there is no cluster-wide default
    underneath it, because a cluster-level failure policy would let an operator
    arm a rejection the deployer never sees stated. Absent means absent: no
    constraint.

    **Two independent questions, deliberately two fields.**

    - ``layer`` — how close do you want them? A *target*.
    - ``strategy`` — and if that cannot be met? ``MustGather`` refuses;
      ``PreferGather`` deploys anyway.

    Read the pair as a failure policy layered on a target, not as a placement
    instruction: the solver always takes the tightest domain that fits, so
    neither field makes a group land any closer than it otherwise would.
    What they decide is what happens when the target is missed.

    ``PreferGather`` **with** a ``layer`` is the combination most deployments
    want — "aim for X but ship it either way" — and it needs both fields to
    be expressible at all: place as usual, and if the group ends up looser
    than ``layer``, say so on the model as a ``gather_unmet`` degradation. A
    target without a threat attached, which is the normal way to ask for
    something.
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    strategy: Optional[GatherStrategyEnum] = None
    """None means no gather requirement at all. `PreferGather` keeps widening
    to the cluster root and reports a miss; `MustGather` stops at `layer` and
    refuses the deployment instead."""

    layer: Optional[str] = None
    """The layer `strategy` applies to: a layer id of the cluster's chain, or
    the built-in node layer, which is the leaf.

    Required under `MustGather` — a refusal needs something to refuse below.
    Optional under `PreferGather`, where it is the target a miss is reported
    against; omitted there, nothing is reported and any placement is fine.

    Every value here names a layer the cluster actually declares, including
    `accelerator_domain`, which is an ordinary custom layer like any other.
    `routes.models.validate_gather_layer` refuses anything else — the solver
    stands an unknown layer down rather than failing, so an unchecked name
    would be a `MustGather` nothing enforces.
    """

    @model_validator(mode="after")
    def check_layer_accompanies_must(self) -> "GatherSpec":
        """`MustGather` without a layer has nothing to stop at.

        Left unchecked it reads as "refuse if it does not fit" with no
        definition of "fit", and the solver's `_enforced_gather` would stand
        the requirement down — so the deployment would be accepted under a
        promise that was never in force. Refusing at the edge is the whole
        difference between a constraint and a decoration.
        """
        if self.strategy == GatherStrategyEnum.MUST_GATHER and not self.layer:
            raise ValueError("gather strategy 'MustGather' requires a layer")
        if self.layer and self.strategy is None:
            raise ValueError(
                "gather layer is set without a strategy; there is nothing to "
                "apply it to"
            )
        # `PreferGather` + a layer is deliberately NOT refused: that pair is
        # "aim for this, ship it anyway, tell me if you missed" and is the
        # common case. It is also why this check is one-directional — a
        # strategy needs no layer, only `MustGather` does.
        return self


class DisaggregationSpec(BaseModel):
    mode: PDModeEnum

    vendor: Optional[str] = None
    """Which accelerator vendor this group runs on, e.g. "nvidia".

    Only needed in a cluster with more than one vendor partition that could
    host the group. A PD group cannot span vendors -- the KV path differs
    (HCCL/MemFabric vs UCX/RDMA verbs) -- so this is a *placement* constraint,
    not a preference, and it doubles as the key the recipe is derived from.

    None in a single-vendor cluster, where it is derived. Deliberately not
    guessed in a mixed one: the platform picking "the partition with the most
    cards" would override a user who wants the idle partition instead.

    It lives here rather than on the model because the constraint exists
    *because of* PD. If a non-PD workload ever needs the same thing, promote
    it to a model-level `gpu_filters` -- the two shapes match.
    """

    readiness: Literal["any_per_role", "all"] = "any_per_role"
    """Whether every role member has to be ready, or one per role is enough.

    `all` is for a deployment sized to a known load, where a partial group
    degrades into queueing rather than into reduced throughput — it moves the
    shortfall out of `degradations` and into `state`, so the endpoint stops
    taking traffic during a scale-up instead of serving through it. Judged in
    `derive_model_state`; not surfaced in the deployment form yet.
    """

    kv_load_failure_policy: Literal["fail", "recompute"] = "fail"
    """What decode does when the KV it was promised does not load.

    Only `vllm-nixl` renders it — SGLang has no equivalent concept and
    Mooncake's connector does not read the key — so a non-default value is
    refused at admission on the other modes rather than stored and ignored.

    `recompute` degrades silently by design, so choosing it means committing
    to watch decode's recompute share; the PD-effectiveness ratio is where
    that shows.
    """

    # `router_kind` was here: an escape hatch for swapping the router
    # implementation out of a mode's bundle. Both things that would have used
    # it arrived instead -- the `custom` mode, and per-role image/run_command
    # overrides -- and either expresses more than a name ever could. What was
    # left accepted any string, changed nothing, and appeared in every PD
    # model's API response, which reads as an offer to swap routers there.


class ModelStateEnum(str, Enum):
    """Model-level lifecycle. Deliberately *not* a copy of
    ``ModelInstanceStateEnum`` — this is an aggregate, not a per-process
    lifecycle, so it has no download/start phases.

    Degradation is not a value here. Cache not attached, bandwidth below the
    measured baseline, ratio unmet — all of those coexist with a servable
    group, so they live in the orthogonal ``degradations`` marker instead.
    """

    PENDING = "pending"
    """Nothing ready yet."""
    PARTIAL = "partial"
    """Members are up and the deployment still cannot serve — for a group, a
    role with zero ready members, or an upstream registration that has not
    succeeded.

    **Unreachable for a role-less model**: one ready replica serves, so there
    is no such condition. Being short of the requested count is
    `degradations: [ratio_unmet]` beside a RUNNING state, not this."""
    RUNNING = "running"
    """Servable: every role has at least one ready member *and* the upstream
    registration succeeded."""
    ERROR = "error"
    """A member has failed in a way it can't recover from."""

    def __str__(self):
        return self.value


class RoleStatus(BaseModel):
    """Per-role readiness detail.

    Carried on the Model row rather than computed per request because the
    list endpoint returns `ModelPublic` without instances, and the UI needs
    per-role detail on a row it hasn't expanded.
    """

    desired: int = 0

    ready: int = 0
    """Members of this role that are RUNNING **and taking traffic**.

    A member inside its drain window is excluded, and that is the whole
    difference from a plain RUNNING count: the router dropped it from its
    member list the moment the window opened, so it answers only what it had
    already taken. Counting it here made a role scaled from 3 to 2 report
    `ready=3, desired=2` for the length of the window — more members than were
    asked for, which reads as an edit that did not take.

    This is what `model_role_ready_instances` publishes, so that series dips
    while a role is draining. That is the intended reading: the dip is real
    capacity leaving."""

    draining: int = 0
    """Members inside their scale-down drain window, whatever their state.

    Separate from `ready` rather than derivable from it, because a draining
    member need not be RUNNING: victim selection scores a broken member zero
    and so picks it first, which makes "draining and also ERROR" the ordinary
    case rather than a corner.

    **It is also the only member a reader cannot account for.** Someone
    looking at an expanded group counts rows, and during a scale-down the rows
    outnumber `ready`. Every other kind of surplus explains itself — a member
    that is still starting is the gap between `ready` and `desired`, and the
    row says «Starting» — but a drained member sits outside both numbers. So
    this is what the cell prints beside the fraction, and printing the member
    total instead would leave the reader to do the subtraction and still not
    know what the extra one was doing.

    Defaults to 0, which is what a role that has never had a drain window
    reads back as."""


class DegradationReasonEnum(str, Enum):
    """Reasons a group is servable but worse than asked for.

    Orthogonal to `state`, following the precedent set by `stale`: "config
    changed *and* still serving" has to be expressible as one fact, and so
    does "running but the cache never attached".
    """

    CACHE_NOT_INJECTED = "cache_not_injected"
    # There is deliberately no bandwidth or PD-effectiveness marker here.
    # Both need traffic to have happened, and the counters behind them live in
    # Prometheus, where the worker's aggregator publishes them labelled and
    # over a path that handles tunnelled hosts. A marker on the row would be a
    # second, staler copy of that answer.
    RATIO_UNMET = "ratio_unmet"
    # No prefill and decode member share a host, so no request's KV can avoid
    # the network. Placement-only and knowable at admission, unlike the
    # bandwidth markers above which need traffic to have happened.
    PAIRING_REMOTE = "pairing_remote"

    GATHER_UNMET = "gather_unmet"
    """The group is serving, but looser than the layer it asked to sit in.

    Only ever set under `PreferGather`: that strategy ships whatever it can
    place, so without this marker "I wanted same-rack" and "I got same-rack"
    are indistinguishable afterwards — the request is in the spec and the
    outcome is nowhere. `MustGather` needs no marker, having refused instead.

    Placement-only and knowable as soon as the members are bound, like
    `PAIRING_REMOTE` beside it: no traffic has to happen for it to be true."""

    GATHER_BLOCKED_SCALE_OUT = "gather_blocked_scale_out"
    """A member cannot be placed without leaving the domain this deployment is
    pinned to, and the deployment asked to be refused rather than spread.

    The other half of `MustGather`, and the half nobody could see. Under it
    `GatherFloorFilter` refuses every worker outside the domain the running
    members occupy -- the strategy working exactly as specified -- but the
    refusal lands on one pending instance's `state_message` and nowhere else:
    the model stays RUNNING with an empty `degradations` list, so a scale-up
    that will never complete is indistinguishable from one still in flight
    unless the user opens each member in turn. `GATHER_UNMET` does not cover
    it either, by construction: that one fires only once the floor is ALREADY
    broken, which under `MustGather` is an invariant check rather than a
    report.

    **It does not claim the floor is the cause.** A group with no room left
    anywhere -- inside the domain or outside it -- presents identically: a
    weight-bearing member placed, a sibling pending, nothing moving. The two
    are not separable from the model row, and pretending otherwise would send
    the operator to edit a gather policy when the answer was to add a worker.
    The wording is therefore true under both readings: a member is not being
    placed, and this deployment is one that would rather wait than spread.
    Which of the two it is, the member's `state_message` says.

    Deliberately absent during formation, where no member is placed yet and
    the solver -- not the filter -- is the one refusing; a group that cannot
    form fails scheduling with the shortfall named, which is a different
    report with a different audience. And deliberately delayed by a dwell, so
    an ordinary scale-up does not wear the marker for the seconds between the
    row being created and the scheduler reaching it."""

    ENGINE_VERSION_BELOW_RECIPE_FLOOR = "engine_version_below_recipe_floor"
    """The pinned engine version is below the floor this PD recipe declares.

    The floor is not stylistic. `sglang-nixl` and `sglang-mooncake` declare
    `>=0.5.7` because a member's id stopped being its URL and became a UUID
    the registry mints at that version: on an older build `DELETE
    /workers/{url}` answers 400, so a scaled-down member is never removed from
    the router's registry and keeps taking traffic after GPUStack believes it
    is gone.

    **A degradation and not a 400, unlike a cache provider's `versions`.**
    The cache range is enforced at admission because an out-of-range engine
    there is handed injected args it cannot parse and fails to start -- a
    refusal costs nothing, since the deployment was not going to run. This
    floor is different: the group runs, and the version string it was pinned
    to may be a self-built image with a private number that happens to carry
    the fix. Rejecting would break those deployments to prevent a failure they
    do not have.

    Set only for a version `version_in_range` positively reports as out of
    range. Unpinned, unparseable and unknown-to-us all leave it unset -- the
    same fail-open the cache check takes, and for the same reason: an exotic
    version string must never be the thing that condemns a deployment.

    Two parseable shapes are let through as well, because sorting them below
    the floor answers a question they were never asked: a local version
    (`0.5.6+ourfix`) is by PEP 440's own definition an official release plus
    whatever the packager put on top of it -- backporting the very fix the
    floor wants is a common reason to cut one -- and a pre-release
    (`0.5.7-rc1`, `0.5.6.dev0`) is cut from a branch rather than from a
    release line the floor was ever measured against."""

    PAIRING_UNVERIFIED = "pairing_unverified"
    """A pairing factor whose two effective values GPUStack could not compare —
    usually one role declaring it and the other going silent.

    The factors prefill and decode must share — the context window, the tensor
    parallelism, the dtypes, the block size, the KV cache layout. Comparing
    them only when *both* roles write them down would pass a group where one
    side is silent by not checking it, and that is the ordinary way a group is
    misconfigured: edit prefill, leave decode alone.

    **A marker and not a 400, because the silent side's value is genuinely
    unknown here.** Substituting the engines' defaults was the obvious repair
    and is wrong for every one of these: an unwritten `--tensor-parallel-size`
    is the member's card count and not 1, since GPUStack injects it itself; an
    unwritten `--dtype` is `auto`, which needs the checkpoint's config to
    resolve and admission has no session to fetch one; `--block-size` and
    `--kv-cache-layout` are settled by the platform and the attention backend,
    `VLLM_KV_CACHE_LAYOUT` included, so any constant written down would drift
    into a false alarm on a later vLLM. A pair this marker describes is very
    often correct — what it reports is that nothing verified it, which is not
    the same claim as "this is broken".

    **An explicit `auto` is silence, not a third value.** `--dtype auto`
    against `--dtype float16` lands here rather than being refused: on a float16
    checkpoint the two are the same run, and telling them apart needs the config
    file admission cannot open. Writing `auto` down does not turn a question
    into an answer.

    A divergence GPUStack *can* prove — two concrete values that differ — is
    still refused at admission. Both sides silent is deliberately not marked:
    two roles taking the same default from the same engine on the same model
    agree whatever it resolves to."""

    PAIRING_TP_MISPLACED = "pairing_tp_misplaced"
    """The cards the members actually got break the recipe's tensor-parallel
    direction.

    The admission check can only read the spec, and the spec routinely does not
    contain the number: a role that writes no `--tensor-parallel-size` and pins
    no cards runs whatever the scheduler gives it. Placement is where that
    stops being unknown — every member's `gpu_indexes` is written down, the
    engines derive tp from exactly that for a single-worker member, and the
    direction the recipe declares (`PDMode.pairing.tensor_parallel`) can
    finally be applied to the deployment that exists rather than the one that
    was described.

    **Marked, never enforced.** By the time this is knowable the members are
    placed and, usually, serving; failing them would take down a group to
    report a shape it is already running. Under NIXL the shape does break —
    a decode narrower than its prefill raises an `IndexError` inside decode on
    first transfer — but that failure belongs to the engine and arrives with
    its own message. This is the attribution: the reason that IndexError exists
    is a placement, and the placement is written on the model."""

    def __str__(self):
        return self.value


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

    # Empty `roles` is the backward-compatibility baseline: behaviour is
    # byte-for-byte unchanged. `roles` without `disaggregation` is plain
    # multi-role orchestration; both together is PD.
    roles: Optional[List[RoleSpec]] = Field(
        default=None,
        sa_column=Column(pydantic_column_type(List[RoleSpec]), nullable=True),
    )
    disaggregation: Optional[DisaggregationSpec] = Field(
        sa_type=pydantic_column_type(DisaggregationSpec), default=None
    )
    # Beside `roles` rather than inside `disaggregation`: gather describes how
    # far apart the group's *members* may sit, and members come from `roles`.
    # A role-bearing model without disaggregation is a valid shape (plain
    # multi-role orchestration), and it wants this just as much. The
    # deployment form still shows the control in the PD block, which is a
    # placement decision about the form, not about the field.
    gather: Optional[GatherSpec] = Field(
        sa_type=pydantic_column_type(GatherSpec), default=None
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

    # Server-owned status. Declared here and on `ModelPublic`, deliberately
    # *not* on `ModelBase`: `ModelUpdate` inherits `ModelBase`, and the UI
    # issues whole-object PUTs (start/stop, inline replica edits), so
    # anything reachable from `ModelBase` gets written back by the client.
    # `ready_replicas` sits on `ModelSpecBase` for historical reasons and the
    # frontend has to strip it by hand — don't grow that list.
    #
    # One writer only: `sync_model_status` computes all five from a single
    # scan of the model's instances, in one transaction behind one change
    # gate. There is no second owner.
    # String, not sa.Enum — following CacheService.state. A bare
    # `Optional[ModelStateEnum]` maps to `sa.Enum(name="modelstateenum")`, and
    # that breaks twice over: asyncpg then renders `$1::modelstateenum` on
    # every read and write, against a column the migration created as VARCHAR;
    # and sa.Enum persists member *names*, so the row would hold "RUNNING"
    # while the API, the enum's own value and the `?state=` filter all say
    # "running".
    state: Optional[ModelStateEnum] = Field(
        default=None, sa_column=Column(String(length=64), nullable=True)
    )
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    role_status: Optional[Dict[str, RoleStatus]] = Field(
        default=None,
        sa_column=Column(pydantic_column_type(Dict[str, RoleStatus]), nullable=True),
    )
    stale: Optional[bool] = Field(default=None)
    """A member's `spec_digest` differs from the model's current one, so the
    running group predates the config it's shown with. Orthogonal to `state`:
    a stale group is usually still serving."""
    degradations: Optional[List[str]] = Field(sa_type=JSON, default=None)
    """`DegradationReasonEnum` values. A list, because they coexist."""
    restarting_since: Optional[datetime] = Field(
        sa_column=Column(UTCDateTime), default=None
    )
    """When `POST /{id}/restart` last tore this deployment down, cleared once
    it is serving again. The window a second restart must be refused in.

    **It exists because the fact is not derivable.** The rows cannot be asked:
    live members spanning more than one `spec_digest` reads like
    "mid-replacement" and never happens, because the teardown is synchronous
    and the reconcile rebuilds from the same target digest, so the two
    generations are never in the table at once. Without the timestamp, a
    second click landing while the replacements are still starting deletes
    exactly those replacements and costs the group another full startup, with
    nothing in the UI to say why it went back to pending.

    Cleared by `sync_model_status` on reaching RUNNING, and lapsing on its own
    after `RESTART_IN_FLIGHT_LAPSE_SECONDS`. The lapse is not a tidy-up: a
    group that never converges is precisely the one an operator needs to
    restart again, and a guard with no expiry would answer 409 forever."""

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
    # Read-only status, mirrored from `Model`. Absent from `ModelBase` so
    # `ModelUpdate` can't accept it — see the note on `Model`.
    state: Optional[ModelStateEnum] = None
    state_message: Optional[str] = None
    role_status: Optional[Dict[str, RoleStatus]] = None
    stale: Optional[bool] = None
    degradations: Optional[List[str]] = None
    # Exposed so the Restart control can be disabled while one is in flight,
    # rather than letting the click through to a 409 the user has to read.
    restarting_since: Optional[datetime] = None
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


class RoleEffectiveModel(ModelBase):
    """A `Model` as one role sees it — see `role_effective_model`.

    Non-table on purpose, and both reasons are load-bearing:

    * **A projection must never reach the database.** `Model.model_copy()`
      looks like the obvious way to build one, but the copy *shares the
      original's* `_sa_instance_state` — it is the same ORM identity, so the
      projection would sit one session flush away from writing a role's
      overrides onto the Model row. A non-table class cannot be added to a
      session at all, so the rule holds by construction rather than by
      everyone remembering it.
    * It records the direction of the data: nothing reads a projection back.

    It carries the spec, not the aggregate status: `state` / `role_status` /
    `degradations` live on `Model` and `ModelPublic` only. A worker or a
    scheduling pass acting on a model-wide aggregate would be reading the
    wrong thing anyway.
    """

    id: Optional[int] = None

    # Deliberately left unhashable, which is what `Model` is too — SQLModel
    # sets `__hash__ = None` on a table class the same way pydantic does for
    # any mutable model, and `ModelInstance` has to override it explicitly to
    # go into a queue. So a projection behaves like the thing it stands in
    # for, and a reader that starts hashing models fails for both rather than
    # only for role-bearing deployments.


# The RoleSpec fields that describe the role itself rather than override a
# Model field. Everything else is an override, derived rather than listed so
# that adding one to RoleSpec cannot silently fail to be projected.
_ROLE_OWN_FIELDS = frozenset({"name", "dependencies", "resources"})

_ROLE_OVERRIDE_FIELDS = frozenset(RoleSpec.model_fields) - _ROLE_OWN_FIELDS


def find_role(model, role_name: Optional[str]) -> Optional[RoleSpec]:
    """The named role of `model`, or None if it has no roles or no match."""
    if not role_name:
        return None
    for role in model.roles or []:
        if role.name == role_name:
            return role
    return None


def servable_instances(model, instances):
    """The members that can answer a whole request for `model`.

    For a role-bearing group that is the router alone. Every member serves an
    OpenAI-shaped API on its own port, so handing a request to any of them
    succeeds — a prefill returns after a single token, a decode runs without
    the prefix its KV was meant to carry, and both answer 200 with plausible
    text. Balancing across the group therefore does not fail; it silently
    answers two thirds of requests wrongly.

    One function, because there are two places that route to an instance — the
    gateway's upstream registration and the direct proxy — and a rule this
    consequential must not be able to hold in one and not the other.

    A group with no running router yields nothing rather than falling back to
    its GPU members: there is no member of a group that can serve alone, so an
    empty result is the honest answer and the caller reports the group as
    unavailable.
    """
    if not getattr(model, "roles", None):
        return list(instances)
    return [
        instance
        for instance in instances
        if getattr(instance, "role", None) == RoleNameEnum.ROUTER.value
    ]


def member_worker_ids(entry) -> List[int]:
    """Every machine one member occupies, not just the one it is filed under.

    **One reader, because there were four and they disagreed.** A member that
    spans machines records the extra ones on
    `distributed_servers.subordinate_workers`; `worker_id` alone is the machine
    its row is filed under. Everything that asks "where is this member" was
    reading that one field: the gather floor, the breach report it is meant to
    be caught by, the pairing-locality sum, and the proximity scorer. So a
    member on three machines was invisible on two of them to all four —
    anchoring a floor on the wrong domain, under-reporting the breach, and
    scoring a candidate against a group it could not fully see.

    Why the whole span and not the primary: a member wide enough to straddle
    machines holds cards on all of them, so it pairs from all of them. A decode
    spread over two hosts gives a prefill on either one somewhere local to
    fetch from, and counting only the primary makes the other host look empty.

    **Duck-typed across the two shapes it is asked of**, because the ledger has
    to read the same for a member already placed and a member being considered
    — otherwise scale-out and scale-down stop being inverses. A stored
    `ModelInstance` carries `worker_id` and keeps its other halves under
    `distributed_servers`; a fresh `ModelInstanceScheduleCandidate` carries
    `worker` and `subordinate_workers` directly.

    Reachable today without any cross-node support in the group solver: a
    scaled-out member goes down the per-instance path with the whole worker
    list, and `distributed_inference_across_workers` defaults to true for
    vLLM, SGLang and MindIE.

    Order is the member's own — primary first — because that is the order the
    ranks are laid out in, and a caller that cares which machine holds rank 0
    must not have to guess.
    """
    primary = getattr(entry, "worker_id", None)
    if primary is None:
        worker = getattr(entry, "worker", None)
        primary = getattr(worker, "id", None) if worker is not None else None
    out: List[int] = [] if primary is None else [primary]

    subordinates = getattr(entry, "subordinate_workers", None)
    if subordinates is None:
        servers = getattr(entry, "distributed_servers", None)
        subordinates = (
            getattr(servers, "subordinate_workers", None) if servers else None
        )
    for subordinate in subordinates or []:
        worker_id = getattr(subordinate, "worker_id", None)
        if worker_id is not None and worker_id not in out:
            out.append(worker_id)
    return out


def role_takes_no_accelerator(model, role_name: Optional[str]) -> bool:
    """Whether this role should be placed without claiming any GPU.

    The router, and only the router. It is a proxy — it forwards requests to
    the members that hold the weights and loads none itself — so this is a
    property of what the role *is*, and the answer cannot depend on anyone
    remembering to tick anything. That holds for a router the user brings
    themselves (image *and* command) too: the only sizing available is
    `estimate_model_vram`, which returns the model's weights, so giving any
    router a card would book a proxy at the whole model — leaving it
    unschedulable beside the prefill and decode it serves.
    """
    role = find_role(model, role_name)
    if role is None:
        return False
    return role.name == RoleNameEnum.ROUTER.value


def role_container_resources(model, role_name: Optional[str]) -> RoleResources:
    """CPU and memory for a role that claims no accelerator.

    Returns the declared values where given and the router floor otherwise, so
    a caller never has to know whether the deployment said anything. Callers
    that also handle accelerator-bearing roles must gate on
    `role_takes_no_accelerator` first: this returns the floor for any role
    name, and applying it to prefill would override what sizing computed.
    """
    role = find_role(model, role_name)
    declared = role.resources if role else None
    return RoleResources(
        cpu=(declared.cpu if declared and declared.cpu else ROUTER_DEFAULT_CPU),
        memory=(
            declared.memory if declared and declared.memory else ROUTER_DEFAULT_MEMORY
        ),
    )


def role_effective_model(model, role_name: Optional[str]):
    """Return `model` with the named role's overrides applied.

    A `RoleSpec` field left as None means "inherit the Model field of the same
    name". Nothing downstream performs that merge: the worker's start path
    reads `self._model.<field>` in dozens of places and the scheduler's
    filters, selectors and scorers read a Model in dozens more, all of them
    expecting a single set of values. So the merge happens once, here, at the
    two points where a Model is handed to those readers — `get_model()` on the
    worker and `find_candidate()` on the server.

    `replicas` is projected too, and unconditionally: it is never None, and
    for a role-bearing model `Model.replicas` is a 0/1 deployment switch while
    `roles[].replicas` is the count. Inside these two read paths the role's
    count is the right answer — it is what decides how many GPUs one replica
    gets and whether the multi-replica overcommit rule applies. Outside them
    `Model.replicas` keeps its switch meaning, which is why this projection
    deliberately does not reach the evaluator's `set_model_gpus_per_replica`:
    that one writes back.

    Returns `model` itself when there is nothing to project, so a role-less
    deployment takes byte-for-byte the path it takes today.

    One known edge: `distributed_inference_across_workers` is defaulted from
    the *Model's* backend by `ModelSpecBase.set_defaults`, which runs before a
    role's `backend` override is applied. A role that switches engines
    therefore inherits the Model's value rather than one derived from its own
    backend. That only arises under `pd_mode=custom`, the one mode that
    permits a mixed-engine group, and there the user is already supplying the
    connection state by hand — so set it explicitly on the Model in that case.
    """
    role = find_role(model, role_name)
    if role is None:
        return model

    projected = RoleEffectiveModel.model_validate(model)
    for field in _ROLE_OVERRIDE_FIELDS:
        value = getattr(role, field, None)
        if value is None:
            continue
        # Copy, so that mutating a projected list in place — the worker
        # substitutes `{data_dir}` into `backend_parameters` that way — cannot
        # reach back into the role held by `model.roles`.
        setattr(projected, field, copy.deepcopy(value))
    return projected


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

    role: Optional[str] = None
    """Which role of the parent Model this instance serves. None for a plain
    single-role deployment."""
    group_id: Optional[str] = Field(default=None, index=True)
    """Shared by every member of one group. A group is a *generation*, not a
    replica: one group_id is one `spec_digest`.

    Pairing binds to this rather than to peer addresses because serving ports
    were measured to change on every rebuild; addresses get resolved when the
    router config is rendered."""
    spec_digest: Optional[str] = None
    """The generation this instance was created from. Differing from the
    model's current digest is what makes the model `stale`."""
    named_ports: Optional[Dict[str, PortBand]] = Field(
        default=None,
        sa_column=Column(pydantic_column_type(Dict[str, PortBand]), nullable=True),
    )
    """Connector ports by declared name. Values are bands, not points."""

    draining_since: Optional[datetime] = Field(
        sa_column=Column(UTCDateTime), default=None
    )
    """Set when scale-down picked this member, cleared if it is kept.

    A prefill cannot be told "stop accepting work and exit once the blocks you
    hold have been fetched" — the engine has no such shutdown, and waiting for
    it is upstream WIP. So the wait happens here instead: the member is taken
    out of the router's registry immediately (`pd_membership.desired_members`
    skips it) and its container is left running for a window, which is what
    lets the decodes that are mid-request finish pulling from it.

    **The row is what makes this survive a server restart.** Held in memory,
    a restart mid-window would leave a member that no router knows about and
    nothing will ever delete — serving nothing, holding its cards. With the
    timestamp on the row the reaper picks it up again, and a window that
    elapsed while the server was down simply reaps on the next pass.

    Clearing it is the rollback, and it needs no second mechanism: the next
    membership reconcile sees the member as ordinary and re-registers it.

    **`UTCDateTime`, not a bare `datetime`.** The column is TIMESTAMP WITHOUT
    TIME ZONE and the writer builds `datetime.now(timezone.utc)`, which is
    aware. SQLite stores that without complaint; asyncpg refuses it outright
    (`DataError: invalid input for query argument`), and the refusal surfaces
    as a reconcile that fails after `find_scale_down_candidates` has already
    picked its victim — so on PostgreSQL every per-role scale-down retried
    forever and no surplus member was ever taken out of rotation, while
    `role_status` and `degradations` went on reporting the group as converged.
    The type strips the zone on the way in and puts UTC back on the way out,
    which is why every other timestamp on this table already uses it."""

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
