import random
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional
from pydantic import BaseModel
from sqlalchemy import JSON, Column
from sqlmodel import Field, ForeignKey, Integer, SQLModel, Text
from sqlmodel.sql.sqltypes import AutoString

from gpustack.schemas.common import (
    ListParams,
    PaginatedList,
    pydantic_column_type,
)
from gpustack.mixins import BaseModelMixin
from gpustack.schemas.models import (
    ComputedResourceClaim,
    ExtendedKVCacheConfig,
    SpeculativeConfig,
)

from gpustack.schemas.workers import GPUDeviceInfo, OperatingSystemInfo

DATASET_RANDOM = "Random"
DATASET_SHAREGPT = "ShareGPT"

# Random-dataset seed range. The upper bound stays well below numpy's 2**32 seed
# ceiling because a multi-stage run derives each stage's seed as base + stage
# index (see dataset_seed_increment) — a base near the ceiling would overflow on
# the last stages. Keep it in sync with genDatasetSeed() in the UI.
DATASET_SEED_MIN = 1
DATASET_SEED_MAX = 2**31 - 1


def generate_dataset_seed() -> int:
    """A fresh Random-dataset seed, so two runs of the same config send
    different synthetic prompts instead of replaying each other's prefix cache.
    """
    return random.randint(DATASET_SEED_MIN, DATASET_SEED_MAX)


class BenchmarkLoadTypeEnum(str, Enum):
    r"""
    The load axis (knob) a benchmark ramps or pins.

    - fixed_rate: guidellm `constant` — open-loop, N requests per second offered
      regardless of whether the server keeps up.
    - concurrency: guidellm `concurrent` — closed-loop, N requests in flight.

    An enum rather than a free string because both the runner and the analysis
    compare it exactly: a typo in a free string falls through to the fixed_rate
    branch, so a run labelled "concurrency" silently executes as a rate sweep.
    """

    FIXED_RATE = "fixed_rate"
    CONCURRENCY = "concurrency"


class BenchmarkTargetModeEnum(str, Enum):
    r"""What the load is aimed at: one instance, or the deployment's entrance.

    The two are different measurements, and the difference is not overhead —
    it is what the number means.

    ``instance`` measures an ENGINE. The load goes straight at a member's own
    port, so nothing but the engine is in the path. It is the right mode for
    tuning engine parameters. For a group the member is its router (no other
    member can answer a whole request); for a plain model it is one replica.

    ``route`` measures a DEPLOYMENT, through the route clients actually call.
    A plain model with four replicas is four replicas here and one replica in
    ``instance`` mode — which is the whole reason this exists: comparing a
    2P2D group against a four-replica deployment is only meaningful when both
    sides are driven as deployments.

    ``route`` puts the server's proxy in the path, and at high rates the
    proxy can be the bottleneck rather than the deployment. That is the reason
    ``instance`` stays the default and stays available for groups.
    """

    INSTANCE = "instance"
    ROUTE = "route"


class BenchmarkLoadModeEnum(str, Enum):
    r"""
    Which of the three mutually-exclusive load shapes a benchmark runs.

    Derived from the config rather than stored: `auto_tune` and `stages` are
    independent columns whose combination decides the shape, and three separate
    call sites (command building, result collection, ready-file counting) used to
    re-derive that precedence with their own if/elif chain. See
    :func:`benchmark_load_mode`.
    """

    AUTO_TUNE = "auto_tune"  # adaptive ramp over the load axis
    STAGES = "stages"  # one single-rate run per user-specified stage
    SINGLE = "single"  # one run at `request_rate`


def benchmark_load_mode(benchmark) -> BenchmarkLoadModeEnum:
    """The load shape this benchmark runs, from its config.

    Single source of the auto_tune > stages > single precedence. Creation
    validation rejects setting both auto_tune and stages, so for anything created
    through the API the precedence is never actually exercised; it stays here for
    rows that predate that check.
    """
    if getattr(benchmark, "auto_tune", None):
        return BenchmarkLoadModeEnum.AUTO_TUNE
    if getattr(benchmark, "stages", None):
        return BenchmarkLoadModeEnum.STAGES
    return BenchmarkLoadModeEnum.SINGLE


def benchmark_load_axis(benchmark) -> str:
    """The benchmark-runner `--axis` value for this benchmark's load type.

    The runner names the rate axis "rate" while the column stores "fixed_rate",
    so the two are not interchangeable; anything other than an explicit
    concurrency load type is the rate axis.
    """
    if getattr(benchmark, "load_type", None) == BenchmarkLoadTypeEnum.CONCURRENCY:
        return "concurrency"
    return "rate"


@dataclass(frozen=True)
class SLOThreshold:
    r"""
    One optional latency SLO target ("this metric must stay <= N ms").

    Single source of truth for the 3 metrics x 3 aggregations grid, consumed by
    everything that has to walk it:

    - the runner, to forward the threshold to benchmark-runner (`flag`);
    - the analysis, to decide whether a measured point meets the SLO and how much
      of its budget the point used (`metric`, `scale`).

    One table rather than a hand-written copy of the nine rows in each of
    those: with several lists, adding an aggregation silently loses the
    threshold in whichever one is missed.

    `scale` takes the stored metric to milliseconds: request_latency is stored in
    seconds, TTFT / TPOT already in ms.

    NOT the source for the migration's column list — a migration is a frozen
    historical step and must not change behaviour when this table grows.
    """

    attr: str
    metric: str
    scale: float
    flag: str
    # Second column to read when `metric` was not measured on a point (absent or
    # non-positive). Only the TPOT rows need one — see SLO_THRESHOLDS.
    fallback: Optional[str] = None


SLO_THRESHOLDS: List[SLOThreshold] = [
    SLOThreshold(
        "slo_avg_ttft_ms", "time_to_first_token_mean", 1.0, "--slo-avg-ttft-ms"
    ),
    SLOThreshold(
        "slo_p95_ttft_ms", "time_to_first_token_p95", 1.0, "--slo-p95-ttft-ms"
    ),
    SLOThreshold(
        "slo_p99_ttft_ms", "time_to_first_token_p99", 1.0, "--slo-p99-ttft-ms"
    ),
    # TPOT thresholds bound the DECODE-ONLY per-token time, which guidellm files
    # under `inter_token_latency_ms` — (last_token - first_token) / (tokens - 1),
    # the quantity vLLM and genai-perf report as TPOT.
    #
    # Deliberately NOT `time_per_output_token_*`, guidellm's OTHER per-token
    # metric: (last_token - request_start) / tokens, i.e. TTFT folded into the
    # decode average. That charges prefill and queue wait to the decode loop;
    # the error is TTFT / (n * TPOT), so ~5% on a 128-token run and ~40% at 16
    # output tokens, and it grows with load exactly where the SLO decides
    # capacity.
    #
    # It stays as the FALLBACK because the decode-only metric is not always
    # measurable: when a server answers without streaming incrementally — one
    # chunk carrying the whole output, which is common at low load — the first and
    # last token iteration share a timestamp, so guidellm reports 0. There is no
    # observable gap between tokens in that case, and total-time-over-tokens is
    # the only per-token number that exists. Falling back keeps such a point
    # judged instead of failing it (a threshold that fails wherever the server
    # batched its stream would bracket the ramp on its first point) and without
    # waiving it (0 ms would clear every budget).
    # This table is also where gpustack's vocabulary meets benchmark-runner's.
    # gpustack says TPOT (the API field, the CLI, the form, the column), the
    # runner and guidellm below it say ITL — both names for the decode-only
    # per-token time. The flag column carries the translation, so it happens
    # once, here, instead of being re-derived at each call site: everything to
    # the left of it is gpustack's own naming, everything sent to the right is
    # the runner's.
    SLOThreshold(
        "slo_avg_tpot_ms",
        "inter_token_latency_mean",
        1.0,
        "--slo-avg-itl-ms",
        fallback="time_per_output_token_mean",
    ),
    SLOThreshold(
        "slo_p95_tpot_ms",
        "inter_token_latency_p95",
        1.0,
        "--slo-p95-itl-ms",
        fallback="time_per_output_token_p95",
    ),
    SLOThreshold(
        "slo_p99_tpot_ms",
        "inter_token_latency_p99",
        1.0,
        "--slo-p99-itl-ms",
        fallback="time_per_output_token_p99",
    ),
    SLOThreshold(
        "slo_avg_latency_ms", "request_latency_mean", 1000.0, "--slo-avg-latency-ms"
    ),
    SLOThreshold(
        "slo_p95_latency_ms", "request_latency_p95", 1000.0, "--slo-p95-latency-ms"
    ),
    SLOThreshold(
        "slo_p99_latency_ms", "request_latency_p99", 1000.0, "--slo-p99-latency-ms"
    ),
]


class BenchmarkStateEnum(str, Enum):
    r"""
    Enum for Benchmark State

    Transitions:

       |- - Server - -|- - - - - - - Worker - - - - - - -|
       |              |                                  |
    PENDING ---> ---> ---> QUEUED ---> RUNNING ---> COMPLETED/STOPPED/ERROR
                              ^          ^
                              |          |
                              |----------|
                                         |
                                         |(Worker unreachable)
                                         v
                                     UNREACHABLE
    """

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"
    UNREACHABLE = "unreachable"

    def __str__(self):
        return self.value


class ModelInstanceRuntimeInfo(BaseModel):
    computed_resource_claim: Optional[ComputedResourceClaim]
    ports: Optional[List[int]]

    worker_id: Optional[int] = None
    worker_name: Optional[str] = None
    worker_ip: Optional[str] = None
    gpu_type: Optional[str] = None
    gpu_indexes: Optional[List[int]] = None
    gpu_ids: Optional[List[str]] = None


class ModelInstanceSnapshot(ModelInstanceRuntimeInfo):
    id: int
    name: str
    resolved_path: Optional[str] = None
    # Which role of its model this member served. None for a plain deployment,
    # and the only thing that tells a group's members apart once the run is
    # over: a report that lists three machines cannot otherwise say which one
    # was the router the load was sent to and which held the cards.
    role: Optional[str] = None

    # resource info
    state: Optional[str] = None
    state_message: Optional[str] = None

    # backend info
    backend: Optional[str] = None
    backend_version: Optional[str] = None
    api_detected_backend_version: Optional[str] = None
    backend_parameters: Optional[List[str]] = Field(sa_type=JSON, default=None)
    injected_backend_parameters: Optional[List[str]] = Field(sa_type=JSON, default=None)
    image_name: Optional[str] = None
    run_command: Optional[str] = Field(sa_type=Text, default=None)
    env: Optional[Dict[str, str]] = Field(sa_type=JSON, default=None)

    # Extended KV Cache configuration. Maps to LMCache in vLLM, and to SGLang's
    # native HiCache (LMCache in shared mode).
    extended_kv_cache: Optional[ExtendedKVCacheConfig] = Field(
        sa_type=pydantic_column_type(ExtendedKVCacheConfig), default=None
    )

    cache_service_name: Optional[str] = None
    """Name of the attached shared cache service at benchmark time — the
    config above stores only the id, and the snapshot must keep naming
    the service after it is deleted."""

    speculative_config: Optional[SpeculativeConfig] = Field(
        sa_type=pydantic_column_type(SpeculativeConfig), default=None
    )

    # subordinate workers info
    subordinate_workers: Optional[List[ModelInstanceRuntimeInfo]] = None


class WorkerSnapshot(BaseModel):
    id: int
    name: str
    cpu_total: Optional[int] = None
    memory_total: Optional[int] = None
    os: Optional[OperatingSystemInfo] = None


class GPUSnapshot(GPUDeviceInfo):
    id: str
    worker_id: int
    worker_name: str
    memory_total: Optional[int] = None
    core_total: Optional[int] = None


@dataclass
class BenchmarkDeploymentMetadata:
    name: str
    labels: dict[str, str]


class BenchmarkBase(SQLModel):
    name: str = Field(index=True, unique=True)
    description: Optional[str] = Field(
        sa_type=Text,
        nullable=True,
        default=None,
    )

    profile: Optional[str] = Field(default="Custom")
    dataset_name: Optional[str] = Field(
        default=None
    )  # type selector / denormalized name: Random / ShareGPT
    dataset_input_tokens: Optional[int] = Field(default=None)
    dataset_output_tokens: Optional[int] = Field(default=None)
    # The seed actually used by this run — the single source of truth. Normally
    # filled by the client (the UI shows the value before submit); left None it
    # is generated server-side at creation, then frozen (a benchmark runs once).
    dataset_seed: Optional[int] = Field(default=None)
    # Provenance of dataset_seed, NOT part of choosing it: True = randomly
    # generated, False = pinned by the user for a reproducible re-run. Drives the
    # form's checkbox state and, on clone, whether the seed is re-rolled or kept.
    dataset_seed_random: Optional[bool] = Field(default=True)
    # Multi-stage (ramp / manual) seed policy: True = each stage's seed is
    # base + stage_index (stages differ, spreading prefix/KV-cache reuse);
    # False = all stages share the base seed. Only meaningful for the Random
    # synthetic dataset — file datasets read in file order regardless of seed
    # until shuffle lands (design known-limit).
    dataset_seed_increment: Optional[bool] = Field(default=True)

    # Data distribution: spread token lengths around the mean instead of a
    # single fixed value, for more realistic load.
    # Maps to guidellm's prompt_tokens_stdev/min/max + output_tokens_stdev/min/max.
    dataset_input_stdev: Optional[int] = Field(default=None)
    dataset_input_min: Optional[int] = Field(default=None)
    dataset_input_max: Optional[int] = Field(default=None)
    dataset_output_stdev: Optional[int] = Field(default=None)
    dataset_output_min: Optional[int] = Field(default=None)
    dataset_output_max: Optional[int] = Field(default=None)

    cluster_id: int = Field(default=None)
    # What the load is aimed at. A column rather than a derivation: it is part
    # of the configuration a clone or an export has to carry, and two runs of
    # one model in different modes are not comparable, so the report has to be
    # able to say which one it was.
    # Stored as a plain string for the same reason as `load_type` below: the
    # default enum column keys on member NAMES (`INSTANCE`), while the
    # migration backfills, the API accepts and the UI sends the VALUE
    # (`instance`). Measured consequence of the mismatch: every read of a
    # pre-existing row raised `'instance' is not among the defined enum
    # values`, which killed the benchmark watch stream and left the worker
    # re-subscribing every five seconds.
    target_mode: BenchmarkTargetModeEnum = Field(
        default=BenchmarkTargetModeEnum.INSTANCE, sa_type=AutoString
    )
    model_id: Optional[int] = Field(default=None)
    model_name: Optional[str] = Field(
        default=None
    )  # denormalized field for easier query
    model_instance_name: str

    request_rate: int = Field(default=10)  # requests per second
    total_requests: Optional[int] = Field(
        default=None
    )  # total number of requests to send
    # Global duration cap (guidellm --max-seconds) for non-stage runs
    # (throughput / custom-sweep). Stage runs carry max_seconds per stage instead.
    max_seconds: Optional[float] = Field(default=None)

    # The load axis (knob) — see BenchmarkLoadTypeEnum. The latency-SLO scenario
    # is concurrency + any slo_* threshold set.
    #
    # Typed as the enum so an unknown value is a 422 at the API boundary instead
    # of silently falling through to the rate axis, but stored as a plain string
    # (sa_type) to keep the column an AutoString holding the enum VALUE — the
    # default enum column would store member NAMES and diverge from the migration,
    # the runner's --axis values and the UI.
    load_type: Optional[BenchmarkLoadTypeEnum] = Field(default=None, sa_type=AutoString)

    # Stages: per-stage independent constraints, so rate 1 and rate 1000 can
    # carry different limits. Each item: {rate: float, max_requests?: int,
    # max_seconds?: float}. Used only when auto_tune is off (Custom manual mode);
    # the runner does one single-rate guidellm run per stage. Mutually exclusive
    # with auto_tune (rejected at creation); see benchmark_load_mode.
    stages: Optional[List[Dict[str, Any]]] = Field(sa_type=JSON, default=None)

    # Auto-tune (adaptive ramp): when true, the runner ramps the load_type axis
    # (fixed_rate=req/s, concurrency=streams) with a geometric bracket + binary
    # search instead of running user-specified stages, and auto-detects the
    # answer. Target is derived: slo_* set -> SLO boundary (max knob meeting SLO);
    # otherwise -> throughput saturation (peak output tok/s). Replaces the old
    # guidellm `sweep` profile (removed).
    auto_tune: Optional[bool] = Field(default=None)
    # Auto-tune budget / bounds (used when auto_tune=true). None -> runner default.
    # The values themselves are deliberately NOT named here: the runner's
    # AutoTuneConfig owns them, the shipped presets in profiles_config.yaml
    # override them per goal, and a copy in this comment would be a second
    # source of truth that drifts the moment either is retuned.
    lower_bound: Optional[float] = Field(default=None)  # knob floor
    upper_bound: Optional[float] = Field(default=None)  # knob ceiling (anti-runaway)
    # Per-point requests = max(min_requests, round(knob * multiplier)) is computed
    # by the runner's ramp engine from its own defaults (see AutoTuneConfig);
    # neither knob is surfaced or stored here.
    max_points: Optional[int] = Field(default=None)  # max measured points
    max_total_seconds: Optional[float] = Field(default=None)  # whole-run cap

    # Latency SLO: optional "<= threshold" targets used to pick the max load that
    # still meets the SLO. Each is independent; a point meets the SLO when every
    # SET threshold holds (AND) and success >= 95%. `slo_avg_ttft_ms` / `slo_avg_tpot_ms`
    # are the average TTFT / TPOT; the p95 / p99 and end-to-end latency targets
    # extend them (EvalScope-style latency metrics), giving 3 metrics x 3
    # aggregations = 9 optional thresholds.
    #
    # The columns are declared individually (they are queryable/sortable scalars),
    # but everything that WALKS the grid — the runner's CLI forwarding, the SLO
    # evaluation, the budget-utilization check — reads SLO_THRESHOLDS instead of
    # repeating these nine names. Adding an aggregation means one row there plus
    # the column here and in a migration.
    #
    # "TPOT" here means the DECODE-ONLY per-token time (guidellm's
    # inter_token_latency), matching the industry definition and what the report
    # shows. SLO_THRESHOLDS carries the mapping; the threshold names stay
    # `*_tpot_ms` because that is what the CLI, the API and the form call them.
    slo_avg_ttft_ms: Optional[float] = Field(default=None)  # avg TTFT (ms)
    slo_avg_tpot_ms: Optional[float] = Field(default=None)  # avg TPOT (ms)
    slo_p95_ttft_ms: Optional[float] = Field(default=None)  # p95 TTFT (ms)
    slo_p95_tpot_ms: Optional[float] = Field(default=None)  # p95 TPOT (ms)
    slo_p99_ttft_ms: Optional[float] = Field(default=None)  # p99 TTFT (ms)
    slo_p99_tpot_ms: Optional[float] = Field(default=None)  # p99 TPOT (ms)
    slo_avg_latency_ms: Optional[float] = Field(default=None)  # avg e2e latency (ms)
    slo_p95_latency_ms: Optional[float] = Field(default=None)  # p95 e2e latency (ms)
    slo_p99_latency_ms: Optional[float] = Field(default=None)  # p99 e2e latency (ms)

    # Shared prefix: guidellm prefix_buckets — a list of buckets, each
    # {prefix_tokens, prefix_count, bucket_weight}. A common prompt prefix shared
    # across requests (system prompt / RAG context) to exercise prefix-cache
    # reuse; can mix several prefix lengths by weight (e.g. 70% short / 30% long).
    prefix_buckets: Optional[List[Dict[str, Any]]] = Field(sa_type=JSON, default=None)

    # Multi-turn conversation length (guidellm `--data turns=N`).
    turns: Optional[int] = Field(default=None)
    # Warmup / cooldown (numeric: <1 = percent, >=1 = absolute count/seconds).
    warmup: Optional[float] = Field(default=None)
    cooldown: Optional[float] = Field(default=None)
    # Stopping constraints.
    max_errors: Optional[int] = Field(default=None)
    max_error_rate: Optional[float] = Field(default=None)
    stop_on_saturation: Optional[bool] = Field(default=None)

    # Server-derived from the target model instance on create (see
    # validate_and_mutate_benchmark_in), so a client-supplied value is overwritten
    # rather than honored. Kept here because the column is part of the config the
    # export/clone round-trip carries.
    worker_id: Optional[int] = Field(default=None)

    def get_deployment_metadata(
        self,
    ) -> Optional[BenchmarkDeploymentMetadata]:
        """
        Get the deployment metadata for the benchmark.
        """

        return BenchmarkDeploymentMetadata(
            name=self.name,
            labels={
                "benchmark-name": self.name,
                "model-instance-name": self.model_instance_name or "",
                "type": "benchmark",
            },
        )


ModelInstanceSnapshots = Dict[str, ModelInstanceSnapshot]
WorkerSnapshots = Dict[str, WorkerSnapshot]
GPUSnapshots = Dict[str, GPUSnapshot]


class BenchmarkSnapshot(BaseModel):
    instances: Optional[ModelInstanceSnapshots] = None
    workers: Optional[WorkerSnapshots] = None
    gpus: Optional[GPUSnapshots] = None
    # The deployment generation the run measured, for a model that carries one.
    # Not a scalar column: nothing queries or sorts by it, it is read with the
    # rest of the snapshot when a reader asks what a report was measuring.
    spec_digest: Optional[str] = None
    # The route the load entered through, in `route` mode. Recorded because a
    # route is not a fixed view of a model: its targets and their weights can
    # be edited, and a canary route can send a share of the load somewhere
    # else entirely — so "which route, under what name" is part of what the
    # numbers mean.
    route_name: Optional[str] = None


class BenchmarkMetricsLite(SQLModel):
    requests_per_second_mean: Optional[float] = Field(
        default=None, description="Mean requests per second (unit: req/s)"
    )
    request_latency_mean: Optional[float] = Field(
        default=None, description="Mean request latency (unit: seconds)"
    )
    # guidellm reports two per-output-token latencies and names them the reverse
    # of the industry's convention. Keep both, but know which is which:
    #
    #   inter_token_latency_ms   = (last_token - first_token) / (tokens - 1)
    #     Decode only. This IS the industry's TPOT (vLLM, genai-perf). It is what
    #     the report displays as "TPOT" and what the tpot SLO thresholds bound.
    #   time_per_output_token_ms = (last_token - request_start) / tokens
    #     Includes TTFT, so it bills prefill + queue wait to the decode loop. No
    #     standard name. Recorded for completeness; not displayed, not judged on.
    time_per_output_token_mean: Optional[float] = Field(
        default=None,
        description=(
            "Mean per-output-token time INCLUDING the first token, i.e. "
            "guidellm's time_per_output_token (unit: ms). Not the industry TPOT "
            "— see inter_token_latency_mean"
        ),
    )
    inter_token_latency_mean: Optional[float] = Field(
        default=None,
        description=(
            "Mean decode-only time per output token, i.e. guidellm's "
            "inter_token_latency and the industry's TPOT (unit: ms)"
        ),
    )
    time_to_first_token_mean: Optional[float] = Field(
        default=None, description="Mean time to first token (unit: ms)"
    )
    # P95 / P99 percentiles for the SLO-relevant latency metrics (populated from
    # guidellm's per-point percentiles). Used to evaluate tail SLO thresholds.
    time_to_first_token_p95: Optional[float] = Field(
        default=None, description="P95 time to first token (unit: ms)"
    )
    inter_token_latency_p95: Optional[float] = Field(
        default=None, description="P95 decode-only time per output token (unit: ms)"
    )
    # Kept because it is cheap and already collected, but nothing reads it: the
    # tpot SLO thresholds moved to the decode-only metric above.
    time_per_output_token_p95: Optional[float] = Field(
        default=None,
        description="P95 per-output-token time including the first token (unit: ms)",
    )
    request_latency_p95: Optional[float] = Field(
        default=None, description="P95 request latency (unit: seconds)"
    )
    time_to_first_token_p99: Optional[float] = Field(
        default=None, description="P99 time to first token (unit: ms)"
    )
    inter_token_latency_p99: Optional[float] = Field(
        default=None, description="P99 decode-only time per output token (unit: ms)"
    )
    time_per_output_token_p99: Optional[float] = Field(
        default=None,
        description="P99 per-output-token time including the first token (unit: ms)",
    )
    request_latency_p99: Optional[float] = Field(
        default=None, description="P99 request latency (unit: seconds)"
    )
    # ── Real ITL: the per-INTERVAL distribution ───────────────────────────────
    # Everything above is one value per REQUEST. These four summarize the
    # measured gaps BETWEEN consecutive streamed outputs — one sample per gap,
    # pooled across requests — which is what vLLM / SGLang / evalscope report as
    # ITL, and the only reading here that can show a single decode stall (a
    # per-request average divides it away by that request's other gaps).
    #
    # Deliberately not named `inter_token_latency_*`: those columns above hold
    # guidellm's field of that name, which is the industry's TPOT. Two names one
    # letter apart for two different metrics is how a report ends up comparing
    # the wrong pair.
    #
    # None means NOT MEASURED — a point from before benchmark-runner started
    # recording the gaps, or a non-streaming run — never "the gaps were 0 ms".
    # `_max` is carried (unlike every other metric here) because the worst
    # single gap IS the finding for a stall hunt; SGLang reports Max ITL for the
    # same reason.
    itl_per_chunk_mean: Optional[float] = Field(
        default=None, description="Mean measured inter-token gap (unit: ms)"
    )
    itl_per_chunk_p95: Optional[float] = Field(
        default=None, description="P95 measured inter-token gap (unit: ms)"
    )
    itl_per_chunk_p99: Optional[float] = Field(
        default=None, description="P99 measured inter-token gap (unit: ms)"
    )
    itl_per_chunk_max: Optional[float] = Field(
        default=None, description="Largest measured inter-token gap (unit: ms)"
    )
    tokens_per_second_mean: Optional[float] = Field(
        default=None, description="Mean tokens per second (unit: tok/s)"
    )
    output_tokens_per_second_mean: Optional[float] = Field(
        default=None, description="Mean output tokens per second (unit: tok/s)"
    )
    input_tokens_per_second_mean: Optional[float] = Field(
        default=None, description="Mean prompt tokens per second (unit: tok/s)"
    )
    request_concurrency_mean: Optional[float] = Field(
        default=None,
        description="Mean request concurrency (unit: number of concurrent requests)",
    )
    request_concurrency_max: Optional[float] = Field(
        default=None,
        description="Max request concurrency (unit: number of concurrent requests)",
    )
    request_total: Optional[int] = Field(
        default=None, description="Total number of requests made"
    )
    request_successful: Optional[int] = Field(
        default=None, description="Total number of successful requests"
    )
    request_errored: Optional[int] = Field(
        default=None, description="Total number of errored requests"
    )
    request_incomplete: Optional[int] = Field(
        default=None, description="Total number of incomplete requests"
    )


class BenchmarkMetrics(BenchmarkMetricsLite):
    raw_metrics: Optional[Dict[str, Any]] = Field(
        sa_column=Column(JSON), default=None
    )  # deferred loading of potentially large field


class BenchmarkResultGrid(BenchmarkMetricsLite):
    """
    One measured point of a benchmark task: a single (input_tokens, rate) cell,
    without the parent link.

    Split out from :class:`BenchmarkResultBase` so the write contract
    (:class:`BenchmarkResultCreate`) can reuse the coordinates without inheriting
    `benchmark_id` — the server derives that from the request path, and a client
    must not be able to state it.
    """

    # Grid coordinates
    input_tokens: Optional[int] = Field(
        default=None
    )  # input-length axis (one guidellm run)
    rate: Optional[float] = Field(
        default=None
    )  # concurrency (concurrent) or req/s (constant/poisson)
    strategy_type: Optional[str] = Field(
        default=None
    )  # concurrent / constant / poisson / ...
    sequence: int = Field(default=0)  # run_index, for ordering / aligning benchmarks[i]


class BenchmarkResultBase(BenchmarkResultGrid):
    """
    A measured point together with the benchmark it belongs to.

    A benchmark task produces N x M of these (N input lengths x M rates). The
    parent `Benchmark` row keeps a single "representative" point (global throughput
    peak) in its flat metric columns for list/sort; the full grid lives here.
    """

    benchmark_id: int = Field(
        sa_column=Column(
            Integer,
            ForeignKey("benchmarks.id", ondelete="CASCADE"),
            index=True,
            nullable=False,
        )
    )


class BenchmarkResult(BenchmarkResultBase, BaseModelMixin, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    raw_metrics: Optional[Dict[str, Any]] = Field(
        sa_column=Column(JSON), default=None
    )  # this point's benchmarks[i] dump (includes percentiles)

    __tablename__ = 'benchmark_results'


class BenchmarkResultCreate(BenchmarkResultGrid):
    """
    The write contract for one uploaded point.

    Deliberately narrower than the table model: it carries the measurement and
    nothing else. `benchmark_id` comes from the request path, and `id` /
    `created_at` / `updated_at` / `deleted_at` are the server's to assign — an
    untyped dict body let a client set any of them (a pinned primary key turned
    into a 500, and `deleted_at` hid the row it had just written).
    """

    raw_metrics: Optional[Dict[str, Any]] = None


class BenchmarkResultPublic(BenchmarkResultBase):
    id: int
    created_at: datetime
    updated_at: datetime
    # This point's benchmarks[i] dump (includes percentiles) for stage drill-down.
    # No request samples are stored (runner uses --sample-requests 0), so it stays
    # a moderate size per stage.
    raw_metrics: Optional[Dict[str, Any]] = None


class BenchmarkRuntime(SQLModel):
    """What the RUN writes about itself: progress, outcome, conclusion.

    Deliberately NOT on :class:`BenchmarkBase`, which is what `BenchmarkCreate`
    inherits wholesale. `validate_and_mutate_benchmark_in` builds the row with
    ``Benchmark(**benchmark_in.model_dump())`` and then overrides only the fields it
    derives (worker/cluster/owner, model, seed), so anything else on the create
    schema lands in the row exactly as the client sent it — a `POST` carrying
    ``{"peak_rate": 9999, "validity": {"sufficient": true}, "state": "completed"}``
    would create a benchmark that already displays a conclusion nobody measured.

    These are only ever written by the worker, through `BenchmarkStateUpdate` on
    `PATCH /{id}/state`. Mixed into `BenchmarkWithSnapshots` rather than the base:
    the table and both Public schemas descend from it, so readers keep every field
    while the create body loses them. Same reasoning as `owner_principal_id`, and as
    `BenchmarkResultCreate` narrowing the per-point upload.
    """

    state: BenchmarkStateEnum = Field(
        default=BenchmarkStateEnum.PENDING,
        index=True,
    )
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    progress: Optional[float] = Field(default=None)
    pid: Optional[int] = Field(default=None)

    # Max rate that still meets the SLO targets (computed by the worker).
    slo_met_rate: Optional[float] = Field(default=None)
    # Recommended concurrency from over-saturation detection.
    recommended_rate: Optional[float] = Field(default=None)
    # Best operating points (computed by the worker from the stage grid).
    peak_rate: Optional[float] = Field(default=None)  # rate at throughput peak
    # Test-coverage validity, computed by the worker from the stage grid:
    # {"sufficient": bool, "warnings": [{"code": str, "params": {...}}]}. Drives
    # the detail page's coverage warning banner.
    validity: Optional[Dict[str, Any]] = Field(sa_type=JSON, default=None)


class BenchmarkWithSnapshots(BenchmarkBase, BenchmarkRuntime):
    snapshot: Optional[BenchmarkSnapshot] = Field(
        default=None,
        sa_column=Column(pydantic_column_type(BenchmarkSnapshot)),
    )
    gpu_summary: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    gpu_vendor_summary: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )


class Benchmark(BenchmarkWithSnapshots, BenchmarkMetrics, BaseModelMixin, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)

    # Tenant scope. Server-derived from cluster on creation.
    owner_principal_id: Optional[int] = Field(
        default=None,
        sa_column=Column(Integer, ForeignKey("principals.id"), nullable=True),
    )
    __tablename__ = 'benchmarks'


class BenchmarkListParams(ListParams):
    sortable_fields: ClassVar[List[str]] = [
        "name",
        "dataset_name",
        "model_name",
        "state",
        "created_at",
        "updated_at",
        # metrics fields
        "requests_per_second_mean",
        "request_latency_mean",
        "time_per_output_token_mean",
        "inter_token_latency_mean",
        "time_to_first_token_mean",
        # Tails of the latency metrics the list shows, and the measured
        # per-interval ITL. "Which run has the worst TTFT p99" is a sort, not a
        # read of every row.
        "time_to_first_token_p95",
        "time_to_first_token_p99",
        "inter_token_latency_p95",
        "inter_token_latency_p99",
        "itl_per_chunk_mean",
        "itl_per_chunk_p95",
        "itl_per_chunk_p99",
        "tokens_per_second_mean",
        "output_tokens_per_second_mean",
        "input_tokens_per_second_mean",
        "request_concurrency_mean",
        "request_concurrency_max",
        "request_total",
        "request_successful",
        "request_errored",
        "request_incomplete",
    ]


class BenchmarkCreate(BenchmarkBase):
    # The route to drive, in `route` mode. Input only — the row keys on the
    # model, and the name is recorded on the snapshot rather than as a column:
    # what a route resolves to is editable (targets, weights, a canary), so it
    # is a fact about the run, not a handle the run is addressed by.
    #
    # Optional, because an API client that names only a model still gets the
    # route derived for it. The form sends it because it lists routes: a model
    # can sit behind more than one, and picking for the user would measure
    # whichever the server happened to choose.
    route_name: Optional[str] = None

    # A run targets a MODEL. Naming a member is still accepted — the instance
    # list page's "run benchmark" action does exactly that, and a client that
    # names one is naming its model too — but it is not required, and under
    # PD it is not a choice a client can make correctly: a group answers
    # only through its router. The server resolves the endpoint and writes the
    # name it resolved, so the column stays as non-null as it ever was.
    model_instance_name: Optional[str] = None


class BenchmarkUpdate(SQLModel):
    name: str = Field(index=True, unique=True)
    description: Optional[str] = Field(
        sa_type=Text,
        nullable=True,
        default=None,
    )


class BenchmarkStateUpdate(SQLModel):
    state: Optional[BenchmarkStateEnum] = None
    state_message: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    pid: Optional[int] = Field(default=None)
    progress: Optional[float] = None
    # Best operating points (computed by the worker from the stage grid; only
    # the explicitly-set ones are persisted via model_fields_set).
    peak_rate: Optional[float] = None
    slo_met_rate: Optional[float] = None
    recommended_rate: Optional[float] = None
    validity: Optional[Dict[str, Any]] = None


class BenchmarkFullPublic(
    BenchmarkWithSnapshots,
    BenchmarkMetrics,
):
    id: int
    # The owning Org. Server-derived from the cluster on create and
    # therefore kept out of BenchmarkBase / Create — declared on the
    # Public schemas so readers can render the owning Org.
    owner_principal_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    gpu_summary: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    gpu_vendor_summary: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )


class BenchmarkPublic(
    BenchmarkWithSnapshots,
    BenchmarkMetricsLite,
):
    id: int
    owner_principal_id: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    gpu_summary: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )
    gpu_vendor_summary: Optional[str] = Field(
        default=None, sa_column=Column(Text, nullable=True)
    )


BenchmarksPublic = PaginatedList[BenchmarkPublic]
