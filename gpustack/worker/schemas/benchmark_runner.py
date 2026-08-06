# The data structures in this file are adapted from:
# https://github.com/vllm-project/guidellm/blob/62b0f8e01f5c558920fd5d02fe828459264b4f87/src/guidellm/benchmark/schemas/generative/report.py#L58
# Modifications have been made to fit project requirements.

import json
import uuid
from pathlib import Path
from typing import Any, Dict, Generic, Literal, Optional, Self, TypeVar
from pydantic import BaseModel, Field

from gpustack.schemas.benchmark import BenchmarkMetrics

BaseModelT = TypeVar("BaseModelT", bound=BaseModel)
RegisterClassT = TypeVar("RegisterClassT", bound=type)
SuccessfulT = TypeVar("SuccessfulT")
ErroredT = TypeVar("ErroredT")
IncompleteT = TypeVar("IncompleteT")
TotalT = TypeVar("TotalT")

GenerativeRequestType = Literal[
    "text_completions",
    "chat_completions",
    "audio_transcriptions",
    "audio_translations",
]


class StatusBreakdown(BaseModel, Generic[SuccessfulT, ErroredT, IncompleteT, TotalT]):
    """
    Generic model for organizing results by processing status.

    Provides structured categorization of results into successful, errored,
    incomplete, and total status groups. Supports flexible typing for each
    status category to accommodate different result types while maintaining
    consistent organization patterns across the application.

    Example:
    ::
        from guidellm.utils import StatusBreakdown

        # Define a breakdown for request counts
        breakdown = StatusBreakdown[int, int, int, int](
            successful=150,
            errored=5,
            incomplete=10,
            total=165
        )
    """

    successful: SuccessfulT = Field(
        description="Results or metrics for requests with successful completion status",
        default=None,  # type: ignore[assignment]
    )
    errored: ErroredT = Field(
        description="Results or metrics for requests with error completion status",
        default=None,  # type: ignore[assignment]
    )
    incomplete: IncompleteT = Field(
        description="Results or metrics for requests with incomplete processing status",
        default=None,  # type: ignore[assignment]
    )
    total: TotalT = Field(
        description="Aggregated results or metrics combining all status categories",
        default=None,  # type: ignore[assignment]
    )


class SchedulerMetrics(BaseModel):
    """
    Scheduler timing and performance statistics.

    Tracks overall benchmark timing, request counts by status, and detailed internal
    scheduler performance metrics including queue times, processing delays, and
    request execution statistics. Used to analyze scheduler efficiency and identify
    bottlenecks in request processing pipelines.
    """

    # Overall timings for the scheduler
    start_time: float = Field(
        description="Unix timestamp when the benchmark run started"
    )
    request_start_time: float = Field(
        description="Unix timestamp when first request was made"
    )
    measure_start_time: float = Field(
        description="Unix timestamp when measurement period started"
    )
    measure_end_time: float = Field(
        description="Unix timestamp when measurement period ended"
    )
    request_end_time: float = Field(
        description="Unix timestamp when last request completed"
    )
    end_time: float = Field(description="Unix timestamp when the benchmark run ended")


class Percentiles(BaseModel):
    """
    Standard percentile values for probability distributions.

    Captures key percentile points from 0.1th to 99.9th percentile for comprehensive
    distribution analysis, enabling assessment of central tendency, spread, and tail
    behavior in benchmark metrics.

    Only the percentiles the report actually renders are declared: pydantic drops
    undeclared keys on validation, and `raw_metrics` is a re-dump of THIS model, so
    a percentile missing here is gone from the database, not merely unused. guidellm
    emits p001/p01/p05/p10/p25/p50/p75/p90/p95/p99/p999; p25 and p75 are the IQR
    band of the latency-distribution charts (a robust spread signal — "the band
    widens" only means divergence when it comes from real quantiles, never from
    mean x a constant factor), and p90/p95/p99 carry the tail / SLA thresholds.
    """

    # All optional, and none of them defaulted to 0. Two reasons, and the second is
    # the one that made p50/p90/p95/p99 join p25/p75: a report written by an older
    # guidellm (or a row already in the database) still validates, and a report that
    # omits ONE percentile no longer costs the whole point. Required fields put that
    # decision in the wrong place — `model_validate` raises, `_aggregate_points`
    # skips the file it came from, and a twelve-point curve quietly becomes eleven,
    # surfacing only as "1 of 12 measured point(s) could not be read" in
    # `state_message`. Absence renders as "no data"; 0 would read as a real
    # measurement.
    p25: Optional[float] = Field(default=None, description="25th percentile value")
    p50: Optional[float] = Field(
        default=None, description="50th percentile (median) value"
    )
    p75: Optional[float] = Field(default=None, description="75th percentile value")
    p90: Optional[float] = Field(default=None, description="90th percentile value")
    p95: Optional[float] = Field(default=None, description="95th percentile value")
    p99: Optional[float] = Field(default=None, description="99th percentile value")


class DistributionSummary(BaseModel):
    """
    Comprehensive statistical summary of a probability distribution.

    Captures central tendency (mean, median, mode), spread (variance, std_dev),
    extrema (min, max), and percentile information with optional probability density
    function. Supports creation from raw values, PDFs, or time-based event data for
    rate and concurrency analysis in benchmark metrics.

    guidellm additionally reports `mode`, `variance`, `total_sum` and `pdf`. `pdf` is
    the per-value probability density array and is deliberately NOT declared: it
    dwarfs the rest of the point and nothing renders it. Leaving it undeclared is
    what keeps it out of `raw_metrics`.
    """

    mean: float = Field(description="Mean/average value")
    median: float = Field(description="Median (50th percentile) value")
    min: float = Field(description="Minimum value")
    max: float = Field(description="Maximum value")
    # Sample size of this distribution. A point with few samples has p99 == max, so
    # the report greys out its tail percentiles instead of reading them as an SLA
    # conclusion. Optional for the same backward-compatibility reason as p25/p75.
    count: Optional[int] = Field(default=None, description="Number of observations")
    std_dev: Optional[float] = Field(default=None, description="Standard deviation")
    percentiles: Percentiles = Field(description="Standard percentile values")


class StatusDistributionSummary(
    StatusBreakdown[
        DistributionSummary,
        DistributionSummary,
        DistributionSummary,
        DistributionSummary,
    ]
):
    """
    Distribution summaries broken down by request status categories.

    Provides separate statistical analysis for successful, incomplete, and errored
    requests with total aggregate statistics. Enables status-aware performance analysis
    and SLO validation across different request outcomes in benchmark results.
    """

    pass


class GenerativeMetrics(BaseModel):
    """
    Comprehensive metrics for generative AI benchmarks.

    Aggregates request statistics, token metrics, timing distributions, and
    domain-specific measurements across text, image, video, and audio modalities.
    Provides detailed statistical summaries including distribution analysis for
    throughput, latency, concurrency, and resource utilization metrics across
    successful, incomplete, and errored requests.
    """

    # Request stats
    request_totals: StatusBreakdown[int, int, int, int] = Field(
        description="Request counts by status: successful, incomplete, errored, total"
    )
    requests_per_second: StatusDistributionSummary = Field(
        description="Distribution of requests per second across benchmark execution"
    )
    request_concurrency: StatusDistributionSummary = Field(
        description="Distribution of concurrent request counts during execution"
    )
    request_latency: StatusDistributionSummary = Field(
        description="Distribution of request latencies for completed requests"
    )
    request_streaming_iterations_count: StatusDistributionSummary = Field(
        description="Distribution of stream iterations for completed requests"
    )

    # General token stats
    prompt_token_count: StatusDistributionSummary = Field(
        description="Distribution of prompt token counts by request status"
    )
    output_token_count: StatusDistributionSummary = Field(
        description="Distribution of output token counts by request status"
    )
    total_token_count: StatusDistributionSummary = Field(
        description="Distribution of total token counts by request status"
    )
    time_to_first_token_ms: StatusDistributionSummary = Field(
        description="Distribution of first token latencies in milliseconds"
    )
    # guidellm's two per-output-token latencies, named the reverse of the
    # industry's convention:
    #   time_per_output_token_ms = (last_token - request_start) / tokens
    #     -> includes TTFT. No standard name. Recorded, not judged on, not shown.
    #   inter_token_latency_ms   = (last_token - first_token) / (tokens - 1)
    #     -> decode only. THIS is the industry TPOT, and what the report shows and
    #        the `sla_*_tpot_ms` thresholds bound.
    # Both are per-REQUEST values whose distribution is token-weighted, so their
    # percentiles rank requests by their own average decode speed; neither can
    # show a single-token stall (guidellm keeps only the first/last token
    # timestamp per request, so the per-chunk gaps do not survive collection).
    time_per_output_token_ms: StatusDistributionSummary = Field(
        description=(
            "Distribution of average time per output token INCLUDING the first "
            "token, in milliseconds"
        )
    )
    inter_token_latency_ms: StatusDistributionSummary = Field(
        description=(
            "Distribution of decode-only time per output token (the industry's "
            "TPOT), in milliseconds"
        )
    )
    # Token-rate distributions. The MEAN is sound (duration-weighted, so it comes
    # out as total tokens over the event span); the PERCENTILES are artifacts and
    # the report does not show them. guidellm keeps only the first and last token
    # timestamp per request and reconstructs the rest by np.linspace, spreading the
    # tokens evenly, while the whole prompt lands at the first-token instant — a
    # real 480-request stage reported tokens_per_second p99 = 689,483 tok/s and
    # max = 9.6M tok/s. Kept in the archive as guidellm reported them.
    prompt_tokens_per_second: StatusDistributionSummary = Field(
        description="Distribution of prompt token processing rates"
    )
    output_tokens_per_second: StatusDistributionSummary = Field(
        description="Distribution of output token generation rates"
    )
    tokens_per_second: StatusDistributionSummary = Field(
        description="Distribution of total token throughput including prompt and output"
    )
    output_tokens_per_iteration: StatusDistributionSummary = Field(
        description="Distribution of output tokens generated per streaming iteration"
    )
    iter_tokens_per_iteration: StatusDistributionSummary = Field(
        description=(
            "Distribution of output tokens (without first) generated per "
            "streaming iteration"
        )
    )


class RequestTimings(BaseModel):
    """
    Timing measurements for tracking request lifecycle events.

    Provides comprehensive timing data for distributed request processing, capturing
    key timestamps from initial targeting through final completion. Essential for
    performance analysis, SLA monitoring, and debugging request processing bottlenecks
    across scheduler workers and backend systems.
    """

    targeted_start: float | None = Field(
        default=None,
        description="Unix timestamp when request was initially targeted for execution",
    )
    queued: float | None = Field(
        default=None,
        description="Unix timestamp when request was placed into processing queue",
    )
    dequeued: float | None = Field(
        default=None,
        description="Unix timestamp when request was removed from queue for processing",
    )
    scheduled_at: float | None = Field(
        default=None,
        description="Unix timestamp when the request was scheduled for processing",
    )
    resolve_start: float | None = Field(
        default=None,
        description="Unix timestamp when backend resolution of the request began",
    )
    request_start: float | None = Field(
        default=None,
        description="Unix timestamp when the backend began processing the request",
    )
    first_request_iteration: float | None = Field(
        default=None,
    )
    first_token_iteration: float | None = Field(
        default=None,
    )
    last_token_iteration: float | None = Field(
        default=None,
    )
    last_request_iteration: float | None = Field(
        default=None,
    )
    request_iterations: int = Field(
        default=0,
    )
    token_iterations: int = Field(
        default=0,
    )
    request_end: float | None = Field(
        default=None,
        description="Unix timestamp when the backend completed processing the request",
    )
    resolve_end: float | None = Field(
        default=None,
        description="Unix timestamp when backend resolution of the request completed",
    )
    finalized: float | None = Field(
        default=None,
        description="Unix timestamp when request was processed by the scheduler",
    )


class RequestInfo(BaseModel):
    """
    Complete information about a request in the scheduler system.

    Encapsulates all metadata, status tracking, and timing information for requests
    processed through the distributed scheduler. Provides comprehensive lifecycle
    tracking from initial queuing through final completion, including error handling
    and node identification for debugging and performance analysis.

    Example:
    ::
        request = RequestInfo()
        request.status = "in_progress"
        start_time = request.started_at
        completion_time = request.completed_at
    """

    request_id: str = Field(
        description="Unique identifier for the request",
        default_factory=lambda: str(uuid.uuid4()),
    )
    status: Literal[
        "queued", "pending", "in_progress", "completed", "errored", "cancelled"
    ] = Field(description="Current processing status of the request", default="queued")
    scheduler_node_id: int = Field(
        description="ID/rank of the scheduler node handling the request",
        default=-1,
    )
    scheduler_process_id: int = Field(
        description="ID/rank of the node's scheduler process handling the request",
        default=-1,
    )
    scheduler_start_time: float = Field(
        description="Unix timestamp when scheduler processing began",
        default=-1,
    )
    timings: RequestTimings = Field(
        default_factory=RequestTimings,
        description="Timing measurements for the request lifecycle",
    )

    error: str | None = Field(
        default=None, description="Error message if the request status is 'errored'"
    )
    traceback: str | None = Field(
        default=None,
        description="Full traceback of the error if the request status is 'errored'",
    )


class UsageMetrics(BaseModel):
    """
    Multimodal usage metrics for generation requests.

    Tracks resource consumption across different modalities including text, images,
    video, and audio. Provides granular metrics for tokens, bytes, duration, and
    format-specific measurements to enable comprehensive usage monitoring and billing.
    """

    # Text stats
    text_tokens: int | None = Field(
        default=None, description="Number of text tokens processed/generated."
    )
    text_words: int | None = Field(
        default=None, description="Number of text words processed/generated."
    )
    text_characters: int | None = Field(
        default=None, description="Number of text characters processed/generated."
    )


class GenerativeRequestStats(BaseModel):
    """
    Request statistics for generative AI text generation workloads.

    Captures comprehensive performance metrics for individual generative requests,
    including token counts, timing measurements, and derived performance statistics.
    Provides computed properties for latency analysis, throughput calculations,
    and token generation metrics essential for benchmark evaluation.

    Example:
    ::
        stats = GenerativeRequestStats(
            request_id="req_123",
            request_type="text_completion",
            info=request_info,
            input_metrics=input_usage,
            output_metrics=output_usage
        )
        throughput = stats.output_tokens_per_second
    """

    type_: Literal["generative_request_stats"] = "generative_request_stats"
    request_id: Optional[str] = Field(
        default=None, description="Unique identifier for the request"
    )
    # Optional: truncated/errored request samples may omit this field.
    request_type: Optional[GenerativeRequestType | str] = Field(
        default=None,
        description="Type of generative request (text_completion or chat_completion)",
    )
    response_id: str | None = Field(
        default=None, description="Unique identifier matching vLLM Response ID"
    )
    request_args: str | None = Field(
        default=None, description="Backend arguments used for this request"
    )
    output: str | None = Field(
        default=None, description="Generated text output from the request"
    )
    info: RequestInfo = Field(description="Request metadata and timing information")
    input_metrics: UsageMetrics = Field(
        description="Token usage statistics for the input prompt"
    )
    output_metrics: UsageMetrics = Field(
        description="Token usage statistics for the generated output"
    )


class BenchmarkStrategy(BaseModel):
    """Scheduling strategy of a single benchmark run (guidellm config.strategy)."""

    type_: Optional[str] = None  # concurrent / constant / poisson / ...
    streams: Optional[int] = None  # concurrent: the concurrency level
    rate: Optional[float] = None  # constant / poisson: requests per second
    max_concurrency: Optional[int] = None


class BenchmarkRunConfig(BaseModel):
    """Per-run config of a benchmark in the report (guidellm benchmark.config)."""

    run_index: int = 0
    strategy: Optional[BenchmarkStrategy] = None


class GenerativeBenchmark(BaseModel):
    """
    Complete generative AI benchmark results with specialized metrics.

    Encapsulates comprehensive performance data from scheduler-driven generative
    workload executions including request-level statistics, token/latency distributions,
    throughput analysis, and concurrency patterns. Provides computed fields for temporal
    analysis and status-grouped request details for detailed post-execution reporting.
    """

    config: Optional[BenchmarkRunConfig] = Field(
        default=None,
        description="Per-run config including scheduling strategy and run index",
    )
    scheduler_metrics: SchedulerMetrics = Field(
        description="Scheduler timing and performance statistics",
    )
    metrics: GenerativeMetrics = Field(
        description="Performance metrics and statistical distributions",
    )
    start_time: float = Field(
        description="Benchmark start time in seconds since epoch",
    )
    end_time: float = Field(
        description="Benchmark end time in seconds since epoch",
    )
    duration: float = Field(
        description="Total benchmark execution duration in seconds",
    )
    requests_truncated: StatusBreakdown[
        list[GenerativeRequestStats],
        list[GenerativeRequestStats],
        list[GenerativeRequestStats],
        None,
    ] = Field(
        default_factory=lambda: StatusBreakdown(
            successful=[],
            errored=[],
            incomplete=[],
            total=None,
        ),
        description=(
            "Request details grouped by status: successful, incomplete, errored"
        ),
    )
    scheduler_state: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Scheduler end-state. end_processing_constraints names the constraint "
            "that actually stopped this point (max_requests = normal; "
            "requests_exhausted = dataset ran out; max_seconds/max_errors = early "
            "stop). Kept only to derive the termination reason, then dropped."
        ),
    )


class GenerativeBenchmarksReport(BaseModel):
    """
    Container for multiple benchmark results with load/save functionality.

    Aggregates multiple generative benchmark executions into a single report,
    providing persistence through JSON and YAML file formats. Enables result
    collection, storage, and retrieval across different execution sessions with
    automatic file type detection and path resolution.

    """

    benchmarks: list[GenerativeBenchmark] = Field(
        description="List of completed benchmarks in the report",
        default_factory=list,
    )

    @staticmethod
    def _termination_reason(bm: "GenerativeBenchmark") -> Optional[dict]:
        """Which constraint stopped this point, for the UI to flag early stops.

        guidellm records the satisfied end constraints in
        ``scheduler_state.end_processing_constraints`` (keyed by constraint name).
        We pick the one that actually halted processing (request_processing in
        stop/stop_local), falling back to the first. ``max_requests`` means the
        point hit its target normally; anything else (``requests_exhausted`` =
        dataset too small, ``max_seconds`` = timed out, ``max_errors*`` = too many
        errors) is an early stop worth surfacing.
        """
        ss = bm.scheduler_state
        if not isinstance(ss, dict):
            return None
        end = ss.get("end_processing_constraints") or ss.get("end_queuing_constraints")
        if not isinstance(end, dict) or not end:
            return None
        name, item = None, None
        for k, v in end.items():
            rp = v.get("request_processing") if isinstance(v, dict) else None
            if rp in ("stop", "stop_local"):
                name, item = k, v
                break
        if name is None:
            name, item = next(iter(end.items()))
        meta = (item.get("metadata") if isinstance(item, dict) else {}) or {}
        # Target = the max_requests constraint's configured value (this point's
        # goal, e.g. 960), NOT the triggering constraint's own count (which for
        # requests_exhausted is just the dataset size = processed).
        target = None
        sc = ss.get("scheduler_constraints")
        if isinstance(sc, dict) and isinstance(sc.get("max_requests"), dict):
            target = (sc["max_requests"].get("metadata") or {}).get("max_requests")
        if target is None:
            target = meta.get("max_requests") or meta.get("num_requests")
        return {
            "reason": name,
            "requested": target,
            "processed": meta.get("processed_requests"),
        }

    @staticmethod
    def _point_metrics_kwargs(bm: "GenerativeBenchmark") -> dict:
        """Flat BenchmarkMetricsLite kwargs for one benchmark point."""
        m = bm.metrics
        return dict(
            requests_per_second_mean=m.requests_per_second.successful.mean,
            request_latency_mean=m.request_latency.successful.mean,
            time_per_output_token_mean=m.time_per_output_token_ms.successful.mean,
            inter_token_latency_mean=m.inter_token_latency_ms.successful.mean,
            time_to_first_token_mean=m.time_to_first_token_ms.successful.mean,
            # p95 / p99 percentiles for the SLA-relevant latency metrics. The tpot
            # thresholds are evaluated on inter_token_latency_* (decode only);
            # time_per_output_token_* is kept for reference — see SLA_THRESHOLDS.
            time_to_first_token_p95=(
                m.time_to_first_token_ms.successful.percentiles.p95
            ),
            inter_token_latency_p95=(
                m.inter_token_latency_ms.successful.percentiles.p95
            ),
            time_per_output_token_p95=(
                m.time_per_output_token_ms.successful.percentiles.p95
            ),
            request_latency_p95=m.request_latency.successful.percentiles.p95,
            time_to_first_token_p99=(
                m.time_to_first_token_ms.successful.percentiles.p99
            ),
            inter_token_latency_p99=(
                m.inter_token_latency_ms.successful.percentiles.p99
            ),
            time_per_output_token_p99=(
                m.time_per_output_token_ms.successful.percentiles.p99
            ),
            request_latency_p99=m.request_latency.successful.percentiles.p99,
            tokens_per_second_mean=m.tokens_per_second.successful.mean,
            output_tokens_per_second_mean=m.output_tokens_per_second.successful.mean,
            input_tokens_per_second_mean=m.prompt_tokens_per_second.successful.mean,
            request_concurrency_max=m.request_concurrency.successful.max,
            request_concurrency_mean=m.request_concurrency.successful.mean,
            request_total=m.request_totals.total,
            request_successful=m.request_totals.successful,
            request_errored=m.request_totals.errored,
            request_incomplete=m.request_totals.incomplete,
        )

    def _peak_benchmark(self) -> Optional["GenerativeBenchmark"]:
        """Global throughput-peak point (representative of the whole task)."""
        best = None
        for bm in self.benchmarks:
            if bm.metrics is None:
                continue
            tps = bm.metrics.tokens_per_second.successful.mean or 0
            if best is None or tps > best[0]:
                best = (tps, bm)
        return best[1] if best else None

    def to_metrics(self) -> Optional[BenchmarkMetrics]:
        """
        Representative metrics for the parent benchmark row = the global
        throughput-peak point across all (input_tokens, rate) cells. For a
        single-rate run this is just the only point, preserving legacy behavior.
        """
        bm = self._peak_benchmark()
        if bm is None:
            return None
        # scheduler_state is bulky and only needed to derive per-point termination
        # (done in to_results); strip it from the full report dump kept here.
        raw = self.model_dump()
        for b in raw.get("benchmarks", []):
            if isinstance(b, dict):
                b.pop("scheduler_state", None)
        return BenchmarkMetrics(
            raw_metrics=raw,
            **self._point_metrics_kwargs(bm),
        )

    def to_results(
        self, input_tokens: Optional[int] = None, sequence_start: int = 0
    ) -> list[dict]:
        """
        One dict per measured (input_tokens, rate) grid point, for upload into the
        benchmark_results sub-table. `benchmark_id` is filled in by the server from
        the request path; `raw_metrics` carries this point's benchmarks[i] dump.

        `sequence` is the point's position in the probe order, which is what the
        results API orders by. A multi-point run writes ONE report file per point,
        each holding a single benchmark whose `config.run_index` is therefore
        always 0 — so the caller passes `sequence_start` to continue the numbering
        across files instead (see BenchmarkManager._sync_benchmark_metrics).
        """
        results: list[dict] = []
        for bm in self.benchmarks:
            if bm.metrics is None:
                continue
            strat = bm.config.strategy if bm.config else None
            strategy_type = strat.type_ if strat else None
            if strat is not None and strat.type_ == "concurrent":
                rate = strat.streams
            elif strat is not None:
                rate = strat.rate
            else:
                rate = None
            sequence = sequence_start + len(results)
            # Keep a compact termination reason in raw_metrics; drop the bulky
            # scheduler_state that produced it.
            raw = bm.model_dump()
            raw.pop("scheduler_state", None)
            termination = self._termination_reason(bm)
            if termination:
                raw["termination"] = termination
            results.append(
                dict(
                    sequence=sequence,
                    strategy_type=strategy_type,
                    rate=float(rate) if rate is not None else None,
                    input_tokens=input_tokens,
                    raw_metrics=raw,
                    **self._point_metrics_kwargs(bm),
                )
            )
        return results

    @classmethod
    def load_file(cls, path: str) -> Self:
        """
        Load report from JSON or YAML file.

        :param path: File path or directory containing DEFAULT_FILE to load from
        :param type_: File format override ('json' or 'yaml'), auto-detected from
            extension if None
        :return: Loaded report instance with benchmarks and configuration
        :raises ValueError: If file type is unsupported or cannot be determined
        :raises FileNotFoundError: If specified file does not exist
        """
        file_path = Path(path)
        file_type = file_path.suffix.lower()[1:]

        with open(file_path, "r", encoding="utf-8") as metrics_file:
            if file_type == "json":
                model_dict = json.loads(metrics_file.read())
            else:
                raise ValueError(f"Unsupported file type: {file_type} for {file_path}.")

        return cls.model_validate(model_dict)
