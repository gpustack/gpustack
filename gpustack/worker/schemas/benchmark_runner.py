# The data structures in this file are adapted from:
# https://github.com/vllm-project/guidellm/blob/62b0f8e01f5c558920fd5d02fe828459264b4f87/src/guidellm/benchmark/schemas/generative/report.py#L58
# Modifications have been made to fit project requirements.

import json
from pathlib import Path
from typing import Generic, Optional, Self, TypeVar
from pydantic import BaseModel, Field

from gpustack.schemas.benchmark import BenchmarkMetrics

BaseModelT = TypeVar("BaseModelT", bound=BaseModel)
RegisterClassT = TypeVar("RegisterClassT", bound=type)
SuccessfulT = TypeVar("SuccessfulT")
ErroredT = TypeVar("ErroredT")
IncompleteT = TypeVar("IncompleteT")
TotalT = TypeVar("TotalT")


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
    """

    p50: float = Field(description="50th percentile (median) value")
    p90: float = Field(description="90th percentile value")
    p99: float = Field(description="99th percentile value")


class DistributionSummary(BaseModel):
    """
    Comprehensive statistical summary of a probability distribution.

    Captures central tendency (mean, median, mode), spread (variance, std_dev),
    extrema (min, max), and percentile information with optional probability density
    function. Supports creation from raw values, PDFs, or time-based event data for
    rate and concurrency analysis in benchmark metrics.
    """

    mean: float = Field(description="Mean/average value")
    median: float = Field(description="Median (50th percentile) value")
    min: float = Field(description="Minimum value")
    max: float = Field(description="Maximum value")
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
    time_per_output_token_ms: StatusDistributionSummary = Field(
        description="Distribution of average time per output token in milliseconds"
    )
    inter_token_latency_ms: StatusDistributionSummary = Field(
        description="Distribution of inter-token latencies in milliseconds"
    )
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


class GenerativeBenchmark(BaseModel):
    """
    Complete generative AI benchmark results with specialized metrics.

    Encapsulates comprehensive performance data from scheduler-driven generative
    workload executions including request-level statistics, token/latency distributions,
    throughput analysis, and concurrency patterns. Provides computed fields for temporal
    analysis and status-grouped request details for detailed post-execution reporting.
    """

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

    def to_metrics(self) -> Optional[BenchmarkMetrics]:
        """
        Convert the report to a gpustack benchmark metrics object.
        """
        if not self.benchmarks:
            return None

        if self.benchmarks[0].metrics is None:
            return None

        fbm = self.benchmarks[0].metrics
        return BenchmarkMetrics(
            raw_metrics=self.model_dump(),
            requests_per_second_mean=fbm.requests_per_second.successful.mean,
            request_latency_mean=fbm.request_latency.successful.mean,
            time_per_output_token_mean=fbm.time_per_output_token_ms.successful.mean,
            inter_token_latency_mean=fbm.inter_token_latency_ms.successful.mean,
            time_to_first_token_mean=fbm.time_to_first_token_ms.successful.mean,
            tokens_per_second_mean=fbm.tokens_per_second.successful.mean,
            output_tokens_per_second_mean=fbm.output_tokens_per_second.successful.mean,
            input_tokens_per_second_mean=fbm.prompt_tokens_per_second.successful.mean,
        )

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
