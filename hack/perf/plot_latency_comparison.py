"""
Plot latency comparison between baseline and optimized benchmark results.

Examples:
  uv run python hack/perf/plot_latency_comparison.py \
    --metric latency \
    --baseline baseline_r1.json baseline_r4.json baseline_r8.json baseline_r16.json \
    --optimized optimized_r1.json optimized_r4.json optimized_r8.json optimized_r16.json \
    --output latency_comparison.png

  uv run python hack/perf/plot_latency_comparison.py \
    --metric ttft \
    --baseline baseline_r1.json baseline_r4.json baseline_r8.json baseline_r16.json \
    --optimized optimized_r1.json optimized_r4.json optimized_r8.json optimized_r16.json \
    --output ttft_comparison.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


METRIC_FIELDS = {
    "latency": ("request_latency_mean", "Latency", "s"),
    "ttft": ("time_to_first_token_mean", "TTFT", "ms"),
    "itl": ("inter_token_latency_mean", "ITL", "ms"),
    "tpot": ("time_per_output_token_mean", "TPOT", "ms"),
}


@dataclass(frozen=True)
class BenchmarkPoint:
    request_rate: float
    request_latency_mean: float
    request_concurrency_mean: float
    time_to_first_token_mean: float
    inter_token_latency_mean: float
    time_per_output_token_mean: float
    source: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Compare two groups of benchmark result JSON files and plot "
            "request rate vs. mean latency."
        ),
        epilog="""\
Examples:
  uv run python hack/perf/plot_latency_comparison.py \
    --metric latency \
    --baseline baseline_r1.json baseline_r4.json baseline_r8.json baseline_r16.json \
    --optimized optimized_r1.json optimized_r4.json optimized_r8.json optimized_r16.json \
    --output latency_comparison.png

  uv run python hack/perf/plot_latency_comparison.py \
    --metric ttft \
    --baseline baseline_r1.json baseline_r4.json baseline_r8.json baseline_r16.json \
    --optimized optimized_r1.json optimized_r4.json optimized_r8.json optimized_r16.json \
    --output ttft_comparison.png
""",
    )
    parser.add_argument(
        "--baseline",
        nargs="+",
        required=True,
        help="Baseline benchmark JSON files.",
    )
    parser.add_argument(
        "--optimized",
        nargs="+",
        required=True,
        help="GPUStack-Optimized benchmark JSON files.",
    )
    parser.add_argument(
        "--output",
        default="latency_comparison.png",
        help="Output image path. Default: latency_comparison.png",
    )
    parser.add_argument(
        "--title",
        default="Latency Comparison: Baseline vs. GPUStack-Optimized",
        help="Plot title.",
    )
    parser.add_argument(
        "--figsize",
        nargs=2,
        type=float,
        metavar=("WIDTH", "HEIGHT"),
        default=(11, 7),
        help="Figure size in inches. Default: 11 7",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_FIELDS.keys()),
        default="latency",
        help="Metric to plot on y-axis. Default: latency",
    )
    return parser.parse_args()


def load_point(path_str: str) -> BenchmarkPoint:
    path = Path(path_str)
    with path.open() as f:
        payload = json.load(f)

    required_fields = [
        "request_rate",
        "request_latency_mean",
        "request_concurrency_mean",
        "time_to_first_token_mean",
        "inter_token_latency_mean",
        "time_per_output_token_mean",
    ]
    missing = [field for field in required_fields if payload.get(field) is None]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"{path} is missing required fields: {missing_str}")

    return BenchmarkPoint(
        request_rate=float(payload["request_rate"]),
        request_latency_mean=float(payload["request_latency_mean"]),
        request_concurrency_mean=float(payload["request_concurrency_mean"]),
        time_to_first_token_mean=float(payload["time_to_first_token_mean"]),
        inter_token_latency_mean=float(payload["inter_token_latency_mean"]),
        time_per_output_token_mean=float(payload["time_per_output_token_mean"]),
        source=path,
    )


def load_series(paths: list[str]) -> list[BenchmarkPoint]:
    points = [load_point(path) for path in paths]
    return sorted(points, key=lambda point: point.request_rate)


def metric_value(point: BenchmarkPoint, metric: str) -> float:
    field_name = METRIC_FIELDS[metric][0]
    return float(getattr(point, field_name))


def metric_label(metric: str) -> str:
    title, unit = METRIC_FIELDS[metric][1], METRIC_FIELDS[metric][2]
    return f"{title} ({unit})"


def annotate_series(
    ax: Axes,
    points: list[BenchmarkPoint],
    color: str,
    direction: int,
    metric: str,
) -> None:
    x_offsets = [10, 18, -54, -62, 14, -48]
    y_magnitudes = [18, 30, 22, 34, 26, 38]
    short_name, unit = METRIC_FIELDS[metric][1].lower(), METRIC_FIELDS[metric][2]
    for index, point in enumerate(points):
        x_offset = x_offsets[index % len(x_offsets)]
        y_offset = y_magnitudes[index % len(y_magnitudes)] * direction
        label = (
            f"{short_name}={metric_value(point, metric):.2f}{unit}\n"
            f"rps={point.request_rate:.0f}\n"
            f"conc={point.request_concurrency_mean:.2f}"
        )
        ax.annotate(
            label,
            xy=(point.request_rate, metric_value(point, metric)),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            ha="left" if x_offset >= 0 else "right",
            va="bottom" if direction > 0 else "top",
            fontsize=9,
            color=color,
            bbox={
                "boxstyle": "round,pad=0.25",
                "fc": "white",
                "ec": color,
                "alpha": 0.85,
            },
            arrowprops={"arrowstyle": "-", "color": color, "alpha": 0.5},
        )


def annotate_speedup_arrows(
    ax: Axes,
    baseline_points: list[BenchmarkPoint],
    optimized_points: list[BenchmarkPoint],
    metric: str,
) -> None:
    baseline_by_rate = {point.request_rate: point for point in baseline_points}
    optimized_by_rate = {point.request_rate: point for point in optimized_points}
    shared_rates = sorted(set(baseline_by_rate) & set(optimized_by_rate))

    for request_rate in shared_rates:
        baseline_point = baseline_by_rate[request_rate]
        optimized_point = optimized_by_rate[request_rate]
        optimized_metric_value = metric_value(optimized_point, metric)
        baseline_metric_value = metric_value(baseline_point, metric)
        if optimized_metric_value <= 0:
            continue

        speedup = baseline_metric_value / optimized_metric_value
        mid_y = (baseline_metric_value + optimized_metric_value) / 2
        if speedup >= 1:
            speedup_label = f"x{speedup:.2f} faster"
        else:
            speedup_label = f"x{(1 / speedup):.2f} slower"

        ax.annotate(
            "",
            xy=(request_rate, optimized_metric_value),
            xytext=(request_rate, baseline_metric_value),
            arrowprops={
                "arrowstyle": "->",
                "color": "#54A24B",
                "lw": 2,
                "alpha": 0.9,
            },
        )
        ax.text(
            request_rate,
            mid_y,
            speedup_label,
            ha="left",
            va="center",
            fontsize=9,
            color="#54A24B",
            bbox={
                "boxstyle": "round,pad=0.2",
                "fc": "white",
                "ec": "#54A24B",
                "alpha": 0.9,
            },
        )


def plot_series(
    baseline_points: list[BenchmarkPoint],
    optimized_points: list[BenchmarkPoint],
    title: str,
    output_path: str,
    figsize: tuple[float, float],
    metric: str,
) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    baseline_x = [point.request_rate for point in baseline_points]
    baseline_y = [metric_value(point, metric) for point in baseline_points]
    optimized_x = [point.request_rate for point in optimized_points]
    optimized_y = [metric_value(point, metric) for point in optimized_points]

    ax.plot(
        baseline_x,
        baseline_y,
        marker="o",
        linewidth=2,
        color="#4C78A8",
        label="Baseline",
    )
    ax.plot(
        optimized_x,
        optimized_y,
        marker="o",
        linewidth=2,
        color="#F58518",
        label="GPUStack-Optimized",
    )

    annotate_series(ax, baseline_points, color="#4C78A8", direction=1, metric=metric)
    annotate_series(ax, optimized_points, color="#F58518", direction=-1, metric=metric)
    annotate_speedup_arrows(ax, baseline_points, optimized_points, metric=metric)

    all_latencies = baseline_y + optimized_y
    if all_latencies:
        min_latency = min(all_latencies)
        max_latency = max(all_latencies)
        latency_span = max_latency - min_latency
        padding = max(1.0, latency_span * 0.35)
        lower_bound = max(0, min_latency - padding * 0.45)
        upper_bound = max_latency + padding
        ax.set_ylim(lower_bound, upper_bound)

    ax.set_xlabel("Request/s")
    ax.set_ylabel(metric_label(metric))
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")


def main() -> None:
    args = parse_args()
    baseline_points = load_series(args.baseline)
    optimized_points = load_series(args.optimized)
    plot_series(
        baseline_points=baseline_points,
        optimized_points=optimized_points,
        title=args.title,
        output_path=args.output,
        figsize=tuple(args.figsize),
        metric=args.metric,
    )


if __name__ == "__main__":
    main()
