#!/usr/bin/env python3
"""
Generate a latency optimization markdown document from benchmark result JSON files.

Example:
  python3 hack/perf/generate_latency_optimization_doc.py \
    --title "Optimizing Qwen3-8B Latency" \
    --model Qwen/Qwen3-8B-FP8 \
    --baseline-file ./output/baseline.json \
    --optimized-file ./output/optimized.json \
    --group "Inference Engine=./output/vllm.json,./output/sglang.json" \
    --group "Inference Engine=vLLM::./output/vllm.json,SGLang::./output/sglang.json" \
    --group "Speculative Decoding=./output/speculative.json" \
    --other-group "ShareGPT BS=1=./output/sharegpt_bs1_baseline.json,./output/sharegpt_bs1_optimized.json" \
    --other-group "ShareGPT BS=2=./output/sharegpt_bs2_baseline.json,./output/sharegpt_bs2_optimized.json" \
    --output ./output/latency_doc.md
"""

from __future__ import annotations

from perf_doc_common import (
    BenchmarkRecord,
    build_arg_parser,
    collect_experimental_setup,
    collect_profiles_from_results,
    fmt_number,
    load_record,
    load_grouped_records,
    request_success_rate,
    render_benchmark_method_intro,
    render_benchmark_result_block,
    render_backend_parameters,
    render_profile_config,
    render_recommended_configuration,
    render_record_section,
    render_result_image,
    render_setup_list,
    render_standard_note_block,
)


def fmt_latency_compare(baseline_value: float, candidate_value: float) -> str:
    if baseline_value <= 0 or candidate_value <= 0:
        return "N/A"
    ratio = baseline_value / candidate_value
    if ratio >= 1:
        return (
            f'<span style="background-color:lightgreen;">({ratio:.2f}x faster)</span>'
        )
    return f'<span style="background-color:#ffd6d6;">({(1 / ratio):.2f}x slower)</span>'


def is_successful_record(record: BenchmarkRecord) -> bool:
    success_rate = request_success_rate(record)
    return success_rate is None or success_rate >= 1


def latency_annotation(
    baseline_latency: float,
    candidate_latency: float,
    candidate_record: BenchmarkRecord,
) -> str:
    success_rate = request_success_rate(candidate_record)
    if success_rate is not None and success_rate < 1:
        return (
            '<span style="background-color:#ffd6d6;">'
            f'(Success rate: {success_rate * 100:.1f}%, optimization skipped)'
            "</span>"
        )
    return fmt_latency_compare(baseline_latency, candidate_latency)


def format_latency_delta_summary(
    metric_name: str, faster_value: float, slower_value: float
) -> str:
    delta = slower_value - faster_value
    if faster_value <= 0 or slower_value <= 0:
        return f"{metric_name} = {fmt_number(faster_value)} ms vs {fmt_number(slower_value)} ms"
    if abs(delta) < 1e-9:
        return f"{metric_name} = {fmt_number(faster_value)} ms vs {fmt_number(slower_value)} ms, unchanged"
    if delta > 0:
        ratio = slower_value / faster_value
        return (
            f"{metric_name} = {fmt_number(faster_value)} ms vs {fmt_number(slower_value)} ms, "
            f"reduced by {fmt_number(delta)} ms ({fmt_number(ratio)}x faster)"
        )

    increase = abs(delta)
    ratio = faster_value / slower_value
    return (
        f"{metric_name} = {fmt_number(faster_value)} ms vs {fmt_number(slower_value)} ms, "
        f"increased by {fmt_number(increase)} ms ({fmt_number(ratio)}x slower)"
    )


def build_summary_rows(
    baseline: BenchmarkRecord, records: list[BenchmarkRecord]
) -> list[str]:
    baseline_latency = float(baseline.payload.get("request_latency_mean") or 0)
    best_by_group: dict[str, BenchmarkRecord] = {}
    for record in records:
        if not is_successful_record(record):
            continue
        current_best = best_by_group.get(record.group_name)
        record_latency = float(
            record.payload.get("request_latency_mean") or float("inf")
        )
        if current_best is None:
            best_by_group[record.group_name] = record
            continue
        current_best_latency = float(
            current_best.payload.get("request_latency_mean") or float("inf")
        )
        if record_latency < current_best_latency:
            best_by_group[record.group_name] = record

    rows = [
        "| Benchmark Case | Group | Optimized | Baseline | Comparison |",
        "|---|---|---:|---:|---|",
    ]
    for record in best_by_group.values():
        candidate_latency = float(record.payload.get("request_latency_mean") or 0)
        benchmark_case = record.profile
        if record.request_rate is not None:
            benchmark_case = f"{benchmark_case} (r={record.request_rate})"
        rows.append(
            "| "
            + f"{benchmark_case} | {record.group_name} | "
            + f"{candidate_latency:.2f}s | {baseline_latency:.2f}s | "
            + f"{latency_annotation(baseline_latency, candidate_latency, record)} |"
        )
    return rows


def build_profile_comparison_rows(
    other_grouped_records: list[tuple[str, list[BenchmarkRecord]]],
) -> list[str]:
    rows = [
        "| Benchmark Case | Baseline (vLLM without any optimizations) | Optimized |",
        "|----------|-------------------------------------------|-----------|",
    ]
    for profile_name, records in other_grouped_records:
        if len(records) < 2:
            continue
        baseline, optimized = records[0], records[1]
        baseline_latency = float(baseline.payload.get("request_latency_mean") or 0)
        optimized_latency = float(optimized.payload.get("request_latency_mean") or 0)
        rows.append(
            "| "
            + f"**{profile_name}** | "
            + f"Mean latency: {baseline_latency:.2f}s/req | "
            + f"Mean latency: {optimized_latency:.2f}s/req "
            + f"{latency_annotation(baseline_latency, optimized_latency, optimized)} |"
        )
    return rows


def record_display_name(record: BenchmarkRecord) -> str:
    return record.custom_title or record.name


def render_group_latency_summary(records: list[BenchmarkRecord]) -> str:
    successful_records = [record for record in records if is_successful_record(record)]
    if len(successful_records) < 2:
        return ""

    ranked_records = sorted(
        successful_records,
        key=lambda record: float(
            record.payload.get("request_latency_mean") or float("inf")
        ),
    )
    fastest = ranked_records[0]
    slowest = ranked_records[-1]

    fastest_latency = float(fastest.payload.get("request_latency_mean") or 0)
    slowest_latency = float(slowest.payload.get("request_latency_mean") or 0)
    fastest_ttft = float(fastest.payload.get("time_to_first_token_mean") or 0)
    slowest_ttft = float(slowest.payload.get("time_to_first_token_mean") or 0)
    fastest_tpot = float(fastest.payload.get("time_per_output_token_mean") or 0)
    slowest_tpot = float(slowest.payload.get("time_per_output_token_mean") or 0)

    if fastest_latency <= 0 or slowest_latency <= 0:
        return ""

    latency_delta = slowest_latency - fastest_latency
    latency_ratio = slowest_latency / fastest_latency

    summary = (
        f"- Summary: `{record_display_name(fastest)}` Mean Latency = {fmt_number(fastest_latency)}s, "
        f"`{record_display_name(slowest)}` Mean Latency = {fmt_number(slowest_latency)}s. "
        f"`{record_display_name(fastest)}` is faster by {fmt_number(latency_delta)}s "
        f"({fmt_number(latency_ratio)}x faster)."
    )

    extra_metrics: list[str] = []
    if slowest_ttft > 0 and fastest_ttft > 0:
        extra_metrics.append(
            format_latency_delta_summary("TTFT", fastest_ttft, slowest_ttft)
        )
    if slowest_tpot > 0 and fastest_tpot > 0:
        extra_metrics.append(
            format_latency_delta_summary("TPOT", fastest_tpot, slowest_tpot)
        )
    if extra_metrics:
        summary += " " + "; ".join(extra_metrics) + "."

    return summary


def render_other_group_pair_section(
    group_name: str, records: list[BenchmarkRecord]
) -> str:
    if not records:
        return ""

    lines = [f"#### {group_name}", ""]

    if len(records) >= 1:
        baseline = records[0]
        baseline_instance = baseline.payload.get("snapshot", {}).get("instances", {})
        if isinstance(baseline_instance, dict) and baseline_instance:
            baseline_instance = next(iter(baseline_instance.values()))
        else:
            baseline_instance = {}
        lines.extend(
            [
                f"- Baseline Backend Parameters:{render_backend_parameters(baseline_instance)}",
                "",
                render_benchmark_result_block(baseline.payload).replace(
                    '??? info "Benchmark result"',
                    '??? info "Baseline benchmark result"',
                    1,
                ),
                "",
            ]
        )

    if len(records) >= 2:
        optimized = records[1]
        optimized_instance = optimized.payload.get("snapshot", {}).get("instances", {})
        if isinstance(optimized_instance, dict) and optimized_instance:
            optimized_instance = next(iter(optimized_instance.values()))
        else:
            optimized_instance = {}
        lines.extend(
            [
                f"- Optimized Backend Parameters:{render_backend_parameters(optimized_instance)}",
                "",
                render_benchmark_result_block(optimized.payload).replace(
                    '??? info "Benchmark result"',
                    '??? info "Optimized benchmark result"',
                    1,
                ),
                "",
            ]
        )

    return "\n".join(lines)


def generate_latency_markdown(
    title: str,
    baseline: BenchmarkRecord,
    grouped_records: list[tuple[str, list[BenchmarkRecord]]],
    other_grouped_records: list[tuple[str, list[BenchmarkRecord]]],
    declared_models: list[str] | None = None,
    optimized_record: BenchmarkRecord | None = None,
    image_name: str = "replace-this-image.png",
) -> str:
    summary_records = [record for _, records in grouped_records for record in records]
    all_records = summary_records + [
        record for _, records in other_grouped_records for record in records
    ]
    summary_rows = build_profile_comparison_rows(other_grouped_records)
    optimization_rows = build_summary_rows(baseline, summary_records)
    models, hardware, engine_versions = collect_experimental_setup(
        all_records, declared_models=declared_models
    )
    profile_configs = collect_profiles_from_results(all_records)
    used_profiles = list(profile_configs.keys())

    sections = [
        f"# {title}",
        "",
        "## Conclusion",
        "",
        render_result_image("Latency Optimization Result", image_name),
        "",
        *render_recommended_configuration(
            "latency", optimized_record, declared_models=declared_models
        ),
        "Comparison of benchmark results before and after optimization:",
        "",
        *summary_rows,
        "",
        *render_standard_note_block(),
        "## Experimental Setup",
        "",
        *render_setup_list("Model", models),
        *render_setup_list("Hardware", hardware),
        *render_setup_list("Engine Version", engine_versions),
        *render_benchmark_method_intro(),
    ]

    for profile_name in used_profiles:
        sections.extend(
            render_profile_config(profile_name, profile_configs.get(profile_name))
        )

    sections.extend(
        [
            "## Experiment Results",
            "",
        ]
    )

    for group_name, records in grouped_records:
        sections.append(f"### {group_name}")
        sections.append("")
        include_subheading = len(records) > 1
        for record in records:
            sections.append(
                render_record_section(record, include_heading=include_subheading)
            )
        group_summary = render_group_latency_summary(records)
        if group_summary:
            sections.extend([group_summary, ""])

    sections.extend(
        [
            "### Summary of Optimization Options",
            "",
            "| Benchmark Case | Group | Optimized | Baseline | Comparison |",
            "|---|---|---:|---:|---|",
            *optimization_rows[2:],
            "",
        ]
    )

    if other_grouped_records:
        sections.extend(
            [
                "### Other Benchmark Cases",
                "",
            ]
        )
        for group_name, records in other_grouped_records:
            rendered = render_other_group_pair_section(group_name, records)
            if rendered:
                sections.append(rendered)

    return "\n".join(sections).rstrip() + "\n"


def main() -> None:
    args = build_arg_parser("latency").parse_args()
    baseline_record, grouped_records, other_grouped_records = load_grouped_records(
        args.baseline_file, args.group, args.other_group
    )
    optimized_record = load_record(args.optimized_file) if args.optimized_file else None
    output_path = args.output
    from pathlib import Path

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        generate_latency_markdown(
            args.title,
            baseline_record,
            grouped_records,
            other_grouped_records,
            declared_models=args.model,
            optimized_record=optimized_record,
            image_name=args.image_name,
        ),
        encoding="utf-8",
    )
    print(f"Generated markdown document at {output_file}")


if __name__ == "__main__":
    main()
