#!/usr/bin/env python3
"""
Generate a throughput optimization markdown document from benchmark result JSON files.

Example:
  python3 hack/perf/generate_throughput_optimization_doc.py \
    --title "Optimizing Qwen3.5-35B-A3B Throughput" \
    --model Qwen/Qwen3.5-35B-A3B-FP8 \
    --baseline-file ./output/baseline.json \
    --optimized-file ./output/optimized.json \
    --group "Baseline of the Inference Engine=./output/baseline.json,./output/sglang.json" \
    --group "Choosing the Inference Engine=vLLM:: ./output/baseline.json,SGLang:: ./output/sglang.json" \
    --group "Quantization=./output/fp8.json" \
    --other-group "ShareGPT=./output/sharegpt_baseline.json,./output/sharegpt_optimized.json" \
    --other-group "Long Context=./output/long_context_baseline.json,./output/long_context_optimized.json" \
    --output ./output/throughput_doc.md
"""

from __future__ import annotations

from pathlib import Path

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


def fmt_tps_compare(baseline_value: float, candidate_value: float) -> str:
    if baseline_value <= 0 or candidate_value <= 0:
        return "N/A"
    ratio = (candidate_value - baseline_value) / baseline_value * 100
    if ratio >= 0:
        return f'<span style="background-color:lightgreen;">(+{ratio:.2f}%)</span>'
    return f'<span style="background-color:#ffd6d6;">({ratio:.2f}%)</span>'


def throughput_annotation(
    baseline_tps: float,
    candidate_tps: float,
    candidate_record: BenchmarkRecord,
) -> str:
    success_rate = request_success_rate(candidate_record)
    if success_rate is not None and success_rate < 1:
        return (
            '<span style="background-color:#ffd6d6;">'
            f'(Success rate: {success_rate * 100:.1f}%, optimization skipped)'
            "</span>"
        )
    return fmt_tps_compare(baseline_tps, candidate_tps)


def format_tpot_delta_summary(faster_value: float, slower_value: float) -> str:
    delta = slower_value - faster_value
    if faster_value <= 0 or slower_value <= 0:
        return f"Mean TPOT = {fmt_number(faster_value)} ms vs {fmt_number(slower_value)} ms"
    if abs(delta) < 1e-9:
        return f"Mean TPOT = {fmt_number(faster_value)} ms vs {fmt_number(slower_value)} ms, unchanged"
    if delta > 0:
        ratio = delta / slower_value * 100
        return (
            f"Mean TPOT = {fmt_number(faster_value)} ms vs {fmt_number(slower_value)} ms, "
            f"reduced by {fmt_number(delta)} ms ({fmt_number(ratio)}%)"
        )

    increase = abs(delta)
    ratio = increase / slower_value * 100
    return (
        f"Mean TPOT = {fmt_number(faster_value)} ms vs {fmt_number(slower_value)} ms, "
        f"increased by {fmt_number(increase)} ms ({fmt_number(ratio)}% slower)"
    )


def build_summary_rows(
    baseline: BenchmarkRecord, records: list[BenchmarkRecord]
) -> list[str]:
    baseline_tps = float(baseline.payload.get("tokens_per_second_mean") or 0)
    baseline_tpot = float(baseline.payload.get("time_per_output_token_mean") or 0)
    records_by_group: dict[str, list[BenchmarkRecord]] = {}

    for record in records:
        records_by_group.setdefault(record.group_name, []).append(record)

    best_by_group: dict[str, BenchmarkRecord] = {}

    for group_name, group_records in records_by_group.items():
        # Skip the engine baseline comparison group from the optimization summary.
        if any(record.path == baseline.path for record in group_records):
            continue

        best_record: BenchmarkRecord | None = None
        best_tps = float("-inf")
        for record in group_records:
            record_tps = float(record.payload.get("tokens_per_second_mean") or 0)
            if record_tps > best_tps:
                best_tps = record_tps
                best_record = record
        if best_record is not None:
            best_by_group[group_name] = best_record

    rows = [
        "| Benchmark Case | Group | Optimized | Baseline |",
        "|---|---|---|---|",
    ]
    for record in best_by_group.values():
        benchmark_case = record.profile
        if record.request_rate is not None:
            benchmark_case = f"{benchmark_case} (r={record.request_rate})"

        candidate_tps = float(record.payload.get("tokens_per_second_mean") or 0)
        candidate_tpot = float(record.payload.get("time_per_output_token_mean") or 0)
        rows.append(
            "| "
            + f"{benchmark_case} | {record.group_name} | "
            + f"Total TPS: {candidate_tps:.2f} {throughput_annotation(baseline_tps, candidate_tps, record)}"
            + f"<br>Mean TPOT(ms): {fmt_number(candidate_tpot)} | "
            + f"Total TPS: {baseline_tps:.2f}<br>Mean TPOT(ms): {fmt_number(baseline_tpot)} |"
        )
    return rows


def build_optimization_option_rows(
    baseline: BenchmarkRecord, records: list[BenchmarkRecord]
) -> list[str]:
    baseline_tps = float(baseline.payload.get("tokens_per_second_mean") or 0)
    baseline_tpot = float(baseline.payload.get("time_per_output_token_mean") or 0)
    records_by_group: dict[str, list[BenchmarkRecord]] = {}

    for record in records:
        records_by_group.setdefault(record.group_name, []).append(record)

    rows: list[str] = []
    for group_name, group_records in records_by_group.items():
        best_record: BenchmarkRecord | None = None
        best_tps = float("-inf")
        for record in group_records:
            record_tps = float(record.payload.get("tokens_per_second_mean") or 0)
            if record_tps > best_tps:
                best_tps = record_tps
                best_record = record

        if best_record is None:
            continue

        candidate_tps = float(best_record.payload.get("tokens_per_second_mean") or 0)
        candidate_tpot = float(
            best_record.payload.get("time_per_output_token_mean") or 0
        )
        rows.append(
            "| "
            + f"{group_name} | "
            + f"Total TPS: {candidate_tps:.2f} {throughput_annotation(baseline_tps, candidate_tps, best_record)}"
            + f"<br>Mean TPOT(ms): {fmt_number(candidate_tpot)} | "
            + f"Total TPS: {baseline_tps:.2f}<br>Mean TPOT(ms): {fmt_number(baseline_tpot)} |"
        )
    return rows


def build_profile_comparison_rows(
    other_grouped_records: list[tuple[str, list[BenchmarkRecord]]],
) -> list[str]:
    rows = [
        "| Benchmark Case | baseline (vLLM without any optimizations) | Optimized |",
        "|----------|-------------------------------------------|-----------|",
    ]
    for profile_name, records in other_grouped_records:
        if len(records) < 2:
            continue
        baseline, optimized = records[0], records[1]
        baseline_tps = float(baseline.payload.get("tokens_per_second_mean") or 0)
        baseline_tpot = float(baseline.payload.get("time_per_output_token_mean") or 0)
        optimized_tps = float(optimized.payload.get("tokens_per_second_mean") or 0)
        optimized_tpot = float(optimized.payload.get("time_per_output_token_mean") or 0)

        rows.append(
            "| "
            + f"**{profile_name}** | "
            + f"Total TPS: {baseline_tps:.2f}<br>Mean TPOT(ms): {fmt_number(baseline_tpot)} | "
            + f"Total TPS: {optimized_tps:.2f} {throughput_annotation(baseline_tps, optimized_tps, optimized)}"
            + f"<br>Mean TPOT(ms): {fmt_number(optimized_tpot)} |"
        )
    return rows


def record_display_name(record: BenchmarkRecord) -> str:
    return record.custom_title or record.name


def render_group_speed_summary(records: list[BenchmarkRecord]) -> str:
    if len(records) < 2:
        return ""

    ranked_records = sorted(
        records,
        key=lambda record: float(record.payload.get("tokens_per_second_mean") or 0),
    )
    slowest = ranked_records[0]
    fastest = ranked_records[-1]

    slowest_tps = float(slowest.payload.get("tokens_per_second_mean") or 0)
    fastest_tps = float(fastest.payload.get("tokens_per_second_mean") or 0)
    slowest_tpot = float(slowest.payload.get("time_per_output_token_mean") or 0)
    fastest_tpot = float(fastest.payload.get("time_per_output_token_mean") or 0)

    if slowest_tps <= 0 or fastest_tps <= 0:
        return ""

    tps_gap = fastest_tps - slowest_tps
    tps_ratio = tps_gap / slowest_tps * 100

    summary = (
        f"- Summary: `{record_display_name(fastest)}` Total TPS = {fmt_number(fastest_tps)}, "
        f"`{record_display_name(slowest)}` Total TPS = {fmt_number(slowest_tps)}. "
        f"`{record_display_name(fastest)}` is faster by {fmt_number(tps_gap)} tok/s "
        f"({fmt_number(tps_ratio)}%)"
    )
    if slowest_tpot > 0 and fastest_tpot > 0:
        summary += "; " + format_tpot_delta_summary(fastest_tpot, slowest_tpot) + "."
    else:
        summary += "."
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


def generate_throughput_markdown(
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
    conclusion_rows = build_profile_comparison_rows(other_grouped_records)
    optimization_rows = build_optimization_option_rows(baseline, summary_records)
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
        render_result_image("Throughput Optimization Result", image_name),
        "",
        *render_recommended_configuration(
            "throughput", optimized_record, declared_models=declared_models
        ),
        "Comparison of benchmark results before and after optimization:",
        "",
        *conclusion_rows,
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
        group_summary = render_group_speed_summary(records)
        if group_summary:
            sections.extend([group_summary, ""])

    sections.extend(
        [
            "### Summary of Optimization Options",
            "",
            "| Benchmark Cases | Optimized | Baseline |",
            "| --------------------------------------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------ |",
            *optimization_rows,
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
    args = build_arg_parser("throughput").parse_args()
    baseline_record, grouped_records, other_grouped_records = load_grouped_records(
        args.baseline_file, args.group, args.other_group
    )
    optimized_record = load_record(args.optimized_file) if args.optimized_file else None
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        generate_throughput_markdown(
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
