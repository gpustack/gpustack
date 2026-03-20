from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class GroupFileSpec:
    path: Path
    title: str | None = None


@dataclass(frozen=True)
class GroupSpec:
    name: str
    files: list[GroupFileSpec]


@dataclass(frozen=True)
class BenchmarkRecord:
    group_name: str
    path: Path
    payload: dict[str, Any]
    custom_title: str | None = None

    @property
    def name(self) -> str:
        return str(self.payload.get("name") or self.path.stem)

    @property
    def profile(self) -> str:
        return str(self.payload.get("profile") or "N/A")

    @property
    def request_rate(self) -> Any:
        return self.payload.get("request_rate")


def request_success_rate(record: BenchmarkRecord) -> float | None:
    successful = record.payload.get("request_successful")
    total = record.payload.get("request_total")
    if total is None:
        total = record.payload.get("total_requests")
    if successful is None or total in (None, 0):
        return None
    try:
        return float(successful) / float(total)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def build_arg_parser(metric_name: str) -> argparse.ArgumentParser:
    throughput_example = """\
Examples:
  python3 hack/perf/generate_throughput_optimization_doc.py \\
    --title "Optimizing Qwen3.5-35B-A3B Throughput" \\
    --model Qwen/Qwen3.5-35B-A3B-FP8 \\
    --baseline-file .cache/plan/benchmark/output/baseline.json \\
    --optimized-file .cache/plan/benchmark/output/optimized.json \\
    --group "Baseline of the Inference Engine=.cache/plan/benchmark/output/baseline.json,.cache/plan/benchmark/output/sglang.json" \\
    --group "Quantization=.cache/plan/benchmark/output/fp8.json" \\
    --group "Performance Mode=.cache/plan/benchmark/output/performance_mode.json" \\
    --other-group "ShareGPT=.cache/plan/benchmark/output/sharegpt_baseline.json,.cache/plan/benchmark/output/sharegpt_optimized.json" \\
    --other-group "Long Context=.cache/plan/benchmark/output/long_context_baseline.json,.cache/plan/benchmark/output/long_context_optimized.json" \\
    --output .cache/plan/benchmark/output/throughput_doc.md
"""
    default_example = f"""\
Examples:
  python3 hack/perf/generate_{metric_name}_optimization_doc.py \\
    --title "Optimizing Qwen3.5-35B-A3B {metric_name.title()}" \\
    --model Qwen/Qwen3.5-35B-A3B-FP8 \\
    --baseline-file .cache/plan/benchmark/output/baseline.json \\
    --optimized-file .cache/plan/benchmark/output/optimized.json \\
    --group "Baseline=.cache/plan/benchmark/output/baseline.json" \\
    --group "DP Attention=.cache/plan/benchmark/output/opt_r1.json,.cache/plan/benchmark/output/opt_r4.json" \\
    --output .cache/plan/benchmark/output/{metric_name}_doc.md
"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            f"Generate a markdown {metric_name} optimization document from benchmark JSON files."
        ),
        epilog=throughput_example if metric_name == "throughput" else default_example,
    )
    parser.add_argument("--title", required=True, help="Document title.")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Explicit model name to render in the Model section. Can be repeated. "
            "If omitted, the script falls back to auto-detecting models from benchmark results."
        ),
    )
    parser.add_argument(
        "--optimized-file",
        help=(
            "Path to the benchmark result JSON for the final recommended optimized "
            "configuration. If provided, the document will include a Serving Command section."
        ),
    )
    parser.add_argument(
        "--baseline-file",
        required=True,
        help="Path to the benchmark result JSON used as the baseline.",
    )
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        help=(
            "Benchmark group in the format GroupName=file1,file2,... Can be repeated. "
            "To override a per-file heading, use GroupName=Title::file1,Other Title::file2."
        ),
    )
    parser.add_argument(
        "--other-group",
        action="append",
        default=[],
        help=(
            "Benchmark group in the format GroupName=file1,file2,... that should be "
            "placed under Other Benchmark Cases. To override a per-file heading, use "
            "GroupName=Title::file1,Other Title::file2. For latency and throughput docs, "
            "pass exactly two files in the order baseline,optimized so the same "
            "pair can also be used to build the comparison table in Conclusion."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output markdown file path.",
    )
    parser.add_argument(
        "--image-name",
        default="replace-this-image.png",
        help=(
            "Image file name placed under ../../assets/performance-lab/ in the "
            "generated document. Defaults to replace-this-image.png."
        ),
    )
    return parser


def parse_group_file_spec(raw_item: str) -> GroupFileSpec:
    item = raw_item.strip()
    if not item:
        raise ValueError("Empty file entry in group specification.")
    if "::" not in item:
        return GroupFileSpec(path=Path(item))

    title_part, path_part = item.split("::", 1)
    title = title_part.strip()
    path_text = path_part.strip()
    if not title or not path_text:
        raise ValueError(
            f"Invalid group file entry '{raw_item}'. Expected file or Title::file."
        )
    return GroupFileSpec(path=Path(path_text), title=title)


def parse_group_spec(raw_group: str) -> GroupSpec:
    if "=" not in raw_group:
        raise ValueError(
            f"Invalid --group value '{raw_group}'. Expected GroupName=file1,file2,..."
        )
    name, files_part = raw_group.split("=", 1)
    files = [
        parse_group_file_spec(item) for item in files_part.split(",") if item.strip()
    ]
    if not name.strip() or not files:
        raise ValueError(
            f"Invalid --group value '{raw_group}'. Expected GroupName=file1,file2,..."
        )
    return GroupSpec(name=name.strip(), files=files)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def first_benchmark_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    benchmarks = payload.get("raw_metrics", {}).get("benchmarks", [])
    if not benchmarks:
        return {}
    return benchmarks[0]


def first_instance(payload: dict[str, Any]) -> dict[str, Any]:
    instances = payload.get("snapshot", {}).get("instances", {})
    if isinstance(instances, dict) and instances:
        return next(iter(instances.values()))
    if isinstance(instances, list) and instances:
        return instances[0]
    return {}


def load_record(
    path: str | Path, group_name: str = "", custom_title: str | None = None
) -> BenchmarkRecord:
    resolved = Path(path).resolve()
    payload = load_json(resolved)
    return BenchmarkRecord(
        group_name=group_name,
        path=resolved,
        payload=payload,
        custom_title=custom_title,
    )


def fmt_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def normalize_engine_version(value: Any) -> str | None:
    if value is None:
        return None
    version = str(value).strip()
    if not version:
        return None
    if not version.lower().startswith("v"):
        return f"v{version}"
    return version


def metric_block(
    metrics: dict[str, Any], metric_key: str, category: str = "successful"
) -> dict[str, Any]:
    return metrics.get("metrics", {}).get(metric_key, {}).get(category, {})


def metric_percentile(metric: dict[str, Any], percentile: str) -> Any:
    return metric.get("percentiles", {}).get(percentile)


def render_benchmark_result_block(payload: dict[str, Any]) -> str:
    benchmark = first_benchmark_metrics(payload)
    metrics = benchmark.get("metrics", {})
    scheduler_metrics = benchmark.get("scheduler_metrics", {})

    request_totals = metrics.get("request_totals", {})
    request_latency = metric_block(benchmark, "request_latency")
    ttft = metric_block(benchmark, "time_to_first_token_ms")
    tpot = metric_block(benchmark, "time_per_output_token_ms")
    itl = metric_block(benchmark, "inter_token_latency_ms")
    requests_per_second = metric_block(benchmark, "requests_per_second")
    request_concurrency = metric_block(benchmark, "request_concurrency")
    output_tps = metric_block(benchmark, "output_tokens_per_second")
    total_tps = metric_block(benchmark, "tokens_per_second")
    prompt_token_count = metric_block(benchmark, "prompt_token_count")
    output_token_count = metric_block(benchmark, "output_token_count")

    duration = benchmark.get("duration")
    if duration is None:
        start = scheduler_metrics.get("measure_start_time")
        end = scheduler_metrics.get("measure_end_time")
        if start is not None and end is not None:
            duration = end - start
    successful_requests = request_totals.get(
        "successful", payload.get("request_successful", 0)
    )
    total_input_tokens = None
    if prompt_token_count.get("mean") is not None:
        total_input_tokens = prompt_token_count.get("mean") * successful_requests
    total_generated_tokens = None
    if output_token_count.get("mean") is not None:
        total_generated_tokens = output_token_count.get("mean") * successful_requests

    lines = [
        "============ Serving Benchmark Result ============",
        f"Successful requests:                     {successful_requests}",
        f"Maximum request concurrency:             {fmt_number(payload.get('max_concurrency') or payload.get('request_concurrency_max'), 0)}",
        f"Benchmark duration (s):                  {fmt_number(duration)}",
        f"Total input tokens:                      {fmt_number(total_input_tokens, 0)}",
        f"Total generated tokens:                  {fmt_number(total_generated_tokens, 0)}",
        f"Request throughput (req/s):              {fmt_number(requests_per_second.get('mean'))}",
        f"Output token throughput (tok/s):         {fmt_number(payload.get('output_tokens_per_second_mean') or output_tps.get('mean'))}",
        f"Peak output token throughput (tok/s):    {fmt_number(output_tps.get('max'))}",
        f"Peak concurrent requests:                {fmt_number(payload.get('request_concurrency_max') or request_concurrency.get('max'))}",
        f"Total Token throughput (tok/s):          {fmt_number(payload.get('tokens_per_second_mean') or total_tps.get('mean'))}",
        "----------------------Latency---------------------",
        f"Mean Latency(s):                          {fmt_number(request_latency.get('mean'))}",
        f"Median Latency(s):                        {fmt_number(request_latency.get('median'))}",
        f"P95 Latency(s):                           {fmt_number(metric_percentile(request_latency, 'p95'))}",
        f"P99 Latency(s):                           {fmt_number(metric_percentile(request_latency, 'p99'))}",
        "---------------Time to First Token----------------",
        f"Mean TTFT (ms):                          {fmt_number(ttft.get('mean'))}",
        f"Median TTFT (ms):                        {fmt_number(ttft.get('median'))}",
        f"P95 TTFT (ms):                           {fmt_number(metric_percentile(ttft, 'p95'))}",
        f"P99 TTFT (ms):                           {fmt_number(metric_percentile(ttft, 'p99'))}",
        "-----Time per Output Token (excl. 1st token)------",
        f"Mean TPOT (ms):                          {fmt_number(tpot.get('mean'))}",
        f"Median TPOT (ms):                        {fmt_number(tpot.get('median'))}",
        f"P95 TPOT (ms):                           {fmt_number(metric_percentile(tpot, 'p95'))}",
        f"P99 TPOT (ms):                           {fmt_number(metric_percentile(tpot, 'p99'))}",
        "---------------Inter-token Latency----------------",
        f"Mean ITL (ms):                           {fmt_number(itl.get('mean'))}",
        f"Median ITL (ms):                         {fmt_number(itl.get('median'))}",
        f"P95 ITL (ms):                            {fmt_number(metric_percentile(itl, 'p95'))}",
        f"P99 ITL (ms):                            {fmt_number(metric_percentile(itl, 'p99'))}",
        "==================================================",
    ]
    formatted = "\n".join(lines)
    indented = textwrap.indent(f"```\n{formatted}\n```", "    ")
    return f'??? info "Benchmark result"\n{indented}'


def render_backend_parameters(instance: dict[str, Any]) -> str:
    params = instance.get("backend_parameters") or []
    if not params:
        return "N/A"
    formatted = "\n".join(str(item) for item in params)
    indented = textwrap.indent(f"```bash\n{formatted}\n```", "  ")
    return f"\n{indented}"


def render_record_section(
    record: BenchmarkRecord,
    heading_level: int = 4,
    include_heading: bool = True,
) -> str:
    payload = record.payload
    instance = first_instance(payload)
    request_rate = payload.get("request_rate")
    subtitle_bits = [record.profile]
    if request_rate is not None:
        subtitle_bits.append(f"request_rate={request_rate}")
    subtitle = ", ".join(subtitle_bits)
    lines: list[str] = []
    if include_heading:
        heading = "#" * heading_level
        heading_text = record.custom_title or f"{record.group_name}: {subtitle}"
        lines.extend(
            [
                f"{heading} {heading_text}",
                "",
            ]
        )

    lines.extend(
        [
            f"- Profile: `{payload.get('profile') or payload.get('dataset_name') or 'N/A'}`",
            f"- Backend Parameters:{render_backend_parameters(instance)}",
            "",
            render_benchmark_result_block(payload),
            "",
        ]
    )
    return "\n".join(lines)


def collect_experimental_setup(
    records: list[BenchmarkRecord],
    declared_models: list[str] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    models: list[str] = []
    hardware: list[str] = []
    engine_versions: list[str] = []

    if declared_models:
        for model_name in declared_models:
            model_text = str(model_name).strip()
            if model_text and model_text not in models:
                models.append(model_text)

    for record in records:
        payload = record.payload
        instance = first_instance(payload)

        if not models:
            model_name = payload.get("model_name")
            if model_name:
                model_text = str(model_name)
                if model_text not in models:
                    models.append(model_text)

        hardware_name = payload.get("gpu_summary")
        if hardware_name:
            hardware_text = str(hardware_name)
            if hardware_text not in hardware:
                hardware.append(hardware_text)

        engine_name = instance.get("backend") or payload.get("backend")
        engine_version = normalize_engine_version(
            instance.get("backend_version")
            or instance.get("api_detected_backend_version")
            or payload.get("backend_version")
        )
        engine_text = " ".join(
            part
            for part in [
                str(engine_name) if engine_name else None,
                str(engine_version) if engine_version else None,
            ]
            if part
        )
        if engine_text and engine_text not in engine_versions:
            engine_versions.append(engine_text)

    return models, hardware, engine_versions


def render_setup_list(title: str, values: list[str]) -> list[str]:
    lines = [f"### {title}", ""]
    if not values:
        lines.append("N/A")
        lines.append("")
        return lines
    if len(values) == 1:
        lines.append(values[0])
        lines.append("")
        return lines
    lines.extend(f"- {value}" for value in values)
    lines.append("")
    return lines


def render_standard_note_block() -> list[str]:
    return [
        "!!! note",
        "    1. Our benchmark tests do not cover all possible optimization combinations. For example, we select the inference engine that performs best under its default configuration as the starting point for further tuning. This pruning approach yields a local optimum, which may not be the global optimum.",
        "    2. There are other optimization methods that depend on specific user scenarios, including max batch size, schedule configuration, extended KV cache, CUDA graph, etc. The conclusions in this document can serve as a starting point for more targeted optimizations.",
        "    3. The tests are conducted on specific hardware and software setups. Advances in the inference engine may lead to new conclusions.",
        "    4. Although using quantization may impact accuracy. FP8 quantization can achieve less than 1% accuracy drop for most models. See the [evaluation results](https://github.com/Tencent/AngelSlim/blob/main/README_en.md#-benchmark) for more details. Therefore, it is highly recommended to use FP8 quantization for low-latency serving scenarios.",
        "    5. Speculative decoding can significantly reduce latency for low-concurrency requests. However, the acceleration effect may vary depending on the data distribution of different benchmark datasets and the choice of draft models. For example, the chosen draft model here is trained on English data, which may lead to suboptimal performance on other languages.",
        "",
        "If there are any missing points or updates reflecting new changes, please [let us know](https://github.com/gpustack/gpustack/issues/new/choose).",
        "",
    ]


def render_result_image(image_alt_text: str, image_name: str) -> str:
    image_file_name = Path(image_name).name.strip() or "replace-this-image.png"
    return f"![{image_alt_text}](../../assets/performance-lab/{image_file_name})"


def detect_serving_model(
    record: BenchmarkRecord, declared_models: list[str] | None = None
) -> str:
    if declared_models:
        for model_name in declared_models:
            model_text = str(model_name).strip()
            if model_text:
                return model_text
    payload_model_name = str(record.payload.get("model_name") or "").strip()
    if payload_model_name:
        return payload_model_name
    return record.path.stem


def render_serving_command(instance: dict[str, Any], model_name: str) -> str:
    backend = str(instance.get("backend") or "").strip().lower()
    params = [
        str(item).strip()
        for item in (instance.get("backend_parameters") or [])
        if str(item).strip()
    ]
    if backend == "sglang":
        command_lines = [
            "python3 -m sglang.launch_server \\",
            f"    --model {model_name} \\",
        ]
    else:
        command_lines = [f"vllm serve {model_name} \\"]

    for index, param in enumerate(params):
        suffix = " \\" if index < len(params) - 1 else ""
        command_lines.append(f"    {param}{suffix}")

    if not params:
        command_lines[-1] = command_lines[-1].removesuffix(" \\")

    indented = textwrap.indent("```bash\n" + "\n".join(command_lines) + "\n```", "    ")
    return '???+ tip "Serving Command"\n' + indented


def render_recommended_configuration(
    metric_name: str,
    optimized_record: BenchmarkRecord | None,
    declared_models: list[str] | None = None,
) -> list[str]:
    if optimized_record is None:
        return []

    instance = first_instance(optimized_record.payload)
    backend = str(instance.get("backend") or "").strip()
    if not backend:
        return []

    model_name = detect_serving_model(optimized_record, declared_models=declared_models)
    hardware_name = str(optimized_record.payload.get("gpu_summary") or "").strip()

    sentence = f"Recommended configuration for optimizing {metric_name}"
    if model_name:
        sentence += f" of {model_name}"
    if hardware_name:
        sentence += f" on {hardware_name}"
    sentence += ":"

    return [
        sentence,
        "",
        render_serving_command(instance, model_name),
        "",
    ]


def render_benchmark_method_intro() -> list[str]:
    return [
        "### Benchmark Method",
        "",
        "This project uses GPUStack's one-click benchmark capability for serving workloads. The benchmark tests in this document were executed with that workflow.",
        "",
        "GPUStack's benchmark implementation is built on top of [guidellm](https://github.com/vllm-project/guidellm) via the wrapper project [benchmark-runner](https://github.com/gpustack/benchmark-runner).",
        "",
        "GPUStack handles model deployment, benchmark job submission, and result collection for the benchmark configurations listed below.",
        "",
        "#### Benchmark Profiles",
        "",
    ]


def render_open_source_replacement(
    profiles: dict[str, dict[str, Any]],
) -> list[str]:
    dataset_names = {
        str(profile.get("dataset_name") or "").strip().lower()
        for profile in profiles.values()
        if profile
    }
    lines = [
        "### Open-Source Replacement",
        "",
        "If you do not use GPUStack, you can replace the GPUStack benchmark workflow with direct `guidellm benchmark` commands.",
        "",
    ]
    if "sharegpt" in dataset_names:
        lines.extend(
            [
                "For profiles with `dataset_name: ShareGPT`:",
                "",
                "```bash",
                "guidellm benchmark \\",
                "  --target ${target} \\",
                "  --profile constant \\",
                "  --rate ${request_rate} \\",
                "  --max-requests ${total_request} \\",
                "  --processor ${model_path} \\",
                "  --data ./ShareGPT_V3_unfiltered_cleaned_split.json",
                "```",
                "",
            ]
        )
    if "random" in dataset_names:
        lines.extend(
            [
                "For profiles with `dataset_name: Random`:",
                "",
                "```bash",
                "guidellm benchmark \\",
                "  --target ${target} \\",
                "  --profile constant \\",
                "  --rate ${request_rate} \\",
                "  --max-requests ${total_request} \\",
                "  --processor ${model_path} \\",
                '  --data "prompt_tokens=${dataset_input_tokens},output_tokens=${dataset_output_tokens}" \\',
                "  --random-seed ${dataset_seed:-42}",
                "```",
                "",
            ]
        )
    return lines


def collect_profiles_from_results(
    records: list[BenchmarkRecord],
) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for record in records:
        payload = record.payload
        profile_name = payload.get("profile")
        if not profile_name:
            continue
        text = str(profile_name)
        if text in profiles:
            continue

        profile: dict[str, Any] = {"name": text}
        if payload.get("description"):
            profile["description"] = payload["description"]
        if payload.get("dataset_name") is not None:
            profile["dataset_name"] = payload["dataset_name"]
        if payload.get("dataset_input_tokens") is not None:
            profile["dataset_input_tokens"] = payload["dataset_input_tokens"]
        if payload.get("dataset_output_tokens") is not None:
            profile["dataset_output_tokens"] = payload["dataset_output_tokens"]
        if payload.get("dataset_seed") is not None:
            profile["dataset_seed"] = payload["dataset_seed"]
        if payload.get("dataset_shared_prefix_tokens") is not None:
            profile["dataset_shared_prefix_tokens"] = payload[
                "dataset_shared_prefix_tokens"
            ]
        if payload.get("request_rate") is not None:
            profile["request_rate"] = payload["request_rate"]
        if payload.get("total_requests") is not None:
            profile["total_requests"] = payload["total_requests"]
        if payload.get("max_concurrency") is not None:
            profile["max_concurrency"] = payload["max_concurrency"]
        profiles[text] = profile
    return profiles


def render_profile_config(
    profile_name: str, profile: dict[str, Any] | None
) -> list[str]:
    lines = [f"##### {profile_name}", ""]
    if not profile:
        lines.append("Profile config not found.")
        lines.append("")
        return lines

    lines.append("```yaml")
    for key, value in profile.items():
        if key == "name":
            continue
        lines.extend(render_yaml_lines(key, value))
    lines.append("```")
    lines.append("")
    return lines


def render_yaml_lines(key: str, value: Any, indent: int = 0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, list):
        lines = [f"{prefix}{key}:"]
        for item in value:
            lines.append(f"{prefix}- {item}")
        return lines
    if isinstance(value, dict):
        lines = [f"{prefix}{key}:"]
        for child_key, child_value in value.items():
            lines.extend(render_yaml_lines(str(child_key), child_value, indent + 2))
        return lines
    if isinstance(value, str) and (":" in value or len(value) > 80):
        return [f"{prefix}{key}: >", f"{prefix}  {value}"]
    if value is None:
        return [f"{prefix}{key}: null"]
    return [f"{prefix}{key}: {value}"]


def load_grouped_records(
    baseline_file: str,
    raw_groups: list[str],
    raw_other_groups: list[str],
) -> tuple[
    BenchmarkRecord,
    list[tuple[str, list[BenchmarkRecord]]],
    list[tuple[str, list[BenchmarkRecord]]],
]:
    group_specs = [parse_group_spec(raw_group) for raw_group in raw_groups]
    other_group_specs = [parse_group_spec(raw_group) for raw_group in raw_other_groups]
    baseline_path = Path(baseline_file).resolve()

    grouped_records: list[tuple[str, list[BenchmarkRecord]]] = []
    other_grouped_records: list[tuple[str, list[BenchmarkRecord]]] = []
    baseline_record: BenchmarkRecord | None = None

    for group_spec in group_specs:
        records: list[BenchmarkRecord] = []
        for file_spec in group_spec.files:
            resolved = file_spec.path.resolve()
            payload = load_json(resolved)
            record = BenchmarkRecord(
                group_name=group_spec.name,
                path=resolved,
                payload=payload,
                custom_title=file_spec.title,
            )
            records.append(record)
            if resolved == baseline_path:
                baseline_record = record
        grouped_records.append((group_spec.name, records))

    for group_spec in other_group_specs:
        records: list[BenchmarkRecord] = []
        for file_spec in group_spec.files:
            resolved = file_spec.path.resolve()
            payload = load_json(resolved)
            record = BenchmarkRecord(
                group_name=group_spec.name,
                path=resolved,
                payload=payload,
                custom_title=file_spec.title,
            )
            records.append(record)
        other_grouped_records.append((group_spec.name, records))

    if baseline_record is None:
        raise SystemExit(
            "--baseline-file must be included in one of the --group entries"
        )

    return baseline_record, grouped_records, other_grouped_records


def generate_markdown(
    title: str,
    baseline: BenchmarkRecord,
    grouped_records: list[tuple[str, list[BenchmarkRecord]]],
    other_grouped_records: list[tuple[str, list[BenchmarkRecord]]],
    summary_rows_builder: Callable[[BenchmarkRecord, list[BenchmarkRecord]], list[str]],
    image_alt_text: str,
    image_name: str = "replace-this-image.png",
    declared_models: list[str] | None = None,
) -> str:
    summary_records = [record for _, records in grouped_records for record in records]
    all_records = summary_records + [
        record for _, records in other_grouped_records for record in records
    ]
    summary_rows = summary_rows_builder(baseline, summary_records)
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
        render_result_image(image_alt_text, image_name),
        "",
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

    sections.extend(render_open_source_replacement(profile_configs))

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

    if other_grouped_records:
        sections.extend(
            [
                "### Other Benchmark Cases",
                "",
            ]
        )
        for group_name, records in other_grouped_records:
            sections.append(f"#### {group_name}")
            sections.append("")
            for record in records:
                sections.append(render_record_section(record, heading_level=5))

    return "\n".join(sections).rstrip() + "\n"
