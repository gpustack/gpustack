"""
Compare benchmark results between a base model and one or more LoRA adapters.

In vLLM, the base model is served at its original model name, while mounted
LoRA adapters are addressed by the adapter names passed to --lora-modules.

Two benchmark modes are supported:
  sequential  — models are benchmarked one at a time (default).
                Use this to isolate per-model performance without interference.
  concurrent  — all models are benchmarked with a single shared worker pool.
                Requests from all models are interleaved in round-robin order so
                every model faces the same server competition from the very first
                request.  Each model gets `concurrency` (-c) simultaneous in-flight
                requests, for a total of concurrency * n_models connections.

Usage examples:
    # Sequential mode (default)
    python benchmark_lora_compare.py \
        --server-url http://<host>:<port> \
        --base-model Qwen/Qwen3-0.6B \
        --lora-models qwen3-lora-medical qwen3-lora-legal \
        --api-key <key> \
        -n 50 -c 5

    # Concurrent mode — all models run at the same time
    python benchmark_lora_compare.py \
        --server-url http://<host>:<port> \
        --base-model Qwen/Qwen3-0.6B \
        --lora-models qwen3-lora-medical qwen3-lora-legal \
        --api-key <key> \
        -n 50 -c 5 \
        --mode concurrent
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from typing import List

import aiohttp
from aiohttp import ClientSession
from httpx_aiohttp import AiohttpTransport
from openai import AsyncOpenAI, DefaultAsyncHttpxClient
from tqdm import tqdm

from benchmark_serving import (
    BenchmarkResults,
    calculate_results,
    main as run_benchmark,
    make_chat_completion_request,
    output_benchmark_results_pretty,
    preflight_check,
    set_headers,
)

LABEL_W = 30
COL_W = 22
DELTA_W = 18


def fmt_val(val, fmt=".2f", unit="") -> str:
    if isinstance(val, float):
        return f"{val:{fmt}}{unit}"
    return f"{val}{unit}"


def fmt_delta(base_val, cmp_val, higher_is_better: bool = True) -> str:
    if base_val is None or base_val == 0:
        return "N/A"
    delta_pct = (cmp_val - base_val) / base_val * 100
    sign = "+" if delta_pct >= 0 else ""
    arrow = "↑" if delta_pct >= 0 else "↓"
    good = (delta_pct >= 0) == higher_is_better
    color = "\033[92m" if good else "\033[91m"
    reset = "\033[0m"
    return f"{color}{sign}{delta_pct:.1f}%{arrow}{reset}"


def print_comparison(
    base: BenchmarkResults,
    lora_results: List[BenchmarkResults],
    mode: str = "sequential",
):
    n_lora = len(lora_results)
    # Total line width: label + base col + n_lora * (value col + delta col)
    total_w = LABEL_W + COL_W + n_lora * (COL_W + DELTA_W)
    sep = "=" * total_w
    thin = "-" * total_w

    mode_tag = (
        "Sequential"
        if mode == "sequential"
        else "Concurrent (all models ran simultaneously)"
    )

    def header_row():
        h = f"  {'Metric':<{LABEL_W}} {'Base':<{COL_W}}"
        for r in lora_results:
            # Truncate long names for header display
            name = (
                r.model if len(r.model) <= COL_W - 1 else r.model[: COL_W - 4] + "..."
            )
            h += f" {name:<{COL_W - 1}} {'Δ vs Base':<{DELTA_W}}"
        return h

    def data_row(label, base_val, lora_vals, fmt=".2f", higher_is_better=True, unit=""):
        line = f"  {label:<{LABEL_W}} {fmt_val(base_val, fmt, unit):<{COL_W}}"
        for lv in lora_vals:
            val_str = fmt_val(lv, fmt, unit)
            delta_str = fmt_delta(
                base_val if isinstance(base_val, (int, float)) else None,
                lv if isinstance(lv, (int, float)) else None,
                higher_is_better=higher_is_better,
            )
            line += f" {val_str:<{COL_W - 1}} {delta_str:<{DELTA_W}}"
        print(line)

    def section(title):
        print(f"  {title}")

    print(sep)
    print(
        f"  LoRA vs Base Model Benchmark Comparison  ({n_lora} LoRA adapter(s))  [{mode_tag}]"
    )
    print(sep)
    print(header_row())
    print(thin)

    base_output_tps = (
        base.total_completion_tokens / base.total_time if base.total_time > 0 else 0.0
    )
    base_total_tps = base.total_tokens / base.total_time if base.total_time > 0 else 0.0

    section("[Throughput]")
    data_row(
        "Total tokens",
        base.total_tokens,
        [r.total_tokens for r in lora_results],
        fmt="d",
    )
    data_row(
        "Total input tokens",
        base.total_prompt_tokens,
        [r.total_prompt_tokens for r in lora_results],
        fmt="d",
    )
    data_row(
        "Total output tokens",
        base.total_completion_tokens,
        [r.total_completion_tokens for r in lora_results],
        fmt="d",
    )
    data_row(
        "Output tok/s",
        base_output_tps,
        [r.total_completion_tokens / r.total_time for r in lora_results],
        unit=" tok/s",
    )
    data_row(
        "Total tok/s",
        base_total_tps,
        [r.total_tokens / r.total_time for r in lora_results],
        unit=" tok/s",
    )
    data_row(
        "Requests/s",
        base.requests_per_second,
        [r.requests_per_second for r in lora_results],
        unit=" req/s",
    )

    section("[Time to First Token (ms)]  — lower is better")
    data_row(
        "  Average TTFT",
        base.time_to_first_token.average,
        [r.time_to_first_token.average for r in lora_results],
        unit=" ms",
        higher_is_better=False,
    )
    data_row(
        "  P50 TTFT",
        base.time_to_first_token.p50,
        [r.time_to_first_token.p50 for r in lora_results],
        unit=" ms",
        higher_is_better=False,
    )
    data_row(
        "  P95 TTFT",
        base.time_to_first_token.p95,
        [r.time_to_first_token.p95 for r in lora_results],
        unit=" ms",
        higher_is_better=False,
    )
    data_row(
        "  P99 TTFT",
        base.time_to_first_token.p99,
        [r.time_to_first_token.p99 for r in lora_results],
        unit=" ms",
        higher_is_better=False,
    )

    section("[Per-request Output Tokens/s]  — higher is better")
    data_row(
        "  Average TPS",
        base.completion_tokens_per_second.average,
        [r.completion_tokens_per_second.average for r in lora_results],
        unit=" tok/s",
    )
    data_row(
        "  P50 TPS",
        base.completion_tokens_per_second.p50,
        [r.completion_tokens_per_second.p50 for r in lora_results],
        unit=" tok/s",
    )
    data_row(
        "  P95 TPS",
        base.completion_tokens_per_second.p95,
        [r.completion_tokens_per_second.p95 for r in lora_results],
        unit=" tok/s",
    )
    data_row(
        "  P99 TPS",
        base.completion_tokens_per_second.p99,
        [r.completion_tokens_per_second.p99 for r in lora_results],
        unit=" tok/s",
    )

    section("[Request Latency (s)]  — lower is better")
    data_row(
        "  Average latency",
        base.latency.average,
        [r.latency.average for r in lora_results],
        unit=" s",
        higher_is_better=False,
    )
    data_row(
        "  P50 latency",
        base.latency.p50,
        [r.latency.p50 for r in lora_results],
        unit=" s",
        higher_is_better=False,
    )
    data_row(
        "  P95 latency",
        base.latency.p95,
        [r.latency.p95 for r in lora_results],
        unit=" s",
        higher_is_better=False,
    )
    data_row(
        "  P99 latency",
        base.latency.p99,
        [r.latency.p99 for r in lora_results],
        unit=" s",
        higher_is_better=False,
    )

    print(sep)
    print(
        f"  Base : {base.model}  ({base.successful_requests}/{base.total_requests} ok)"
    )
    for i, r in enumerate(lora_results, 1):
        print(f"  LoRA {i}: {r.model}  ({r.successful_requests}/{r.total_requests} ok)")
    print(sep)


async def run_sequential(
    shared: dict, base_model: str, lora_models: List[str]
) -> tuple[BenchmarkResults, List[BenchmarkResults]]:
    """Run benchmarks one model at a time."""
    total = 1 + len(lora_models)

    print(f"\n[1/{total}] Benchmarking BASE model: {base_model}")
    base_results = await run_benchmark(model=base_model, **shared)

    lora_results = []
    for i, lora_model in enumerate(lora_models, start=2):
        print(f"\n[{i}/{total}] Benchmarking LoRA model: {lora_model}")
        result = await run_benchmark(model=lora_model, **shared)
        lora_results.append(result)

    return base_results, lora_results


async def run_concurrent(
    shared: dict, base_model: str, lora_models: List[str]
) -> tuple[BenchmarkResults, List[BenchmarkResults]]:
    """Run all models through a single shared worker pool.

    Requests are enqueued in round-robin order (model0, model1, model2, model0,
    …) so every model faces identical competition from the very first request.
    Total concurrency = concurrency_per_model * n_models, matching the load
    each model would receive in sequential mode but applied simultaneously.
    """
    all_models = [base_model] + lora_models
    n_models = len(all_models)
    num_requests = shared["num_requests"]
    concurrency_per_model = shared["concurrency"]
    total_concurrency = concurrency_per_model * n_models
    request_timeout = shared["request_timeout"]
    max_completion_tokens = shared["max_completion_tokens"]
    ignore_eos = shared["ignore_eos"]
    prompt_multiplier = shared["prompt_multiplier"]

    MIN_AIOHTTP_CONNECTIONS = 2000

    connector = aiohttp.TCPConnector(
        limit=max(MIN_AIOHTTP_CONNECTIONS, total_concurrency * 4),
        force_close=True,
    )
    async with ClientSession(connector=connector, trust_env=True) as aiohttp_session:
        if shared.get("headers"):
            set_headers(aiohttp_session, shared["headers"])
        transport = AiohttpTransport(client=aiohttp_session)
        httpx_client = DefaultAsyncHttpxClient(
            transport=transport, timeout=request_timeout
        )
        client = AsyncOpenAI(
            base_url=f"{shared['server_url']}/v1",
            api_key=shared["api_key"],
            http_client=httpx_client,
            max_retries=0,
        )

        print(f"\n[concurrent] Pre-flight check for {n_models} models...")
        for model in all_models:
            ok = await preflight_check(client, model)
            if not ok:
                raise Exception(f"Preflight check failed for model: {model}")
            print(f"  OK {model}")

        # Round-robin queue: (model0,req0),(model1,req0),(model2,req0),(model0,req1)...
        # This guarantees every model gets equal exposure to server competition.
        queue: asyncio.Queue = asyncio.Queue()
        for i in range(num_requests):
            for model in all_models:
                await queue.put(model)
        for _ in range(total_concurrency):
            await queue.put(None)  # sentinel per worker

        results_per_model: dict[str, list] = {m: [] for m in all_models}
        total_jobs = num_requests * n_models

        pbar = tqdm(
            total=total_jobs,
            desc=f"Running concurrent benchmark ({n_models} models × {num_requests} req)",
            unit="request",
            dynamic_ncols=True,
        )

        async def worker():
            while True:
                model = await queue.get()
                if model is None:
                    queue.task_done()
                    break
                result = await make_chat_completion_request(
                    client,
                    model,
                    max_completion_tokens,
                    ignore_eos,
                    request_timeout,
                    prompt_multiplier,
                )
                if result:
                    results_per_model[model].append(result)
                else:
                    # Keep failed requests counted for accurate success_rate
                    results_per_model[model].append(None)
                queue.task_done()
                pbar.update(1)

        worker_tasks = [asyncio.create_task(worker()) for _ in range(total_concurrency)]
        wall_start = time.time()
        await queue.join()
        await asyncio.gather(*worker_tasks)
        wall_elapsed = time.time() - wall_start
        pbar.close()

        # Filter out None (failed) results before passing to calculate_results
        all_results = []
        for model in all_models:
            valid = [r for r in results_per_model[model] if r is not None]
            r = calculate_results(
                model,
                concurrency_per_model,
                request_timeout,
                max_completion_tokens,
                wall_elapsed,
                num_requests,
                valid,
            )
            all_results.append(r)

        print(
            f"\n[concurrent] Wall-clock time: {wall_elapsed:.1f}s  |  "
            f"Total concurrency: {total_concurrency} "
            f"({concurrency_per_model}/model × {n_models} models)"
        )
        return all_results[0], all_results[1:]


async def compare(args):
    shared = dict(
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        request_timeout=args.request_timeout,
        max_completion_tokens=args.max_completion_tokens,
        ignore_eos=args.ignore_eos,
        server_url=args.server_url,
        api_key=args.api_key,
        headers=args.headers,
        embeddings=False,
        prompt_multiplier=args.prompt_multiplier,
    )

    if args.mode == "concurrent":
        base_results, lora_results = await run_concurrent(
            shared, args.base_model, args.lora_models
        )
    else:
        base_results, lora_results = await run_sequential(
            shared, args.base_model, args.lora_models
        )

    print()
    if args.verbose:
        print("===== Base model full results =====")
        output_benchmark_results_pretty(base_results)
        for i, (name, result) in enumerate(zip(args.lora_models, lora_results), 1):
            print(f"\n===== LoRA {i}: {name} full results =====")
            output_benchmark_results_pretty(result)
        print()

    print_comparison(base_results, lora_results, mode=args.mode)

    if args.result_file:
        output = {
            "mode": args.mode,
            "base": asdict(base_results),
            "lora": [asdict(r) for r in lora_results],
        }
        with open(args.result_file, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nJSON results saved to {args.result_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare benchmark results: base model vs one or more LoRA adapters"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        required=True,
        help="Model name for the base model (no LoRA), as registered in vLLM",
    )
    parser.add_argument(
        "--lora-models",
        type=str,
        nargs="+",
        required=True,
        metavar="LORA_MODEL",
        help=(
            "One or more LoRA adapter names as set via --lora-modules in vLLM. "
            "Example: --lora-models lora-medical lora-legal lora-code"
        ),
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://127.0.0.1:8000",
        help="URL of the vLLM server (default: http://127.0.0.1:8000)",
    )
    parser.add_argument("--api-key", type=str, default="fake", help="API key")
    parser.add_argument(
        "-n",
        "--num-requests",
        type=int,
        default=50,
        help="Number of requests per model (default: 50)",
    )
    parser.add_argument(
        "-c",
        "--concurrency",
        type=int,
        default=5,
        help="Concurrent requests (default: 5)",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=300,
        help="Per-request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=512,
        help="Max output tokens per request (default: 512)",
    )
    parser.add_argument(
        "--prompt-multiplier",
        type=int,
        default=1,
        help="Repeat the randomly selected prompt N times to create longer inputs",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Force model to generate up to max_completion_tokens (disables EOS stop)",
    )
    parser.add_argument(
        "-H",
        "--header",
        action="append",
        dest="headers",
        help="Custom HTTP header Key:Value. May be specified multiple times.",
    )
    parser.add_argument(
        "--result-file",
        type=str,
        help="Save full JSON results to this file",
    )
    parser.add_argument(
        "--mode",
        choices=["sequential", "concurrent"],
        default="sequential",
        help=(
            "Benchmark execution mode. "
            "'sequential' (default): benchmark each model one at a time for isolated results. "
            "'concurrent': benchmark all models simultaneously to measure performance under "
            "shared GPU pressure."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also print individual full benchmark reports before the comparison table",
    )

    args = parser.parse_args()

    try:
        asyncio.run(compare(args))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
