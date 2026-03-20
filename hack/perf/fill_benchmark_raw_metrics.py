#!/usr/bin/env python3
"""
Redump complete benchmark detail JSON from GPUStack into local benchmark result files.

Examples:
  python3 hack/perf/fill_benchmark_raw_metrics.py \
    --dir $RESULTS_DIR \
    --gpustack-url $GPUSTACK_URL \
    --gpustack-token $GPUSTACK_TOKEN

  python3 hack/perf/fill_benchmark_raw_metrics.py \
    --dir $RESULTS_DIR \
    --gpustack-url $GPUSTACK_URL \
    --gpustack-token $GPUSTACK_TOKEN \
    --force
"""

from __future__ import annotations

import argparse
import json
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict


METRIC_FIELDS = [
    "raw_metrics",
    "requests_per_second_mean",
    "request_latency_mean",
    "time_per_output_token_mean",
    "inter_token_latency_mean",
    "time_to_first_token_mean",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan local benchmark result JSON files and redump the full benchmark "
            "detail from the GPUStack server using each file's benchmark id."
        )
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing benchmark result JSON files. The scan is recursive.",
    )
    parser.add_argument("--gpustack-url", required=True, help="GPUStack base URL.")
    parser.add_argument(
        "--gpustack-token", required=True, help="GPUStack API token for authentication."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redump files even if local metrics are already populated.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds for each benchmark detail request.",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification. Use only for self-signed servers.",
    )
    return parser


def create_ssl_context(insecure: bool) -> ssl.SSLContext:
    context = ssl.create_default_context()
    if insecure:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    return context


def fetch_benchmark_detail(
    base_url: str,
    token: str,
    benchmark_id: int,
    timeout: int,
    ssl_context: ssl.SSLContext,
) -> Dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v2/benchmarks/{benchmark_id}"
    request = urllib.request.Request(
        url=url,
        headers={
            "Accept": "application/json, text/plain, */*",
            "Authorization": f"Bearer {token}",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(
            request, timeout=timeout, context=ssl_context
        ) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"GET {url} failed: {exc.code} {details}") from exc


def should_redump(payload: Dict[str, Any], force: bool) -> bool:
    if force:
        return True
    return any(payload.get(field) is None for field in METRIC_FIELDS)


def process_file(
    file_path: Path,
    *,
    base_url: str,
    token: str,
    timeout: int,
    ssl_context: ssl.SSLContext,
    force: bool,
) -> str:
    with file_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    benchmark_id = payload.get("id")
    if not isinstance(benchmark_id, int):
        return "skip: missing integer id"

    if not should_redump(payload, force):
        return "skip: metrics already populated"

    remote_payload = fetch_benchmark_detail(
        base_url=base_url,
        token=token,
        benchmark_id=benchmark_id,
        timeout=timeout,
        ssl_context=ssl_context,
    )
    if remote_payload.get("id") != benchmark_id:
        return "skip: server returned mismatched benchmark id"

    if payload == remote_payload:
        return "skip: local file already matches server"

    with file_path.open("w", encoding="utf-8") as file:
        json.dump(remote_payload, file, indent=2, ensure_ascii=False)
        file.write("\n")
    return "updated: full benchmark json"


def main() -> None:
    args = build_parser().parse_args()
    directory = Path(args.dir).resolve()
    if not directory.is_dir():
        raise SystemExit(f"Directory not found: {directory}")

    ssl_context = create_ssl_context(args.insecure)
    json_files = sorted(directory.rglob("*.json"))
    if not json_files:
        raise SystemExit(f"No JSON files found under: {directory}")

    updated = 0
    skipped = 0
    failed = 0

    for file_path in json_files:
        try:
            result = process_file(
                file_path,
                base_url=args.gpustack_url,
                token=args.gpustack_token,
                timeout=args.timeout,
                ssl_context=ssl_context,
                force=args.force,
            )
            print(f"{file_path}: {result}")
            if result.startswith("updated:"):
                updated += 1
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            print(f"{file_path}: error: {exc}")

    print(
        f"Completed. total={len(json_files)} updated={updated} skipped={skipped} failed={failed}"
    )


if __name__ == "__main__":
    main()
