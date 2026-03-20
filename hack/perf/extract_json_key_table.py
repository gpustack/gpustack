#!/usr/bin/env python3
"""
Extract one or more JSON-path values from all JSON files under a directory and print
a markdown table.

Examples:
  python3 hack/perf/extract_json_key_table.py \
    --dir ./output \
    --path request_successful

  python3 hack/perf/extract_json_key_table.py \
    --dir ./output \
    --path request_successful \
    --path request_total

  python3 hack/perf/extract_json_key_table.py \
    --dir ./output \
    --path raw_metrics.benchmarks[0].duration

  python3 hack/perf/extract_json_key_table.py \
    --dir hack/perf/run-cases/high-throughput/qwen_3.5_35b_fp8/output \
    --path snapshot.instances.*.computed_resource_claim.is_unified_memory

  python3 hack/perf/extract_json_key_table.py \
    --dir ./output \
    --path request_successful \
    --sort-by request_successful \
    --sort-order desc \
    --output /tmp/request_successful.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan all JSON files under a directory and print a markdown table for "
            "the given JSON path."
        )
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="Directory containing JSON files. The scan is recursive.",
    )
    parser.add_argument(
        "--path",
        action="append",
        required=True,
        help=(
            "JSON path such as key.subkey or list[0].field. "
            "Use * to match all values under an object level, for example "
            "snapshot.instances.*.computed_resource_claim.is_unified_memory. "
            "Can be repeated to add multiple columns."
        ),
    )
    parser.add_argument(
        "--output",
        help="Optional output file path. If omitted, print to stdout.",
    )
    parser.add_argument(
        "--sort-by",
        help=(
            "Optional sort column. Use 'file_name' to sort by file name, or pass "
            "one of the values given to --path."
        ),
    )
    parser.add_argument(
        "--sort-order",
        choices=["asc", "desc"],
        default="asc",
        help="Sort order for --sort-by. Default: asc.",
    )
    return parser


def parse_json_path(path: str) -> list[str | int]:
    tokens: list[str | int] = []
    current = ""
    i = 0

    while i < len(path):
        char = path[i]
        if char == ".":
            if current:
                tokens.append(current)
                current = ""
            i += 1
            continue
        if char == "*":
            if current:
                tokens.append(current)
                current = ""
            tokens.append("*")
            i += 1
            continue
        if char == "[":
            if current:
                tokens.append(current)
                current = ""
            end = path.find("]", i)
            if end == -1:
                raise ValueError(f"Invalid JSON path: {path}")
            index_text = path[i + 1 : end].strip()
            if index_text == "*":
                tokens.append("*")
                i = end + 1
                continue
            if not index_text.isdigit():
                raise ValueError(f"Invalid list index in JSON path: {path}")
            tokens.append(int(index_text))
            i = end + 1
            continue
        current += char
        i += 1

    if current:
        tokens.append(current)
    return tokens


def get_path_value(payload: Any, path_tokens: list[str | int]) -> Any:
    current_items = [payload]
    for token in path_tokens:
        next_items: list[Any] = []
        for current in current_items:
            if token == "*":
                if isinstance(current, dict):
                    next_items.extend(current.values())
                elif isinstance(current, list):
                    next_items.extend(current)
                continue
            if isinstance(token, int):
                if not isinstance(current, list) or token >= len(current):
                    continue
                next_items.append(current[token])
                continue
            if not isinstance(current, dict) or token not in current:
                continue
            next_items.append(current[token])
        if not next_items:
            return None
        current_items = next_items

    if len(current_items) == 1:
        return current_items[0]
    return current_items


def format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, list):
        if not value:
            return "N/A"
        if all(not isinstance(item, (dict, list)) for item in value):
            return ", ".join(str(item) for item in value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def escape_cell(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", " ")


def normalize_sort_value(value: Any) -> tuple[int, Any]:
    if isinstance(value, bool):
        return (0, int(value))
    if isinstance(value, (int, float)):
        return (1, value)
    if isinstance(value, str):
        return (2, value)
    if isinstance(value, list):
        return (3, json.dumps(value, ensure_ascii=True, sort_keys=True))
    if isinstance(value, dict):
        return (4, json.dumps(value, ensure_ascii=True, sort_keys=True))
    return (5, str(value))


def collect_rows(
    directory: Path, json_paths: list[str]
) -> list[tuple[str, list[Any], list[str]]]:
    path_tokens_by_path = {
        json_path: parse_json_path(json_path) for json_path in json_paths
    }
    rows: list[tuple[str, list[Any], list[str]]] = []

    for file_path in sorted(directory.rglob("*.json")):
        with file_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        raw_values: list[Any] = []
        values: list[str] = []
        for json_path in json_paths:
            value = get_path_value(payload, path_tokens_by_path[json_path])
            raw_values.append(value)
            values.append(format_value(value))
        rows.append((file_path.name, raw_values, values))

    return rows


def sort_rows(
    rows: list[tuple[str, list[Any], list[str]]],
    json_paths: list[str],
    sort_by: str | None,
    sort_order: str,
) -> list[tuple[str, list[Any], list[str]]]:
    if not sort_by:
        return rows

    if sort_by == "file_name":
        present_rows = [row for row in rows if row[0] is not None]
        return sorted(
            present_rows,
            key=lambda row: row[0],
            reverse=sort_order == "desc",
        )

    if sort_by not in json_paths:
        raise SystemExit(
            f"Invalid --sort-by value: {sort_by}. Expected 'file_name' or one of: "
            + ", ".join(json_paths)
        )

    sort_index = json_paths.index(sort_by)
    present_rows = [row for row in rows if row[1][sort_index] is not None]
    missing_rows = [row for row in rows if row[1][sort_index] is None]

    present_rows = sorted(
        present_rows,
        key=lambda row: normalize_sort_value(row[1][sort_index]),
        reverse=sort_order == "desc",
    )
    return [*present_rows, *missing_rows]


def render_markdown_table(
    rows: list[tuple[str, list[Any], list[str]]], json_paths: list[str]
) -> str:
    header_cells = ["File Name", *json_paths]
    lines = [
        "| " + " | ".join(escape_cell(cell) for cell in header_cells) + " |",
        "|" + "|".join("---" for _ in header_cells) + "|",
    ]
    for file_name, _, values in rows:
        cells = [file_name, *values]
        lines.append("| " + " | ".join(escape_cell(cell) for cell in cells) + " |")
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    directory = Path(args.dir).resolve()

    if not directory.is_dir():
        raise SystemExit(f"Directory not found: {directory}")

    rows = collect_rows(directory, args.path)
    if not rows:
        raise SystemExit(f"No JSON files found under: {directory}")

    rows = sort_rows(rows, args.path, args.sort_by, args.sort_order)
    output = render_markdown_table(rows, args.path)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
        print(f"Saved markdown table to {output_path}")
        return

    print(output)


if __name__ == "__main__":
    main()
