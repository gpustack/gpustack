"""CLI controller: prepare, research, validate and apply a catalog proposal."""

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import List

from hack.catalog_sync.http import PublicClient
from hack.catalog_sync.models import (
    CATALOGS,
    REPO_FIELDS,
    Proposal,
    digest,
    parse_targets,
    require,
)
from hack.catalog_sync.validate import (
    check_card_identity,
    check_citations,
    load_catalog,
    render_catalogs,
    validate_entry,
)


def read_json(path: Path) -> dict:
    require(path.is_file() and not path.is_symlink(), "Missing regular JSON file")
    require(path.stat().st_size <= 16 * 1024 * 1024, "JSON artifact is too large")
    return json.loads(path.read_text())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def prepare(root: Path, output: Path, inputs: str) -> dict:
    request = {"base_sha": git_head(root), "targets": parse_targets(inputs)}
    request["request_id"] = digest(request)
    write_json(output / "request.json", request)
    return request


def _existing(root: Path) -> dict:
    return {
        source: load_catalog((root / path).read_text())["model_sets"]
        for source, path in CATALOGS.items()
    }


def _has_target(target: str, cards: dict) -> bool:
    prefix, repo = target.split(":", 1)
    source = "huggingface" if prefix == "hf" else "model_scope"
    return any(
        spec.get(REPO_FIELDS[source]) == repo
        for card in cards[source]
        for spec in card["specs"]
    )


def _check_candidate(candidate, existing, client):
    require(
        set(candidate.hubs) == set(CATALOGS), "Assess both Hugging Face and ModelScope"
    )
    sources = [entry.source for entry in candidate.entries]
    require(len(sources) == len(set(sources)), "Only one card per hub and target")
    if candidate.status != "ready":
        require(not candidate.entries, "Only ready candidates may contain edits")
        if candidate.status == "unchanged":
            require(
                _has_target(candidate.target, existing),
                "Unchanged target is not in the catalog",
            )
        return
    require(bool(candidate.entries), "Ready candidate has no entries")
    prefix, repo = candidate.target.split(":", 1)
    source = "huggingface" if prefix == "hf" else "model_scope"
    require(
        any(
            entry.source == source
            and any(
                spec.get(REPO_FIELDS[source]) == repo
                for spec in entry.card.get("specs", [])
            )
            for entry in candidate.entries
        ),
        "Proposal does not include the requested repository",
    )
    for hub, assessment in candidate.hubs.items():
        if assessment.evidence:
            check_citations(client, assessment.evidence)
        if hub in sources:
            require(assessment.status == "available", "Cannot add an unverified hub")
            entry = next(entry for entry in candidate.entries if entry.source == hub)
            require(
                any(
                    spec.get(REPO_FIELDS[hub]) == assessment.repo_id
                    for spec in entry.card["specs"]
                ),
                "Hub assessment does not identify a proposed repository",
            )


def _candidate_result(candidate, root, existing, client):
    outcome = {
        "target": candidate.target,
        "status": candidate.status,
        "reason": candidate.reason,
        "hubs": {key: value.model_dump() for key, value in candidate.hubs.items()},
        "ascend_910b": candidate.ascend_910b,
        "entries": [],
    }
    entries, icons = [], {}
    try:
        _check_candidate(candidate, existing, client)
        for entry in candidate.entries:
            check_card_identity(entry.card, entry.source, existing[entry.source])
            report, downloaded = validate_entry(entry, root, client)
            for path, data in downloaded.items():
                require(
                    path not in icons or icons[path] == data, "Conflicting icon content"
                )
            icons.update(downloaded)
            entries.append(entry)
            outcome["entries"].append(report)
    except Exception as error:
        # One unavailable model must not discard independently verified models.
        outcome.update(
            status="blocked",
            reason=f"Validation: {type(error).__name__}: {error}",
            entries=[],
        )
        return outcome, [], {}
    return outcome, entries, icons


def report_markdown(request: dict, results: List[dict]) -> str:
    lines = [
        "## Model catalog update",
        "",
        "Adds or updates the requested checkpoints using independently checked "
        "public metadata.",
        "Parameter counts prefer official statements. Recipes prefer broad "
        "compatibility; Ascend research prioritizes 910B.",
        "",
        f"Base revision: `{request['base_sha']}`.",
        "",
        "No GPU/NPU inference was executed. Source quotations require human semantic review.",
    ]
    for result in results:
        lines += [
            "",
            f"### {result['target']} — {result['status']}",
            "",
            result["reason"],
            "",
            f"Ascend 910B: {result['ascend_910b']}",
        ]
        for source, assessment in result["hubs"].items():
            lines += ["", f"- {source}: {assessment['status']}. {assessment['reason']}"]
        for entry in result["entries"]:
            lines += [
                "",
                f"**{entry['source']} / {entry['name']}**",
                "",
                f"Parameters: {entry['parameter_method']}. {entry['parameter_notes']}",
            ]
            for snapshot, runner, proof in zip(
                entry["snapshots"], entry["runners"], entry["spec_evidence"]
            ):
                lines += [
                    "",
                    f"- Weights: `{snapshot['repo_id']}` at `{snapshot['revision']}`; "
                    f"{snapshot['weight_bytes']:,} bytes across "
                    f"{len(snapshot['files'])} files.",
                    f"- Runner: `{runner['image']}` ({runner['hardware']}). "
                    f"{proof['compatibility']}",
                ]
            for field, citations in entry["evidence"].items():
                lines += [
                    f"- {field}: "
                    + "; ".join(f"[{c['explanation']}]({c['url']})" for c in citations)
                ]
            for proof in entry["spec_evidence"]:
                for field in ("deployment", "quantization"):
                    lines += [
                        f"- {field}: "
                        + "; ".join(
                            f"[{c['explanation']}]({c['url']})" for c in proof[field]
                        )
                    ]
    lines += [
        "",
        "Full weight manifests, quotations and hardware matching results "
        "are in the workflow artifacts.",
        "",
    ]
    # Repository policy leaves issue closure to QA. Avoid mentions from upstream text.
    body = "\n".join(lines)
    body = re.sub(
        r"(?i)\b(close[sd]?|fix(?:es|ed)?|resolve[sd]?)\s+#(\d+)", r"ref #\2", body
    )
    body = body.replace("@", "＠").replace("<!--", "&lt;!--")
    require(len(body.encode()) < 60000, "PR report is too large; reduce the batch")
    return body


def validate(
    root: Path, request: dict, proposal: Proposal, output: Path, client=None
) -> dict:
    require(
        git_head(root) == request["base_sha"],
        "Validation checkout differs from the research base",
    )
    require(
        sorted(c.target for c in proposal.candidates) == sorted(request["targets"]),
        "Proposal must assess each requested target exactly once",
    )
    client = client or PublicClient()
    existing = _existing(root)
    entries, icons, results = [], {}, []
    for candidate in proposal.candidates:
        result, accepted, downloaded = _candidate_result(
            candidate, root, existing, client
        )
        for path, data in downloaded.items():
            require(
                path not in icons or icons[path] == data,
                "Conflicting icons between targets",
            )
        entries.extend(accepted)
        icons.update(downloaded)
        results.append(result)
    files = {**render_catalogs(root, entries), **icons}
    body = report_markdown(request, results)
    bundle = {
        "request": request,
        "body": body,
        "files": {
            path: base64.b64encode(data).decode() for path, data in files.items()
        },
        "before": {
            path: (
                hashlib.sha256((root / path).read_bytes()).hexdigest()
                if (root / path).exists()
                else None
            )
            for path in files
        },
    }
    bundle["sha256"] = digest(bundle)
    write_json(output / "validated.json", bundle)
    write_json(output / "report.json", {"request": request, "results": results})
    (output / "pr-body.md").write_text(body)
    result = {
        "changed": bool(files),
        "partial": any(row["status"] == "blocked" for row in results),
        "sha256": bundle["sha256"],
    }
    write_json(output / "result.json", result)
    return result


def apply(root: Path, bundle: dict, expected_digest: str, write: bool = True) -> None:
    checksum = bundle.pop("sha256")
    require(
        checksum == expected_digest == digest(bundle),
        "Validated artifact digest mismatch",
    )
    require(
        git_head(root) == bundle["request"]["base_sha"], "Publication checkout changed"
    )
    decoded = {}
    for path, content in bundle["files"].items():
        require(
            path in CATALOGS.values()
            or re.fullmatch(
                r"static/catalog_icons/[A-Za-z0-9][A-Za-z0-9_-]*\.png", path
            ),
            "Artifact contains a path outside the catalog allowlist",
        )
        target = root / path
        require(not target.is_symlink(), "Artifact destination is a symlink")
        require(
            target.resolve().is_relative_to(root.resolve()),
            "Artifact escapes the checkout",
        )
        before = (
            hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None
        )
        require(
            before == bundle["before"][path],
            "Publication would overwrite changed content",
        )
        decoded[path] = base64.b64decode(content, validate=True)
    if write:
        for path, data in decoded.items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


def preview(root: Path, output: Path) -> dict:
    """Run the local proposal pipeline in order, stopping at the first failure."""
    from hack.catalog_sync.agent import research

    require(not any(output.iterdir()), "Use a new empty preview output directory")
    request = prepare(root, output / "request", os.environ.get("CATALOG_MODEL_IDS", ""))
    research_output = output / "research"
    research_output.mkdir()
    research(root, request, research_output)
    proposal = Proposal.model_validate(read_json(research_output / "proposal.json"))
    validated_output = output / "validated"
    validated_output.mkdir()
    result = validate(root, request, proposal, validated_output)
    print(json.dumps(result, indent=2))
    print(f"Review: {validated_output / 'pr-body.md'}")
    report = read_json(validated_output / "report.json")
    for candidate in report["results"]:
        if candidate["status"] == "blocked":
            print(f"Blocked {candidate['target']}: {candidate['reason']}")
    require(
        result["changed"] or not result["partial"],
        "No changes passed validation; see the report",
    )
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["prepare", "research", "validate", "verify", "apply", "preview"],
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--proposal", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "preview":
        preview(args.root, args.output)
    elif args.command == "prepare":
        prepare(args.root, args.output, os.environ["CATALOG_MODEL_IDS"])
    elif args.command == "research":
        from hack.catalog_sync.agent import research

        research(args.root, read_json(args.request), args.output)
    elif args.command == "validate":
        result = validate(
            args.root,
            read_json(args.request),
            Proposal.model_validate(read_json(args.proposal)),
            args.output,
        )
        if os.getenv("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
                for key, value in result.items():
                    stream.write(f"{key}={str(value).lower()}\n")
    else:
        bundle = read_json(args.output / "validated.json")
        apply(
            args.root,
            bundle,
            os.environ["CATALOG_BUNDLE_SHA256"],
            write=args.command == "apply",
        )
        (args.output / "pr-body.md").write_text(bundle["body"])


def main() -> None:
    try:
        _main()
    except ValueError as error:
        print(f"Catalog sync failed: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
