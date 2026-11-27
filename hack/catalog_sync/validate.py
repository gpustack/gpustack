"""Independently verify proposals using trusted code and fresh public evidence."""

import html
import io
import json
import math
import re
import subprocess
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import yaml
from gpustack_runner import list_runners
from packaging.specifiers import SpecifierSet
from PIL import Image

from gpustack.schemas.catalog_source import normalize_catalog_yaml
from gpustack.schemas.model_sets import ModelSet
from gpustack.schemas.models import CategoryEnum
from gpustack.utils.model_catalog import filter_specs_by_gpu
from hack.catalog_sync.http import PublicClient
from hack.catalog_sync.hubs import collect
from hack.catalog_sync.models import (
    CATALOGS,
    REPO_FIELDS,
    Citation,
    Entry,
    require,
)

SPEC_FIELDS = {
    "mode",
    "quantization",
    "gpu_filters",
    "source",
    "huggingface_repo_id",
    "model_scope_model_id",
    "huggingface_filename",
    "model_scope_file_path",
    "backend",
    "backend_version",
    "image_name",
    "backend_parameters",
    "env",
}
CARD_FIELDS = {
    "name",
    "description",
    "home",
    "icon",
    "categories",
    "capabilities",
    "size",
    "activated_size",
    "size_unit",
    "licenses",
    "release_date",
    "specs",
}


def _plain(text: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]*>", " ", text)).split())


def check_citations(client: PublicClient, citations: List[Citation]) -> None:
    require(bool(citations), "Missing evidence")
    for citation in citations:
        document = client.get(citation.url).decode("utf-8")
        require(
            _plain(citation.quote) in _plain(document),
            f"Quoted evidence not found: {citation.url}",
        )


def check_download_layout(spec: dict) -> None:
    """Keep directory-based backends on the worker's full-repository download path."""
    if spec["backend"].lower() in {"vllm", "sglang", "mindie"}:
        require(
            not spec.get("huggingface_filename")
            and not spec.get("model_scope_file_path"),
            f"{spec['backend']} recipes require a full repository download; "
            "omit huggingface_filename and model_scope_file_path so configuration, "
            "tokenizer and other model assets are downloaded with the weights",
        )


def check_card(entry: Entry) -> ModelSet:
    card = entry.card
    require(not (set(card) - CARD_FIELDS), "Unsupported catalog card fields")
    require(card.get("name") and card.get("description"), "Missing name or description")
    require(
        card.get("categories") and card.get("licenses"), "Missing category or license"
    )
    require(
        set(card["categories"]) <= {category.value for category in CategoryEnum},
        "Unknown model category",
    )
    require(
        type(card.get("size")) in (float, int)
        and math.isfinite(card["size"])
        and card["size"] > 0,
        "Missing parameter size",
    )
    release_date = card.get("release_date")
    require(isinstance(release_date, str), "Release date must be an ISO date string")
    released = date.fromisoformat(release_date)
    require(released <= date.today(), "Release date is in the future")
    if card.get("activated_size") is not None:
        total_billions = (
            card["size"] * {"M": 0.001, "B": 1, "T": 1000}[card.get("size_unit", "B")]
        )
        require(
            0 < card["activated_size"] <= total_billions,
            "Active parameter size must be positive and no greater than total size",
        )
    require(len(card["specs"]) == len(entry.specs), "Every spec needs its own proof")
    for spec in card["specs"]:
        require(not (set(spec) - SPEC_FIELDS), "Unsupported deployment fields")
        require(
            spec.get("mode", "standard") in {"standard", "throughput", "latency"},
            "Unknown deployment mode",
        )
        require(spec.get("source") == entry.source, "Spec uses the wrong hub")
        require(spec.get(REPO_FIELDS[entry.source]), "Spec has no repository ID")
        other = "model_scope" if entry.source == "huggingface" else "huggingface"
        require(REPO_FIELDS[other] not in spec, "Spec mixes hub IDs")
        require(
            spec.get("backend") and spec.get("quantization"),
            "Missing deployment metadata",
        )
        check_download_layout(spec)
        gf = spec.get("gpu_filters") or {}
        require(
            not (set(gf) - {"vendor", "vendor_variant", "compute_capability"}),
            "Unknown GPU filter",
        )
        if gf.get("compute_capability"):
            SpecifierSet(gf["compute_capability"])
    # Strict parsing rejects fields the application would otherwise silently drop.
    normalize_catalog_yaml(yaml.safe_dump({"model_sets": [card]}), strict=True)
    return ModelSet.model_validate(card)


def check_parameters(entry: Entry, snapshots: List[dict]) -> None:
    if entry.parameter_method == "official":
        require(
            bool(entry.evidence.get("size")),
            "Official parameter count needs a citation",
        )
        return
    # Packed quantized tensors count storage elements, not original parameters.
    require(
        all(
            not s["documents"].get("config.json", {}).get("quantization_config")
            and s.get("safetensors_parameters")
            and all(name.endswith(".safetensors") for name in s["files"])
            for s in snapshots
        ),
        "Parameter fallback needs unquantized safetensors metadata; "
        "use official evidence otherwise",
    )
    scale = {"M": 1e6, "B": 1e9, "T": 1e12}[entry.card.get("size_unit", "B")]
    expected = entry.card["size"] * scale
    require(
        all(
            abs(s["safetensors_parameters"] - expected) <= scale * 0.000001
            for s in snapshots
        ),
        "Computed parameter size does not match safetensors metadata",
    )


def _inspect_image(image: str) -> None:
    manifest = subprocess.run(
        ["docker", "manifest", "inspect", image],
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )
    require(
        json.loads(manifest.stdout).get("schemaVersion") == 2,
        "Runner registry returned an invalid manifest",
    )


def _check_official_image(spec, proof, vendor):
    repositories = {
        ("vllm", "nvidia"): "vllm/vllm-openai",
        ("vllm", "ascend"): "quay.io/ascend/vllm-ascend",
        ("sglang", "nvidia"): "lmsysorg/sglang",
    }
    repository = repositories.get((spec.backend.lower(), vendor))
    require(repository is not None, "No supported official image publisher")
    require(
        bool(
            re.fullmatch(
                re.escape(repository)
                + r":[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}"
                + r"(?:@sha256:[a-f0-9]{64})?",
                spec.image_name,
            )
        ),
        "Use an explicitly tagged image from the backend's official publisher",
    )
    require(
        spec.image_name == proof.runner_image, "Deployment image differs from proof"
    )
    reference = spec.image_name.split("@", 1)[0]
    require(
        any(reference in citation.quote for citation in proof.deployment),
        "Deployment evidence must name the official image tag",
    )
    _inspect_image(spec.image_name)
    return {
        "image": spec.image_name,
        "service_version": spec.backend_version,
        "hardware": proof.hardware,
        "verification": "official recipe image and registry manifest; "
        "runtime compatibility requires citation review; no hardware execution",
    }


def check_runner(spec, proof) -> dict:
    vendor, _, variant = proof.hardware.partition(":")
    variant = "a3" if variant == "910c" else variant
    gf = spec.gpu_filters
    require(gf and gf.vendor == [vendor], "Scope the spec to its evidenced GPU vendor")
    if vendor == "ascend":
        require(
            gf.vendor_variant == [variant], "Scope Ascend to its evidenced CANN variant"
        )
    if spec.image_name:
        return _check_official_image(spec, proof, vendor)
    require(
        bool(spec.backend_version),
        "Pin backend_version to the evidenced runner's service version; "
        "automatic selection can choose a different version on another host",
    )
    matches = [
        row
        for row in list_runners(with_deprecated=True)
        if row.docker_image == proof.runner_image
        and row.service.lower() == spec.backend.lower()
        and row.backend == ("cuda" if vendor == "nvidia" else "cann")
        and (not variant or row.backend_variant == variant)
        and row.service_version == spec.backend_version
    ]
    require(bool(matches), "No matching packaged runner for the recipe")
    row = matches[0]
    _inspect_image(row.docker_image)
    return {
        "image": row.docker_image,
        "service_version": row.service_version,
        "runtime": row.original_backend_version,
        "platform": row.platform,
        "hardware": proof.hardware,
        "verification": "packaged runner and registry manifest; no hardware execution",
    }


def check_matching(model: ModelSet) -> dict:
    probes = {
        f"nvidia:{cc}": SimpleNamespace(
            vendor="nvidia", compute_capability=cc, arch_family=None
        )
        for cc in (
            "7.0",
            "7.5",
            "8.0",
            "8.6",
            "8.9",
            "9.0",
            "10.0",
            "10.3",
            "11.0",
            "12.0",
            "12.1",
        )
    }
    probes.update(
        {
            "ascend:910b": SimpleNamespace(
                vendor="ascend", compute_capability=None, arch_family="Ascend910B2"
            ),
            "ascend:910c": SimpleNamespace(
                vendor="ascend", compute_capability=None, arch_family="Ascend910_9391"
            ),
        }
    )
    selected = {
        key: [
            model.specs.index(spec) for spec in filter_specs_by_gpu([gpu], model.specs)
        ]
        for key, gpu in probes.items()
    }
    reachable = {index for values in selected.values() for index in values}
    require(
        reachable == set(range(len(model.specs))),
        "A spec is shadowed or matches no representative hardware",
    )
    return selected


def acquire_icon(client: PublicClient, entry: Entry, root: Path) -> Tuple[str, bytes]:
    icon = entry.icon
    require(icon is not None, "Missing icon download metadata")
    check_citations(client, icon.evidence)
    if icon.github_org:
        org = client.json(f"https://api.github.com/orgs/{icon.github_org}")
        require(
            org.get("type") == "Organization", "Icon owner is not a GitHub organization"
        )
        require(
            icon.url == org.get("avatar_url"),
            "Use the official organization's avatar_url",
        )
    path = f"static/catalog_icons/{icon.filename}"
    require(
        not (root / path).exists(), "Reuse existing icons instead of overwriting them"
    )
    require(entry.card.get("icon") == f"/{path}", "Icon path does not match download")
    raw = client.get(icon.url, limit=2 * 1024 * 1024)
    with Image.open(io.BytesIO(raw)) as picture:
        require(
            picture.format in {"PNG", "JPEG", "WEBP"}, "Icon must be a raster image"
        )
        require(
            16 <= min(picture.size) and max(picture.size) <= 2048,
            "Unexpected icon dimensions",
        )
        picture.load()
        picture.thumbnail((256, 256))
        output = io.BytesIO()
        picture.convert("RGBA").save(output, format="PNG")
    return path, output.getvalue()


def validate_entry(
    entry: Entry, root: Path, client: PublicClient
) -> Tuple[dict, Dict[str, bytes]]:
    model = check_card(entry)
    required = {"identity", "release_date", "licenses", "size"}
    if model.capabilities:
        required.add("capabilities")
    if model.activated_size is not None:
        required.add("activated_size")
    require(required <= set(entry.evidence), "Missing metadata evidence fields")
    for citations in entry.evidence.values():
        check_citations(client, citations)
    snapshots, runners = [], []
    for spec, proof in zip(model.specs, entry.specs):
        check_citations(client, proof.deployment)
        check_citations(client, proof.quantization)
        pattern = (
            spec.huggingface_filename
            if entry.source == "huggingface"
            else spec.model_scope_file_path
        )
        snapshot = collect(
            client,
            entry.source,
            getattr(spec, REPO_FIELDS[entry.source]),
            proof.revision,
            pattern,
        )
        require(
            snapshot["snapshot"] == proof.snapshot,
            "Weight/config snapshot changed or was not collected",
        )
        snapshots.append(snapshot)
        runners.append(check_runner(spec, proof))
    check_parameters(entry, snapshots)
    matching = check_matching(model)
    icons = {}
    if entry.icon:
        path, data = acquire_icon(client, entry, root)
        icons[path] = data
    else:
        path = model.icon or ""
        require(
            bool(re.fullmatch(r"/static/catalog_icons/[A-Za-z0-9_.-]+", path)),
            "Use a local catalog icon",
        )
        require((root / path.lstrip("/")).is_file(), "Catalog icon does not exist")
    return {
        "source": entry.source,
        "name": model.name,
        "parameter_method": entry.parameter_method,
        "parameter_notes": entry.parameter_notes,
        "snapshots": snapshots,
        "runners": runners,
        "matching": matching,
        "evidence": {
            key: [c.model_dump() for c in values]
            for key, values in entry.evidence.items()
        },
        "spec_evidence": [proof.model_dump() for proof in entry.specs],
    }, icons


def load_catalog(text: str) -> dict:
    """Reject malformed catalog structure before matching or rendering cards."""
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid catalog YAML: {error}") from None
    require(isinstance(document, dict), "Catalog must be a YAML mapping")
    cards = document.get("model_sets")
    require(isinstance(cards, list), "Catalog must contain a model_sets sequence")
    names = set()
    for card in cards:
        require(isinstance(card, dict), "Catalog entry must be a mapping")
        name = card.get("name")
        require(isinstance(name, str) and name.strip(), "Catalog entry is missing name")
        require(name not in names, f"Duplicate catalog name: {name}")
        names.add(name)
        specs = card.get("specs")
        require(
            isinstance(specs, list) and all(isinstance(spec, dict) for spec in specs),
            f"Catalog entry {name} must contain a specs sequence of mappings",
        )
    return document


def replace_cards(text: str, cards: List[dict]) -> str:
    """Replace YAML sequence nodes while preserving unrelated bytes and anchors."""
    load_catalog(text)
    document = yaml.compose(text)
    node = next(
        (value for key, value in document.value if key.value == "model_sets"), None
    )
    require(
        isinstance(node, yaml.SequenceNode),
        "Catalog needs an explicit model_sets sequence",
    )
    if node.flow_style:
        require(not node.value, "Nonempty flow-style catalogs are not supported")
        start = node.start_mark.index
        while text[start - 1] == " ":
            start -= 1
        replacement = "\n" + yaml.safe_dump(
            cards, sort_keys=False, allow_unicode=True
        ).rstrip("\n")
        result = text[:start] + replacement + text[node.end_mark.index :]
        normalize_catalog_yaml(result, strict=True)
        return result
    existing = {}
    for item in node.value:
        name = next(
            (value.value for key, value in item.value if key.value == "name"), None
        )
        require(isinstance(name, str) and name, "Catalog entry needs an explicit name")
        require(name not in existing, f"Duplicate catalog name: {name}")
        existing[name] = item
    edits, additions = [], []
    for card in cards:
        replacement = yaml.safe_dump(
            [card], sort_keys=False, allow_unicode=True, width=1000
        )
        if card["name"] in existing:
            item = existing[card["name"]]
            start, end = item.start_mark.index, item.end_mark.index
            # Sequence MappingNodes start after the '- ' on their first line.
            start = text.rfind("\n", 0, start) + 1
            edits.append((start, end, replacement))
        else:
            additions.append(replacement)
    if additions:
        edits.append((node.end_mark.index, node.end_mark.index, "".join(additions)))
    for start, end, replacement in sorted(edits, reverse=True):
        text = text[:start] + replacement + text[end:]
    normalize_catalog_yaml(text, strict=True)
    return text


def check_card_identity(card: dict, source: str, existing: List[dict]) -> None:
    """Require repository continuity when replacing a named catalog entry."""
    field = REPO_FIELDS[source]
    repos = {spec.get(field) for spec in card.get("specs", [])} - {None, ""}
    for old in existing:
        overlaps = any(spec.get(field) in repos for spec in old["specs"])
        same_name = old["name"] == card["name"]
        require(
            not overlaps or same_name,
            "Preserve the existing card name for a known repository",
        )
        require(
            not same_name or overlaps,
            f"Catalog name {card['name']} belongs to unrelated repositories; "
            "use a distinct name for the new model",
        )


def render_catalogs(root: Path, entries: List[Entry]) -> Dict[str, bytes]:
    result = {}
    for source, path in CATALOGS.items():
        cards = [entry.card for entry in entries if entry.source == source]
        names = [card["name"] for card in cards]
        require(
            len(names) == len(set(names)), "Two targets propose the same catalog card"
        )
        if cards:
            before = (root / path).read_text()
            original = load_catalog(before)
            for card in cards:
                check_card_identity(card, source, original["model_sets"])
            after = replace_cards(before, cards)
            rendered = yaml.safe_load(after)
            expected = {card["name"]: card for card in original["model_sets"]}
            expected.update({card["name"]: card for card in cards})
            require(
                {card["name"]: card for card in rendered["model_sets"]} == expected,
                "Rendering changed an unrelated card",
            )
            require(
                {k: v for k, v in original.items() if k != "model_sets"}
                == {k: v for k, v in rendered.items() if k != "model_sets"},
                "Rendering changed shared catalog metadata",
            )
            if before != after:
                result[path] = after.encode()
    return result
