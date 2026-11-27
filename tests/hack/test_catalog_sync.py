import base64
import copy
import io
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

from PIL import Image
import pytest
import yaml

from hack.catalog_sync import agent, hubs, run, validate
from hack.catalog_sync.http import PublicClient, public_url
from hack.catalog_sync.models import (
    CATALOGS,
    Citation,
    Entry,
    Proposal,
    digest,
    parse_targets,
)

FIXTURES = Path(__file__).parents[1] / "fixtures/catalog_sync"


@pytest.fixture
def proposal():
    return Proposal.model_validate_json((FIXTURES / "ready.json").read_text())


@pytest.fixture
def entry(proposal):
    return proposal.candidates[0].entries[0]


@pytest.fixture
def client():
    result = Mock(spec=PublicClient)
    result.get.return_value = (FIXTURES / "model-card.md").read_bytes()
    return result


@pytest.fixture
def catalog_root(tmp_path):
    for path in CATALOGS.values():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("draft_models: []\nmodel_sets: []\n")
    icon = tmp_path / "static/catalog_icons/example.png"
    icon.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(icon)
    return tmp_path


@pytest.fixture
def snapshot():
    return {
        "source": "huggingface",
        "repo_id": "publisher/example",
        "revision": "1234567890abcdef1234567890abcdef12345678",
        "files": {"model.safetensors": {"size": 14000000080, "oid": "sha"}},
        "weight_bytes": 14000000080,
        "documents": {"config.json": {}},
        "safetensors_parameters": 7001000000,
        "snapshot": "a" * 64,
    }


@pytest.fixture
def verified_io(monkeypatch, snapshot):
    monkeypatch.setattr(validate, "collect", lambda *args: copy.deepcopy(snapshot))
    row = SimpleNamespace(
        docker_image="example.org/runner:vllm-cuda",
        service="vllm",
        backend="cuda",
        backend_variant="",
        service_version="0.20.0",
        original_backend_version="12.8",
        platform="linux/amd64",
    )
    monkeypatch.setattr(validate, "list_runners", lambda **kwargs: [row])
    monkeypatch.setattr(
        validate.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout='{"schemaVersion": 2}'),
    )
    monkeypatch.setattr(run, "git_head", lambda root: "b" * 40)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("org/model, hf:org/model\nms:other/model", ["hf:org/model", "ms:other/model"]),
        (
            "https://huggingface.co/org/model https://modelscope.cn/models/other/model",
            ["hf:org/model", "ms:other/model"],
        ),
    ],
)
def test_parse_targets(value, expected):
    assert parse_targets(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "org/model;echo",
        "hf:org/../model",
        "https://evil.test/org/model",
        "https://huggingface.co/org/model?revision=x",
        "$(touch /tmp/x)",
        ",".join(f"org/model{i}" for i in range(11)),
    ],
)
def test_reject_invalid_targets(value):
    with pytest.raises(ValueError):
        parse_targets(value)


@pytest.mark.parametrize("index_name", hubs.SAFETENSORS_INDEXES)
def test_hf_collection_uses_index_once_and_keeps_byte_counts_separate(index_name):
    client = Mock(spec=PublicClient)
    client.json.side_effect = [
        {
            "sha": "revision",
            "safetensors": {"total": 7},
            "siblings": [
                {
                    "rfilename": index_name,
                    "size": 100,
                    "blobId": "index",
                },
                {"rfilename": "a.safetensors", "size": 15, "lfs": {"sha256": "a"}},
                {"rfilename": "b.safetensors", "size": 16, "lfs": {"sha256": "b"}},
                {"rfilename": "pytorch_model.bin", "size": 31, "blobId": "bin"},
            ],
        },
        {
            "metadata": {"total_size": 14},
            "weight_map": {
                "a": "a.safetensors",
                "b": "b.safetensors",
                "c": "a.safetensors",
            },
        },
    ]
    result = hubs.collect(client, "huggingface", "org/model")
    assert result["weight_bytes"] == 31
    index = result["documents"][index_name]
    assert index["metadata"]["total_size"] == 14
    assert index["tensor_count"] == 3
    assert index["shards"] == ["a.safetensors", "b.safetensors"]
    assert "weight_map" not in index
    assert client.json.call_args.kwargs == {"limit": hubs.INDEX_LIMIT}
    assert set(result["files"]) == {"a.safetensors", "b.safetensors"}
    assert result["revision"] == "revision"
    assert result["snapshot"] == digest(
        {k: v for k, v in result.items() if k != "snapshot"}
    )


def test_compact_index_binds_tensor_mapping_not_just_shard_names():
    first = {"model.safetensors.index.json": {"weight_map": {"a": "shard.safetensors"}}}
    second = {
        "model.safetensors.index.json": {"weight_map": {"b": "shard.safetensors"}}
    }
    assert digest(hubs._compact_documents(first)) != digest(
        hubs._compact_documents(second)
    )


@pytest.mark.parametrize("size", [None, 0, hubs.INDEX_LIMIT + 1])
def test_invalid_index_size_is_rejected_before_download(size):
    client = Mock(spec=PublicClient)
    with pytest.raises(ValueError, match="Invalid index size"):
        hubs._documents(
            client,
            "huggingface",
            "org/model",
            "rev",
            {"model.safetensors.index.json": {"size": size}},
        )
    client.json.assert_not_called()


@pytest.mark.parametrize("index", [{}, {"weight_map": {}}, {"weight_map": {"a": 1}}])
def test_invalid_index_mapping_is_rejected(index):
    client = Mock(spec=PublicClient)
    client.json.return_value = index
    with pytest.raises(ValueError, match="Invalid weight index mapping"):
        hubs._documents(
            client,
            "huggingface",
            "org/model",
            "rev",
            {"model.safetensors.index.json": {"size": 100}},
        )


def test_modelscope_collection_keeps_file_hashes():
    client = Mock(spec=PublicClient)
    client.json.return_value = {
        "Code": 200,
        "Data": {
            "Files": [
                {"Path": "dir", "Type": "tree"},
                {"Path": "model.safetensors", "Size": 123, "Sha256": "file-digest"},
            ]
        },
    }
    result = hubs.collect(client, "model_scope", "org/model")
    assert result["revision"] == "master"
    assert result["files"]["model.safetensors"]["oid"] == "file-digest"
    assert "Recursive=true" in client.json.call_args.args[0]


@pytest.mark.parametrize(
    "files,pattern,indexes",
    [
        ({"a.gguf": {"size": 1}, "b.gguf": {"size": 2}}, None, {}),
        ({"a.bin": {"size": 1}, "b.safetensors": {"size": 2}}, "*", {}),
        ({"model.safetensors": {"size": None}}, None, {}),
        (
            {},
            None,
            {
                "model.safetensors.index.json": {
                    "weight_map": {"x": "missing.safetensors"}
                }
            },
        ),
    ],
)
def test_ambiguous_missing_or_duplicate_weight_formats_are_rejected(
    files, pattern, indexes
):
    with pytest.raises(ValueError):
        hubs.select_weights(files, pattern, indexes)


def test_official_parameter_count_takes_precedence(entry, snapshot):
    validate.check_parameters(entry, [snapshot])
    entry.parameter_method = "safetensors_metadata"
    with pytest.raises(ValueError, match="does not match"):
        validate.check_parameters(entry, [snapshot])
    entry.card["size"] = 7.001
    validate.check_parameters(entry, [snapshot])
    snapshot["documents"]["config.json"]["quantization_config"] = {
        "quant_method": "fp8"
    }
    with pytest.raises(ValueError, match="unquantized"):
        validate.check_parameters(entry, [snapshot])


def test_broad_nvidia_recipe_and_ascend_matching(entry):
    spec = entry.card["specs"][0]
    ascend = copy.deepcopy(spec)
    ascend["gpu_filters"] = {"vendor": "ascend", "vendor_variant": "910b"}
    entry.card["specs"].insert(0, ascend)
    entry.specs.insert(0, entry.specs[0].model_copy())
    model = validate.check_card(entry)
    matches = validate.check_matching(model)
    assert matches["nvidia:7.0"] == matches["nvidia:12.1"] == [1]
    assert matches["ascend:910b"] == [0]
    assert matches["ascend:910c"] == []
    entry.card["specs"].append(copy.deepcopy(spec))
    entry.specs.append(entry.specs[-1].model_copy())
    with pytest.raises(ValueError, match="shadowed"):
        validate.check_matching(validate.check_card(entry))


def test_910c_uses_a3_runtime_variant(entry, monkeypatch):
    entry.card["specs"][0]["gpu_filters"] = {"vendor": "ascend", "vendor_variant": "a3"}
    entry.specs[0].hardware = "ascend:910c"
    model = validate.check_card(entry)
    assert validate.check_matching(model)["ascend:910c"] == [0]


@pytest.mark.parametrize(
    "change",
    [
        {"release_date": "2999-01-01"},
        {"release_date": None},
        {"release_date": 20260101},
        {"weight_size": 123},
        {"size": -7},
    ],
)
def test_card_rejects_bad_metadata(entry, change):
    entry.card.update(change)
    with pytest.raises(ValueError):
        validate.check_card(entry)


def test_active_parameters_use_catalog_billions_convention(entry):
    entry.card.update(size=1.02, size_unit="T", activated_size=42)
    assert validate.check_card(entry).activated_size == 42
    entry.card["activated_size"] = 1021
    with pytest.raises(ValueError, match="Active parameter"):
        validate.check_card(entry)


def test_weight_selection_cannot_omit_indexed_shards():
    files = {"a.safetensors": {"size": 12}, "b.safetensors": {"size": 15}}
    indexes = {
        "model.safetensors.index.json": {
            "weight_map": {
                "a": "a.safetensors",
                "b": "b.safetensors",
            }
        }
    }
    with pytest.raises(ValueError, match="omits indexed"):
        hubs.select_weights(files, "a.safetensors", indexes)


def test_additional_components_require_explicit_selection():
    files = {
        "model.safetensors": {"size": 12},
        "vision/model.safetensors": {"size": 15},
    }
    with pytest.raises(ValueError, match="Additional weights"):
        hubs.select_weights(files, None, {})


def test_evidence_quote_is_retrieved(client):
    citation = Citation(
        url="https://example.org/model",
        quote="A fabricated publisher statement",
        explanation="Supposed official parameter statement.",
    )
    with pytest.raises(ValueError, match="not found"):
        validate.check_citations(client, [citation])


def test_validation_rechecks_snapshot(entry, catalog_root, client, verified_io):
    report, icons = validate.validate_entry(entry, catalog_root, client)
    assert report["snapshots"][0]["weight_bytes"] == 14000000080
    assert not icons
    entry.specs[0].snapshot = "c" * 64
    with pytest.raises(ValueError, match="snapshot changed"):
        validate.validate_entry(entry, catalog_root, client)


@pytest.mark.parametrize("source", list(CATALOGS))
@pytest.mark.parametrize("backend", ["vLLM", "SGLang", "MindIE"])
@pytest.mark.parametrize("pattern", ["model.safetensors", "*.safetensors", "*"])
def test_directory_backend_rejects_filtered_download_before_collecting_evidence(
    entry, catalog_root, client, verified_io, monkeypatch, source, backend, pattern
):
    entry.source = source
    spec = entry.card["specs"][0]
    spec.update(source=source, backend=backend)
    if source == "model_scope":
        spec["model_scope_model_id"] = spec.pop("huggingface_repo_id")
    field = (
        "huggingface_filename" if source == "huggingface" else "model_scope_file_path"
    )
    spec[field] = pattern
    collect = Mock(wraps=validate.collect)
    monkeypatch.setattr(validate, "collect", collect)
    with pytest.raises(ValueError, match="require a full repository download"):
        validate.validate_entry(entry, catalog_root, client)
    collect.assert_not_called()
    client.get.assert_not_called()


@pytest.mark.parametrize("backend", ["vLLM", "SGLang", "MindIE"])
@pytest.mark.parametrize("pattern", [None, ""])
def test_directory_backend_accepts_full_repository_download(entry, backend, pattern):
    entry.card["specs"][0].update(backend=backend, huggingface_filename=pattern)
    assert validate.check_card(entry).specs[0].huggingface_filename == pattern


@pytest.mark.parametrize("source", list(CATALOGS))
def test_file_based_gguf_recipe_retains_filename_selection(entry, source):
    entry.source = source
    spec = entry.card["specs"][0]
    spec.update(source=source, backend="llama-box", quantization="Q4_K_M")
    if source == "model_scope":
        spec["model_scope_model_id"] = spec.pop("huggingface_repo_id")
    field = (
        "huggingface_filename" if source == "huggingface" else "model_scope_file_path"
    )
    spec[field] = "model-Q4_K_M.gguf"
    model = validate.check_card(entry)
    assert getattr(model.specs[0], field) == "model-Q4_K_M.gguf"


def test_no_cross_vendor_recipe(entry, verified_io):
    entry.card["specs"][0].pop("gpu_filters")
    with pytest.raises(ValueError, match="GPU vendor"):
        validate.check_runner(validate.check_card(entry).specs[0], entry.specs[0])


@pytest.mark.parametrize("version", [None, "", "0.17.1", "0.20.0"])
def test_deployment_version_must_match_evidenced_runner(
    entry, verified_io, monkeypatch, version
):
    spec = entry.card["specs"][0]
    if version is None:
        spec.pop("backend_version")
    else:
        spec["backend_version"] = version
    inspect = Mock(return_value=SimpleNamespace(stdout='{"schemaVersion": 2}'))
    monkeypatch.setattr(validate.subprocess, "run", inspect)
    model = validate.check_card(entry)
    if version == "0.20.0":
        result = validate.check_runner(model.specs[0], entry.specs[0])
        assert result["service_version"] == spec["backend_version"]
        inspect.assert_called_once()
    else:
        message = (
            "Pin backend_version" if not version else "No matching packaged runner"
        )
        with pytest.raises(ValueError, match=message):
            validate.check_runner(model.specs[0], entry.specs[0])
        inspect.assert_not_called()


@pytest.mark.parametrize(
    "image,proof_image,quote,error",
    [
        (
            "vllm/vllm-openai:model",
            "vllm/vllm-openai:model",
            "Use vllm/vllm-openai:model for this checkpoint.",
            None,
        ),
        (
            "vllm/vllm-openai:model",
            "vllm/vllm-openai:other",
            "Use vllm/vllm-openai:model for this checkpoint.",
            "differs from proof",
        ),
        (
            "vllm/vllm-openai:model",
            "vllm/vllm-openai:model",
            "Use a model-specific image for this checkpoint.",
            "must name",
        ),
        (
            "untrusted/vllm:model",
            "untrusted/vllm:model",
            "Use untrusted/vllm:model for this checkpoint.",
            "official publisher",
        ),
        (
            "vllm/vllm-openai",
            "vllm/vllm-openai",
            "Use vllm/vllm-openai for this checkpoint.",
            "explicitly tagged",
        ),
    ],
)
def test_official_image_requires_matching_evidence_without_backend_version(
    entry, monkeypatch, image, proof_image, quote, error
):
    spec = entry.card["specs"][0]
    spec.pop("backend_version")
    spec["image_name"] = image
    proof = entry.specs[0]
    proof.runner_image = proof_image
    proof.deployment[0].quote = quote
    inspect = Mock(return_value=SimpleNamespace(stdout='{"schemaVersion": 2}'))
    monkeypatch.setattr(validate.subprocess, "run", inspect)
    model = validate.check_card(entry)
    if error:
        with pytest.raises(ValueError, match=error):
            validate.check_runner(model.specs[0], proof)
        inspect.assert_not_called()
    else:
        result = validate.check_runner(model.specs[0], proof)
        assert result["image"] == model.specs[0].image_name
        assert result["service_version"] is None
        assert inspect.call_args.args[0] == ["docker", "manifest", "inspect", image]


def test_official_image_must_exist_in_registry(entry, monkeypatch):
    spec = entry.card["specs"][0]
    spec["image_name"] = "vllm/vllm-openai:model"
    proof = entry.specs[0]
    proof.runner_image = spec["image_name"]
    proof.deployment[0].quote = "Use vllm/vllm-openai:model for this checkpoint."
    monkeypatch.setattr(
        validate.subprocess,
        "run",
        Mock(side_effect=validate.subprocess.CalledProcessError(1, "docker")),
    )
    with pytest.raises(validate.subprocess.CalledProcessError):
        validate.check_runner(validate.check_card(entry).specs[0], proof)


def test_yaml_edit_preserves_anchors_and_unrelated_cards(entry):
    header = '.version: &version "0.20.0"\ndraft_models: []\nmodel_sets:\n'
    first = yaml.safe_dump([entry.card], sort_keys=False)
    tail = (
        '- name: Unrelated\n  description: "keep quoting"\n  specs:\n'
        '  - backend: vLLM\n    backend_version: *version\n'
        '    source: huggingface\n    huggingface_repo_id: org/other\n'
    )
    source = header + first + tail
    changed = copy.deepcopy(entry.card)
    changed["description"] = "Updated model description."
    output = validate.replace_cards(source, [changed])
    assert output.startswith(header)
    assert output.endswith(tail)
    assert (
        yaml.safe_load(output)["model_sets"][0]["description"] == changed["description"]
    )
    assert validate.replace_cards(output, [changed]) == output


def test_append_to_empty_catalog(entry):
    source = "draft_models: []\nmodel_sets: []\n"
    output = validate.replace_cards(source, [entry.card])
    assert yaml.safe_load(output)["model_sets"] == [entry.card]


@pytest.mark.parametrize("source", list(CATALOGS))
def test_render_against_packaged_catalog_preserves_all_other_content(entry, source):
    root = Path(__file__).parents[2]
    entry.source = source
    spec = entry.card["specs"][0]
    spec["source"] = source
    if source == "model_scope":
        spec["model_scope_model_id"] = spec.pop("huggingface_repo_id")
    rendered = validate.render_catalogs(root, [entry])
    before = (root / CATALOGS[source]).read_bytes()
    assert rendered[CATALOGS[source]].startswith(before)
    assert (
        len(yaml.safe_load(rendered[CATALOGS[source]])["model_sets"])
        == len(yaml.safe_load(before)["model_sets"]) + 1
    )


@pytest.mark.parametrize("source", list(CATALOGS))
def test_unrelated_checkpoint_cannot_replace_packaged_card_by_name(entry, source):
    root = Path(__file__).parents[2]
    entry.source = source
    entry.card["name"] = "Qwen3-0.6B"
    spec = entry.card["specs"][0]
    spec["source"] = source
    if source == "model_scope":
        spec["model_scope_model_id"] = spec.pop("huggingface_repo_id")
    before = (root / CATALOGS[source]).read_bytes()
    with pytest.raises(ValueError, match="belongs to unrelated repositories"):
        validate.render_catalogs(root, [entry])
    assert (root / CATALOGS[source]).read_bytes() == before


@pytest.mark.parametrize("source", list(CATALOGS))
def test_existing_checkpoint_can_update_in_place_but_cannot_be_renamed(
    entry, catalog_root, source
):
    entry.source = source
    spec = entry.card["specs"][0]
    spec["source"] = source
    if source == "model_scope":
        spec["model_scope_model_id"] = spec.pop("huggingface_repo_id")
    path = CATALOGS[source]
    (catalog_root / path).write_text(yaml.safe_dump({"model_sets": [entry.card]}))
    entry.card["description"] = "Updated description for the same checkpoint."
    rendered = validate.render_catalogs(catalog_root, [entry])
    cards = yaml.safe_load(rendered[path])["model_sets"]
    assert cards == [entry.card]
    assert cards[0]["specs"][0]["backend_version"] == "0.20.0"
    entry.card["name"] = "Renamed checkpoint"
    with pytest.raises(ValueError, match="Preserve the existing card name"):
        validate.render_catalogs(catalog_root, [entry])


@pytest.mark.parametrize("blocked_reason", ["name_collision", "automatic_version"])
def test_invalid_target_does_not_discard_other_validated_changes(
    proposal, catalog_root, client, verified_io, tmp_path, blocked_reason
):
    candidate = proposal.candidates[0]
    existing = copy.deepcopy(candidate.entries[0].card)
    existing["name"] = "Existing model"
    existing["specs"][0]["huggingface_repo_id"] = "other/existing"
    path = CATALOGS["huggingface"]
    (catalog_root / path).write_text(yaml.safe_dump({"model_sets": [existing]}))
    other = candidate.model_copy(deep=True)
    other.target = "hf:publisher/other"
    other.hubs["huggingface"].repo_id = "publisher/other"
    card = other.entries[0].card
    card["specs"][0]["huggingface_repo_id"] = "publisher/other"
    card["name"] = "Existing model" if blocked_reason == "name_collision" else "Other"
    if blocked_reason == "automatic_version":
        card["specs"][0].pop("backend_version")
    proposal.candidates.append(other)
    request = {"base_sha": "b" * 40, "targets": [c.target for c in proposal.candidates]}
    output = tmp_path / "validated"
    output.mkdir()
    result = run.validate(catalog_root, request, proposal, output, client)
    assert result["changed"] and result["partial"]
    reports = run.read_json(output / "report.json")["results"]
    assert [report["status"] for report in reports] == ["ready", "blocked"]
    reason = (
        "belongs to unrelated repositories"
        if blocked_reason == "name_collision"
        else "Pin backend_version"
    )
    assert reason in reports[1]["reason"]
    bundle = run.read_json(output / "validated.json")
    rendered = yaml.safe_load(base64.b64decode(bundle["files"][path]))["model_sets"]
    assert rendered == [existing, candidate.entries[0].card]


def test_partial_batch_preserves_verified_models(
    proposal, catalog_root, client, verified_io, tmp_path
):
    other = proposal.candidates[0].model_copy(deep=True)
    other.target = "hf:publisher/unverified"
    other.entries[0].card["specs"][0]["huggingface_repo_id"] = "publisher/unverified"
    other.hubs["huggingface"].repo_id = "publisher/unverified"
    other.entries[0].specs[0].snapshot = "f" * 64
    proposal.candidates.append(other)
    request = {"base_sha": "b" * 40, "targets": [c.target for c in proposal.candidates]}
    output = tmp_path / "result"
    output.mkdir()
    result = run.validate(catalog_root, request, proposal, output, client)
    assert result["changed"] and result["partial"]
    bundle = run.read_json(output / "validated.json")
    assert set(bundle["files"]) == {CATALOGS["huggingface"]}
    run.apply(catalog_root, bundle, result["sha256"])
    cards = yaml.safe_load((catalog_root / CATALOGS["huggingface"]).read_text())[
        "model_sets"
    ]
    assert len(cards) == 1
    reports = run.read_json(output / "report.json")["results"]
    assert [row["status"] for row in reports] == ["ready", "blocked"]
    assert "No GPU/NPU inference was executed" in (output / "pr-body.md").read_text()


def test_proposal_must_cover_requested_targets(
    proposal, catalog_root, client, verified_io, tmp_path
):
    with pytest.raises(ValueError, match="each requested target"):
        run.validate(
            catalog_root,
            {"base_sha": "b" * 40, "targets": ["hf:other/model"]},
            proposal,
            tmp_path,
            client,
        )


@pytest.mark.parametrize(
    "path",
    ["../escape.png", ".github/workflows/evil.yml", "static/catalog_icons/../evil.png"],
)
def test_publication_rejects_paths_even_in_a_digest_valid_bundle(
    path, tmp_path, verified_io
):
    bundle = {
        "request": {"base_sha": "b" * 40},
        "files": {path: base64.b64encode(b"x").decode()},
        "before": {path: None},
    }
    checksum = digest(bundle)
    bundle["sha256"] = checksum
    with pytest.raises(ValueError, match="allowlist"):
        run.apply(tmp_path, bundle, checksum)


def test_publication_rejects_tampered_artifact(tmp_path, verified_io):
    bundle = {"request": {"base_sha": "b" * 40}, "files": {}, "before": {}}
    checksum = digest(bundle)
    bundle.update(sha256=checksum, files={"unexpected": "x"})
    with pytest.raises(ValueError, match="digest"):
        run.apply(tmp_path, bundle, checksum)


def test_publication_refuses_new_local_changes(tmp_path, verified_io):
    path = CATALOGS["huggingface"]
    destination = tmp_path / path
    destination.parent.mkdir(parents=True)
    destination.write_text("Local edits must survive")
    bundle = {
        "request": {"base_sha": "b" * 40},
        "files": {path: base64.b64encode(b"new").decode()},
        "before": {path: "0" * 64},
    }
    checksum = digest(bundle)
    bundle["sha256"] = checksum
    with pytest.raises(ValueError, match="overwrite changed"):
        run.apply(tmp_path, bundle, checksum)
    assert destination.read_text() == "Local edits must survive"


def test_official_github_icon_is_downloaded_as_local_png(entry, catalog_root, client):
    image = io.BytesIO()
    Image.new("RGB", (512, 512), "red").save(image, format="JPEG")
    citation = {
        "url": "https://example.org/model",
        "quote": "The publisher's official GitHub organization is example-publisher",
        "explanation": "The model publisher links this organization.",
    }
    entry.icon = Entry.model_validate(
        {
            **entry.model_dump(),
            "icon": {
                "filename": "publisher.png",
                "url": "https://avatars.githubusercontent.com/u/123",
                "github_org": "example-publisher",
                "evidence": [citation],
                "reason": "No matching local icon exists for this publisher.",
            },
        }
    ).icon
    entry.card["icon"] = "/static/catalog_icons/publisher.png"
    client.json.return_value = {"type": "Organization", "avatar_url": entry.icon.url}
    client.get.side_effect = [
        (FIXTURES / "model-card.md").read_bytes(),
        image.getvalue(),
    ]
    path, data = validate.acquire_icon(client, entry, catalog_root)
    assert path == "static/catalog_icons/publisher.png"
    with Image.open(io.BytesIO(data)) as picture:
        assert picture.format == "PNG" and picture.size == (256, 256)


@pytest.mark.parametrize(
    "url",
    [
        "http://example.org",
        "https://user:password@example.org",
        "https://127.0.0.1",
        "https://[::1]",
    ],
)
def test_fetch_rejects_nonpublic_destinations(url, monkeypatch):
    monkeypatch.setattr(
        "hack.catalog_sync.http.socket.getaddrinfo",
        lambda *a, **kw: [(None, None, None, None, ("127.0.0.1", 443))],
    )
    with pytest.raises(ValueError):
        public_url(url)


@pytest.mark.parametrize("address", ["198.18.0.1", "198.19.255.254"])
def test_fake_ip_proxy_requires_explicit_exact_host(monkeypatch, address):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("CATALOG_FAKE_IP_HOSTS", raising=False)
    monkeypatch.setattr(
        "hack.catalog_sync.http.socket.getaddrinfo",
        lambda *a, **kw: [(None, None, None, None, (address, 443))],
    )
    with pytest.raises(ValueError, match="non-public address"):
        public_url("https://huggingface.co/model")
    monkeypatch.setenv("CATALOG_FAKE_IP_HOSTS", " HuggingFace.co, api.github.com ")
    public_url("https://huggingface.co/model")
    with pytest.raises(ValueError, match="non-public address"):
        public_url("https://huggingface.co.example.org/model")
    with pytest.raises(ValueError, match="non-public address"):
        public_url("https://cdn.huggingface.co/model")


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.0.0.1",
        "192.168.1.1",
        "169.254.169.254",
        "198.51.100.1",
        "::1",
        "fc00::1",
    ],
)
def test_fake_ip_option_does_not_allow_other_reserved_addresses(monkeypatch, address):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv("CATALOG_FAKE_IP_HOSTS", "huggingface.co")
    monkeypatch.setattr(
        "hack.catalog_sync.http.socket.getaddrinfo",
        lambda *a, **kw: [(None, None, None, None, (address, 443))],
    )
    with pytest.raises(ValueError, match="non-public address"):
        public_url("https://huggingface.co/model")


@pytest.mark.parametrize(
    "hosts",
    [
        "*",
        "*.huggingface.co",
        "https://huggingface.co",
        "huggingface.co:443",
        "198.18.0.1",
        "huggingface.co,",
    ],
)
def test_fake_ip_option_rejects_ambiguous_host_configuration(monkeypatch, hosts):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv("CATALOG_FAKE_IP_HOSTS", hosts)
    with pytest.raises(ValueError, match="exact hostnames"):
        public_url("https://huggingface.co")


def test_fake_ip_option_is_unavailable_in_github_actions(monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("CATALOG_FAKE_IP_HOSTS", "huggingface.co")
    with pytest.raises(ValueError, match="local testing only"):
        public_url("https://huggingface.co")


def test_redirect_cannot_extend_fake_ip_host_exception(monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv("CATALOG_FAKE_IP_HOSTS", "huggingface.co")
    monkeypatch.setattr(
        "hack.catalog_sync.http.socket.getaddrinfo",
        lambda *a, **kw: [(None, None, None, None, ("198.18.0.1", 443))],
    )
    response = MagicMock()
    response.__enter__.return_value = response
    response.is_redirect = True
    response.headers = {"Location": "https://unlisted.example.org/file"}
    client = PublicClient()
    fetch = Mock(return_value=response)
    monkeypatch.setattr(client.session, "get", fetch)
    with pytest.raises(ValueError, match="unlisted.example.org"):
        client.get("https://huggingface.co/model")
    assert fetch.call_count == 1


@pytest.fixture
def research_network(monkeypatch):
    monkeypatch.setattr(agent, "check_network", lambda: None)


def test_research_exports_only_typed_proposal(
    tmp_path, proposal, monkeypatch, research_network
):
    root = Path(__file__).parents[2]
    for key, value in {
        "CATALOG_LLM_MODEL": "test",
        "CATALOG_LLM_URL": "https://llm.test/v1",
        "CATALOG_LLM_TOKEN": "secret-token",
        "GITHUB_TOKEN": "write-token",
        "GH_TOKEN": "gh-write-token",
        "AWS_SECRET_ACCESS_KEY": "cloud-secret",
        "SSH_AUTH_SOCK": "/tmp/host-agent.sock",
        "CATALOG_FAKE_IP_HOSTS": "huggingface.co,api.github.com",
    }.items():
        monkeypatch.setenv(key, value)
    observed = []

    def fake_run(workspace, env, prompt, deadline):
        assert "GITHUB_TOKEN" not in env
        assert not {"GH_TOKEN", "AWS_SECRET_ACCESS_KEY", "SSH_AUTH_SOCK"} & env.keys()
        assert "write-token" not in json.dumps(env)
        assert env["CATALOG_FAKE_IP_HOSTS"] == "huggingface.co,api.github.com"
        assert env["GIT_CONFIG_GLOBAL"] == "/dev/null"
        assert env["GIT_CONFIG_VALUE_0"] == "/dev/null"
        home = Path(env["HOME"])
        assert home.stat().st_mode & 0o777 == 0o700
        assert (home / ".qwen/settings.json").stat().st_mode & 0o777 == 0o600
        assert (workspace / ".github/skills/model-catalog-update/SKILL.md").is_file()
        (workspace / "proposal.json").write_text(proposal.model_dump_json())
        (workspace / "untrusted.py").write_text("raise Exception('never import')")
        observed.append(prompt)

    monkeypatch.setattr(agent, "_run", fake_run)
    agent.research(root, {"targets": ["hf:publisher/example"]}, tmp_path)
    assert observed
    assert sorted(p.name for p in tmp_path.iterdir()) == ["proposal.json"]


def test_repair_restores_settings_and_removes_workspace_startup_files(
    tmp_path, proposal, monkeypatch, research_network
):
    root = Path(__file__).parents[2]
    monkeypatch.setenv("CATALOG_LLM_MODEL", "test")
    monkeypatch.setenv("CATALOG_LLM_URL", "https://llm.test/v1")
    monkeypatch.setenv("CATALOG_LLM_TOKEN", "secret-token")
    attempts = []

    def fake_run(workspace, env, prompt, deadline):
        config = Path(env["HOME"]) / ".qwen/settings.json"
        assert json.loads(config.read_text()) == agent.settings(
            "test", "https://llm.test/v1"
        )
        assert all(not (workspace / name).exists() for name in agent.STARTUP_CONFIG)
        if not attempts:
            config.write_text('{"untrusted": true}')
            (workspace / ".mcp.json").write_text('{"untrusted": true}')
            (workspace / "proposal.json").write_text('{}')
        else:
            (workspace / "proposal.json").write_text(proposal.model_dump_json())
        attempts.append(deadline)

    monkeypatch.setattr(agent, "_run", fake_run)
    agent.research(root, {"targets": ["hf:publisher/example"]}, tmp_path)
    assert len(attempts) == 2 and attempts[0] == attempts[1]
    assert run.read_json(tmp_path / "proposal.json")["candidates"]


def test_research_failure_redacts_credentials(tmp_path, monkeypatch, research_network):
    root = Path(__file__).parents[2]
    monkeypatch.setenv("CATALOG_LLM_MODEL", "test")
    monkeypatch.setenv("CATALOG_LLM_URL", "https://llm.test/v1")
    monkeypatch.setenv("CATALOG_LLM_TOKEN", "secret-token")

    def failed_run(*args):
        raise ValueError("Authentication failed for secret-token")

    monkeypatch.setattr(agent, "_run", failed_run)
    with pytest.raises(ValueError, match="research-error.txt"):
        agent.research(root, {"targets": ["hf:publisher/example"]}, tmp_path)
    diagnostic = (tmp_path / "research-error.txt").read_text()
    assert "secret-token" not in diagnostic and "[REDACTED]" in diagnostic


def test_provider_error_is_reported_without_reasoning_or_tool_transcripts():
    transcript = "\n".join(
        [
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "thinking", "thinking": "private reasoning"}
                        ]
                    },
                }
            ),
            json.dumps({"type": "user", "message": {"content": "private tool output"}}),
            json.dumps(
                {
                    "type": "result",
                    "error": {
                        "message": "API Error: 429 insufficient balance secret-token"
                    },
                }
            ),
        ]
    )
    summary = agent.failure_summary(transcript, 1, "secret-token")
    assert "429 insufficient balance" in summary
    assert "[REDACTED]" in summary
    assert "private" not in summary and "secret-token" not in summary


def test_missing_structured_error_does_not_dump_agent_output():
    assert "private" not in agent.failure_summary(
        'truncated private reasoning', 1, "token"
    )


def test_research_deadline_terminates_supervised_process(tmp_path, monkeypatch):
    executable = tmp_path / "qwen"
    pid_file = tmp_path / "research.pid"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, time\n"
        f"pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    executable.chmod(0o700)
    monkeypatch.setattr(agent.shutil, "which", lambda name: str(executable))
    monkeypatch.setenv("CATALOG_LLM_MODEL", "test")
    with pytest.raises(ValueError, match="deadline"):
        agent._run(
            tmp_path,
            {"PATH": os.environ["PATH"], "CATALOG_LLM_TOKEN": "test-secret"},
            "Offline supervision test",
            time.monotonic() + 5,
        )
    assert pid_file.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_preflight_stops_before_spending_llm_requests(tmp_path, monkeypatch):
    for key in ("CATALOG_LLM_MODEL", "CATALOG_LLM_URL", "CATALOG_LLM_TOKEN"):
        monkeypatch.setenv(key, "test")

    def reserved_address(url):
        raise ValueError("Evidence URL resolves to a non-public address")

    monkeypatch.setattr(agent, "public_url", reserved_address)
    call = Mock()
    monkeypatch.setattr(agent, "_run", call)
    with pytest.raises(ValueError, match="huggingface.co.*non-public address"):
        agent.research(tmp_path, {"targets": ["hf:org/model"]}, tmp_path)
    call.assert_not_called()


def test_preview_stops_after_research_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("CATALOG_MODEL_IDS", "hf:org/model")
    monkeypatch.setattr(run, "git_head", lambda root: "a" * 40)
    monkeypatch.setattr(
        agent,
        "research",
        Mock(side_effect=ValueError("API Error: 429 insufficient balance")),
    )
    validator = Mock()
    monkeypatch.setattr(run, "validate", validator)
    with pytest.raises(ValueError, match="429 insufficient balance"):
        run.preview(tmp_path, tmp_path)
    validator.assert_not_called()
    assert not (tmp_path / "validated").exists()


def test_preview_refuses_stale_output(tmp_path, monkeypatch):
    (tmp_path / "proposal.json").write_text("stale result")
    call = Mock()
    monkeypatch.setattr(agent, "research", call)
    with pytest.raises(ValueError, match="new empty"):
        run.preview(tmp_path, tmp_path)
    call.assert_not_called()


@pytest.mark.parametrize("blocked", [False, True])
def test_preview_reports_validation_without_applying(
    tmp_path, monkeypatch, proposal, capsys, blocked
):
    monkeypatch.setenv("CATALOG_MODEL_IDS", "hf:publisher/example")
    monkeypatch.setattr(run, "git_head", lambda root: "a" * 40)

    def research(root, request, output):
        (output / "proposal.json").write_text(proposal.model_dump_json())

    def validate(root, request, proposed, output):
        assert proposed == proposal
        (output / "pr-body.md").write_text("Review this proposal")
        run.write_json(
            output / "report.json",
            {
                "results": [
                    {
                        "target": "hf:publisher/example",
                        "status": "blocked" if blocked else "ready",
                        "reason": "Non-public address: cdn-thumbnails.huggingface.co",
                    }
                ]
            },
        )
        return {"changed": not blocked, "partial": blocked}

    monkeypatch.setattr(agent, "research", research)
    monkeypatch.setattr(run, "validate", validate)
    apply = Mock()
    monkeypatch.setattr(run, "apply", apply)
    if blocked:
        with pytest.raises(ValueError, match="No changes passed validation"):
            run.preview(tmp_path, tmp_path)
        assert (
            "Blocked hf:publisher/example: Non-public address: "
            "cdn-thumbnails.huggingface.co"
        ) in capsys.readouterr().out
    else:
        assert run.preview(tmp_path, tmp_path)["changed"]
        assert "Blocked" not in capsys.readouterr().out
    assert (tmp_path / "validated/pr-body.md").read_text() == "Review this proposal"
    apply.assert_not_called()


def test_cli_reports_failure_without_traceback(monkeypatch, capsys):
    monkeypatch.setattr(
        run,
        "_main",
        Mock(side_effect=ValueError("API Error: 429 insufficient balance")),
    )
    with pytest.raises(SystemExit) as failure:
        run.main()
    assert failure.value.code == 1
    error = capsys.readouterr().err
    assert "429 insufficient balance" in error and "Traceback" not in error


def test_report_never_closes_issues_or_mentions_accounts():
    result = {
        "target": "hf:org/model",
        "status": "blocked",
        "reason": "Fixes #123 @person",
        "hubs": {},
        "entries": [],
        "ascend_910b": "No public evidence yet.",
    }
    report = run.report_markdown({"base_sha": "a" * 40}, [result])
    assert "ref #123" in report
    assert "Fixes #123" not in report and "@person" not in report
