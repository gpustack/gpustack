import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml
from transformers import dynamic_module_utils
from transformers.models.auto import configuration_auto

from gpustack.policies import utils as policy_utils
from gpustack.policies.candidate_selectors import (
    SGLangResourceFitSelector,
    VLLMResourceFitSelector,
)
from gpustack.policies.candidate_selectors.base_candidate_selector import (
    ModelParameters,
)
from gpustack.policies.resource_view import ResourceView
from gpustack.scheduler import calculator, evaluator, scheduler
from gpustack.schemas.models import Model, SourceEnum, is_omni_model
from gpustack.utils.model_config import safe_pretrained_config_from_dict
from gpustack.utils.hybrid_attention import is_hybrid_attention
from tests.fixtures.workers.fixtures import linux_nvidia_22_H100_80gx8

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "model_configs" / "catalog"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())
CASES = MANIFEST["models"]


@pytest.fixture(scope="module")
def catalog_sets():
    catalog = yaml.safe_load((ROOT / MANIFEST["catalog"]).read_text())
    return {item["name"]: item for item in catalog["model_sets"]}


@pytest.fixture(autouse=True)
def offline_metadata_only(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", True)

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Catalog evaluation must not load repository Python or use the network"
        )

    monkeypatch.setattr(
        dynamic_module_utils, "get_class_from_dynamic_module", forbidden
    )
    monkeypatch.setattr(configuration_auto, "get_class_from_dynamic_module", forbidden)
    monkeypatch.setattr("httpx.Client.send", forbidden)
    monkeypatch.setattr("requests.Session.request", forbidden)


@pytest.fixture
def catalog_model(request, catalog_sets, tmp_path):
    case = request.param
    model_set = catalog_sets[case["name"]]
    specs = [
        spec
        for spec in model_set["specs"]
        if spec.get("huggingface_repo_id") == case["repo_id"]
        and spec.get("backend") == case["backend"]
    ]
    specs.sort(
        key=lambda spec: (
            spec.get("gpu_filters", {}).get("vendor") == "ascend",
            spec.get("mode") != "standard",
        )
    )
    assert specs, f"Catalog recipe missing for {case['name']}"
    spec = dict(specs[0])
    spec.update(
        name=case["name"], source=SourceEnum.LOCAL_PATH, local_path=str(tmp_path)
    )
    shutil.copy(FIXTURES / case["file"], tmp_path / "config.json")
    return case, Model(**spec)


def test_catalog_matrix_has_thirty_distinct_architectures(catalog_sets):
    assert len(CASES) == 30
    assert len({tuple(case["architectures"]) for case in CASES}) == 30
    dates = [case["release_date"] for case in CASES]
    assert dates == sorted(dates, reverse=True)
    for case in CASES:
        model_set = catalog_sets[case["name"]]
        assert model_set["release_date"] == case["release_date"]
        assert model_set["categories"] == case["categories"]
        content = (FIXTURES / case["file"]).read_bytes()
        assert hashlib.sha256(content).hexdigest() == case["sha256"]
        assert json.loads(content)["architectures"] == case["architectures"]


@pytest.mark.parametrize(
    "catalog_model", CASES, indirect=True, ids=lambda case: case["name"]
)
@pytest.mark.parametrize("location", ["server", "worker"])
@pytest.mark.asyncio
async def test_catalog_metadata_preserves_evaluation_fields(
    catalog_model, location, tmp_path, monkeypatch
):
    case, model = catalog_model
    raw = json.loads((FIXTURES / case["file"]).read_text())
    if location == "worker":
        model.local_path = str(tmp_path / "worker-only")
        monkeypatch.setattr(
            calculator, "read_local_path_file_from_workers", AsyncMock(return_value=raw)
        )
        monkeypatch.setattr(
            scheduler,
            "check_diffusers_model_index_from_workers",
            AsyncMock(return_value=False),
        )
    await scheduler.evaluate_pretrained_config(model)
    assert model.categories == case["categories"]
    parsed = await calculator.get_pretrained_config_with_workers(model)
    assert parsed.architectures == case["architectures"]
    assert is_hybrid_attention(parsed) == case["expected"]["hybrid_attention"]
    if "quantization_config" in raw:
        assert parsed.quantization_config == raw["quantization_config"]

    if is_omni_model(model):
        # The TTS runtime owns the talker; the selector intentionally skips LLM shapes.
        parsed = safe_pretrained_config_from_dict(raw["talker_config"])
    params = ModelParameters()
    params.from_model_pretrained_config(model, parsed)
    if not is_omni_model(model):
        assert params.architectures == case["architectures"]
    for field, expected in case["expected"].items():
        if field != "hybrid_attention":
            assert getattr(params, field) == expected, field


@pytest.mark.parametrize(
    "catalog_model", CASES, indirect=True, ids=lambda case: case["name"]
)
@pytest.mark.asyncio
async def test_catalog_recipe_produces_candidates(config, catalog_model, monkeypatch):
    case, model = catalog_model
    valid, messages = await evaluator.evaluate_model_metadata(config, model, [])
    assert valid, messages
    # Audio recipes can supply categories before the metadata evaluator is called.
    await scheduler.evaluate_pretrained_config(model)
    assert model.categories == case["categories"]

    selector_class = (
        SGLangResourceFitSelector
        if model.backend == "SGLang"
        else VLLMResourceFitSelector
    )
    worker = linux_nvidia_22_H100_80gx8()
    model.backend_parameters = [
        *(model.backend_parameters or []),
        "--tensor-parallel-size=2",
    ]
    # Fix only the filesystem weight-size boundary; exercise the real VRAM estimator.
    weight_size = AsyncMock(return_value=20 * 1024**3)
    monkeypatch.setattr(policy_utils, "get_local_model_weight_size", weight_size)
    selector = selector_class(config, model, [], resource_view=ResourceView([]))
    candidates = await selector.select_candidates([worker])
    weight_size.assert_awaited()
    assert candidates, selector._messages
    assert all(len(candidate.gpu_indexes) == 2 for candidate in candidates)
    assert all(
        all(value > 0 for value in candidate.computed_resource_claim.vram.values())
        for candidate in candidates
    )
    for gpu in worker.status.gpu_devices:
        gpu.memory.total = 1024**3
    selector = selector_class(config, model, [], resource_view=ResourceView([]))
    assert await selector.select_candidates([worker]) == []
    assert selector._messages
    if not is_omni_model(model):
        assert selector._num_attention_heads == case["expected"]["num_attention_heads"]
        model.backend_parameters[-1] = "--tensor-parallel-size=5"
        selector = selector_class(config, model, [], resource_view=ResourceView([]))
        with pytest.raises(ValueError, match="attention heads"):
            await selector.select_candidates([worker])
