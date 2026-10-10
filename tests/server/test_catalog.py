import os
import time
import pytest
from tenacity import retry, stop_after_attempt, wait_fixed
from gpustack.routes.model_sets import filter_specs_by_gpu
from gpustack.schemas.catalog_source import (
    KIND_MODEL_SET,
    build_catalog_entries,
    normalize_catalog_yaml,
)
from gpustack.schemas.model_sets import ModelSet
from gpustack.schemas.gpu_devices import GPUDevice
from gpustack.schemas.models import CategoryEnum, ModelCreate, SourceEnum
from gpustack.server.catalog import read_builtin_catalog_text
from gpustack.schemas.source import SourceContent, SourceTypeEnum
from gpustack.utils.hub import match_hugging_face_files, match_model_scope_file_paths
from gpustack.utils.compat_importlib import pkg_resources
from gpustack.utils.command import resolve_executor_backend
from gpustack.utils.vllm_topology import (
    parse_user_parallelism,
    validate_multinode_topology,
)
from huggingface_hub import HfApi
from modelscope.hub.api import HubApi


def _packaged_model_sets(catalog_file=None):
    """Model set name -> model set, loaded from the packaged catalog via the source
    pipeline (no DB). Mirrors what CatalogSourceController materializes."""
    content = normalize_catalog_yaml(read_builtin_catalog_text(catalog_file))
    entries = build_catalog_entries(
        [SourceContent("builtin", SourceTypeEnum.BUILTIN, content)]
    )
    return {
        entry.name: ModelSet(**entry.payload)
        for entry in entries
        if entry.kind == KIND_MODEL_SET
    }


def _packaged_model_set_specs(catalog_file=None):
    return {
        name: model_set.specs
        for name, model_set in _packaged_model_sets(catalog_file).items()
    }


@pytest.mark.parametrize(
    "catalog_file", ["model-catalog.yaml", "model-catalog-modelscope.yaml"]
)
def test_packaged_catalog_model_release_dates(catalog_file):
    catalog_path = pkg_resources.files("gpustack.assets").joinpath(catalog_file)
    model_sets = _packaged_model_sets(str(catalog_path))
    # Dates identify the open-weight releases of these specific checkpoints.
    release_dates = {
        "Kimi-K3": "2026-07-27",
        "DeepSeek-V4.1-Flash": "2026-09-10",
        "GLM-5.3": "2026-08-28",
        "GLM-5.3-Flash": "2026-08-26",
        "Muse-Glimmer-30B": "2026-08-10",
        "Qwen3.8-27B": "2026-08-14",
        "MiMo-V2.6-Pro-MOPD": "2026-09-27",
    }
    for name, expected in release_dates.items():
        assert model_sets[name].release_date is not None
        assert model_sets[name].release_date.isoformat() == expected


@pytest.mark.parametrize(
    "catalog_file,source,repo_field,organizations",
    [
        (
            "model-catalog.yaml",
            SourceEnum.HUGGING_FACE,
            "huggingface_repo_id",
            ["moonshotai", "deepseek-ai", "zai-org", "meta-models"],
        ),
        (
            "model-catalog-modelscope.yaml",
            SourceEnum.MODEL_SCOPE,
            "model_scope_model_id",
            ["MoonshotAI", "deepseek-ai", "ZhipuAI", "meta-models"],
        ),
    ],
)
def test_packaged_agentic_model_specs(catalog_file, source, repo_field, organizations):
    catalog_path = pkg_resources.files("gpustack.assets").joinpath(catalog_file)
    model_sets = _packaged_model_sets(str(catalog_path))
    models = [
        ("Kimi-K3", "kimi_k3", "MXFP4"),
        ("DeepSeek-V4.1-Flash", "deepseek_v41", "MXFP4"),
        ("GLM-5.3", "glm47", "FP8"),
        ("Muse-Glimmer-30B", "muse_glimmer", "BF16"),
    ]
    for (name, parser, quantization), organization in zip(models, organizations):
        model_set = model_sets[name]
        assert model_set.categories == [CategoryEnum.LLM]
        assert "reasoning" not in model_set.capabilities
        specs = filter_specs_by_gpu(
            [GPUDevice(vendor="nvidia", compute_capability="10.0")], model_set.specs
        )
        assert specs
        for spec in specs:
            assert spec.source == source
            assert getattr(spec, repo_field) == f"{organization}/{name}"
            assert spec.quantization == quantization
            assert spec.backend == "vLLM"
            assert spec.backend_version is None
            assert f"--tool-call-parser={parser}" in spec.backend_parameters
            assert f"--reasoning-parser={parser}" in spec.backend_parameters
            assert "--enable-auto-tool-choice" in spec.backend_parameters
            if name.startswith("DeepSeek"):
                assert "--tokenizer-mode=deepseek_v41" in spec.backend_parameters
            elif name == "GLM-5.3":
                assert "--kv-cache-dtype=fp8" in spec.backend_parameters

    deepseek = model_sets["DeepSeek-V4.1-Flash"]
    assert "vision" in deepseek.capabilities
    assert deepseek.size == 552
    assert deepseek.activated_size == 16
    assert model_sets["GLM-5.3"].size == 743
    assert model_sets["Muse-Glimmer-30B"].size == 30
    assert not filter_specs_by_gpu(
        [GPUDevice(vendor="ascend", arch_family="Ascend910_9391")],
        model_sets["Muse-Glimmer-30B"].specs,
    )
    assert "DeepSeek-V4-Flash-0731" not in model_sets
    assert "DeepSeek-V4-Pro-0813" not in model_sets


@pytest.mark.parametrize(
    "catalog_file,source,repo_field,glm_organization",
    [
        (
            "model-catalog.yaml",
            SourceEnum.HUGGING_FACE,
            "huggingface_repo_id",
            "zai-org",
        ),
        (
            "model-catalog-modelscope.yaml",
            SourceEnum.MODEL_SCOPE,
            "model_scope_model_id",
            "ZhipuAI",
        ),
    ],
)
@pytest.mark.parametrize(
    "name,organization,size,size_unit,activated_size,quantization,context,vision,reasoning,tools",
    [
        (
            "GLM-5.3-Flash",
            None,
            320,
            None,
            18,
            "FP8",
            "1M",
            True,
            "glm47",
            "glm47",
        ),
        (
            "Qwen3.8-27B",
            "Qwen",
            27,
            None,
            None,
            "BF16",
            "256K",
            True,
            "qwen3",
            "qwen3_coder",
        ),
        (
            "MiMo-V2.6-Pro-MOPD",
            "XiaomiMiMo",
            1.02,
            "T",
            42,
            "FP8",
            "1M",
            True,
            "mimo",
            "mimo",
        ),
    ],
)
def test_packaged_additional_model_specs(
    catalog_file,
    source,
    repo_field,
    glm_organization,
    name,
    organization,
    size,
    size_unit,
    activated_size,
    quantization,
    context,
    vision,
    reasoning,
    tools,
):
    catalog_path = pkg_resources.files("gpustack.assets").joinpath(catalog_file)
    model_set = _packaged_model_sets(str(catalog_path))[name]
    assert model_set.size == size
    assert model_set.size_unit == size_unit
    assert model_set.activated_size == activated_size
    assert model_set.categories == [CategoryEnum.LLM]
    assert f"context/{context}" in model_set.capabilities
    assert ("vision" in model_set.capabilities) is vision
    assert "tools" in model_set.capabilities
    assert "reasoning" not in model_set.capabilities
    specs = filter_specs_by_gpu(
        [GPUDevice(vendor="nvidia", compute_capability="10.0")], model_set.specs
    )
    assert specs
    for spec in specs:
        assert spec.source == source
        assert getattr(spec, repo_field) == f"{organization or glm_organization}/{name}"
        assert spec.quantization == quantization
        assert spec.backend == "vLLM"
        assert spec.backend_version is None
        assert not spec.gpu_filters.vendor_variant
        assert "--max-model-len=65536" in spec.backend_parameters
        assert f"--reasoning-parser={reasoning}" in spec.backend_parameters
        assert f"--tool-call-parser={tools}" in spec.backend_parameters
        assert "--enable-auto-tool-choice" in spec.backend_parameters
        assert not any("parallel-size" in arg for arg in spec.backend_parameters)
        if name == "MiMo-V2.6-Pro-MOPD":
            assert "--trust-remote-code" in spec.backend_parameters
            assert "--gpu-memory-utilization=0.95" in spec.backend_parameters
            assert "--generation-config=vllm" in spec.backend_parameters


@pytest.mark.parametrize(
    "catalog_file", ["model-catalog.yaml", "model-catalog-modelscope.yaml"]
)
@pytest.mark.parametrize("compute_capability", ["8.0", "9.0", "10.0"])
def test_packaged_glm53_flash_hardware_spec(catalog_file, compute_capability):
    catalog_path = pkg_resources.files("gpustack.assets").joinpath(catalog_file)
    model_set = _packaged_model_sets(str(catalog_path))["GLM-5.3-Flash"]
    specs = filter_specs_by_gpu(
        [GPUDevice(vendor="nvidia", compute_capability=compute_capability)],
        model_set.specs,
    )
    if compute_capability == "8.0":
        assert not specs
        return
    assert len(specs) == 1
    parameters = specs[0].backend_parameters
    if compute_capability == "9.0":
        assert "--no-enable-flashinfer-autotune" in parameters
        assert not any(arg.startswith("--kv-cache-dtype") for arg in parameters)
    else:
        assert "--kv-cache-dtype=fp8" in parameters


@pytest.mark.parametrize(
    "catalog_file", ["model-catalog.yaml", "model-catalog-modelscope.yaml"]
)
@pytest.mark.parametrize("compute_capability", ["8.0", "9.0", "10.0"])
def test_packaged_kimi_k3_hardware_spec(catalog_file, compute_capability):
    catalog_path = pkg_resources.files("gpustack.assets").joinpath(catalog_file)
    model_set = _packaged_model_sets(str(catalog_path))["Kimi-K3"]
    specs = filter_specs_by_gpu(
        [GPUDevice(vendor="nvidia", compute_capability=compute_capability)],
        model_set.specs,
    )
    if compute_capability == "8.0":
        assert not specs
        return

    assert len(specs) == 1
    spec = specs[0]
    assert spec.quantization == "MXFP4"
    assert spec.backend_version is None
    assert not any(
        arg.startswith("--tokenizer-mode") for arg in spec.backend_parameters
    )
    assert not any("parallel-size" in arg for arg in spec.backend_parameters)
    assert "--max-model-len=65536" in spec.backend_parameters
    assert not any(
        arg.startswith(("--max-num-seqs", "--max-num-batched-tokens"))
        for arg in spec.backend_parameters
    )
    if compute_capability == "9.0":
        assert "--moe-backend=marlin" in spec.backend_parameters
        assert "--attention-backend=FLASHMLA" in spec.backend_parameters
        assert "--gpu-memory-utilization=0.95" in spec.backend_parameters
        assert "--tool-call-parser=kimi_k3" in spec.backend_parameters
        assert "--reasoning-parser=kimi_k3" in spec.backend_parameters
        assert "--enable-auto-tool-choice" in spec.backend_parameters
    else:
        assert "--moe-backend=marlin" not in spec.backend_parameters


@pytest.mark.parametrize("arch_family", [None, "Ascend910B3"])
def test_packaged_glm53_ascend_spec(arch_family):
    catalog_path = pkg_resources.files("gpustack.assets").joinpath(
        "model-catalog-modelscope.yaml"
    )
    model_set = _packaged_model_sets(str(catalog_path))["GLM-5.3"]
    specs = filter_specs_by_gpu(
        [GPUDevice(vendor="ascend", arch_family=arch_family)], model_set.specs
    )
    assert len(specs) == 1
    spec = specs[0]
    assert spec.gpu_filters.vendor == ["ascend"]
    assert not spec.gpu_filters.vendor_variant
    assert spec.source == SourceEnum.MODEL_SCOPE
    assert spec.model_scope_model_id == "Eco-Tech/GLM-5.3-w8a8c8"
    assert spec.quantization == "W8A8"
    assert spec.backend == "vLLM"
    assert spec.backend_version is None
    assert spec.image_name == "quay.io/ascend/vllm-ascend:v0.23.0"
    assert "--quantization=ascend" in spec.backend_parameters
    assert "--enable-expert-parallel" in spec.backend_parameters
    assert not any("parallel-size" in arg for arg in spec.backend_parameters)
    assert not any(
        arg.startswith("--additional-config") for arg in spec.backend_parameters
    )
    assert "--enable-auto-tool-choice" in spec.backend_parameters
    assert not any("fp8" in arg for arg in spec.backend_parameters)
    assert "--kv-cache-dtype=int8" in spec.backend_parameters
    assert "--reasoning-parser=glm45" in spec.backend_parameters
    assert "--tool-call-parser=glm47" in spec.backend_parameters


@pytest.mark.parametrize(
    "name,repo_id,image_tag,reasoning",
    [
        ("Qwen3.8-27B", "Qwen3.8-27B-w8a8", "qwen3.8-a2", "qwen3"),
        ("GLM-5.3-Flash", "GLM-5.3-Flash-w8a8", "glm-5.3-flash", "glm45"),
        ("GLM-5.3", "GLM-5.3-w8a8c8", "v0.23.0", "glm45"),
        (
            "DeepSeek-V4.1-Flash",
            "DeepSeek-V4.1-Flash-w8a8",
            "deepseek-v4.1-flash",
            "deepseek_v41",
        ),
    ],
)
def test_packaged_a2_quantized_deployment(name, repo_id, image_tag, reasoning):
    catalog_path = pkg_resources.files("gpustack.assets").joinpath(
        "model-catalog-modelscope.yaml"
    )
    model_set = _packaged_model_sets(str(catalog_path))[name]
    specs = filter_specs_by_gpu(
        [GPUDevice(vendor="ascend", arch_family="Ascend910B3")], model_set.specs
    )
    assert len(specs) == 1
    spec = specs[0]
    assert not spec.gpu_filters.vendor_variant
    assert spec.quantization == "W8A8"

    # The deployment payload preserves the image and quantized weight source.
    model = ModelCreate(**{**spec.model_dump(), "name": "a2-model"})
    assert model.source == SourceEnum.MODEL_SCOPE
    assert model.model_scope_model_id == f"Eco-Tech/{repo_id}"
    assert model.backend == "vLLM"
    assert model.backend_version is None
    assert model.image_name == f"quay.io/ascend/vllm-ascend:{image_tag}"
    parameters = model.backend_parameters
    assert "--quantization=ascend" in parameters
    assert "--max-model-len=65536" in parameters
    assert f"--reasoning-parser={reasoning}" in parameters
    assert "--enable-auto-tool-choice" in parameters
    assert not any("parallel-size" in arg for arg in parameters)
    assert not any("fp8" in arg or "flashinfer" in arg for arg in parameters)
    assert not any(arg.startswith("--speculative-config") for arg in parameters)
    if name in ("GLM-5.3-Flash", "DeepSeek-V4.1-Flash"):
        assert model.env["VLLM_USE_V2_MODEL_RUNNER"] == "0"

    if name != "Qwen3.8-27B":
        # Custom images use native MP; EP shards MoE weights across DP ranks.
        assert model.distributed_inference_across_workers
        assert "--enable-expert-parallel" in parameters
        assert (
            resolve_executor_backend(
                parameters, model.backend_version, model.image_name
            )
            == "mp"
        )
        nodes = 2 if name == "GLM-5.3-Flash" else 4
        topology = validate_multinode_topology(
            [8] * nodes, parse_user_parallelism(parameters)
        )
        assert topology.tp == 8
        assert topology.dp == nodes
        assert topology.dpl_per_node == [1] * nodes


def test_packaged_qwen38_huggingface_a2_deployment():
    model_set = _packaged_model_sets()["Qwen3.8-27B"]
    specs = filter_specs_by_gpu(
        [GPUDevice(vendor="ascend", arch_family="Ascend910B3")], model_set.specs
    )
    assert len(specs) == 1
    spec = specs[0]
    assert spec.source == SourceEnum.HUGGING_FACE
    assert spec.huggingface_repo_id == "Qwen/Qwen3.8-27B"
    assert spec.quantization == "BF16"
    assert spec.image_name == "quay.io/ascend/vllm-ascend:qwen3.8-a2"
    assert spec.backend_version is None
    assert "--quantization=ascend" not in spec.backend_parameters


@pytest.mark.parametrize(
    "catalog_file", ["model-catalog.yaml", "model-catalog-modelscope.yaml"]
)
def test_packaged_models_without_a2_recipe(catalog_file):
    catalog_path = pkg_resources.files("gpustack.assets").joinpath(catalog_file)
    model_sets = _packaged_model_sets(str(catalog_path))
    for name in ("Kimi-K3", "MiMo-V2.6-Pro-MOPD", "Muse-Glimmer-30B"):
        assert not filter_specs_by_gpu(
            [GPUDevice(vendor="ascend", arch_family="Ascend910B3")],
            model_sets[name].specs,
        )


@pytest.mark.skipif(
    os.getenv("HF_TOKEN") is None,
    reason="Skipped by default unless HF_TOKEN is set. Unauthed requests are rate limited.",
)
def test_model_catalog():
    model_set_specs = _packaged_model_set_specs()

    Hfapi = HfApi()

    model_name_filter = os.getenv("TEST_CATALOG_MODEL_NAME_FILTER")
    for model_set_name, model_specs in model_set_specs.items():
        assert model_set_name
        assert len(model_specs) > 0
        for model_spec in model_specs:
            assert (
                model_spec.source == SourceEnum.HUGGING_FACE
            ), f"Expected huggingface source but got: {model_spec.source}"

            if (
                model_name_filter is not None
                and model_name_filter not in model_spec.huggingface_repo_id
            ):
                continue

            time.sleep(0.01)  # mitigate rate limit

            print(model_spec.huggingface_repo_id, model_spec.huggingface_filename)
            if model_spec.huggingface_filename is None:
                model_info = Hfapi.model_info(model_spec.huggingface_repo_id)
                assert model_info is not None
            else:
                match_files = match_hugging_face_files(
                    model_spec.huggingface_repo_id, model_spec.huggingface_filename
                )
                assert (
                    len(match_files) > 0
                ), f"Failed to find model files: {model_spec.huggingface_repo_id}, {model_spec.huggingface_filename}"


@pytest.mark.skipif(
    os.getenv("HF_TOKEN") is None,
    reason="Skipped by default unless HF_TOKEN is set. Unauthed requests are rate limited.",
)
def test_model_catalog_modelscope():
    modelscope_catalog_file = pkg_resources.files("gpustack.assets").joinpath(
        "model-catalog-modelscope.yaml"
    )

    model_set_specs = _packaged_model_set_specs(str(modelscope_catalog_file))

    Msapi = HubApi()

    model_name_filter = os.getenv("TEST_CATALOG_MODEL_NAME_FILTER")
    for model_set_name, model_specs in model_set_specs.items():
        assert model_set_name
        assert len(model_specs) > 0
        for model_spec in model_specs:
            assert (
                model_spec.source == SourceEnum.MODEL_SCOPE
            ), f"Expected modelscope source but got: {model_spec.source}"

            if (
                model_name_filter is not None
                and model_name_filter not in model_spec.model_scope_model_id
            ):
                continue

            print(model_spec.model_scope_model_id, model_spec.model_scope_file_path)
            if model_spec.model_scope_file_path is None:
                model_info = Msapi.get_model(model_spec.model_scope_model_id)
                assert model_info is not None
            else:
                match_files = match_model_scope_file_paths_with_retry(
                    model_spec.model_scope_model_id,
                    model_spec.model_scope_file_path,
                )
                assert (
                    len(match_files) > 0
                ), f"Failed to find model files: {model_spec.model_scope_model_id}, {model_spec.model_scope_file_path}"


@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def match_model_scope_file_paths_with_retry(
    model_scope_model_id, model_scope_file_path
):
    return match_model_scope_file_paths(model_scope_model_id, model_scope_file_path)
