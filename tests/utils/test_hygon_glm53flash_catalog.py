from functools import lru_cache

import pytest
import yaml

from gpustack.schemas.model_sets import Catalog
from gpustack.schemas.workers import GPUDeviceStatus
from gpustack.utils.compat_importlib import pkg_resources
from gpustack.utils.model_catalog import filter_specs_by_gpu


@lru_cache(maxsize=1)
def _glm_flash_specs():
    path = pkg_resources.files("gpustack.assets").joinpath(
        "model-catalog-modelscope.yaml"
    )
    catalog = Catalog.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    return next(
        model.specs for model in catalog.model_sets if model.name == "GLM-5.3-Flash"
    )


def _hygon_spec():
    return next(
        spec
        for spec in filter_specs_by_gpu(
            [GPUDeviceStatus(vendor="hygon")], _glm_flash_specs()
        )
        if spec.mode == "Hygon-BW1100-TP8"
    )


@pytest.mark.parametrize("vendor", ["hygon", "nvidia", "amd", "ascend"])
def test_hygon_glm53flash_vendor_selection(vendor):
    specs = filter_specs_by_gpu([GPUDeviceStatus(vendor=vendor)], _glm_flash_specs())
    matches = [spec for spec in specs if spec.mode == "Hygon-BW1100-TP8"]
    assert len(matches) == (1 if vendor == "hygon" else 0)


def test_hygon_glm53flash_topology_and_multivalue_arguments():
    spec = _hygon_spec()
    assert spec.source == "model_scope"
    assert spec.model_scope_model_id == "ZhipuAI/GLM-5.3-Flash"
    assert spec.backend == "SGLang"
    assert spec.quantization == "FP8"
    assert spec.image_name == "nas.lvjianjin.cn:8500/yangkun/glm53flash:20261111-opt"
    assert spec.gpu_selector.gpus_per_replica == 8
    assert spec.distributed_inference_across_workers is False
    args = spec.backend_parameters
    assert "--tp-size=8" in args
    numa = args.index("--numa-node")
    assert args[numa + 1 : numa + 9] == ["3", "1", "1", "0", "7", "5", "5", "4"]
    batches = args.index("--cuda-graph-bs-decode")
    assert args[batches + 1 : batches + 9] == [
        "1",
        "2",
        "4",
        "8",
        "16",
        "32",
        "64",
        "128",
    ]
    assert "--speculative-algorithm=EAGLE" in args
    assert "--speculative-adaptive" in args
    assert "--context-length=1048576" in args
    assert not any(
        arg.startswith(("--port", "--model-path", "--served-model-name"))
        for arg in args
    )


def test_hygon_glm53flash_runtime_and_portable_environment():
    env = _hygon_spec().env
    assert env["GLM53F_RUNTIME_ROOT"] == "/workspace/runtime/glm53f"
    assert env["PYTHONPATH"] == (
        "/workspace/runtime/glm53f/bootstrap:/workspace/runtime/glm53f/overlay"
    )
    assert env["HIP_KERNEL_EVENT_SYSTENFENCE"] == "1"
    assert env["SGLANG_DSA_FUSE_TOPK"] == "1"
    assert env["GLM53F_SPEC_MODE"] == "converged"
    assert env["GLM53F_FIXED_STEP"] == "3"
    for key in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "NCCL_SOCKET_IFNAME",
        "GLOO_SOCKET_IFNAME",
        "HIP_VISIBLE_DEVICES",
        "HSA_FORCE_FINE_GRAIN_PCIE",
    ):
        assert key not in env
    for key in (
        "SGLANG_TORCH_PROFILER_DIR",
        "GLM53F_SPAWN_TMP",
        "GLM53F_DIAGNOSTIC_DIR",
        "GLM53F_AUDIT_DIR",
    ):
        assert env[key].startswith("/tmp/glm53f-runtime/")
    assert all(isinstance(value, str) for value in env.values())
