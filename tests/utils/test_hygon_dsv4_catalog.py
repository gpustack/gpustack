from functools import lru_cache

import pytest
import yaml

from gpustack.schemas.model_sets import Catalog
from gpustack.schemas.workers import GPUDeviceStatus
from gpustack.utils.compat_importlib import pkg_resources
from gpustack.utils.model_catalog import filter_specs_by_gpu


@lru_cache(maxsize=1)
def _deepseek_flash_specs():
    path = pkg_resources.files("gpustack.assets").joinpath(
        "model-catalog-modelscope.yaml"
    )
    catalog = Catalog.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    return next(
        model.specs for model in catalog.model_sets if model.name == "DeepSeek-V4-Flash"
    )


@pytest.mark.parametrize("vendor", ["hygon", "nvidia", "amd", "ascend"])
def test_hygon_dsv4_vendor_selection(vendor):
    specs = filter_specs_by_gpu(
        [GPUDeviceStatus(vendor=vendor)], _deepseek_flash_specs()
    )
    hygon_specs = [spec for spec in specs if spec.mode == "Hygon-BW1100-TP4-DP2"]
    assert len(hygon_specs) == (1 if vendor == "hygon" else 0)


def test_hygon_dsv4_single_worker_topology_and_public_image():
    specs = filter_specs_by_gpu(
        [GPUDeviceStatus(vendor="hygon")], _deepseek_flash_specs()
    )
    spec = next(spec for spec in specs if spec.mode == "Hygon-BW1100-TP4-DP2")
    assert spec.gpu_filters.vendor == ["hygon"]
    assert spec.source == "model_scope"
    assert spec.model_scope_model_id == "hygon/DeepSeek-V4-Flash-0731-Channel-FP8-w8a8"
    assert spec.backend == "SGLang"
    assert spec.image_name == "nas.lvjianjin.cn:8500/yangkun/dsv4:bw1100-fp8"
    assert spec.gpu_selector.gpus_per_replica == 8
    assert spec.distributed_inference_across_workers is False
    assert "--tp-size=4" in spec.backend_parameters
    assert "--dp=2" in spec.backend_parameters
    assert "--speculative-algorithm=DSPARK" in spec.backend_parameters
    assert "--reasoning-parser=deepseek-v4" in spec.backend_parameters
    assert "--tool-call-parser=deepseekv4" in spec.backend_parameters
    assert not any(
        arg.startswith(("--port", "--model-path", "--served-model-name"))
        for arg in spec.backend_parameters
    )
    assert spec.env["SGLANG_TORCH_PROFILER_DIR"] == "/tmp/profile"
    assert all(isinstance(value, str) for value in spec.env.values())
