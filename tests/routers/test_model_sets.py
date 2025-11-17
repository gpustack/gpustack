import pytest
from gpustack.routes.model_sets import filter_specs_by_gpu
from gpustack.schemas.gpu_devices import GPUDevice
from gpustack.schemas.model_sets import GPUFilters, ModelSpec
from gpustack.schemas.models import SourceEnum


def make_model_spec(**kwargs):
    return ModelSpec(
        source=SourceEnum.HUGGING_FACE, huggingface_repo_id="Qwen/Qwen3-0.6B", **kwargs
    )


@pytest.mark.parametrize(
    "case_name, gpus, model_specs, filtered_specs_expected",
    [
        (
            "filter by gpu vendor",
            [
                GPUDevice(vendor="nvidia", compute_capability="8.0"),
            ],
            [
                make_model_spec(
                    mode="standard", gpu_filters=GPUFilters(vendor=["nvidia"])
                ),
                make_model_spec(
                    mode="standard", gpu_filters=GPUFilters(vendor=["amd"])
                ),
            ],
            [
                make_model_spec(
                    mode="standard", gpu_filters=GPUFilters(vendor=["nvidia"])
                ),
            ],
        ),
        (
            "filter by gpu vendor ascend",
            [
                GPUDevice(vendor="ascend"),
            ],
            [
                make_model_spec(
                    mode="standard", gpu_filters=GPUFilters(vendor=["nvidia"])
                ),
                make_model_spec(
                    mode="standard", gpu_filters=GPUFilters(vendor=["ascend"])
                ),
                make_model_spec(
                    mode="throughput", gpu_filters=GPUFilters(vendor=["ascend"])
                ),
            ],
            [
                make_model_spec(
                    mode="standard", gpu_filters=GPUFilters(vendor=["ascend"])
                ),
                make_model_spec(
                    mode="throughput", gpu_filters=GPUFilters(vendor=["ascend"])
                ),
            ],
        ),
        (
            "filter by gpu vendor and compute capability",
            [
                GPUDevice(vendor="nvidia", compute_capability="7.0"),
            ],
            [
                make_model_spec(
                    mode="standard",
                    gpu_filters=GPUFilters(
                        vendor=["nvidia"], compute_capability=">=7.0"
                    ),
                ),
                make_model_spec(
                    mode="standard",
                    gpu_filters=GPUFilters(vendor=["amd"], compute_capability=">=7.0"),
                ),
                make_model_spec(
                    mode="throughput",
                    gpu_filters=GPUFilters(
                        vendor=["nvidia"], compute_capability=">=8.0"
                    ),
                ),
                make_model_spec(
                    mode="latency",
                    gpu_filters=GPUFilters(
                        vendor=["nvidia"], compute_capability=">=7.0,<=9.0"
                    ),
                ),
            ],
            [
                make_model_spec(
                    mode="standard",
                    gpu_filters=GPUFilters(
                        vendor=["nvidia"], compute_capability=">=7.0"
                    ),
                ),
                make_model_spec(
                    mode="latency",
                    gpu_filters=GPUFilters(
                        vendor=["nvidia"], compute_capability=">=7.0,<=9.0"
                    ),
                ),
            ],
        ),
        (
            "no gpu filters",
            [
                GPUDevice(vendor="amd", compute_capability=None),
            ],
            [
                make_model_spec(mode="standard", gpu_filters=None),
                make_model_spec(
                    mode="throughput", gpu_filters=GPUFilters(vendor=["nvidia"])
                ),
            ],
            [
                make_model_spec(mode="standard", gpu_filters=None),
            ],
        ),
    ],
)
def test_filter_specs_by_gpu(
    config, case_name, gpus, model_specs, filtered_specs_expected
):
    try:
        assert filter_specs_by_gpu(gpus, model_specs) == filtered_specs_expected
    except AssertionError as e:
        print(f"Test case '{case_name}' failed.")
        raise e
