import pytest
from gpustack.utils.gpu import parse_gpu_id


expected_matched_inputs = {
    "worker1:cuda:0": {"worker_name": "worker1", "device": "cuda", "gpu_index": "0"},
    "worker_name:npu:12": {
        "worker_name": "worker_name",
        "device": "npu",
        "gpu_index": "12",
    },
    "test_worker:rocm:3": {
        "worker_name": "test_worker",
        "device": "rocm",
        "gpu_index": "3",
    },
    "example:musa:7": {"worker_name": "example", "device": "musa", "gpu_index": "7"},
    "name:example:musa:7": {
        "worker_name": "name:example",
        "device": "musa",
        "gpu_index": "7",
    },
    "name:example:mps:100": {
        "worker_name": "name:example",
        "device": "mps",
        "gpu_index": "100",
    },
}

expected_not_matched_inputs = [
    "invalid:abc:1",
    "worker1:cuda:not_a_number",
]


@pytest.mark.unit
def test_parse_gpu_id():
    for input, expected_output in expected_matched_inputs.items():
        is_matched, result = parse_gpu_id(input)
        assert is_matched, f"Expected {input} to be matched but it was not."
        assert result.get("worker_name") == expected_output.get(
            "worker_name"
        ), f"Expected worker_name to be {expected_output.get('worker_name')} but got {result.get('worker_name')}"
        assert result.get("device") == expected_output.get(
            "device"
        ), f"Expected device to be {expected_output.get('device')} but got {result.get('device')}"
        assert result.get("gpu_index") == expected_output.get(
            "gpu_index"
        ), f"Expected gpu_index to be {expected_output.get('gpu_index')} but got {result.get('gpu_index')}"

    for input in expected_not_matched_inputs:
        is_matched, result = parse_gpu_id(input)
        assert not is_matched, f"Expected {input} to not be matched but it was."
        assert result is None, f"Expected result to be None but got {result}"
