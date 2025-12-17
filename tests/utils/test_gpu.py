import pytest
from gpustack.utils.gpu import parse_gpu_id, compare_compute_capability


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


@pytest.mark.parametrize(
    "current, target, expected",
    [
        # Equal cases
        ("8.0", "8.0", 0),
        ("7.5", "7.5", 0),
        # Greater cases
        ("8.0", "7.5", 1),
        ("8.6", "8.0", 1),
        ("9.0", "8.9", 1),
        ("10.0", "9.0", 1),
        # Less cases
        ("7.5", "8.0", -1),
        ("8.0", "8.6", -1),
        ("8.9", "9.0", -1),
        # Invalid current, valid target -> -1
        (None, "8.0", -1),
        ("", "8.0", -1),
        ("   ", "8.0", -1),
        ("invalid", "8.0", -1),
        ("8", "8.0", -1),
        ("8.", "8.0", -1),
        (".0", "8.0", -1),
        ("8.0.1", "8.0", -1),
        # Valid current, invalid target -> 1
        ("8.0", None, 1),
        ("8.0", "", 1),
        ("8.0", "   ", 1),
        ("8.0", "invalid", 1),
        ("8.0", "8", 1),
        ("8.0", "8.", 1),
        ("8.0", ".0", 1),
        ("8.0", "8.0.1", 1),
        # Both invalid -> 0
        (None, None, 0),
        ("", "", 0),
        ("   ", "   ", 0),
        ("invalid", "invalid", 0),
        (None, "", 0),
        ("8", "invalid", 0),
        ("8.", ".0", 0),
        ("-1.0", "-2.0", 0),
        # Whitespace normalization
        (" 8.0 ", "8.0", 0),
        ("8.0", " 8.0 ", 0),
        (" 8.6 ", " 8.0 ", 1),
    ],
)
def test_compare_compute_capability(current, target, expected):
    assert compare_compute_capability(current, target) == expected
