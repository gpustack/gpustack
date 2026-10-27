from types import SimpleNamespace

import pytest

from gpustack.api.exceptions import BadRequestException
from gpustack.routes.benchmarks import _validate_token_windows
from gpustack.schemas.benchmark import DATASET_RANDOM, DATASET_SHAREGPT


@pytest.mark.parametrize("dataset_name", [DATASET_RANDOM, DATASET_SHAREGPT])
@pytest.mark.parametrize(
    "field",
    [
        "dataset_input_min",
        "dataset_input_max",
        "dataset_output_min",
        "dataset_output_max",
        "dataset_input_tokens",
        "dataset_output_tokens",
    ],
)
@pytest.mark.parametrize("value", [0, -1])
def test_token_lengths_must_be_positive_for_every_dataset(dataset_name, field, value):
    benchmark = SimpleNamespace(
        dataset_name=dataset_name,
        dataset_input_min=None,
        dataset_input_max=None,
        dataset_output_min=None,
        dataset_output_max=None,
        dataset_input_tokens=None,
        dataset_output_tokens=None,
    )
    setattr(benchmark, field, value)

    with pytest.raises(BadRequestException) as error:
        _validate_token_windows(benchmark)
    assert error.value.message == f"Field {field} must be > 0"


@pytest.mark.parametrize(
    "minimum,maximum,output",
    [(0, 100, 64), (10, 5, 64), (10, 100, 0)],
)
def test_sharegpt_rejects_invalid_token_settings(minimum, maximum, output):
    benchmark = SimpleNamespace(
        dataset_name=DATASET_SHAREGPT,
        dataset_input_min=minimum,
        dataset_input_max=maximum,
        dataset_output_tokens=output,
        dataset_output_min=None,
        dataset_output_max=None,
    )
    with pytest.raises(BadRequestException):
        _validate_token_windows(benchmark)


@pytest.mark.parametrize(
    "minimum,maximum,output",
    [
        (None, None, None),
        (10, None, None),
        (None, 100, None),
        (10, 100, None),
        (None, None, 64),
        (10, 100, 64),
    ],
)
def test_sharegpt_accepts_optional_token_limits(minimum, maximum, output):
    benchmark = SimpleNamespace(
        dataset_name=DATASET_SHAREGPT,
        dataset_input_min=minimum,
        dataset_input_max=maximum,
        dataset_output_tokens=output,
        dataset_output_min=None,
        dataset_output_max=None,
    )
    _validate_token_windows(benchmark)
