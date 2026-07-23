"""Schema validation for ``GPUInstanceResources`` sliced-percentage fields.

Locks the API-level ``[0, 100]`` bound on the sliced accelerator percentage
fields so invalid requests are rejected up front instead of at the CRD layer.
"""

import pytest
from pydantic import ValidationError

from gpustack.schemas.gpu_instances import GPUInstanceResources


@pytest.mark.parametrize("value", [0, 50, 100])
def test_sliced_percentages_accept_in_range(value):
    r = GPUInstanceResources(
        accelerator_sliced_memory_percentage=value,
        accelerator_sliced_cores_percentage=value,
    )
    assert r.accelerator_sliced_memory_percentage == value
    assert r.accelerator_sliced_cores_percentage == value


@pytest.mark.parametrize(
    "field",
    ["accelerator_sliced_memory_percentage", "accelerator_sliced_cores_percentage"],
)
@pytest.mark.parametrize("value", [-1, 101])
def test_sliced_percentages_reject_out_of_range(field, value):
    with pytest.raises(ValidationError):
        GPUInstanceResources(**{field: value})
