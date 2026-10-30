import pytest

from gpustack.policies.candidate_selectors import GGUFResourceFitSelector
from gpustack.policies.resource_view import ResourceView
from gpustack.scheduler.calculator import (
    Estimate,
    LayerMemoryEstimate,
    MemoryEstimate,
)
from tests.utils.model import new_model


def _estimate(context_size):
    empty = LayerMemoryEstimate(uma=0, nonuma=0, handleLayers=None)
    return Estimate(
        items=[
            MemoryEstimate(
                offloadLayers=0,
                fullOffloaded=True,
                ram=empty,
                vrams=[empty],
            )
        ],
        architecture="llama",
        contextSize=context_size,
    )


def _selector(backend_parameters):
    model = new_model(
        1,
        "test",
        1,
        huggingface_repo_id="test/test",
        backend_parameters=backend_parameters,
    )
    return GGUFResourceFitSelector(model, [], resource_view=ResourceView([]))


@pytest.mark.parametrize(
    'backend_parameters,estimate_context_size,expected',
    [
        # The estimate reports the context size it used.
        (
            [],
            24576,
            "The estimate is based on a context size of 24576 tokens.",
        ),
        # A request that reaches the estimate.
        (
            ["-c 24576"],
            24576,
            "The estimate is based on a context size of 24576 tokens.",
        ),
        # The estimator rounds the context size up to its alignment.
        (
            ["-c 45000"],
            45024,
            "The estimate is based on a context size of 45024 tokens.",
        ),
        # A request that did not reach the estimate is named.
        (
            ["-c 24576"],
            8192,
            "The estimate is based on a context size of 8192 tokens. "
            "The model requests a context size of 24576 tokens.",
        ),
        # The models without a context have no note.
        ([], 0, None),
    ],
)
def test_estimate_context_size_note(
    backend_parameters, estimate_context_size, expected
):
    selector = _selector(backend_parameters)
    selector._full_offload_resource_claim = _estimate(estimate_context_size)
    assert selector._estimate_context_size_note() == expected
