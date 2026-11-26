import pytest

from gpustack.schemas.models import (
    DPRankPlacement,
    ModelInstanceComponent,
    ModelInstanceWorkload,
    ModelInstanceWorkloadStateEnum,
)
from gpustack.server.services import format_workload_failure

WORKLOAD = "qwen3-0.6b-sgl-za5ze"
HINT = "Check the workload logs on that worker."


def _engine() -> ModelInstanceComponent:
    return ModelInstanceComponent(kind="sglang_engine")


def _rank(dp_rank: int, node_rank: int) -> DPRankPlacement:
    return DPRankPlacement(
        dp_rank=dp_rank,
        node_rank=node_rank,
        gpu_indexes=[0],
        api_port=8000 if node_rank == 0 else None,
    )


def _workload(**overrides) -> ModelInstanceWorkload:
    fields = {
        "name": WORKLOAD,
        "worker_id": 1,
        "state": ModelInstanceWorkloadStateEnum.FAILED,
    }
    fields.update(overrides)
    return ModelInstanceWorkload(**fields)


@pytest.mark.parametrize(
    "workload, worker_name, expected",
    [
        pytest.param(
            _workload(component=_engine()),
            None,
            f'Workload "{WORKLOAD}" (sglang_engine) on Worker 1 has failed. {HINT}',
            id="component",
        ),
        pytest.param(
            _workload(component=_engine()),
            "gpu-node-01",
            f'Workload "{WORKLOAD}" (sglang_engine) on Worker 1 (gpu-node-01) '
            f"has failed. {HINT}",
            id="component-with-worker-name",
        ),
        pytest.param(
            _workload(component=_engine()),
            "",
            f'Workload "{WORKLOAD}" (sglang_engine) on Worker 1 has failed. {HINT}',
            id="component-with-empty-worker-name",
        ),
        pytest.param(
            _workload(component=_engine()),
            "   ",
            f'Workload "{WORKLOAD}" (sglang_engine) on Worker 1 has failed. {HINT}',
            id="component-with-blank-worker-name",
        ),
        pytest.param(
            _workload(dp_rank_placements=[_rank(0, 0)]),
            None,
            f'Workload "{WORKLOAD}" on Worker 1 has failed '
            f"(dp_rank=0/node_rank=0). {HINT}",
            id="single-rank",
        ),
        pytest.param(
            _workload(dp_rank_placements=[_rank(0, 0), _rank(1, 1)]),
            "gpu-node-01",
            f'Workload "{WORKLOAD}" on Worker 1 (gpu-node-01) has failed '
            f"(dp_rank=0/node_rank=0, dp_rank=1/node_rank=1). {HINT}",
            id="two-ranks",
        ),
        pytest.param(
            _workload(
                component=_engine(), state=ModelInstanceWorkloadStateEnum.STOPPED
            ),
            None,
            f'Workload "{WORKLOAD}" (sglang_engine) on Worker 1 has stopped. {HINT}',
            id="stopped",
        ),
    ],
)
def test_format_workload_failure(workload, worker_name, expected):
    assert format_workload_failure(workload, worker_name) == expected


def test_component_workload_renders_no_empty_parentheses():
    smg = ModelInstanceComponent(kind="sglang_smg", api_port=8000, metrics_port=9000)

    message = format_workload_failure(_workload(component=smg))

    assert message == (
        f'Workload "{WORKLOAD}" (sglang_smg) on Worker 1 has failed. {HINT}'
    )
