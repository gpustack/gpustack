"""«Will a member of this have to occupy more than one machine?»

The question exists because of one sentence on the deploy form: «at least
about X% of requests will pair on one host». It is worded as a FLOOR, and a
cross-machine member does not loosen that floor, it inverts it — a member too
wide for any machine takes whole machines, so no machine holds both a prefill
and a decode and the true figure is exactly zero. «At least 25%» beside a real
0 is worse than saying nothing.

It is asked of the server rather than computed in the form because the width is
the engine's own arithmetic: vLLM spells it `--tensor-parallel-size`, SGLang
`--tp-size`, and both fold in pipeline and data parallelism. Two readings of
those flags would be two answers to one question.

The other half of the design is what it must NOT be. `gather-feasibility` was
removed for asking a verdict about a finished configuration while the form was
still half-typed, which conflated «this does not fit» with «I cannot tell yet».
So everything here is arithmetic against one fact about the cluster, and every
uncertainty answers the same as «no» — the caller shows what it always showed.
"""

import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from gpustack.validation.pd.placement import roles_that_must_span
from gpustack.schemas.models import Model, RoleSpec


def _worker(gpus: int, vendor="nvidia"):
    return SimpleNamespace(
        id=1,
        status=SimpleNamespace(
            gpu_devices=[SimpleNamespace(index=i, vendor=vendor) for i in range(gpus)]
        ),
    )


async def _ask(
    params, workers, backend="vLLM", roles=("prefill", "decode"), mode=None, vendor=None
):
    model = Model(
        name="m1",
        source="huggingface",
        huggingface_repo_id="x/y",
        backend=backend,
        backend_parameters=list(params),
    )
    model.cluster_id = 1
    model.roles = [RoleSpec(name=name, replicas=1) for name in roles]
    if mode:
        from gpustack.schemas.models import DisaggregationSpec

        model.disaggregation = DisaggregationSpec(mode=mode, vendor=vendor)
    with patch(
        "gpustack.schemas.workers.Worker.all_by_field",
        new=AsyncMock(return_value=workers),
    ):
        return await roles_that_must_span(None, model)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,vendor,workers,widest",
    [
        ("vllm-nixl", None, [_worker(4), _worker(16, "ascend")], 4),
        (
            "vllm-nixl",
            None,
            [
                SimpleNamespace(
                    status=SimpleNamespace(
                        gpu_devices=[
                            *_worker(4).status.gpu_devices,
                            *_worker(5, "ascend").status.gpu_devices,
                        ]
                    )
                )
            ],
            4,
        ),
        ("vllm-ascend-mooncake", None, [_worker(16), _worker(4, "ascend")], 4),
        ("custom", "nvidia", [_worker(4), _worker(16, "ascend")], 4),
    ],
)
async def test_span_width_counts_only_the_pd_groups_eligible_vendors(
    mode, vendor, workers, widest
):
    spanning, capacity = await _ask(
        ["--tensor-parallel-size=8"], workers, mode=mode, vendor=vendor
    )
    assert spanning == [("prefill", 8), ("decode", 8)]
    assert capacity == widest


@pytest.mark.asyncio
async def test_unknown_vendor_devices_do_not_cause_a_false_span_refusal():
    mixed = _worker(4)
    mixed.status.gpu_devices.extend(_worker(4, None).status.gpu_devices)
    spanning, widest = await _ask(
        ["--tensor-parallel-size=8"], [mixed], mode="vllm-nixl"
    )
    assert (spanning, widest) == ([], 8)


@pytest.mark.asyncio
async def test_unconstrained_roles_cannot_combine_vendors_in_one_member():
    mixed = _worker(4)
    mixed.status.gpu_devices.extend(_worker(5, "ascend").status.gpu_devices)
    spanning, widest = await _ask(["--tensor-parallel-size=8"], [mixed], mode="custom")
    assert spanning == [("prefill", 8), ("decode", 8)]
    assert widest == 5


@pytest.mark.asyncio
async def test_host_floor_refusal_names_every_spanning_role():
    from gpustack.api.exceptions import BadRequestException
    from gpustack.schemas.models import GatherSpec
    from gpustack.schemas.clusters import GatherStrategyEnum
    from gpustack.topology.tree import NODE_LAYER
    from gpustack.validation.pd.placement import validate_gather_layer

    model = Model(
        name="pd",
        source="huggingface",
        huggingface_repo_id="org/model",
        backend="vLLM",
        cluster_id=1,
        roles=[
            RoleSpec(name="prefill", backend_parameters=["--tensor-parallel-size=8"]),
            RoleSpec(name="decode", backend_parameters=["--tensor-parallel-size=16"]),
        ],
        gather=GatherSpec(layer=NODE_LAYER, strategy=GatherStrategyEnum.MUST_GATHER),
    )
    with patch(
        "gpustack.schemas.workers.Worker.all_by_field",
        AsyncMock(return_value=[_worker(4)]),
    ):
        with pytest.raises(BadRequestException) as refused:
            await validate_gather_layer(None, model)
    assert "'prefill' needs 8 GPUs" in refused.value.message
    assert "'decode' needs 16 GPUs" in refused.value.message


@pytest.mark.asyncio
async def test_a_role_wider_than_every_machine_is_named_with_its_width():
    """Both numbers come back, because the sentence the form writes needs
    both: «needs 16, the widest machine has 8» is actionable, «it will span»
    is a fact the reader cannot do anything with."""
    spanning, widest = await _ask(["--tensor-parallel-size=16"], [_worker(8)])

    assert spanning == [("prefill", 16), ("decode", 16)]
    assert widest == 8


@pytest.mark.asyncio
async def test_a_role_that_fits_is_not_named():
    spanning, widest = await _ask(["--tensor-parallel-size=8"], [_worker(8)])

    assert spanning == []
    assert widest == 8


@pytest.mark.asyncio
async def test_the_widest_machine_decides_not_the_first_or_the_average():
    """One big machine in a fleet of small ones is enough to keep every member
    on a single host — the solver will find it. Averaging, or reading whichever
    worker came back first, would announce a span that never happens."""
    spanning, widest = await _ask(
        ["--tensor-parallel-size=8"], [_worker(2), _worker(8), _worker(2)]
    )

    assert spanning == []
    assert widest == 8


@pytest.mark.asyncio
async def test_sglang_spells_the_width_differently():
    """`--tp-size`, which is the reason this is asked of the engine's own
    selector instead of re-read from the flags at either end."""
    spanning, _ = await _ask(["--tp-size=16"], [_worker(8)], backend="SGLang")

    assert [name for name, _ in spanning] == ["prefill", "decode"]


@pytest.mark.asyncio
async def test_pipeline_parallelism_counts_toward_the_width():
    """World size, not tensor-parallel size. A role at tp=8 pp=2 needs 16 GPUs
    and spans exactly as surely as one at tp=16."""
    spanning, _ = await _ask(
        ["--tensor-parallel-size=8", "--pipeline-parallel-size=2"], [_worker(8)]
    )

    assert [name for name, _ in spanning] == ["prefill", "decode"]


# --- everything it cannot be sure of answers the same as «no» --------------- #


@pytest.mark.asyncio
async def test_a_width_the_engine_was_never_told_answers_nothing():
    """The failure mode this file is guarding against is not a wrong
    refusal — nothing is refused here — it is a form that quietly says less
    than it should. An unstated width must read as «no span», not as «I
    cannot tell», because the two are rendered differently and only one of
    them is true."""
    spanning, widest = await _ask([], [_worker(8)])

    assert spanning == []
    assert widest == 0


@pytest.mark.asyncio
async def test_a_cluster_whose_workers_cannot_be_read_answers_nothing():
    spanning, widest = await _ask(["--tensor-parallel-size=16"], [])

    assert spanning == []
    assert widest == 0


@pytest.mark.asyncio
async def test_a_worker_reporting_no_cards_does_not_become_the_widest():
    """A worker mid-registration reports zero devices. Counting it would not
    change the max, but a fleet of only such workers must answer «unknown»
    rather than «the widest machine has 0 cards», which would call every role
    spanning."""
    spanning, widest = await _ask(["--tensor-parallel-size=16"], [_worker(0)])

    assert spanning == []
    assert widest == 0


@pytest.mark.asyncio
async def test_a_backend_whose_width_is_not_known_how_to_ask_answers_nothing():
    spanning, _ = await _ask(["--tensor-parallel-size=16"], [_worker(8)], backend="vox")

    assert spanning == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target,message",
    [
        (
            "gpustack.validation.pd.placement.role_effective_model",
            "Could not project model configuration",
        ),
        (
            "gpustack.schemas.models.get_backend",
            "Could not project model configuration",
        ),
        (
            "gpustack.policies.candidate_selectors.VLLMResourceFitSelector."
            "get_world_size_from_backend_parameters",
            "Could not resolve GPU width",
        ),
    ],
)
async def test_unresolved_role_widths_keep_the_failure_available_for_diagnosis(
    monkeypatch, caplog, target, message
):
    failure = RuntimeError("width resolution failed")
    monkeypatch.setattr(target, Mock(side_effect=failure))
    caplog.set_level(logging.DEBUG, logger="gpustack.validation.pd.placement")
    assert await _ask(["--tensor-parallel-size=16"], [_worker(8)]) == ([], 0)
    records = [
        record
        for record in caplog.records
        if record.name == "gpustack.validation.pd.placement"
    ]
    assert len(records) == 2
    for record, role in zip(records, ("prefill", "decode")):
        assert message in record.getMessage()
        assert f"'{role}'" in record.getMessage()
        assert record.levelno == logging.DEBUG
        assert record.exc_info[1] is failure


@pytest.mark.asyncio
async def test_a_deployment_with_no_cluster_yet_answers_nothing():
    """The form asks as soon as it has a width, and the cluster picker may
    still be empty."""
    model = Model(name="m1", source="huggingface", huggingface_repo_id="x/y")
    model.roles = [RoleSpec(name="prefill", replicas=1)]

    spanning, widest = await roles_that_must_span(None, model)

    assert (spanning, widest) == ([], 0)


@pytest.mark.asyncio
async def test_a_plain_deployment_with_no_roles_answers_nothing():
    """Not a PD group at all — this must not start querying workers for every
    ordinary model that happens to carry a parallel size."""
    spanning, widest = await _ask(["--tensor-parallel-size=16"], [_worker(8)], roles=())

    assert (spanning, widest) == ([], 0)


# --- the route around it ---------------------------------------------------- #


@pytest.mark.asyncio
async def test_sparse_host_floor_update_uses_stored_role_widths():
    from gpustack.api.exceptions import BadRequestException
    from gpustack.schemas.models import ModelUpdate, GatherSpec
    from gpustack.schemas.clusters import GatherStrategyEnum
    from gpustack.topology.tree import NODE_LAYER
    from gpustack.validation.pd.placement import validate_gather_layer

    stored = Model(
        name="m",
        source="huggingface",
        huggingface_repo_id="org/model",
        cluster_id=1,
        backend="vLLM",
        backend_parameters=["--tensor-parallel-size=16"],
        roles=[RoleSpec(name="prefill"), RoleSpec(name="decode")],
    )
    update = ModelUpdate(
        name=stored.name,
        source=stored.source,
        huggingface_repo_id=stored.huggingface_repo_id,
        gather=GatherSpec(layer=NODE_LAYER, strategy=GatherStrategyEnum.MUST_GATHER),
    )
    with patch(
        "gpustack.schemas.workers.Worker.all_by_field",
        AsyncMock(return_value=[_worker(8)]),
    ):
        with pytest.raises(BadRequestException) as error:
            await validate_gather_layer(None, update, stored=stored)
    assert "needs 16 GPUs" in error.value.message
    assert update.roles is None
    assert update.cluster_id is None


@pytest.mark.asyncio
@pytest.mark.parametrize("clear_gather", [False, True])
async def test_sparse_gather_update_distinguishes_omission_from_explicit_null(
    clear_gather,
):
    from gpustack.api.exceptions import BadRequestException
    from gpustack.schemas.models import ModelUpdate, GatherSpec
    from gpustack.schemas.clusters import GatherStrategyEnum
    from gpustack.topology.tree import NODE_LAYER
    from gpustack.validation.pd.placement import validate_gather_layer

    stored = Model(
        name="pd",
        source="huggingface",
        huggingface_repo_id="org/model",
        cluster_id=1,
        backend="vLLM",
        backend_parameters=["--tensor-parallel-size=16"],
        roles=[RoleSpec(name="prefill"), RoleSpec(name="decode")],
        gather=GatherSpec(layer=NODE_LAYER, strategy=GatherStrategyEnum.MUST_GATHER),
    )
    update = ModelUpdate(
        name=stored.name,
        source=stored.source,
        huggingface_repo_id=stored.huggingface_repo_id,
        **({"gather": None} if clear_gather else {}),
    )
    workers = AsyncMock(return_value=[_worker(8)])
    with patch("gpustack.schemas.workers.Worker.all_by_field", workers):
        if clear_gather:
            await validate_gather_layer(None, update, stored=stored)
            workers.assert_not_awaited()
        else:
            with pytest.raises(BadRequestException):
                await validate_gather_layer(None, update, stored=stored)
