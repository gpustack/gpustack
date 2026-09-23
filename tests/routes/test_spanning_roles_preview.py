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

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.routes.models import roles_that_must_span
from gpustack.schemas.models import Model, RoleSpec


def _worker(gpus: int):
    return SimpleNamespace(
        id=1,
        status=SimpleNamespace(
            gpu_devices=[SimpleNamespace(index=i) for i in range(gpus)]
        ),
    )


async def _ask(params, workers, backend="vLLM", roles=("prefill", "decode")):
    model = Model(
        name="m1",
        source="huggingface",
        huggingface_repo_id="x/y",
        backend=backend,
        backend_parameters=list(params),
    )
    model.cluster_id = 1
    model.roles = [RoleSpec(name=name, replicas=1) for name in roles]
    with patch(
        "gpustack.schemas.workers.Worker.all_by_field",
        new=AsyncMock(return_value=workers),
    ):
        return await roles_that_must_span(None, model)


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


def _ctx(is_admin=True, accessible_cluster_ids=None):
    from unittest.mock import MagicMock

    from gpustack.api.tenant import TenantContext
    from gpustack.schemas.principals import PrincipalType

    user = MagicMock()
    user.id = 99
    user.is_admin = is_admin
    user.kind = PrincipalType.USER
    return TenantContext(
        user=user,
        is_platform_admin=is_admin,
        current_principal_id=None if is_admin else 7,
        org_role=None,
        accessible_cluster_ids=set(accessible_cluster_ids or []),
    )


def _create(**kwargs):
    from gpustack.schemas.models import ModelCreate, SourceEnum

    return ModelCreate(
        name="m1",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        **kwargs,
    )


@pytest.mark.asyncio
async def test_the_route_answers_with_both_numbers():
    """Both are needed by the one sentence the form writes, so both cross the
    wire — the form must not have to hold half the comparison."""
    from gpustack.routes.models import preview_spanning_roles
    from gpustack.schemas.models import RoleSpec

    model_in = _create(
        cluster_id=1,
        backend="vLLM",
        backend_parameters=["--tensor-parallel-size=16"],
        roles=[RoleSpec(name="prefill", replicas=1)],
    )
    with (
        patch(
            "gpustack.schemas.clusters.Cluster.one_by_id",
            new=AsyncMock(return_value=SimpleNamespace(id=1, owner_principal_id=1)),
        ),
        patch(
            "gpustack.schemas.workers.Worker.all_by_field",
            new=AsyncMock(return_value=[_worker(8)]),
        ),
    ):
        answer = await preview_spanning_roles(None, _ctx(), model_in)

    assert [(r.name, r.gpus) for r in answer.roles] == [("prefill", 16)]
    assert answer.widest_worker_gpus == 8


@pytest.mark.asyncio
async def test_a_cluster_the_caller_cannot_see_is_not_answered_about():
    """The answer describes someone else's machines. It is a small fact — the
    widest card count in a fleet — but it is still a fact about a cluster the
    caller was not given."""
    from gpustack.api.exceptions import NotFoundException
    from gpustack.routes.models import preview_spanning_roles
    from gpustack.schemas.models import RoleSpec

    model_in = _create(
        cluster_id=1,
        backend="vLLM",
        backend_parameters=["--tensor-parallel-size=16"],
        roles=[RoleSpec(name="prefill", replicas=1)],
    )
    with patch(
        "gpustack.schemas.clusters.Cluster.one_by_id",
        new=AsyncMock(return_value=SimpleNamespace(id=1, owner_principal_id=1)),
    ):
        with pytest.raises(NotFoundException):
            await preview_spanning_roles(None, _ctx(is_admin=False), model_in)
