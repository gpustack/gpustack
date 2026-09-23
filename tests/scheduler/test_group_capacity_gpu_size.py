"""One role, one size of card.

A member reserves a fraction of its card's total memory, so its KV cache grows
with that total while the router shares requests between the role's members as
equals. The capacity function is therefore allowed to offer a role only one
band of card sizes, and what it does with the workers it rules out is as much
of the behaviour as which band it keeps: the solver reads an absent worker as
unmeasured and hedges its refusal, so a ruled-out worker has to come back as a
measured zero.
"""

from types import SimpleNamespace

import pytest

from gpustack.schemas.models import ComputedResourceClaim, Model, RoleSpec
from gpustack.scheduler.group_capacity import GroupCapacity, _RoleProjection

GIB = 1024**3
CLAIM = ComputedResourceClaim(vram={0: 36 * GIB}, ram=0)


def _worker(worker_id: int, *gpu_gib: float):
    return SimpleNamespace(
        id=worker_id,
        name=f"w{worker_id}",
        ip=f"10.0.0.{worker_id}",
        labels={},
        status=SimpleNamespace(
            memory=SimpleNamespace(total=64 * GIB),
            gpu_devices=[
                SimpleNamespace(memory=SimpleNamespace(total=int(size * GIB)))
                for size in gpu_gib
            ],
        ),
    )


def _model(prefill=2, decode=1, router=0):
    model = Model(name="pd", source="huggingface", huggingface_repo_id="x/y")
    model.id = 1
    model.roles = [
        RoleSpec(name="prefill", replicas=prefill),
        RoleSpec(name="decode", replicas=decode),
    ]
    if router:
        model.roles.append(RoleSpec(name="router", replicas=router))
    return model


class _OneCardEach:
    """Offers every worker a single member, then reports it full."""

    def __init__(self, instances):
        self._taken = set()
        for entry in instances:
            if getattr(entry, "gpu_indexes", None) is not None:
                self._taken.add(getattr(entry, "worker_id", None))

    async def select_candidates(self, workers):
        worker = workers[0]
        if worker.id in self._taken:
            return []
        return [
            SimpleNamespace(
                worker=worker,
                gpu_indexes=[0],
                gpu_type="cuda",
                gpu_addresses=None,
                computed_resource_claim=CLAIM,
                subordinate_workers=None,
                overcommit=False,
            )
        ]


def _capacity(workers, model=None):
    model = model or _model()
    cap = GroupCapacity(SimpleNamespace(), model, workers, [])
    cap._selector = lambda m, instances, cpu_only, ram_claim=None: _OneCardEach(
        instances
    )
    for role in ("prefill", "decode", "router"):
        cap._eligible[role] = {w.id: w for w in workers}
        cap._projected[role] = _RoleProjection(model, role == "router")
    return cap


@pytest.mark.asyncio
async def test_the_band_holding_the_most_members_is_the_one_offered():
    """Member count decides, not card size: two 32 GiB workers serve a role of
    two, one 48 GiB worker does not."""
    cap = _capacity([_worker(1, 48), _worker(2, 32), _worker(3, 32)])

    slots = await cap("prefill", [1, 2, 3], [])

    assert slots == {1: 0, 2: 1, 3: 1}


@pytest.mark.asyncio
async def test_a_ruled_out_worker_comes_back_as_a_measured_zero():
    """Absence means "could not be measured" to the solver, which turns the
    refusal into a hedge about broken telemetry. These workers were measured
    and then excluded by a rule."""
    cap = _capacity([_worker(1, 48), _worker(2, 32), _worker(3, 32)])

    slots = await cap("prefill", [1, 2, 3], [])

    assert set(slots) == {1, 2, 3}
    assert slots[1] == 0


@pytest.mark.asyncio
async def test_the_bigger_card_wins_a_tie_on_member_count():
    """Equal room either way, so the band that leaves the most headroom for
    whoever comes next is the one kept."""
    cap = _capacity([_worker(1, 32), _worker(2, 48)], model=_model(prefill=1))

    slots = await cap("prefill", [1, 2], [])

    assert slots == {1: 0, 2: 1}


@pytest.mark.asyncio
async def test_a_single_size_fleet_is_offered_whole():
    cap = _capacity([_worker(1, 48), _worker(2, 48), _worker(3, 48)])

    slots = await cap("prefill", [1, 2, 3], [])

    assert slots == {1: 1, 2: 1, 3: 1}
    assert cap.notes_for("prefill") == []


@pytest.mark.asyncio
async def test_a_role_that_holds_no_card_is_never_banded():
    """The router is a proxy with no accelerator, so there is no card size for
    its members to agree on and every worker stays available to it."""
    cap = _capacity(
        [_worker(1, 48), _worker(2, 32), _worker(3, 32)],
        model=_model(router=1),
    )

    slots = await cap("router", [1, 2, 3], [])

    assert slots == {1: 1, 2: 1, 3: 1}


@pytest.mark.asyncio
async def test_a_shortfall_from_mixed_cards_says_so():
    """The refusal has to be distinguishable from a full cluster: nothing here
    is out of room, the role simply cannot have two sizes at once."""
    cap = _capacity([_worker(1, 48), _worker(2, 32)])

    await cap("prefill", [1, 2], [])

    assert cap.notes_for("prefill") == [
        "Role 'prefill' must sit on GPUs of the same size; the largest "
        "matching set holds 1 of the 2 members it needs."
    ]


@pytest.mark.asyncio
async def test_the_sentence_is_written_once_however_often_the_role_is_measured():
    """A solve measures a role several times -- the whole tree, then domain by
    domain -- and each narrower pass would report a worse shortfall about a
    smaller question."""
    cap = _capacity([_worker(1, 48), _worker(2, 32)])

    await cap("prefill", [1, 2], [])
    await cap("prefill", [1, 2], [])

    assert len(cap.notes_for("prefill")) == 1


@pytest.mark.asyncio
async def test_no_sentence_when_the_kept_band_holds_the_whole_role():
    """Nothing was refused, so a line about the rule would sit in whatever
    refusal follows contradicting it."""
    cap = _capacity([_worker(1, 48), _worker(2, 32), _worker(3, 32)])

    await cap("prefill", [1, 2, 3], [])

    assert cap.notes_for("prefill") == []


@pytest.mark.asyncio
async def test_a_worker_reporting_no_card_size_is_not_ruled_out():
    """Its size is unknown rather than different, and zeroing it would take
    away room the selectors did offer."""
    blind = SimpleNamespace(
        id=3,
        name="w3",
        ip="10.0.0.3",
        labels={},
        status=SimpleNamespace(memory=SimpleNamespace(total=64 * GIB)),
    )
    cap = _capacity([_worker(1, 48), _worker(2, 32), blind])

    slots = await cap("prefill", [1, 2, 3], [])

    assert slots[3] == 1


@pytest.mark.asyncio
async def test_sizing_does_not_apply_the_one_gpu_size_rule():
    """The rule picks the band holding the most members, and the domain search
    sizes the whole tree in one sweep. Applied there, a cluster-wide winner
    zeroes every worker of the losing bands -- so a rack built entirely from
    them sizes as empty and is dropped from the search, although its members
    would have sat on one size of card perfectly well.

    The rule belongs where it is decided: inside a domain, which is where all
    of one role's members land.
    """
    workers = [
        _worker(1, 48),
        _worker(2, 48),
        _worker(3, 32),
        _worker(4, 32),
        _worker(5, 32),
    ]
    cap = _capacity(workers)

    banded = await cap("prefill", [1, 2, 3, 4, 5], [])
    unbanded = await cap.sizing("prefill", [1, 2, 3, 4, 5])

    # Three 32 GiB workers against two 48 GiB ones: the smaller band holds
    # more members, wins, and the bigger cards are zeroed.
    assert banded[1] == 0 and banded[2] == 0
    assert banded[3] == 1
    # Sizing sees every worker, so a 48 GiB-only domain is still worth a look.
    assert unbanded[1] == 1 and unbanded[2] == 1
    assert unbanded[3] == 1
