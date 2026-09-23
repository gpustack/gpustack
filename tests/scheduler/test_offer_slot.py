from types import SimpleNamespace

import pytest

from gpustack.policies.base import MemberResourceClaim
from gpustack.policies.utils import compute_worker_allocated
from gpustack.scheduler.offer_slot import count_offer_slots


def worker(id_=1, name="w1"):
    return SimpleNamespace(id=id_, name=name)


def claim(vram_per_gpu: int, gpu_indexes, ram: int = 0):
    return SimpleNamespace(
        ram=ram, vram={i: vram_per_gpu for i in gpu_indexes}, offload_layers=None
    )


def candidate(
    gpu_indexes,
    vram_per_gpu=10,
    ram=0,
    overcommit=False,
    gpu_type="cuda",
    on=None,
):
    # A real candidate always names the worker it was selected on, and the
    # stand-in is keyed off it — with several machines in play, "the worker the
    # caller asked about" and "the worker this candidate is on" stop being the
    # same thing.
    return SimpleNamespace(
        worker=on if on is not None else worker(),
        gpu_indexes=list(gpu_indexes),
        computed_resource_claim=claim(vram_per_gpu, gpu_indexes, ram),
        gpu_type=gpu_type,
        overcommit=overcommit,
        subordinate_workers=None,
    )


class FakeSelector:
    """A selector whose capacity is a fixed pool of GPUs, each usable once.

    Deliberately re-derives what is free from the instance list it was handed,
    the way the real selectors do through ``compute_worker_allocated`` — that
    is the behaviour the loop depends on, so a stub that ignored the list would
    test nothing.
    """

    def __init__(self, instances, free_gpus, per_member=1, overcommit_after=None):
        self._instances = instances
        self._free = list(free_gpus)
        self._per_member = per_member
        self._overcommit_after = overcommit_after

    async def select_candidates(self, workers):
        taken = set()
        for mi in self._instances:
            for index in mi.gpu_indexes or []:
                taken.add(index)
        available = [g for g in self._free if g not in taken]
        if len(available) < self._per_member:
            return []
        chosen = available[: self._per_member]
        over = (
            self._overcommit_after is not None and len(taken) >= self._overcommit_after
        )
        return [candidate(chosen, overcommit=over, on=workers[0])]


def selector_factory(**kwargs):
    return lambda instances: FakeSelector(instances, **kwargs)


# --- the loop itself ------------------------------------------------------- #


@pytest.mark.asyncio
async def test_capacity_is_counted_by_pretending_each_fit_was_taken():
    """Four free cards, one card per member, and the count is four — which is
    only right if each round sees the previous round's placement."""
    result = await count_offer_slots(
        selector_factory(free_gpus=[0, 1, 2, 3]), [worker()], [], limit=10
    )

    assert result.slots == 4


@pytest.mark.asyncio
async def test_a_member_spanning_two_cards_halves_the_count():
    result = await count_offer_slots(
        selector_factory(free_gpus=[0, 1, 2, 3], per_member=2),
        [worker()],
        [],
        limit=10,
    )

    assert result.slots == 2


@pytest.mark.asyncio
async def test_workers_already_carrying_instances_start_from_what_is_left():
    """The instance list is the same one the real scheduler reads, so an
    existing placement narrows capacity without any separate bookkeeping."""
    existing = [
        SimpleNamespace(
            worker_id=1,
            gpu_indexes=[0, 1],
            gpu_type="cuda",
            computed_resource_claim=claim(10, [0, 1]),
            distributed_servers=None,
        )
    ]

    result = await count_offer_slots(
        selector_factory(free_gpus=[0, 1, 2, 3]), [worker()], existing, limit=10
    )

    assert result.slots == 2


@pytest.mark.asyncio
async def test_a_worker_with_no_room_offers_nothing():
    result = await count_offer_slots(
        selector_factory(free_gpus=[]), [worker()], [], limit=4
    )

    assert result.slots == 0
    assert result.placements == []


# --- the bound ------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_counting_stops_at_the_limit_the_caller_asked_for():
    """Capacity beyond what the caller wants to place is not a number anyone
    needs, and each extra slot costs a full selector pass."""
    result = await count_offer_slots(
        selector_factory(free_gpus=list(range(64))), [worker()], [], limit=3
    )

    assert result.slots == 3


@pytest.mark.asyncio
async def test_a_limit_of_zero_does_not_run_the_selector_at_all():
    def explode(_instances):
        raise AssertionError("the selector must not be constructed")

    result = await count_offer_slots(explode, [worker()], [], limit=0)

    assert result.slots == 0


# --- what is refused ------------------------------------------------------- #


@pytest.mark.asyncio
async def test_overcommitted_fits_do_not_count_as_capacity():
    """The selectors offer overcommit so one deployment can start anyway on a
    busy worker. Counting it would let a domain advertise room it does not
    have, and the whole group would start into contention at once."""
    result = await count_offer_slots(
        selector_factory(free_gpus=[0, 1, 2, 3], overcommit_after=2),
        [worker()],
        [],
        limit=10,
    )

    assert result.slots == 2


@pytest.mark.asyncio
async def test_a_selector_that_raises_yields_the_floor_not_zero_and_not_a_crash():
    """Unknown capacity is not zero capacity. Returning what was already proven
    means the domain can only look smaller than it is, which costs a tighter
    placement — never a wrong one."""
    calls = {"n": 0}

    class Flaky(FakeSelector):
        async def select_candidates(self, workers):
            calls["n"] += 1
            if calls["n"] > 2:
                raise RuntimeError("detector went away")
            return await super().select_candidates(workers)

    result = await count_offer_slots(
        lambda instances: Flaky(instances, free_gpus=[0, 1, 2, 3]),
        [worker()],
        [],
        limit=10,
    )

    assert result.slots == 2


# --- what the placements are for ------------------------------------------- #


@pytest.mark.asyncio
async def test_the_placements_are_kept_so_they_need_not_be_derived_twice():
    """The group scheduler writes these GPU indexes onto the members it places.
    Re-deriving them later would be a second placement that can disagree with
    the one the count was based on."""
    result = await count_offer_slots(
        selector_factory(free_gpus=[0, 1, 2, 3], per_member=2),
        [worker()],
        [],
        limit=10,
    )

    assert [c.gpu_indexes for c in result.placements] == [[0, 1], [2, 3]]


@pytest.mark.asyncio
async def test_a_fresh_selector_is_built_each_round():
    """The selectors compute their claims in __init__ and cache them, so a
    reused instance would answer the first round's question forever."""
    built = []

    def factory(instances):
        built.append(len(instances))
        return FakeSelector(instances, free_gpus=[0, 1, 2])

    await count_offer_slots(factory, [worker()], [], limit=10)

    assert built == [0, 1, 2, 3]


# --- multi-worker placements ----------------------------------------------- #


@pytest.mark.asyncio
async def test_a_distributed_placement_keeps_its_other_halves_visible():
    """A distributed candidate eats VRAM on workers besides this one, and
    allocation finds that through `distributed_servers`. Dropping it would let
    a later round hand out the same remote cards twice.

    Asserted through the *remote* worker's allocation, which is the only place
    the omission would show."""
    remote = SimpleNamespace(worker_id=2, computed_resource_claim=claim(10, [0]))
    seen: list = []

    class Distributed(FakeSelector):
        async def select_candidates(self, workers):
            seen.clear()
            seen.extend(self._instances)
            found = await super().select_candidates(workers)
            for c in found:
                c.subordinate_workers = [remote]
            return found

    result = await count_offer_slots(
        lambda instances: Distributed(instances, free_gpus=[0, 1]),
        [worker()],
        [],
        limit=2,
    )

    assert result.slots == 2
    # `seen` is the list the last round was handed: one stand-in, carrying the
    # first round's remote half.
    assert compute_worker_allocated(seen, worker_id=2).vram == {0: 10}


@pytest.mark.asyncio
async def test_the_stand_in_is_readable_by_the_real_allocation_function():
    """The load-bearing claim of this module: the pretence is exact because it
    goes through the same accounting the scheduler will. Asserted against the
    real `compute_worker_allocated`, not a stub of it."""
    from gpustack.scheduler.offer_slot import _stand_in_for

    result = await count_offer_slots(
        selector_factory(free_gpus=[0, 1]), [worker()], [], limit=2
    )
    stand_ins = [_stand_in_for(c) for c in result.placements]

    assert compute_worker_allocated(stand_ins, worker_id=1).vram == {0: 10, 1: 10}


@pytest.mark.asyncio
async def test_a_stopped_count_says_it_was_stopped():
    """`slots` alone cannot tell "this worker is full" from "the selector broke
    before it could say", and the two reach the operator as the same refusal.
    A plain misconfiguration then reads as "the cluster is full", which is the
    one message that stops someone looking for a mistake."""

    class Broken(FakeSelector):
        async def select_candidates(self, workers):
            raise RuntimeError("no global config")

    result = await count_offer_slots(
        lambda instances: Broken(instances, free_gpus=[0, 1]), [worker()], [], limit=4
    )

    assert result.slots == 0
    assert result.counted_to_exhaustion is False
    assert "no global config" in result.unavailable


@pytest.mark.asyncio
async def test_a_worker_that_is_genuinely_full_counted_to_exhaustion():
    """The other side of the same rule: a real zero must stay distinguishable
    from an unknown one."""
    result = await count_offer_slots(
        selector_factory(free_gpus=[]), [worker()], [], limit=4
    )

    assert result.slots == 0
    assert result.counted_to_exhaustion is True
    assert result.unavailable is None


# --- more than one machine at a time ----------------------------------------- #


@pytest.mark.asyncio
async def test_a_set_of_machines_is_one_question():
    """What the signature change buys. The selectors' cross-node branch
    refuses a list shorter than two outright, so while this handed over one
    worker at a time a member needing more cards than any single machine has
    could not be counted at all — not "reported as zero", but unreachable.

    The combinations that come back are disjoint because each is stood in
    before the next is asked for, which is why counting a domain this way
    yields the number of members it holds rather than a number per machine.
    """

    class Spanning:
        """Takes two machines per member, from a shared pool of four."""

        def __init__(self, instances):
            taken = {
                w
                for mi in instances
                for w in [mi.worker_id]
                + [
                    s.worker_id
                    for s in (
                        mi.distributed_servers.subordinate_workers
                        if mi.distributed_servers
                        else []
                    )
                ]
            }
            self._free = [w for w in (1, 2, 3, 4) if w not in taken]

        async def select_candidates(self, workers):
            usable = [w for w in workers if w.id in self._free]
            if len(usable) < 2:
                return []
            primary, second = usable[0], usable[1]
            return [
                SimpleNamespace(
                    worker=primary,
                    gpu_indexes=[0],
                    computed_resource_claim=claim(10, [0]),
                    gpu_type="cuda",
                    overcommit=False,
                    subordinate_workers=[
                        SimpleNamespace(
                            worker_id=second.id,
                            gpu_indexes=[0],
                            computed_resource_claim=claim(10, [0]),
                        )
                    ],
                )
            ]

    fleet = [worker(id_=i, name=f"w{i}") for i in (1, 2, 3, 4)]

    result = await count_offer_slots(lambda i: Spanning(i), fleet, [], limit=4)

    # Four machines, two per member: two members, not four.
    assert result.slots == 2
    assert [c.worker.id for c in result.placements] == [1, 3]


@pytest.mark.asyncio
async def test_one_machine_is_still_one_machine():
    """The compatibility half: handed a single-element list this is what it
    always was, which is what keeps every deployment that exists today on the
    same numbers."""
    result = await count_offer_slots(
        selector_factory(free_gpus=[0, 1, 2]), [worker()], [], limit=10
    )

    assert result.slots == 3
    assert all(c.worker.id == 1 for c in result.placements)


@pytest.mark.asyncio
async def test_what_one_member_costs_is_kept_whether_or_not_any_fit():
    """A group that does not fit has no placement to read a claim off, so the
    counting pass is the only place its members are ever priced. Taken on every
    round, so a role that fits is priced the same way as one that does not."""

    class Priced(FakeSelector):
        def get_resource_claim(self):
            return MemberResourceClaim(vram=40 * 1024**3, ram=1024**3)

    fits = await count_offer_slots(
        lambda i: Priced(i, free_gpus=[0, 1]), [worker()], [], limit=4
    )
    full = await count_offer_slots(
        lambda i: Priced(i, free_gpus=[]), [worker()], [], limit=4
    )

    assert fits.slots == 2
    assert fits.claim.vram == 40 * 1024**3
    assert full.slots == 0
    assert full.claim.ram == 1024**3


@pytest.mark.asyncio
async def test_a_selector_that_prices_nothing_is_not_a_member_that_costs_nothing():
    """`FakeSelector` has no `get_resource_claim`, like the GGUF selector, which
    sizes per layer and per GPU. Absent has to stay absent: a caller showing
    zero bytes would be inventing a figure nobody measured."""
    result = await count_offer_slots(
        selector_factory(free_gpus=[0]), [worker()], [], limit=2
    )

    assert result.claim is None
