"""What a role asks for, when there is no placement to read it off.

A group that fits is described by the claims its members' candidates carry. A
group that does not fit has no candidates, so the only honest source for "one
prefill member wants 40 GiB" is the selector that priced it while counting —
which is what `demand_for` hands back, alongside how many of that role the
cluster was measured to hold.
"""

from types import SimpleNamespace

import pytest

from gpustack.policies.base import MemberResourceClaim
from gpustack.schemas.models import ComputedResourceClaim, Model, RoleSpec
from gpustack.scheduler.group_capacity import GroupCapacity, _RoleProjection

GIB = 1024**3
CLAIM = ComputedResourceClaim(vram={0: 36 * GIB}, ram=0)


def _worker(id: int):
    return SimpleNamespace(
        id=id,
        name=f"w{id}",
        ip=f"10.0.0.{id}",
        labels={},
        status=SimpleNamespace(memory=SimpleNamespace(total=64 * GIB)),
    )


def _model(prefill=2, decode=1):
    model = Model(name="pd", source="huggingface", huggingface_repo_id="x/y")
    model.id = 1
    model.roles = [
        RoleSpec(name="prefill", replicas=prefill),
        RoleSpec(name="decode", replicas=decode),
    ]
    return model


class _PricedSelector:
    """Hands out `cards` placements in total, and prices one member at 40 GiB."""

    cards = 1

    def __init__(self, instances):
        self._taken = len(
            [e for e in instances if getattr(e, "gpu_indexes", None) is not None]
        )

    def get_resource_claim(self):
        return MemberResourceClaim(vram=40 * GIB, ram=GIB)

    async def select_candidates(self, workers):
        if self._taken >= self.cards:
            return []
        return [
            SimpleNamespace(
                worker=workers[0],
                gpu_indexes=[0],
                gpu_type="cuda",
                gpu_addresses=None,
                computed_resource_claim=CLAIM,
                subordinate_workers=None,
                overcommit=False,
            )
        ]


def _capacity(workers, model=None, selector=_PricedSelector):
    model = model or _model()
    cap = GroupCapacity(SimpleNamespace(), model, workers, [])
    cap._selector = lambda m, instances, cpu_only, ram_claim=None: selector(instances)
    for role in ("prefill", "decode"):
        cap._eligible[role] = {w.id: w for w in workers}
        cap._projected[role] = _RoleProjection(model, False)
    return cap


@pytest.mark.asyncio
async def test_a_role_is_priced_even_where_none_of_it_fits():
    """The case the refusal exists for. Nothing fit, and the breakdown still
    has to say how big the thing that did not fit is."""

    class Full(_PricedSelector):
        cards = 0

    cap = _capacity([_worker(1)], selector=Full)

    claim, placeable = await cap.demand_for("prefill", [1])

    assert claim.vram == 40 * GIB
    assert placeable == 0


@pytest.mark.asyncio
async def test_the_count_never_exceeds_what_the_role_asked_for():
    """Slots are counted per worker and summed, so a fleet with room to spare
    would otherwise report more members than the deployment declares — a role
    of 2 reading "3 placeable" is a breakdown nobody can line up against the
    group's own count."""
    cap = _capacity([_worker(1), _worker(2), _worker(3)])

    _claim, placeable = await cap.demand_for("prefill", [1, 2, 3])

    assert placeable == 2


@pytest.mark.asyncio
async def test_roles_are_measured_without_each_other_standing_in():
    """One unit for the whole breakdown. Measured mid-solve, the second role
    would be answered against the cards the first had just taken, and the two
    numbers would sit in one list looking comparable."""
    cap = _capacity([_worker(1)], model=_model(prefill=1, decode=1))

    _p_claim, prefill = await cap.demand_for("prefill", [1])
    _d_claim, decode = await cap.demand_for("decode", [1])

    # One card, and each role is told it can have it.
    assert (prefill, decode) == (1, 1)
