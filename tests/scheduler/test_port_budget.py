"""A host runs out of ports before it runs out of cards.

Ports come out of `service_port_range` (64 by default) and are handed out on
the worker, at container start. A single-role deployment takes one each, so
the scheduler never had to think about it. A Mooncake prefill takes one per
worker rank — nine for TP8 — and a group concentrates its members onto as few
hosts as it can, which is the combination that empties the pool with cards
still free.
"""

from types import SimpleNamespace

import pytest

from gpustack.scheduler import port_budget
from gpustack.schemas.models import RoleNameEnum

MOONCAKE = "vllm-ascend-mooncake"
NIXL = "sglang-nixl"


def _model(mode=None, backend_parameters=None):
    return SimpleNamespace(
        disaggregation=SimpleNamespace(mode=mode) if mode else None,
        backend_parameters=backend_parameters or [],
    )


def _instance(worker_id, ports=None, named_ports=None):
    return SimpleNamespace(
        worker_id=worker_id, ports=ports or [], named_ports=named_ports or {}
    )


def _band(base, count):
    return SimpleNamespace(base=base, count=count)


class TestMemberDemand:
    def test_a_plain_model_takes_one_port(self):
        """No PD mode, no bands — the count the scheduler already assumed."""
        assert port_budget.member_port_demand(_model(), None, 8) == 1

    def test_mooncake_prefill_takes_one_port_per_card(self):
        """Per *worker rank*, not per tensor-parallel rank.

        The two are the same number under TP8/DP1, which is where the earlier
        `{{tensor_parallel_size}}` reading came from and why it survived. A
        DP4xTP4 member binds sixteen.
        """
        demand = port_budget.member_port_demand(
            _model(MOONCAKE), RoleNameEnum.PREFILL.value, cards=8
        )
        assert demand == 1 + 8

    def test_card_count_scales_the_band(self):
        one = port_budget.member_port_demand(
            _model(MOONCAKE), RoleNameEnum.PREFILL.value, cards=1
        )
        four = port_budget.member_port_demand(
            _model(MOONCAKE), RoleNameEnum.PREFILL.value, cards=4
        )
        assert (one, four) == (2, 5)

    def test_a_fixed_width_band_ignores_the_card_count(self):
        demand = port_budget.member_port_demand(
            _model(NIXL), RoleNameEnum.PREFILL.value, cards=8
        )
        assert demand == 1 + 1

    def test_an_unknown_mode_falls_back_to_the_serving_port(self):
        demand = port_budget.member_port_demand(
            _model("no-such-mode"), RoleNameEnum.PREFILL.value, cards=8
        )
        assert demand == 1


def _spanning(worker_id, subordinates, ports=None, named_ports=None):
    return SimpleNamespace(
        worker_id=worker_id,
        ports=ports or [],
        named_ports=named_ports or {},
        distributed_servers=SimpleNamespace(
            subordinate_workers=[SimpleNamespace(worker_id=w) for w in subordinates]
        ),
    )


class TestPortsTaken:
    def test_a_member_on_another_worker_does_not_count(self):
        instances = [_instance(1, ports=[40000, 40001]), _instance(2, ports=[40002])]
        assert port_budget.ports_taken_on(1, instances) == 2

    def test_a_spanning_member_counts_on_every_machine_it_holds(self):
        """A member is one row holding one set of ports, and each host it
        landed on fences that same set: the subordinate's own pass finds the
        ports already assigned and re-registers rather than allocating. Read
        off `worker_id` alone, a running spanning member was invisible on its
        subordinates — so the budget offered ports the allocator would then
        refuse, and the member that took them wedges in `starting`."""
        member = _spanning(
            1,
            [2],
            ports=[40000],
            named_ports={"kv_port": _band(base=40001, count=8)},
        )

        primary = port_budget.ports_taken_on(1, [member])
        subordinate = port_budget.ports_taken_on(2, [member])

        assert primary == 9
        assert subordinate == primary, "the same ports are fenced on both"
        assert port_budget.ports_taken_on(3, [member]) == 0

    def test_a_bands_base_is_not_charged_twice(self):
        """`ports` and `named_ports` overlap by design.

        The base of a band appears in both indexes. Summing their lengths
        would charge a TP8 member ten ports for the nine it holds, and the
        error compounds per member.
        """
        instance = _instance(
            1,
            ports=[40000, 40001],
            named_ports={"kv_port": _band(base=40001, count=8)},
        )
        # 40000 plus the band 40001..40008.
        assert port_budget.ports_taken_on(1, [instance]) == 9

    def test_cache_servers_come_out_of_the_same_pool(self):
        model_instances = [_instance(1, ports=[40000])]
        cache = [SimpleNamespace(worker_id=1, port=40010, metrics_port=40011)]
        assert port_budget.ports_taken_on(1, model_instances, cache) == 3

    def test_an_unplaced_cache_instance_is_skipped(self):
        cache = [SimpleNamespace(worker_id=None, port=40010, metrics_port=None)]
        assert port_budget.ports_taken_on(1, [], cache) == 0


class TestCapacity:
    def test_the_default_pool_holds_seven_tp8_mooncake_members(self):
        """64 ports, 9 each. The number that motivates the whole check: a
        host with eight free cards has ports for seven members."""
        assert port_budget.port_capacity("40000-40063", 9, 0) == 7

    def test_ports_already_taken_reduce_it(self):
        assert port_budget.port_capacity("40000-40063", 9, 20) == 4

    def test_a_full_pool_is_zero_not_negative(self):
        assert port_budget.port_capacity("40000-40063", 9, 64) == 0
        assert port_budget.port_capacity("40000-40063", 9, 100) == 0

    def test_an_unreadable_range_is_no_limit_rather_than_no_capacity(self):
        """Zero is the one answer that stops an operator looking for a
        mistake, so a configuration this cannot read must not produce it."""
        assert port_budget.port_capacity("not-a-range", 9, 0) is None
        assert port_budget.port_capacity(None, 9, 0) is None

    def test_no_demand_is_no_limit(self):
        assert port_budget.port_capacity("40000-40063", 0, 0) is None


@pytest.mark.parametrize(
    "role", [RoleNameEnum.PREFILL.value, RoleNameEnum.DECODE.value]
)
def test_both_weight_holding_roles_declare_the_mooncake_band(role):
    assert port_budget.member_port_demand(_model(MOONCAKE), role, cards=4) > 1


class TestPortsCommittedDuringOneSolve:
    """VRAM crosses roles through `_translate`; ports have to as well.

    `_share_out` deals two roles onto the same roomiest worker on purpose, so
    the second role is priced on a host the first has already taken ports on —
    and pricing it against the pre-solve snapshot is what admits a group for
    more ports than the range holds, leaving its trailing members wedged in
    `starting`.
    """

    @staticmethod
    def _capacity(model):
        from gpustack.scheduler.group_capacity import GroupCapacity

        cap = GroupCapacity(SimpleNamespace(), model, [], [])
        for role in (RoleNameEnum.PREFILL.value, RoleNameEnum.DECODE.value):
            cap._projected[role] = SimpleNamespace(model=model)
        return cap

    def test_an_earlier_roles_members_are_charged_to_the_worker(self):
        model = _model(MOONCAKE)
        cap = self._capacity(model)
        # What the counting pass offered for this (role, worker): a TP8 member.
        cap._offers[(RoleNameEnum.PREFILL.value, 1)] = [
            SimpleNamespace(gpu_indexes=list(range(8)))
        ]

        committed = cap._committed_port_demand(
            [
                SimpleNamespace(worker_id=1, role=RoleNameEnum.PREFILL.value),
                SimpleNamespace(worker_id=1, role=RoleNameEnum.PREFILL.value),
            ]
        )

        # Two members, each one serving port plus one per worker rank.
        assert committed == {1: 2 * (1 + 8)}

    def test_each_member_is_priced_at_its_own_roles_demand(self):
        """A Mooncake prefill takes nine ports and its router two — a serving
        port plus the prometheus band the recipe declares. Charging one at the
        other's rate is how a budget stops being one."""
        model = _model(MOONCAKE)
        cap = self._capacity(model)
        cap._projected[RoleNameEnum.ROUTER.value] = SimpleNamespace(model=model)
        cap._offers[(RoleNameEnum.PREFILL.value, 1)] = [
            SimpleNamespace(gpu_indexes=list(range(8)))
        ]
        cap._offers[(RoleNameEnum.ROUTER.value, 1)] = [SimpleNamespace(gpu_indexes=[])]

        committed = cap._committed_port_demand(
            [
                SimpleNamespace(worker_id=1, role=RoleNameEnum.PREFILL.value),
                SimpleNamespace(worker_id=1, role=RoleNameEnum.ROUTER.value),
            ]
        )

        assert committed == {1: (1 + 8) + (1 + 1)}

    def test_nothing_committed_yet_charges_nothing(self):
        """The first role of a solve, and every single-role deployment."""
        cap = self._capacity(_model(MOONCAKE))
        assert cap._committed_port_demand([]) == {}

    def test_a_member_spanning_two_machines_is_charged_on_both(self):
        """The worker fences one set of ports on every host the member lands on.

        `_assign_named_ports` sizes the bands once, on the primary, and each
        host in the span then re-registers that same set. So a later role
        asking for ports on the subordinate has to be priced against them, or
        the solve promises room the allocator will find taken.
        """
        model = _model(MOONCAKE)
        cap = self._capacity(model)
        cap._offers[(RoleNameEnum.PREFILL.value, 1)] = [
            SimpleNamespace(gpu_indexes=list(range(8)))
        ]
        cap._spans[(RoleNameEnum.PREFILL.value, 1)] = [1, 2]

        committed = cap._committed_port_demand(
            [SimpleNamespace(worker_id=1, role=RoleNameEnum.PREFILL.value)]
        )

        assert committed == {1: 1 + 8, 2: 1 + 8}

    def test_each_combination_is_priced_at_its_own_width(self):
        """The spanning pass makes ONE offer holding every combination it
        found, so reading the first placement's cards would price a 2-card
        combination at an 8-card one's width — in whichever direction the
        first one happens to be wrong."""
        model = _model(MOONCAKE)
        cap = self._capacity(model)
        cap._config = SimpleNamespace(service_port_range="40000-40063")
        cap._ports_taken = {1: 0, 2: 0}
        offer = SimpleNamespace(slots=20, placements=[])

        wide = cap._within_port_budget(RoleNameEnum.PREFILL.value, 1, offer, 8)
        narrow = cap._within_port_budget(RoleNameEnum.PREFILL.value, 2, offer, 2)

        # 64 ports in the range: nine per member at eight cards, three at two.
        assert wide == 64 // (1 + 8)
        assert narrow == 20, "three ports each leaves room for every slot offered"
