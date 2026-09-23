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


class TestPortsTaken:
    def test_only_this_worker_counts(self):
        instances = [_instance(1, ports=[40000, 40001]), _instance(2, ports=[40002])]
        assert port_budget.ports_taken_on(1, instances) == 2

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
