"""A host runs out of ports before it runs out of cards.

Ports come out of `service_port_range` (64 by default) and are handed out on
the worker, at container start. A single-role deployment takes one each, so
the scheduler never had to think about it. A Mooncake prefill takes one per
worker rank — nine for TP8 — and a group concentrates its members onto as few
hosts as it can, which is the combination that empties the pool with cards
still free.
"""

from gpustack.policies.resource_view import ResourceView
from types import SimpleNamespace

import pytest

from gpustack.scheduler import port_budget
from gpustack.schemas.cache_services import CacheServiceInstanceBase
from gpustack.schemas.models import BackendEnum, RoleNameEnum

MOONCAKE = "vllm-ascend-mooncake"
NIXL = "sglang-nixl"


def _model(mode=None, backend_parameters=None):
    return SimpleNamespace(
        roles=[],
        backend=(
            BackendEnum.SGLANG
            if mode and mode.startswith("sglang")
            else BackendEnum.VLLM
        ),
        disaggregation=SimpleNamespace(mode=mode) if mode else None,
        backend_parameters=backend_parameters or [],
        backend_version=None,
        image_name=None,
        categories=[],
    )


def _instance(worker_id, ports=None, named_ports=None):
    return SimpleNamespace(
        worker_id=worker_id, ports=ports or [], named_ports=named_ports or {}
    )


def _band(base, count):
    return SimpleNamespace(base=base, count=count)


def _cache_instance(worker_id, ports=None, port=None):
    return CacheServiceInstanceBase(
        name="cache-instance",
        cache_service_id=1,
        worker_id=worker_id,
        cluster_id=1,
        ports=ports,
        port=port,
    )


class TestMemberDemand:
    @pytest.mark.parametrize('mode', ['external', 'hybrid'])
    def test_disabled_native_lb_flags_keep_internal_port_budget(self, mode):
        parameters = ['-tp=2', '-dp=2']
        assert port_budget.member_port_requests(
            _model(
                backend_parameters=[
                    f'--data-parallel-{mode}-lb',
                    f'--no-data-parallel-{mode}-lb',
                    *parameters,
                ]
            ),
            None,
            [4],
        ) == port_budget.member_port_requests(
            _model(backend_parameters=parameters), None, [4]
        )

    @pytest.mark.parametrize(
        'mode, expected',
        [
            ('internal', 14),
            ('external', 15),
            ('hybrid', 2),
            ('multi_port_external', 16),
        ],
    )
    def test_template_values_cannot_change_vllm_port_demand(self, mode, expected):
        parameters = [
            '--tp=2',
            '--dp=4',
            '--dpl=2',
            '--distributed_executor_backend=mp',
            '--chat-template="You\'re helpful. --data-parallel-hybrid-lb --tp=8"',
        ]
        if mode != 'internal':
            parameters.append(f'--data-parallel-{mode.replace("_", "-")}-lb')
        assert (
            port_budget.member_port_demand(
                _model(backend_parameters=parameters), None, 4, gpu_per_node=[4, 4]
            )
            == expected
        )

    def test_non_vllm_parameters_do_not_use_vllm_mode_validation(self):
        model = _model(
            NIXL,
            ["--data-parallel-external-lb", "--data-parallel-multi-port-external-lb"],
        )
        assert port_budget.member_port_demand(model, "prefill", 4) == 2

    @pytest.mark.parametrize(
        "flag, nodes, parameters, mode, expected",
        [
            ("--data-parallel-external-lb", [4], ["--dp=2", "--tp=2"], None, 13),
            (
                "--data-parallel-multi-port-external-lb",
                [4],
                ["--dp=2", "--tp=2"],
                None,
                14,
            ),
            ("--data-parallel-external-lb", [4], ["--dp=2", "--tp=2"], MOONCAKE, 17),
            ("--data-parallel-external-lb", [4, 4], ["--dp=2", "--tp=4"], MOONCAKE, 21),
            (
                "--data-parallel-external-lb",
                [4, 4],
                ["--dp=2", "--tp=4", "--data-parallel-size-local=2"],
                MOONCAKE,
                23,
            ),
        ],
    )
    def test_managed_demand_includes_rank_and_communication_ports(
        self, flag, nodes, parameters, mode, expected
    ):
        model = _model(mode, [flag, *parameters])
        model.backend = BackendEnum.VLLM

        demand = port_budget.member_port_demand(
            model, "prefill" if mode else None, nodes[0], gpu_per_node=nodes
        )

        assert demand == expected
        assert port_budget.port_capacity("40000-40063", demand, 40) == 24 // expected

    @pytest.mark.parametrize(
        'nodes,parameters,mode,expected',
        [
            ([4], ['--tp=2', '--dp=2'], None, 2),
            ([4, 4], ['--tp=2', '--dp=4'], None, 2),
            ([4, 4], ['--tp=2', '--dp=4', '--dpl=2'], MOONCAKE, 10),
            ([4, 4], ['--tp=2', '--dp=4'], 'vllm-nixl', 6),
            ([2, 2, 2, 2], ['--tp=4', '--dp=2'], None, 14),
            ([4, 4, 4, 4], ['--tp=4', '--pp=2', '--dp=2', '--dpl=1'], MOONCAKE, 30),
        ],
    )
    def test_hybrid_demand_counts_shared_api_and_cross_rank_coordination(
        self, nodes, parameters, mode, expected
    ):
        model = _model(mode, ['--data-parallel-hybrid-lb', *parameters])
        assert (
            port_budget.member_port_demand(
                model, 'prefill' if mode else None, nodes[0], gpu_per_node=nodes
            )
            == expected
        )

    @pytest.mark.parametrize("parameters", [["--tp=16"], ["--dp=0"]])
    def test_invalid_spanning_topology_cannot_produce_a_port_plan(self, parameters):
        model = _model("vllm-nixl", parameters + ["--distributed-executor-backend=mp"])
        with pytest.raises(ValueError):
            port_budget.member_port_requests(model, "prefill", [4, 4])

    @pytest.mark.parametrize(
        "parameters, expected",
        [([], 2), (["--dp=2"], 3), (["--data-parallel-size", "4"], 5)],
    )
    def test_internal_nixl_counts_dp_side_channels(self, parameters, expected):
        model = _model("vllm-nixl", parameters)
        assert port_budget.member_port_demand(model, "prefill", 8) == expected

    @pytest.mark.parametrize("mode, expected", [("vllm-nixl", 18), (MOONCAKE, 22)])
    def test_spanning_member_covers_global_rank_range(self, mode, expected):
        model = _model(mode, ["--tp=2", "--distributed-executor-backend=mp"])
        assert (
            port_budget.member_port_demand(model, "prefill", 4, gpu_per_node=[4, 4])
            == expected
        )

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

    @pytest.mark.parametrize(
        "ports, port, expected",
        [
            ({"service": 40010, "metrics": 40011, "management": 40012}, 40010, 4),
            ({"service": 40010, "metrics": 40011}, None, 3),
            ({"service": 40010, "service_alias": 40010}, 40010, 2),
            ({"service": 40000, "metrics": 40011}, 40000, 2),
            ({"metrics": 40011}, 40010, 3),
            (None, 40010, 2),
            ({}, 40010, 2),
            (None, None, 1),
        ],
    )
    def test_cache_servers_come_out_of_the_same_pool(self, ports, port, expected):
        model_instances = [_instance(1, ports=[40000])]
        cache = [_cache_instance(1, ports=ports, port=port)]
        assert port_budget.ports_taken_on(1, model_instances, cache) == expected

    @pytest.mark.parametrize("worker_id, expected", [(1, 2), (2, 3), (3, 0)])
    def test_cache_ports_are_counted_only_on_the_assigned_worker(
        self, worker_id, expected
    ):
        cache = [
            _cache_instance(1, ports={"service": 40010, "metrics": 40011}, port=40010),
            _cache_instance(
                2,
                ports={"service": 40010, "metrics": 40011, "management": 40012},
                port=40010,
            ),
        ]
        assert port_budget.ports_taken_on(worker_id, [], cache) == expected


class TestCapacity:
    def test_the_default_pool_holds_seven_tp8_mooncake_members(self):
        """64 ports, 9 each. The number that motivates the whole check: a
        host with eight free cards has ports for seven members."""
        assert port_budget.port_capacity("40000-40063", 9, 0) == 7

    def test_ports_already_taken_reduce_it(self):
        assert port_budget.port_capacity("40000-40063", 9, 20) == 4

    def test_all_cache_ports_reduce_member_capacity(self):
        cache = [
            _cache_instance(
                1,
                ports={"service": 40000, "metrics": 40001, "management": 40002},
                port=40000,
            )
        ]
        taken = port_budget.ports_taken_on(1, [], cache)

        assert port_budget.port_capacity("40000-40009", 9, taken) == 0

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

        cap = GroupCapacity(
            SimpleNamespace(), model, [], [], resource_view=ResourceView([])
        )
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

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mode, demand", [("vllm-nixl", 20), (MOONCAKE, 26)])
    @pytest.mark.parametrize("reverse", [False, True])
    async def test_spanning_budget_covers_every_candidate_for_a_primary(
        self, mode, demand, reverse, monkeypatch
    ):
        from unittest.mock import AsyncMock

        from gpustack.scheduler import group_capacity

        model = _model(mode, ["--tp=2", "--distributed-executor-backend=mp"])
        model.distributed_inference_across_workers = True
        cap = self._capacity(model)
        cap._config = SimpleNamespace(service_port_range="40000-40039")
        workers = {
            worker_id: SimpleNamespace(
                id=worker_id,
                status=SimpleNamespace(memory=SimpleNamespace(total=1)),
            )
            for worker_id in [1, 2, 3]
        }
        candidates = [
            SimpleNamespace(
                worker=workers[1],
                gpu_indexes=list(range(4)),
                subordinate_workers=[
                    SimpleNamespace(worker_id=worker_id, gpu_indexes=list(range(cards)))
                ],
            )
            for worker_id, cards in [(2, 8), (3, 4)]
        ]
        if reverse:
            candidates.reverse()
        monkeypatch.setattr(
            group_capacity, "role_takes_no_accelerator", lambda *a: False
        )
        monkeypatch.setattr(
            group_capacity,
            "count_offer_slots",
            AsyncMock(
                return_value=SimpleNamespace(
                    slots=2, placements=candidates, notes=[], claim=None
                )
            ),
        )

        await cap._spanning("prefill", list(workers), workers, [], 2, {})
        cap._committed_ports = cap._committed_port_demand(
            [SimpleNamespace(worker_id=1, role="prefill")]
        )

        assert cap._committed_ports == {1: demand, 2: demand, 3: demand}
        assert cap._within_port_budget(
            "decode", 1, SimpleNamespace(slots=2), 4, gpu_per_node=[4, 4]
        ) == max(0, (40 - demand) // (18 if mode == "vllm-nixl" else 22))


@pytest.mark.parametrize('counts', [[4], [2, 2]])
@pytest.mark.parametrize('explicit', [False, True])
def test_cuda_mooncake_external_budget_includes_each_bootstrap_listener(
    counts, explicit
):
    params = ['--data-parallel-external-lb', '--tp=1'] + (
        ['--dp=4'] if explicit else []
    )
    ordinary = _model(None, params)
    mooncake = _model('vllm-mooncake', params)
    assert (
        port_budget.member_port_demand(
            mooncake, 'prefill', sum(counts), gpu_per_node=counts
        )
        == port_budget.member_port_demand(
            ordinary, 'prefill', sum(counts), gpu_per_node=counts
        )
        + 4
    )


@pytest.mark.parametrize('counts', [[4], [4, 4]])
def test_multi_port_nixl_budget_includes_global_side_channel_band(counts):
    params = ['--data-parallel-multi-port-external-lb', '--tp=2']
    ordinary = _model(None, params)
    nixl = _model('vllm-nixl', params)
    dp = sum(counts) // 2
    assert (
        port_budget.member_port_demand(
            nixl, 'prefill', sum(counts), gpu_per_node=counts
        )
        == port_budget.member_port_demand(
            ordinary, 'prefill', sum(counts), gpu_per_node=counts
        )
        + dp
    )


def _snapshot(worker_id=1, port_range='40000-40063', busy=()):
    from uuid import uuid4
    from gpustack.schemas.port_reservations import WorkerPortAvailability

    return WorkerPortAvailability(
        worker_id=worker_id,
        generation=uuid4(),
        port_range=port_range,
        unavailable_ports=list(busy),
    )


@pytest.mark.parametrize(
    'snapshots,expected',
    [
        ([_snapshot(port_range='40000-40011')], False),
        ([_snapshot()], True),
        ([_snapshot(busy=range(40009, 40064, 10))], False),
        ([_snapshot(), _snapshot(2, busy=range(40009, 40064, 10))], False),
        ([_snapshot(), _snapshot(2, '40000-40031')], True),
    ],
)
def test_external_layout_checks_capacity_fragmentation_and_common_bands(
    snapshots, expected
):
    model = _model(
        backend_parameters=['--data-parallel-external-lb', '--dp', '4', '--tp', '1']
    )
    requests = port_budget.member_port_requests(model, None, [4])
    assert sum(count for count, _ in requests) == 15
    assert port_budget.fits_port_layout(snapshots, requests) is expected


@pytest.mark.parametrize(
    'snapshots',
    [
        [_snapshot(), _snapshot(2, '40100-40163')],
        [_snapshot(port_range='40063-40000')],
    ],
)
def test_invalid_common_port_range_names_workers_and_configuration(snapshots):
    with pytest.raises(ValueError, match='align service_port_range') as error:
        port_budget.fits_port_layout(snapshots, [(1, None)])
    for snapshot in snapshots:
        assert f'{snapshot.worker_id}: {snapshot.port_range}' in str(error.value)


def test_port_budget_import_does_not_require_worker_runtime_initialization():
    from pathlib import Path
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            '-c',
            'from gpustack.scheduler.port_budget import member_port_requests',
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
@pytest.mark.parametrize('role', [None, 'prefill', 'decode'])
@pytest.mark.parametrize('mode', ['external', 'hybrid'])
async def test_candidate_admission_checks_ports_before_scoring(monkeypatch, role, mode):
    from unittest.mock import AsyncMock
    from gpustack.policies.base import ModelInstanceScheduleCandidate

    workers = [SimpleNamespace(id=1), SimpleNamespace(id=2)]
    model = _model(
        backend_parameters=[f'--data-parallel-{mode}-lb', '--dp', '4', '--tp', '1']
    )
    candidates = [
        ModelInstanceScheduleCandidate(w, [0, 1, 2, 3], None) for w in workers
    ]
    snapshots = AsyncMock(
        return_value={
            1: _snapshot(1, '40000-40000'),
            2: _snapshot(2),
        }
    )
    monkeypatch.setattr(port_budget, 'worker_port_snapshots', snapshots)
    accepted, messages = await port_budget.filter_candidates(
        None, model, role, candidates, workers, [], 7
    )
    assert accepted == [candidates[1]]
    assert messages and 'service ports' in messages[0]
    snapshots.assert_awaited_once_with(workers, 7)


@pytest.mark.asyncio
async def test_role_scoped_band_rejects_candidates_with_a_diagnostic(monkeypatch):
    from unittest.mock import AsyncMock
    from gpustack.policies.base import ModelInstanceScheduleCandidate
    from gpustack.schemas.pd_modes import PDPortScopeEnum

    worker = SimpleNamespace(id=1)
    candidate = ModelInstanceScheduleCandidate(worker, [0], None)
    monkeypatch.setattr(
        'gpustack.worker.pd_injection.band_specs_for',
        lambda *args, **kwargs: [
            SimpleNamespace(scope=PDPortScopeEnum.ROLE, name='kv_port')
        ],
    )
    snapshots = AsyncMock()
    monkeypatch.setattr(port_budget, 'worker_port_snapshots', snapshots)

    accepted, messages = await port_budget.filter_candidates(
        None, _model(MOONCAKE), 'prefill', [candidate], [worker], []
    )

    assert accepted == []
    assert len(messages) == 1
    assert "scope 'role'" in messages[0]
    assert 'kv_port' in messages[0]
    snapshots.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_worker_port_snapshot_does_not_admit_candidate(monkeypatch):
    from unittest.mock import AsyncMock
    from gpustack.policies.base import ModelInstanceScheduleCandidate

    worker = SimpleNamespace(id=1)
    model = _model(backend_parameters=['--data-parallel-external-lb', '--dp', '4'])
    monkeypatch.setattr(
        port_budget, 'worker_port_snapshots', AsyncMock(return_value={})
    )
    accepted, messages = await port_budget.filter_candidates(
        None,
        model,
        None,
        [ModelInstanceScheduleCandidate(worker, [0, 1, 2, 3], None)],
        [worker],
        [],
    )
    assert accepted == []
    assert 'Cannot establish' in messages[0]


@pytest.mark.asyncio
async def test_worker_snapshots_use_worker_transport_once_and_exclude_retry(
    monkeypatch,
):
    from unittest.mock import AsyncMock

    workers = [SimpleNamespace(id=1), SimpleNamespace(id=2)]

    async def read(**kwargs):
        return (
            SimpleNamespace(status=200),
            _snapshot(kwargs['worker'].id).model_dump_json().encode(),
        )

    rpc = AsyncMock(side_effect=read)
    monkeypatch.setattr(port_budget, 'request_to_worker', rpc)
    snapshots = await port_budget.worker_port_snapshots(workers, 7)
    assert set(snapshots) == {1, 2}
    assert rpc.await_count == 2
    for call in rpc.call_args_list:
        assert call.kwargs['path'] == 'port-reservations'
        assert call.kwargs['params'] == {'exclude_instance_id': 7}


@pytest.mark.asyncio
@pytest.mark.parametrize('owner', ['model', 'cache'])
@pytest.mark.parametrize('cluster_id', [None, 11])
async def test_port_admission_includes_persisted_reservations(
    monkeypatch, owner, cluster_id
):
    from unittest.mock import AsyncMock
    from gpustack.policies.base import ModelInstanceScheduleCandidate

    worker = SimpleNamespace(id=1)
    model = _model(backend_parameters=['--data-parallel-external-lb', '--dp', '4'])
    model.cluster_id = cluster_id
    occupied = list(range(40000, 40012))
    cache = _cache_instance(1, ports={'api': 40000}, port=40001)
    cache.cluster_id = cluster_id if cluster_id is not None else 1
    cache.ports = {str(port): port for port in occupied}
    monkeypatch.setattr(
        port_budget,
        'worker_port_snapshots',
        AsyncMock(return_value={1: _snapshot(1, '40000-40020')}),
    )
    instances = [_instance(1, occupied)] if owner == 'model' else []
    session = SimpleNamespace(
        exec=AsyncMock(
            return_value=SimpleNamespace(
                all=lambda: [cache] if owner == 'cache' else []
            )
        )
    )
    accepted, _ = await port_budget.filter_candidates(
        session,
        model,
        None,
        [ModelInstanceScheduleCandidate(worker, [0, 1, 2, 3], None)],
        [worker, SimpleNamespace(id=2)],
        instances,
    )
    assert accepted == []
    session.exec.assert_awaited_once()
    statement = session.exec.call_args.args[0]
    query = str(statement.compile(compile_kwargs={'literal_binds': True}))
    assert 'worker_id IN (1)' in query
    if cluster_id is not None:
        assert f'cluster_id = {cluster_id}' in query
    else:
        assert 'cluster_id =' not in query


@pytest.mark.asyncio
async def test_disjoint_port_ranges_do_not_block_other_candidates(monkeypatch):
    from unittest.mock import AsyncMock
    from gpustack.policies.base import ModelInstanceScheduleCandidate
    from gpustack.schemas.models import ModelInstanceSubordinateWorker

    workers = [SimpleNamespace(id=worker_id) for worker_id in (1, 2, 3)]
    candidates = [
        ModelInstanceScheduleCandidate(
            workers[0],
            list(range(4)),
            None,
            subordinate_workers=[
                ModelInstanceSubordinateWorker(
                    worker_id=worker.id, gpu_indexes=list(range(4))
                )
            ],
        )
        for worker in workers[1:]
    ]
    model = _model(backend_parameters=['--data-parallel-hybrid-lb', '--tp=8', '--dp=1'])
    monkeypatch.setattr(
        port_budget,
        'worker_port_snapshots',
        AsyncMock(
            return_value={
                1: _snapshot(1),
                2: _snapshot(2, '40100-40163'),
                3: _snapshot(3),
            }
        ),
    )

    accepted, messages = await port_budget.filter_candidates(
        None, model, None, candidates, workers, []
    )

    assert accepted == [candidates[1]]
    assert len(messages) == 1
    assert '1: 40000-40063' in messages[0]
    assert '2: 40100-40163' in messages[0]
    assert 'align service_port_range' in messages[0]


@pytest.mark.asyncio
@pytest.mark.parametrize('role', ['prefill', 'router'])
async def test_pd_single_listener_still_requires_a_free_port(monkeypatch, role):
    from unittest.mock import AsyncMock
    from gpustack.policies.base import ModelInstanceScheduleCandidate

    worker = SimpleNamespace(id=1)
    candidate = ModelInstanceScheduleCandidate(worker, [], None)
    monkeypatch.setattr(
        port_budget,
        'worker_port_snapshots',
        AsyncMock(return_value={1: _snapshot(1, '40000-40000', [40000])}),
    )
    accepted, _ = await port_budget.filter_candidates(
        None, _model(), role, [candidate], [worker], []
    )
    assert accepted == []


@pytest.mark.asyncio
@pytest.mark.parametrize('conflicting', [False, True])
async def test_spanning_hybrid_candidate_requires_common_contiguous_bands(
    monkeypatch, conflicting
):
    from unittest.mock import AsyncMock
    from gpustack.policies.base import ModelInstanceScheduleCandidate
    from gpustack.schemas.models import ModelInstanceSubordinateWorker

    workers = [SimpleNamespace(id=1), SimpleNamespace(id=2)]
    candidate = ModelInstanceScheduleCandidate(
        workers[0],
        list(range(4)),
        None,
        subordinate_workers=[
            ModelInstanceSubordinateWorker(worker_id=2, gpu_indexes=list(range(4)))
        ],
    )
    model = _model(backend_parameters=['--data-parallel-hybrid-lb', '--tp=8', '--dp=1'])
    snapshots = {
        1: _snapshot(1, '40000-40031', [40010, 40021]),
        2: _snapshot(2, '40000-40031', [40000, 40011, 40022] if conflicting else []),
    }
    monkeypatch.setattr(
        port_budget, 'worker_port_snapshots', AsyncMock(return_value=snapshots)
    )
    accepted, _ = await port_budget.filter_candidates(
        None, model, None, [candidate], workers, []
    )
    assert bool(accepted) is not conflicting


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'parameters,demand',
    [
        (['--dp=4', '--tp=1'], 4),
        (['--dp=4', '--tp=4', '--enable-dp-attention'], 17),
        (['--attn-dp-size=4', '--tp-size=4'], 17),
    ],
)
async def test_sglang_admission_selects_worker_with_complete_component_ports(
    monkeypatch, parameters, demand
):
    from unittest.mock import AsyncMock
    from gpustack.policies.base import ModelInstanceScheduleCandidate

    model = _model(backend_parameters=parameters)
    model.backend = BackendEnum.SGLANG
    workers = [SimpleNamespace(id=1), SimpleNamespace(id=2)]
    candidates = [
        ModelInstanceScheduleCandidate(w, list(range(4)), None) for w in workers
    ]
    insufficient = 2 if demand == 4 else 16
    monkeypatch.setattr(
        port_budget,
        'worker_port_snapshots',
        AsyncMock(
            return_value={
                1: _snapshot(1, f'40000-{40000 + insufficient - 1}'),
                2: _snapshot(2, f'40000-{40000 + demand - 1}'),
            }
        ),
    )

    accepted, messages = await port_budget.filter_candidates(
        None, model, None, candidates, workers, []
    )

    assert accepted == [candidates[1]]
    assert any(f'cannot fit {demand} service ports' in message for message in messages)


@pytest.mark.asyncio
async def test_internal_pd_admission_preserves_the_connector_gap(monkeypatch):
    from unittest.mock import AsyncMock
    from gpustack.policies.base import ModelInstanceScheduleCandidate

    worker = SimpleNamespace(id=1)
    candidate = ModelInstanceScheduleCandidate(worker, list(range(4)), None)
    monkeypatch.setattr(
        port_budget,
        'worker_port_snapshots',
        AsyncMock(return_value={1: _snapshot(1, '40000-40005', [40004])}),
    )

    accepted, messages = await port_budget.filter_candidates(
        None,
        _model('vllm-nixl', ['--dp=4', '--tp=1']),
        'prefill',
        [candidate],
        [worker],
        [],
    )

    assert accepted == [candidate]
    assert messages == []


@pytest.mark.parametrize('mode', ['external', 'hybrid', 'multi-port-external'])
def test_manual_dp_port_budget_uses_single_legacy_entry(mode):
    from gpustack.utils.model_ports import model_port_requests

    model = _model(
        backend_parameters=[f'--data-parallel-{mode}-lb', '--tp=8', '--dp=2', '--dpl=1']
    )
    model.env = {'GPUSTACK_MANUAL_DP': '1'}
    assert model_port_requests(model, None, [8]) == [('_api', 1, None)]
    assert port_budget.member_port_requests(model, None, [8]) == [(1, None)]


def test_running_managed_port_plan_is_not_changed_by_manual_dp_edit():
    from gpustack.utils.model_ports import model_port_requests

    model = _model(
        backend_parameters=['--data-parallel-external-lb', '--tp=2', '--dp=2']
    )
    model.env = {'GPUSTACK_MANUAL_DP': '1'}
    assert ('_vllm_external_api', 2, None) in model_port_requests(
        model, None, [4], managed=True
    )


def test_multi_port_supervisor_alias_reserves_the_requested_fixed_port():
    from gpustack.utils.model_ports import model_port_requests

    model = _model(
        backend_parameters=[
            '--data_parallel_multi_port_external_lb',
            '--tensor_parallel_size=2',
            '--data_parallel_size=2',
            '--data_parallel_supervisor_port=40100',
        ]
    )
    requests = model_port_requests(model, None, [4])
    assert requests[0] == ('_vllm_multi_port_external_supervisor', 1, 40100)
    assert sum(count for _, count, _ in requests) == 14
