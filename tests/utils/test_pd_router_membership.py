from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from gpustack.utils.pd_router_membership import router_peer_snapshot


def peer(**overrides):
    fields = dict(
        id=1,
        role="prefill",
        group_id="g",
        worker_id=1,
        port=8000,
        named_ports={"bootstrap": SimpleNamespace(base=9000)},
        state="running",
        draining_since=None,
        restart_count=0,
        last_restart_time=None,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_snapshot_excludes_other_generations_unready_and_draining_members():
    active = peer()
    ignored = [
        peer(id=2, group_id="other"),
        peer(id=3, state="starting"),
        peer(id=4, draining_since=datetime.now(timezone.utc)),
        peer(id=5, role="router"),
    ]
    assert router_peer_snapshot([active, *ignored], "g", {1: "10.0.0.1"}) == (
        router_peer_snapshot([active], "g", {1: "10.0.0.1"})
    )


def test_address_reuse_bootstrap_change_and_restart_invalidate_snapshot():
    before = router_peer_snapshot([peer()], "g", {1: "10.0.0.1"})
    for changed in [
        peer(id=2),
        peer(restart_count=1),
        peer(last_restart_time=datetime.now(timezone.utc)),
        peer(named_ports={"bootstrap": SimpleNamespace(base=9001)}),
    ]:
        assert router_peer_snapshot([changed], "g", {1: "10.0.0.1"}) != before
    assert before[0]["ports"] == {"bootstrap": 9000}


def test_snapshot_is_order_independent_and_requires_resolved_addresses():
    peers = [peer(), peer(id=2, role="decode", port=8001)]
    assert router_peer_snapshot(peers, "g", {1: "10.0.0.1"}) == (
        router_peer_snapshot(peers[::-1], "g", {1: "10.0.0.1"})
    )
    assert router_peer_snapshot(peers, "g", {}) == []


def test_hybrid_snapshot_preserves_two_worker_addresses_and_shared_bootstrap_on_restart(
    caplog,
):
    from gpustack.schemas.models import (
        DistributedServers,
        ModelInstanceSubordinateWorker,
        PortBand,
    )
    from tests.utils.model import new_hybrid_workload, new_model_instance

    instance = new_model_instance(
        7, 'hybrid', 1, worker_id=1, state='running', gpu_indexes=[0, 1, 2, 3]
    )
    instance.role, instance.group_id, instance.worker_ip = 'prefill', 'g', '10.0.0.1'
    instance.named_ports = {'bootstrap': PortBand(base=9000)}
    instance.distributed_servers = DistributedServers(
        subordinate_workers=[
            ModelInstanceSubordinateWorker(
                worker_id=2, worker_ip='10.0.0.2', gpu_indexes=[0, 1, 2, 3]
            )
        ],
        workloads=[new_hybrid_workload(w, state='running') for w in (1, 2)],
    )
    for task in instance.distributed_servers.workloads:
        task.component.api_port, task.component.ready = 8000, True
    worker_ips = {1: '10.0.0.1', 2: '10.0.0.2'}
    before = router_peer_snapshot([instance], 'g', worker_ips)
    assert before == [
        dict(
            id=7,
            role='prefill',
            ip=ip,
            port=8000,
            ports={'bootstrap': 9000},
            restart_count=0,
            last_restart_time=None,
        )
        for ip in ('10.0.0.1', '10.0.0.2')
    ]
    with caplog.at_level('DEBUG', logger='gpustack.utils.pd_router_membership'):
        assert router_peer_snapshot([instance], 'g', {1: '10.0.0.1'}) == []
    assert 'instance 7: no current address for Workers [2]' in caplog.text
    instance.restart_count = 1
    after = router_peer_snapshot([instance], 'g', worker_ips)
    assert after == [dict(peer, restart_count=1) for peer in before]


@pytest.mark.parametrize('workers', [1, 2])
def test_external_rank_snapshots_pair_each_api_with_its_bootstrap_port(workers):
    from gpustack.schemas.models import DistributedServers, PortBand
    from tests.utils.model import new_model_instance, new_model_instance_workload

    instance = new_model_instance(
        7, 'external', 1, worker_id=1, state='running', gpu_indexes=[0, 1]
    )
    instance.role, instance.group_id, instance.worker_ip = 'prefill', 'g', '10.0.0.1'
    instance.named_ports = {'bootstrap': PortBand(base=9000, count=2)}
    tasks = []
    for rank in range(2):
        task = new_model_instance_workload(
            1 + rank if workers == 2 else 1, rank, state='running'
        )
        task['name'] = f'engine-{rank}'
        task['dp_rank_placements'][0].update(
            api_port=8000 + rank,
            ready=True,
            gpu_indexes=[rank] if workers == 1 else [0],
        )
        tasks.append(task)
    ips = {1: '10.0.0.1', 2: '10.0.0.2'}
    instance.distributed_servers = DistributedServers(
        workloads=tasks,
        subordinate_workers=(
            [dict(worker_id=2, worker_ip=ips[2], gpu_indexes=[0])]
            if workers == 2
            else []
        ),
    )
    before = router_peer_snapshot([instance], 'g', ips)
    assert [(p['ip'], p['port'], p['ports']['bootstrap']) for p in before] == [
        (ips[1 + rank if workers == 2 else 1], 8000 + rank, 9000 + rank)
        for rank in range(2)
    ]
    instance.restart_count = 1
    after = router_peer_snapshot([instance], 'g', ips)
    assert before != after
    assert all(p['restart_count'] == 1 for p in after)
