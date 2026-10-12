from datetime import datetime, timezone
from types import SimpleNamespace

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
