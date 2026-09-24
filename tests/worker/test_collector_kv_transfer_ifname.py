"""Whether the KV-transfer NIC was derived or typed in, readable from the row.

On a multi-NIC Ascend host the control-plane recipes derive
`HCCL_SOCKET_IFNAME` from `Worker.ifname`, and an operator can hand-fill
`--kv-transfer-ifname` with that same name. An e2e that reads
`HCCL_SOCKET_IFNAME=bond1` off a running container therefore cannot tell the
derivation from the workaround — the two produce the same string, and the
rendered env is not readable from the server.

So the input is reported. `status.kv_transfer_ifname is None` beside a non-null `ifname`
can only mean the value was derived; anything else means someone is still
typing it in. That is the whole of what these tests pin.
"""

from types import SimpleNamespace

from gpustack.schemas.workers import WorkerStatus
from gpustack.worker.collector import WorkerStatusCollector


def _collector(kv_transfer_ifname=None):
    collector = WorkerStatusCollector(
        cfg=SimpleNamespace(
            get_gpu_devices=lambda: None,
            get_system_info=lambda: None,
            get_system_reserved=lambda: {},
            advertise_address=None,
            worker_port=10150,
            worker_metrics_port=10151,
            disable_worker_metrics=False,
            proxy_mode="tunnel",
            kv_transfer_ifname=kv_transfer_ifname,
        ),
        worker_ip_getter=lambda: "10.0.0.1",
        worker_ifname_getter=lambda: "bond1",
        worker_id_getter=lambda: 1,
        worker_uuid_getter=lambda: "uuid",
    )
    collector._detector_factory = SimpleNamespace(
        detect_system_info=lambda: SimpleNamespace(model_dump=lambda: {}),
        detect_gpus=lambda: [],
    )
    return collector


def test_a_worker_told_nothing_reports_nothing():
    """The signal the e2e reads. Silence here is what makes a
    `HCCL_SOCKET_IFNAME` equal to `ifname` attributable to the derivation."""
    reported = _collector().collect()

    assert reported.status.kv_transfer_ifname is None
    assert reported.ifname == "bond1"


def test_a_hand_filled_kv_transfer_ifname_is_reported_as_configured():
    """The other half, and the one that must not be inferred from `ifname`: the
    operator's answer on this host was the management NIC's own name, so a
    report that dropped values matching `ifname` would erase exactly the case
    the field exists to expose."""
    reported = _collector(kv_transfer_ifname="bond1").collect()

    assert reported.status.kv_transfer_ifname == "bond1"


def test_a_blank_kv_transfer_ifname_is_reported_rather_than_laundered():
    """`derive_net_device` steps over a whitespace-only value and derives
    anyway. Reporting it as None would agree with that reading and hide a
    configuration mistake on the one row anybody looks at."""
    assert (
        _collector(kv_transfer_ifname="  ").collect().status.kv_transfer_ifname == "  "
    )


def test_a_status_stored_before_this_field_existed_still_loads():
    """The field rides inside the `status` JSON column, so no migration adds
    it and every row written before today has no such key. Absent must read as
    "nobody set this" rather than raise — otherwise the first upgraded server
    cannot read its own fleet."""
    stored = {"cpu": {"total": 8}, "topology_facts": None}

    assert WorkerStatus.model_validate(stored).kv_transfer_ifname is None
