"""What a worker says when half of its self-measurement fails.

When a worker's GPU detection succeeds but its system detection raises, it
ships `WorkerStatus.get_default_status()`: memory 0, CPU 0, OS an empty
string, no filesystems. Without a trace on the row, nothing anywhere says so:
the worker stays READY, the UI shows its cards free, and every group placement
refuses it, because every role wants some RAM and the host looks like it has
none. The reason exists only in that worker's own log, on a machine the
operator has no reason to open.

So the property pinned here is not the detection — it is that a failure to
measure leaves a trace on the row the server and the UI already read.
"""

from types import SimpleNamespace

import pytest

from gpustack.detectors.base import GPUDetectExepction
from gpustack.worker.collector import WorkerStatusCollector


def _collector(system_info_error=None, gpu_error=None):
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
            kv_ifname=None,
        ),
        worker_ip_getter=lambda: "10.0.0.1",
        worker_ifname_getter=lambda: "eth0",
        worker_id_getter=lambda: 1,
        worker_uuid_getter=lambda: "uuid",
    )

    def detect_system_info():
        if system_info_error:
            raise system_info_error
        return SimpleNamespace(model_dump=lambda: {})

    def detect_gpus():
        if gpu_error:
            raise gpu_error
        return []

    collector._detector_factory = SimpleNamespace(
        detect_system_info=detect_system_info,
        detect_gpus=detect_gpus,
    )
    return collector


def test_a_system_info_failure_reaches_the_row():
    """The case above. Without this the only difference between "this host has
    no memory" and "this host could not measure itself" is a log line."""
    collector = _collector(system_info_error=RuntimeError("no /proc"))

    reported = collector.collect()

    assert reported.state_message
    assert "no /proc" in reported.state_message
    # Named consequences, because "detection failed" alone does not tell the
    # reader why their deployments stopped landing here.
    assert "cannot be scheduled" in reported.state_message
    assert reported.status.memory.total == 0


def test_a_gpu_failure_still_reports_the_way_it_did():
    collector = _collector(gpu_error=GPUDetectExepction("no driver"))

    reported = collector.collect()

    assert reported.state_message == "no driver"


def test_both_halves_failing_says_both():
    """They fail independently — the live case had exactly one of them go —
    so neither may overwrite the other."""
    collector = _collector(
        system_info_error=RuntimeError("no /proc"),
        gpu_error=GPUDetectExepction("no driver"),
    )

    reported = collector.collect()

    assert "no /proc" in reported.state_message
    assert "no driver" in reported.state_message


def test_a_healthy_worker_says_nothing():
    """The message is a fault channel. A worker that measured itself must not
    ship prose the UI would have to decide not to show."""
    assert _collector().collect().state_message is None


@pytest.mark.parametrize("initial", [True, False])
def test_the_initial_pass_still_skips_gpu_detection(initial):
    """`initial=True` reports before the GPU detectors are ready; that is
    unchanged, and the system-info half must not start depending on it."""
    collector = _collector(system_info_error=RuntimeError("no /proc"))

    reported = collector.collect(initial=initial)

    assert "no /proc" in reported.state_message
