"""Documentation links in worker state messages point at the bundled docs.

``compute_state`` is the only place the backend hands a documentation URL to the
UI. An air-gapped deployment cannot reach ``docs.gpustack.ai``, so both links
must be root-absolute paths under ``/help/`` — the static docs site the server
mounts itself.
"""

from datetime import datetime, timedelta, timezone

from gpustack.schemas.workers import WorkerStateEnum
from tests.fixtures.workers.fixtures import linux_nvidia_1_4090_24gx1


def test_heartbeat_lost_message_links_to_bundled_docs():
    worker = linux_nvidia_1_4090_24gx1()
    worker.maintenance = None
    worker.state = WorkerStateEnum.READY
    worker.state_message = None
    worker.heartbeat_time = datetime.now(timezone.utc) - timedelta(seconds=600)

    worker.compute_state()

    assert worker.state == WorkerStateEnum.NOT_READY
    assert "/help/troubleshooting/#view-gpustack-logs" in worker.state_message
    assert "docs.gpustack.ai" not in worker.state_message


def test_unreachable_message_links_to_bundled_docs():
    worker = linux_nvidia_1_4090_24gx1()
    worker.maintenance = None
    worker.state = WorkerStateEnum.READY
    worker.state_message = None
    worker.heartbeat_time = datetime.now(timezone.utc)
    worker.unreachable = True

    worker.compute_state()

    assert worker.state == WorkerStateEnum.UNREACHABLE
    assert "/help/installation/requirements/#port-requirements" in worker.state_message
    assert "docs.gpustack.ai" not in worker.state_message
