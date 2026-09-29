"""Run blocking container provisioning with isolated, per-instance output."""

import multiprocessing
import logging
import threading
from pathlib import Path
from typing import Optional

from gpustack_runtime.deployer import WorkloadPlan, create_workload
from gpustack_runtime.logging import setup_logging as setup_runtime_logging

from gpustack.logging import RedirectStdoutStderr, setup_logging
from gpustack.utils.process import add_signal_handlers, terminate_process_tree
from gpustack.worker.log_sources import CappedLogWriter

logger = logging.getLogger(__name__)


def _provision(plan: WorkloadPlan, log_path: str, debug: bool) -> None:
    add_signal_handlers()
    with CappedLogWriter(log_path, append=True) as log_file:
        with RedirectStdoutStderr(log_file):
            setup_logging(debug=debug)
            setup_runtime_logging()
            # Runtime image pull output belongs to this launch, including the
            # stdout progress emitted before any container exists.
            try:
                create_workload(plan)
            except Exception:
                logger.exception("Container provisioning failed")
                raise


def run_provisioning(
    plan: WorkloadPlan,
    log_path: Path,
    cancelled: threading.Event,
    debug: bool = False,
) -> Optional[int]:
    """Provision in a child, returning its exit code or None on cancellation.

    The caller owns the cancellation token and status write-back. A cancelled
    attempt cannot report success or keep creating a container after teardown.
    """
    if cancelled.is_set():
        return None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = multiprocessing.get_context("spawn").Process(
        target=_provision, args=(plan, str(log_path), debug)
    )
    process.start()
    try:
        while process.is_alive():
            if cancelled.wait(0.2):
                terminate_process_tree(process.pid)
                process.join()
                return None
            process.join(timeout=0)
        process.join()
        return None if cancelled.is_set() else process.exitcode
    finally:
        if process.is_alive():
            terminate_process_tree(process.pid)
            process.join()
        process.close()
