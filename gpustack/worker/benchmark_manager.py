import asyncio
import multiprocessing
import setproctitle
import os
import time
from typing import Dict, Optional, Callable, List, Tuple
import logging
from collections import deque

from gpustack_runtime.deployer import (
    delete_workload,
    get_workload,
    WorkloadStatusStateEnum,
)
from gpustack.api.exceptions import raise_if_response_error
from gpustack.config.config import Config
from gpustack.config import registration
from gpustack.logging import RedirectStdoutStderr
from gpustack.schemas.benchmark import (
    Benchmark,
    BenchmarkStateEnum,
)
from gpustack.utils.process import terminate_process_tree, add_signal_handlers
from gpustack.worker.benchmark.runner import BenchmarkRunner
from gpustack.client import ClientSet
from gpustack.server.bus import Event, EventType
from gpustack.worker.schemas.benchmark_runner import (
    GenerativeBenchmarksReport,
    GenerativeRequestStats,
)
from gpustack_runtime.deployer import logs_workload


logger = logging.getLogger(__name__)


class BenchmarkManager:
    @property
    def _worker_id(self) -> int:
        return self._worker_id_getter()

    """
    The ID of current worker.
    """
    _config: Config
    """
    Global configuration.
    """
    _benchmark_log_dir: str
    """
    The directory to store logs of benchmarks(in subprocess).
    """
    _benchmark_dir: str
    """
    The directory to store results of benchmarks(in subprocess).
    """

    @property
    def _clientset(self) -> ClientSet:
        return self._clientset_getter()

    """
    The clientset to access the API server.
    """

    _provisioning_processes: Dict[int, multiprocessing.Process]
    """
    The mapping of benchmark ID to provisioning (sub)process.
    When the (sub)process is alive, the benchmark is provisioning.
    If the (sub)process exited, the benchmark is either running or failed.
    """
    _benchmark_by_id: Dict[int, Benchmark]
    _benchmark_queue: deque
    _queue_lock: asyncio.Lock
    _worker_task: Optional[asyncio.Task]
    _active_benchmark_id: Optional[int]
    _active_benchmark_started_at: Optional[float]

    _clientset_getter: Callable[[], ClientSet]
    _worker_id_getter: Callable[[], int]

    def __init__(
        self,
        worker_id_getter: Callable[[], int],
        clientset_getter: Callable[[], ClientSet],
        cfg: Config,
    ):
        self._worker_id_getter = worker_id_getter
        self._config = cfg
        self._benchmark_log_dir = f"{cfg.log_dir}/benchmarks"
        self._benchmark_dir = f"{cfg.benchmark_dir}"
        self._clientset_getter = clientset_getter

        self._provisioning_processes = {}
        self._benchmark_by_id = {}
        self._benchmark_queue = deque()
        self._queue_lock = asyncio.Lock()
        self._worker_task = None
        self._active_benchmark_id = None
        self._active_benchmark_started_at = None

        os.makedirs(self._benchmark_log_dir, exist_ok=True)
        os.makedirs(self._benchmark_dir, exist_ok=True)

    async def watch_benchmarks_event(self):
        """
        Loop to watch benchmarks' event and handle.
        """
        logger.info("Watching benchmarks event.")
        if not self._worker_task or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._benchmark_queue_worker())
        while True:
            try:
                await self._clientset.benchmarks.awatch(
                    callback=self._handle_benchmark_event
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error watching benchmarks: {e}")
                await asyncio.sleep(5)

    def _handle_benchmark_event(self, event: Event):
        """
        Handle benchmark events.
        Args:
            event: The benchmark event to handle.
        """
        benchmark = Benchmark.model_validate(event.data)
        logger.trace(
            f"Received event: {str(event.type)}, id: {benchmark.id}, name: {benchmark.name}, state: {str(benchmark.state)}"
        )
        is_pending = benchmark.state == BenchmarkStateEnum.PENDING
        is_stopped = benchmark.state == BenchmarkStateEnum.STOPPED

        is_current_worker = benchmark.worker_id == self._worker_id
        if not is_current_worker:
            return

        if event.type == EventType.DELETED:
            self._stop_benchmark(benchmark)
            logger.trace(
                f"DELETED event: stopped deleted benchmark {benchmark.name}(id={benchmark.id})."
            )
            return

        if is_pending:
            asyncio.create_task(self._enqueue_benchmark(benchmark))
            return

        if is_stopped:
            asyncio.create_task(self._handle_stop_benchmark_event(benchmark))

    async def _handle_stop_benchmark_event(self, benchmark: Benchmark):
        try:
            self._dump_benchmark_logs_to_file(benchmark)
            self._stop_benchmark(benchmark)
            self._clear_active_benchmark(benchmark.id)
        except Exception as e:
            logger.error(f"Failed to stop benchmark {benchmark.name}: {e}")

    async def _enqueue_benchmark(self, benchmark: Benchmark):
        async with self._queue_lock:
            if benchmark.id not in [b.id for b in self._benchmark_queue]:
                self._benchmark_queue.append(benchmark)

                patch_dict = {"state": BenchmarkStateEnum.QUEUED}
                await self._update_benchmark_state(benchmark.id, **patch_dict)
                logger.info(
                    f"Enqueued benchmark {benchmark.name}(id={benchmark.id}) and set to QUEUED."
                )

    async def _benchmark_queue_worker(self):
        """
        Process benchmarks in the queue.
        """
        while True:
            benchmark = None
            async with self._queue_lock:
                if self._active_benchmark_id is not None:
                    benchmark = None
                elif self._benchmark_queue:
                    benchmark = self._benchmark_queue.popleft()
            if benchmark:
                try:
                    await self._start_benchmark(benchmark)
                except Exception as e:
                    logger.error(
                        f"Failed to start benchmark {benchmark.name}(id={benchmark.id}): {e}"
                    )
            else:
                await asyncio.sleep(1)

    async def _start_benchmark(self, benchmark: Benchmark):
        """
        Start benchmark through a subprocess.
        Args:
            benchmark: The benchmark to start.
        """
        if benchmark.id in self._provisioning_processes:
            logger.warning(
                f"Benchmark {benchmark.name}(id={benchmark.id}) is provisioning. Skipping start."
            )
            return

        log_file_path = f"{self._benchmark_log_dir}/{benchmark.id}.log"
        try:
            if os.path.exists(log_file_path):
                os.remove(log_file_path)
        except Exception as e:
            logger.warning(f"Failed to remove old log file {log_file_path}: {e}")

        try:
            logger.debug(f"Starting benchmark {benchmark.name}(id={benchmark.id})")
            fallback_registry = registration.determine_default_registry(
                self._config.system_default_container_registry
            )
            process = multiprocessing.Process(
                target=BenchmarkManager._launch_benchmark,
                args=(
                    benchmark,
                    self._clientset.headers,
                    log_file_path,
                    self._config,
                    fallback_registry,
                ),
            )
            process.daemon = False
            process.start()

            self._provisioning_processes[benchmark.id] = process
            self._set_active_benchmark(benchmark.id)
            patch_dict = {
                "state": BenchmarkStateEnum.RUNNING,
                "pid": process.pid,
            }
            await self._update_benchmark_state(benchmark.id, **patch_dict)
            logger.info(f"Started benchmark {benchmark.name}(id={benchmark.id})")

        except Exception as e:
            # Clean up provisioning process if started.
            if benchmark.id in self._provisioning_processes:
                self._stop_benchmark(benchmark)
            patch_dict = {
                "state": BenchmarkStateEnum.ERROR,
                "state_message": f"Failed to start benchmark: {e}",
            }
            await self._update_benchmark_state(benchmark.id, **patch_dict)
            logger.error(
                f"Failed to start benchmark {benchmark.name}(id={benchmark.id}): {e}"
            )

    @staticmethod
    def _launch_benchmark(
        benchmark: Benchmark,
        client_headers: dict,
        log_file_path: str,
        cfg: Config,
        fallback_registry: Optional[str] = None,
    ):
        """
        Serve benchmark in a subprocess.
        Exits the subprocess when serving ends.

        Args:
            benchmark: The benchmark to serve.
            client_headers: The headers for the clientset.
            log_file_path: The path to the log file.
            cfg: The configuration.
            fallback_registry: The fallback container registry to use if needed.
        """

        setproctitle.setproctitle(f"gpustack_benchmark_{benchmark.id}")
        add_signal_handlers()

        clientset = ClientSet(
            base_url=cfg.get_server_url(),
            headers=client_headers,
        )

        with open(log_file_path, "w", buffering=1, encoding="utf-8") as log_file:
            with RedirectStdoutStderr(log_file):
                try:
                    server_ins = BenchmarkRunner(
                        clientset,
                        benchmark,
                        cfg,
                        fallback_registry,
                    )
                    logger.info(
                        f"Provisioning benchmark {benchmark.name}(id={benchmark.id})"
                    )
                    server_ins.start()
                    logger.info(
                        f"Finished provisioning benchmark {benchmark.name}(id={benchmark.id})"
                    )
                except Exception as e:
                    logger.exception(
                        f"Error provisioning benchmark {benchmark.name}(id={benchmark.id}): {e}"
                    )
                    raise e

    async def _update_benchmark_state(self, id: int, **kwargs):
        client = self._clientset.http_client.get_async_httpx_client()
        resp = await client.patch(f"/benchmarks/{id}/state", json=kwargs)
        resp.raise_for_status()

    def _update_benchmark_state_sync(self, id: int, **kwargs):
        client = self._clientset.http_client.get_httpx_client()
        resp = client.patch(f"/benchmarks/{id}/state", json=kwargs)
        resp.raise_for_status()

    def _stop_benchmark(self, benchmark: Benchmark):
        """
        Stop benchmark and clean up.

        Args:
            benchmark: The benchmark to stop.
        """

        logger.debug(f"Stopping benchmark {benchmark.name}(id={benchmark.id})")

        # Teardown provisioning process if still alive.
        if self._is_provisioning(benchmark):
            terminate_process_tree(self._provisioning_processes[benchmark.id].pid)

        # Delete workload.
        delete_workload(benchmark.name)

        # Cleanup internal states.
        self._provisioning_processes.pop(benchmark.id, None)
        self._benchmark_by_id.pop(benchmark.id, None)
        self._clear_active_benchmark(benchmark.id)

        logger.info(f"Stopped benchmark {benchmark.name}(id={benchmark.id})")

    def _is_provisioning(self, benchmark: Benchmark) -> bool:
        """
        Check if the benchmark is still provisioning.

        Args:
            benchmark: The benchmark to check.
        """
        if process := self._provisioning_processes.get(benchmark.id):
            if process.is_alive():
                process.join(timeout=0)
                return process.is_alive()
        return False

    def sync_benchmark_state(self):
        """
        Synchronize benchmarks' state.
        - If the provision process is still alive, skip.
        - If the workload is still launching, skip.
        - If the workload is not existed, unhealthy, failed, update the benchmark state to ERROR.
        - If the workload is inactive, update the benchmark state to COMPLETED.
        """
        benchmarks_page = self._clientset.benchmarks.list(
            params={"worker_id": self._worker_id, "state": BenchmarkStateEnum.RUNNING}
        )
        if not benchmarks_page.items:
            return

        for benchmark in benchmarks_page.items:
            self._sync_single_benchmark_state(benchmark)

    def _sync_single_benchmark_state(self, benchmark: Benchmark):
        """Synchronize a single benchmark's state."""
        # Check for timeout
        if self._is_benchmark_timed_out(benchmark):
            self._handle_benchmark_timeout(benchmark)
            return

        # Skip if still provisioning
        if self._is_provisioning(benchmark):
            logger.trace(
                f"Benchmark {benchmark.name}(id={benchmark.id}) is provisioning. Skipping sync."
            )
            return

        # Get workload and handle based on state
        workload = get_workload(benchmark.name)

        if self._should_skip_workload(benchmark, workload):
            return

        if self._is_workload_completed(workload):
            self._handle_benchmark_completion(benchmark)
            return

        if self._is_workload_failed(workload):
            self._handle_benchmark_failure(benchmark)
            return

    def _should_skip_workload(self, benchmark: Benchmark, workload) -> bool:
        """Check if workload should be skipped (still launching or running)."""
        if not workload:
            return False

        if workload.state in [
            WorkloadStatusStateEnum.PENDING,
            WorkloadStatusStateEnum.INITIALIZING,
        ]:
            logger.trace(
                f"Benchmark {benchmark.name}(id={benchmark.id}) workload is still launching. Skipping sync."
            )
            return True

        if workload.state == WorkloadStatusStateEnum.RUNNING:
            logger.trace(
                f"Benchmark {benchmark.name}(id={benchmark.id}) workload is running. Skipping sync."
            )
            return True

        return False

    def _is_workload_completed(self, workload) -> bool:
        """Check if workload has completed successfully."""
        return workload and workload.state == WorkloadStatusStateEnum.INACTIVE

    def _is_workload_failed(self, workload) -> bool:
        """Check if workload has failed or is unhealthy."""
        if not workload:
            return True
        return workload.state in [
            WorkloadStatusStateEnum.UNKNOWN,
            WorkloadStatusStateEnum.UNHEALTHY,
            WorkloadStatusStateEnum.FAILED,
        ]

    def _handle_benchmark_timeout(self, benchmark: Benchmark):
        """Handle benchmark timeout."""
        patch_dict = {
            "state": BenchmarkStateEnum.ERROR,
            "state_message": "Benchmark timed out.",
        }
        self._update_benchmark_state_sync(benchmark.id, **patch_dict)
        self._dump_benchmark_logs_to_file(benchmark)
        self._stop_benchmark(benchmark)

    def _handle_benchmark_completion(self, benchmark: Benchmark):
        """Handle successful benchmark completion."""
        patch_dict = {
            "state": BenchmarkStateEnum.COMPLETED,
        }
        self._update_benchmark_state_sync(benchmark.id, **patch_dict)
        logger.info(f"Benchmark {benchmark.name} finished.")

        self._dump_benchmark_logs_to_file(benchmark)
        self._sync_benchmark_metrics(benchmark)
        self._stop_benchmark(benchmark)

    def _handle_benchmark_failure(self, benchmark: Benchmark):
        """Handle benchmark failure."""
        patch_dict = {
            "state": BenchmarkStateEnum.ERROR,
            "state_message": "Benchmark exited or unhealthy.",
        }
        self._update_benchmark_state_sync(benchmark.id, **patch_dict)
        self._dump_benchmark_logs_to_file(benchmark)
        self._stop_benchmark(benchmark)

    def _sync_benchmark_metrics(self, benchmark):
        """
        Synchronize benchmarks' metrics.
        """
        metrics = None
        try:
            metrics_file_path = f"{self._benchmark_dir}/{benchmark.id}.json"
            report = GenerativeBenchmarksReport.load_file(metrics_file_path)
            metrics = report.to_metrics()
        except Exception as e:
            logger.error(
                f"Failed to load metrics for benchmark {benchmark.name}(id={benchmark.id}): {e}"
            )
            return

        if not metrics:
            logger.error(
                f"No metrics found for benchmark {benchmark.name}(id={benchmark.id})."
            )
            return

        self._log_request_failures_if_any(
            benchmark=benchmark,
            report=report,
            total=metrics.request_total or 0,
            successful=metrics.request_successful or 0,
            errored=metrics.request_errored or 0,
            incomplete=metrics.request_incomplete or 0,
        )

        resp = self._clientset.http_client.get_httpx_client().post(
            f"/benchmarks/{benchmark.id}/metrics", json=metrics.model_dump()
        )
        raise_if_response_error(resp)

    def _log_request_failures_if_any(
        self,
        benchmark: Benchmark,
        report: GenerativeBenchmarksReport,
        total: int,
        successful: int,
        errored: int,
        incomplete: int,
        limit: int = 5,
    ) -> None:
        if errored <= 0 and incomplete <= 0:
            return

        try:
            errored_samples, incomplete_samples = self._load_request_samples(
                report, limit=limit
            )
        except Exception as e:
            logger.error(
                "Failed to read request error samples for benchmark "
                f"{benchmark.name}(id={benchmark.id}): {e}"
            )
            return

        if not errored_samples and not incomplete_samples:
            return

        lines: List[str] = [
            "",
            "=== BENCHMARK REQUEST FAILURES ===",
            "SUMMARY: "
            f"benchmark={benchmark.name}(id={benchmark.id}) "
            f"total={total} successful={successful} "
            f"errored={errored} incomplete={incomplete} "
            f"showing_up_to={limit}",
        ]

        if errored_samples:
            lines.append("")
            lines.append(f"---- ERRORED REQUESTS (SHOWING UP TO {limit}) ----")
            lines.extend(self._format_request_samples(errored_samples))

        if incomplete_samples:
            lines.append("")
            lines.append(f"---- INCOMPLETE REQUESTS (SHOWING UP TO {limit}) ----")
            lines.extend(self._format_request_samples(incomplete_samples))

        message = "\n".join(lines)
        self._append_benchmark_log(benchmark, message)

    def _load_request_samples(
        self, report: GenerativeBenchmarksReport, limit: int = 5
    ) -> Tuple[List[GenerativeRequestStats], List[GenerativeRequestStats]]:
        if (
            not report.benchmarks
            or len(report.benchmarks) == 0
            or report.benchmarks[0] is None
            or report.benchmarks[0].requests_truncated is None
        ):
            return [], []

        requests = report.benchmarks[0].requests_truncated
        errored = requests.errored or []
        incomplete = requests.incomplete or []

        return errored[:limit], incomplete[:limit]

    def _format_request_samples(
        self, samples: List[GenerativeRequestStats]
    ) -> List[str]:
        lines: List[str] = []
        for idx, sample in enumerate(samples, start=1):
            request_id = sample.request_id or "unknown"
            request_type = sample.request_type or "unknown"
            status = sample.info.status or "unknown"
            error = sample.info.error
            traceback = sample.info.traceback

            base = (
                f"- [{idx}] request_id={request_id} type={request_type} "
                f"status={status}"
            )
            lines.append(base)

            if error:
                lines.append(f"  ERROR: {error}")
            if traceback:
                lines.append("  TRACEBACK:")
                indented = "\n".join(f"    {line}" for line in traceback.splitlines())
                lines.append(indented)
            lines.append("")
        return lines

    def _append_benchmark_log(self, benchmark: Benchmark, message: str) -> None:
        log_file_path = f"{self._benchmark_log_dir}/{benchmark.id}.log"
        try:
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(message)
                if not message.endswith("\n"):
                    f.write("\n")
        except Exception as e:
            logger.error(
                f"Failed to append benchmark log for {benchmark.name}(id={benchmark.id}): {e}"
            )

    def _set_active_benchmark(self, benchmark_id: int):
        self._active_benchmark_id = benchmark_id
        self._active_benchmark_started_at = time.time()

    def _clear_active_benchmark(self, benchmark_id: int):
        if self._active_benchmark_id == benchmark_id:
            self._active_benchmark_id = None
            self._active_benchmark_started_at = None

    def _is_benchmark_timed_out(self, benchmark: Benchmark) -> bool:
        limit = self._config.benchmark_max_duration_seconds
        if not limit:
            return False
        if self._active_benchmark_id != benchmark.id:
            return False
        if self._active_benchmark_started_at is None:
            return False
        return (time.time() - self._active_benchmark_started_at) > limit

    def _dump_benchmark_logs_to_file(
        self,
        benchmark: Benchmark,
    ):
        try:
            logs = logs_workload(
                name=benchmark.name,
            )
        except Exception as e:
            logger.error(
                f"Failed to fetch workload logs for benchmark {benchmark.name}(id={benchmark.id}): {e}"
            )
            return

        log_file_path = f"{self._benchmark_log_dir}/{benchmark.id}.log"
        with open(log_file_path, "a", encoding="utf-8") as f:
            log_str = logs
            if isinstance(log_str, bytes):
                log_str = log_str.decode("utf-8", errors="replace")
            log_str = str(log_str)
            f.write(log_str)
            if not log_str.endswith("\n"):
                f.write("\n")
