import asyncio
import multiprocessing
import setproctitle
import os
import re
import time
from typing import Dict, NamedTuple, Optional, Callable, List, Set, Tuple
import logging
from collections import Counter, OrderedDict, deque

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
    BenchmarkLoadModeEnum,
    BenchmarkStateEnum,
    BenchmarkTargetModeEnum,
    benchmark_load_mode,
)
from gpustack.schemas.models import ModelInstanceStateEnum
from gpustack.utils.process import terminate_process_tree, add_signal_handlers
from gpustack.worker.benchmark import analysis, artifacts
from gpustack.worker.benchmark.runner import BenchmarkRunner
from gpustack.client import ClientSet
from gpustack.server.bus import Event, EventType
from gpustack.worker.schemas.benchmark_runner import (
    GenerativeBenchmarksReport,
    GenerativeRequestStats,
)
from gpustack_runtime.deployer import logs_workload

logger = logging.getLogger(__name__)

HTTP_ERROR_PATTERN = re.compile(
    r"^HTTP\s+(?P<status>\d+):\s+(?P<msg>.*)\s+\(type=(?P<type>[^,]+),\s*code=(?P<code>[^)]+)\)$"
)
TRUNCATION_SUFFIX = "..."
BENCHMARK_STATE_MESSAGE_MAX_LEN = 1024
BENCHMARK_FAILURE_REASON_MAX_LEN = 220
# Snapshot the running container's logs to disk at most this often, so logs are
# preserved even if the container is garbage-collected before we poll a terminal
# state (see _maybe_snapshot_logs).
BENCHMARK_LOG_SNAPSHOT_INTERVAL_SECONDS = 30
# Minimum gap between partial (in-progress) result syncs for a running
# multi-point benchmark. The state poll fires ~1s; re-globbing and replacing the
# whole result set every second would be wasteful, so throttle to this.
BENCHMARK_PARTIAL_SYNC_INTERVAL_SECONDS = 10
# How often a running benchmark re-checks that its target is still there. The
# lookup reads ServeManager's watch-backed instance cache rather than the API
# (both managers share one ClientSet), so the cost is a dict scan — but the
# cache falls back to a full unpaginated GET while the watch is reconnecting,
# and that one should not fire on every 3-second poll.
BENCHMARK_TARGET_CHECK_INTERVAL_SECONDS = 15
# How long the target may stay missing before the run is failed. This is not a
# detection delay to be minimized: an instance can legitimately leave RUNNING
# for a while (a restart, a worker heartbeat blip, a reconnecting watch whose
# cache is still replaying), and killing a run that has been measuring for
# twenty minutes because of a five-second gap is the worse error. Long enough
# to sit out those, short enough to beat the load generator's own per-request
# timeout, which is what the run would otherwise wait on.
BENCHMARK_TARGET_GONE_GRACE_SECONDS = 300


def _without_raw_metrics(results: list) -> list:
    """The point grid with each point's `raw_metrics` dump left out.

    For partial syncs only. `raw_metrics` is the bulk of a point (~15KB against a
    few hundred bytes of columns) and the whole grid is re-posted on every sync, so
    a 12-point run was shipping the same dumps over and over for the life of the
    run — and each POST is a full replace, so the route also rewrote every row.
    The columns are what the running detail page draws from; the dumps back the
    per-point percentile drill-down, which the terminal sync fills in.

    The cost is stated rather than hidden: expanding a finished point mid-run shows
    no percentile table until the run ends. Shallow copies — the same rows go on to
    the analysis, which reads columns only.
    """
    return [{k: v for k, v in row.items() if k != "raw_metrics"} for row in results]


class CollectedResults(NamedTuple):
    """One aggregated read of a benchmark's result files.

    `loaded` and `skipped` are about the READ, not the run: `loaded` gates the
    partial sync (has anything new arrived?) while `skipped` means a file was on
    disk but unusable. During a partial sync that is routine — the newest point is
    still being written — but in the terminal sync it is permanent data loss, which
    is why the two are counted separately instead of just logged.
    """

    results: list
    metrics: Optional[object]
    report: Optional[GenerativeBenchmarksReport]
    loaded: int
    skipped: int
    # Why the first skipped file was skipped. The count alone sends the reader
    # to the worker log for the one fact that decides what to do next, and the
    # worker log is the hardest thing to reach from a browser — the run's own
    # log page shows the CONTAINER's output, not the agent's. Carried so the
    # row can say it.
    skipped_reason: Optional[str] = None


# How many finished benchmark ids to remember for the teardown guard below.
# Only ids are kept, and the memo is consulted on the event path alone, so the
# bound is about not growing without limit over a long worker uptime rather
# than about cost per entry.
_TEARDOWN_MEMO_LIMIT = 1024


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
    _torn_down_ids: "OrderedDict[int, None]"
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
        # Benchmarks stopped / deleted after being QUEUED but before the worker
        # started them. Checked right before start, so a queued-then-stopped run
        # is never launched — the queue snapshot alone is racy (the stop can land
        # between popleft and _start_benchmark).
        self._canceled_ids: Set[int] = set()
        # Ids whose teardown has already run. The watch re-delivers terminal
        # events — every reconnect replays the current rows — and without this
        # set a STOPPED or DELETED replay re-runs the whole terminal path each
        # time: finalize the analysis, fetch the container logs, delete the
        # workload. Once the row and its workload are both gone those three are
        # pure failing I/O, and because each attempt logs and returns rather
        # than raising, nothing breaks the cycle: ~3 teardowns/s for a benchmark
        # that does not exist, which also starves every run queued behind it
        # (each started and immediately marked "exited or unhealthy"). Cleared
        # on re-enqueue so re-running the same
        # row arms it again.
        self._torn_down_ids = OrderedDict()
        self._worker_task = None
        self._active_benchmark_id = None
        self._active_benchmark_started_at = None
        # Per-benchmark: byte offset where the container logs begin (after the
        # provisioning logs the subprocess wrote), and the last snapshot time.
        self._container_log_offset: Dict[int, int] = {}
        self._last_log_snapshot_at: Dict[int, float] = {}
        # Per-benchmark in-progress result streaming: how many point files were
        # in the last successful partial sync, and when it happened.
        self._partial_synced_count: Dict[int, int] = {}
        self._last_partial_sync_at: Dict[int, float] = {}
        # Per-benchmark target liveness: when the target was last looked up, and
        # when it was FIRST seen missing (cleared as soon as it comes back). A
        # timestamp rather than a strike count so the grace period stays the same
        # wall-clock window whatever the check interval is set to.
        self._last_target_check_at: Dict[int, float] = {}
        self._target_missing_since: Dict[int, float] = {}

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
            self._canceled_ids.add(benchmark.id)
            if self._is_torn_down(benchmark.id):
                # A replayed DELETED event. The teardown already ran, and the
                # row and workload it would act on are gone, so re-running it
                # only produces failing I/O — see `_torn_down_ids`.
                return
            self._stop_benchmark(benchmark)
            logger.trace(
                f"DELETED event: stopped deleted benchmark {benchmark.name}(id={benchmark.id})."
            )
            return

        if is_pending:
            asyncio.create_task(self._enqueue_benchmark(benchmark))
            return

        if is_stopped:
            # Record the cancel synchronously (before the async handler runs) so
            # the queue worker can't start it in the meantime.
            self._canceled_ids.add(benchmark.id)
            if self._is_torn_down(benchmark.id):
                # Same as the DELETED branch above: the watch replays STOPPED
                # rows on every reconnect, and the teardown is not repeatable.
                return
            asyncio.create_task(self._handle_stop_benchmark_event(benchmark))

    async def _handle_stop_benchmark_event(self, benchmark: Benchmark):
        try:
            # Before the teardown below, which drops the partial-sync bookkeeping:
            # a stopped run keeps the points it did measure, so their analysis must
            # stop claiming to be provisional. See _finalize_partial_analysis.
            #
            # No state patch of its own — the server already wrote STOPPED, which is
            # what produced this event. So the ordering question the other two
            # terminal handlers answer (state first or analysis first) does not arise
            # here: there is only the analysis left to write.
            self._finalize_partial_analysis(benchmark)
            self._dump_benchmark_logs_to_file(benchmark)
        except Exception as e:
            logger.error(f"Failed to stop benchmark {benchmark.name}: {e}")
        finally:
            # In a finally because the two calls above write the run's leftovers
            # while these two are what lets this worker run anything else. A row
            # stopped by the user is no longer RUNNING, so the state poll will
            # never look at it again and this is the only chance to release
            # `_active_benchmark_id` — an analysis or a log dump that raised used
            # to take the whole queue down with it.
            self._stop_benchmark(benchmark)
            self._clear_active_benchmark(benchmark.id)

    async def _enqueue_benchmark(self, benchmark: Benchmark):
        async with self._queue_lock:
            # A fresh enqueue supersedes any earlier cancel for this id, and
            # re-arms the teardown guard so the new run can be torn down.
            self._canceled_ids.discard(benchmark.id)
            self._torn_down_ids.pop(benchmark.id, None)
            if benchmark.id not in [b.id for b in self._benchmark_queue]:
                self._benchmark_queue.append(benchmark)

                patch_dict = {"state": BenchmarkStateEnum.QUEUED}
                if not await self._update_benchmark_state(benchmark.id, **patch_dict):
                    # Deleted between the PENDING event and this patch. Mark it
                    # canceled rather than popping it here: the queue is a deque
                    # shared with the sync thread, and the existing
                    # `_canceled_ids` guard drops the entry race-safely when the
                    # queue worker reaches it.
                    self._canceled_ids.add(benchmark.id)
                    logger.info(
                        f"Benchmark {benchmark.name}(id={benchmark.id}) was deleted "
                        "before it could be queued; dropping it."
                    )
                    return
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
                if benchmark.id in self._canceled_ids:
                    # Stopped / deleted while it sat in the queue — drop it
                    # instead of starting a run the user already canceled.
                    self._canceled_ids.discard(benchmark.id)
                    logger.info(
                        f"Skipping start of benchmark {benchmark.name}"
                        f"(id={benchmark.id}); it was canceled while queued."
                    )
                    continue
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
            # Remembered for the whole run so the teardown paths that only have
            # an id — `_reap_deleted_active_benchmark` — can still name the
            # workload they have to delete. Dropped by `_stop_benchmark`.
            self._benchmark_by_id[benchmark.id] = benchmark
            self._set_active_benchmark(benchmark.id)
            patch_dict = {
                "state": BenchmarkStateEnum.RUNNING,
                "pid": process.pid,
            }
            if not await self._update_benchmark_state(benchmark.id, **patch_dict):
                # Deleted while we were starting it. Tear it down here and now:
                # the state poll lists rows from the server, so this id is absent
                # from every later tick and nothing else would ever release
                # `_active_benchmark_id`.
                logger.info(
                    f"Benchmark {benchmark.name}(id={benchmark.id}) was deleted "
                    "while starting; stopping it again."
                )
                self._stop_benchmark(benchmark)
                return
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

    async def _update_benchmark_state(self, id: int, **kwargs) -> bool:
        client = self._clientset.http_client.get_async_httpx_client()
        resp = await client.patch(f"/benchmarks/{id}/state", json=kwargs)
        return self._state_patch_landed(id, resp)

    def _update_benchmark_state_sync(self, id: int, **kwargs) -> bool:
        client = self._clientset.http_client.get_httpx_client()
        resp = client.patch(f"/benchmarks/{id}/state", json=kwargs)
        return self._state_patch_landed(id, resp)

    @staticmethod
    def _state_patch_landed(id: int, resp) -> bool:
        """True when the patch landed, False when the row is gone.

        A 404 here is not a failed write, it is the server stating that the
        benchmark no longer exists — the user deleted it, or the workload cleaner
        did. Raising on it aborted whichever caller was mid-teardown, and on the
        state-poll paths the teardown is what releases `_active_benchmark_id`: a
        run deleted while RUNNING therefore left this worker with an active id
        forever, its queue popping nothing while the poll re-entered the same 404
        every 3 seconds. The worker looked healthy throughout (ready, all GPUs,
        instances running) — only the benchmark queue was dead.

        Every caller already has a correct reaction to "this row is gone": the
        same local teardown it runs for a terminal state. So report it and let
        them get on with it. Every other status still raises — that one IS a
        failed write, and the callers that retry should keep retrying it.
        """
        if resp.status_code == 404:
            logger.info(
                f"Benchmark {id} no longer exists on the server; "
                "skipping the state update and cleaning up locally."
            )
            return False
        resp.raise_for_status()
        return True

    def _stop_benchmark(self, benchmark: Benchmark):
        """
        Stop benchmark and clean up.

        Both teardown steps are guarded, because the internal-state cleanup below
        them — `_clear_active_benchmark` above all — must run either way. A
        container-runtime blip that let `delete_workload` raise would leave
        `_active_benchmark_id` set, and `_benchmark_queue_worker` would then stop
        popping ANYTHING, permanently: on the completion path the row is already
        COMPLETED by this point and `sync_benchmark_state` only polls RUNNING rows,
        so nothing ever revisits it and recovery needs a worker restart. A leaked
        workload is the lesser failure and is at least stated in the log.

        Args:
            benchmark: The benchmark to stop.
        """

        # Teardown provisioning process if still alive.
        if self._is_provisioning(benchmark):
            try:
                terminate_process_tree(self._provisioning_processes[benchmark.id].pid)
            except Exception as e:
                logger.error(
                    "Failed to terminate the provisioning process of benchmark "
                    f"{benchmark.name}(id={benchmark.id}): {e}"
                )

        # Delete workload.
        try:
            delete_workload(benchmark.name)
        except Exception as e:
            logger.error(
                "Failed to delete the workload of benchmark "
                f"{benchmark.name}(id={benchmark.id}): {e}"
            )

        # Do NOT touch _benchmark_queue here. This runs on the sync thread
        # (completion/failure/timeout) as well as the event loop (STOP/DELETE),
        # and _queue_lock is an asyncio.Lock that can't serialize against the
        # thread — rebuilding/reassigning the deque here races the loop's
        # _enqueue_benchmark append and can drop a just-queued entry. A
        # canceled-while-queued benchmark is instead skipped race-safely by the
        # _canceled_ids guard in _benchmark_queue_worker; the dead entry is
        # simply discarded when it's popped.

        # Cleanup internal states.
        self._provisioning_processes.pop(benchmark.id, None)
        self._benchmark_by_id.pop(benchmark.id, None)
        self._container_log_offset.pop(benchmark.id, None)
        self._last_log_snapshot_at.pop(benchmark.id, None)
        self._partial_synced_count.pop(benchmark.id, None)
        self._last_partial_sync_at.pop(benchmark.id, None)
        self._last_target_check_at.pop(benchmark.id, None)
        self._target_missing_since.pop(benchmark.id, None)
        self._clear_active_benchmark(benchmark.id)

        self._record_teardown(benchmark.id)

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
        listed_ids = set()
        for benchmark in benchmarks_page.items:
            listed_ids.add(benchmark.id)
            # One row per try/except, so a single run's reconciliation cannot
            # abort the whole round — nor the rounds after it, since a condition
            # that raises lives in the row and not in the round: one benchmark
            # whose state patch keeps failing would otherwise stop every OTHER
            # run on this worker from being reconciled at all.
            try:
                self._sync_single_benchmark_state(benchmark)
            except Exception as e:
                logger.error(
                    "Failed to sync the state of benchmark "
                    f"{benchmark.name}(id={benchmark.id}): {e}"
                )

        self._reap_deleted_active_benchmark(listed_ids)

    def _reap_deleted_active_benchmark(self, listed_ids: Set[int]):
        """Release the queue when the active run's row has been deleted.

        `_active_benchmark_id` is what serializes runs on this worker: while it
        is set, `_benchmark_queue_worker` pops nothing. Every way a run ends
        clears it through `_stop_benchmark`, and a delete does too — via the
        DELETED watch event, if that event is seen. It is not always: a watch
        that was disconnected, reconnecting, or replaying when the row was
        deleted never delivers it, and then nothing revisits the run, because
        this poll lists rows FROM the server and a deleted row is absent from
        every later tick. The worker keeps an active id for a benchmark that no
        longer exists and its queue is stalled for good, while the worker, its
        GPUs and its instances all keep reporting healthy.

        So ask about it directly — but only when the run is missing from the
        RUNNING page above, which costs nothing in the normal case, and only act
        on a definitive 404. A transport error must never look like a deletion,
        or a server blip would tear down a perfectly healthy run.
        """
        benchmark_id = self._active_benchmark_id
        if benchmark_id is None or benchmark_id in listed_ids:
            return

        client = self._clientset.http_client.get_httpx_client()
        try:
            resp = client.get(f"/benchmarks/{benchmark_id}")
        except Exception as e:
            logger.debug(
                f"Could not check whether benchmark {benchmark_id} still exists: {e}"
            )
            return
        if resp.status_code != 404:
            # Either it is still there (any other state than RUNNING — queued,
            # just-stopped, mid-transition) or the server could not answer. Both
            # are somebody else's business.
            return

        logger.warning(
            f"Active benchmark {benchmark_id} no longer exists on the server. "
            "Tearing it down locally so the queue can move on."
        )
        benchmark = self._benchmark_by_id.get(benchmark_id)
        if benchmark is not None:
            self._stop_benchmark(benchmark)
            return

        # No row object to name the workload with, so the full teardown is not
        # available. Release the queue anyway — a leaked container is recoverable
        # and stated in the log; a stalled queue is neither.
        if process := self._provisioning_processes.pop(benchmark_id, None):
            try:
                terminate_process_tree(process.pid)
            except Exception as e:
                logger.error(
                    "Failed to terminate the provisioning process of benchmark "
                    f"{benchmark_id}: {e}"
                )
        self._clear_active_benchmark(benchmark_id)
        self._record_teardown(benchmark_id)

    def _sync_single_benchmark_state(self, benchmark: Benchmark):
        """Synchronize a single benchmark's state."""
        # Never reconcile another worker's run. The list above asks the server
        # to filter by worker_id, but an unknown query parameter is silently
        # dropped rather than rejected, so this check is what keeps a worker
        # from reconciling the whole cluster when that filter is not honoured.
        #
        # For a run it does not own, this method is wrong at every step:
        # `get_workload` reads the *local* container runtime and returns None,
        # `_is_workload_failed(None)` is True by design, and the
        # `_runner_left_terminal_artifact` escape hatch checks the *local*
        # disk, where the owning worker's sidecar obviously is not. The result
        # is `_handle_benchmark_failure`: a healthy run on another worker
        # patched to ERROR mid-run and its workload torn down.
        #
        # Kept as a guard here and not only in the query because it costs one
        # comparison and holds even if the server-side filter regresses.
        if benchmark.worker_id != self._worker_id:
            return

        if self._is_torn_down(benchmark.id):
            # The row can still read RUNNING here after a terminal handler has
            # patched it: this poll lists from the server, and the list can be
            # served before the patch is visible. Without this guard the stale
            # row re-enters the terminal path — one full teardown per tick until
            # the write lands (observed 21 times for a single run).
            return

        # Check for timeout
        if self._is_benchmark_timed_out(benchmark):
            self._handle_benchmark_timeout(benchmark)
            return

        # Check that what the load is aimed at still exists. Before the workload
        # lookup on purpose: the container is perfectly healthy in this failure —
        # it is talking to an endpoint that went away — so every check below this
        # one says "still running" and the run sits there until the load
        # generator's own per-request timeout expires (an hour, by default).
        if self._is_target_gone(benchmark):
            self._handle_target_gone(benchmark)
            return

        # Skip if still provisioning
        if self._is_provisioning(benchmark):
            logger.trace(
                f"Benchmark {benchmark.name}(id={benchmark.id}) is provisioning. Skipping sync."
            )
            return

        # Get workload and handle based on state
        workload = get_workload(benchmark.name)

        # Snapshot container logs while running, so we still have them if the
        # container is garbage-collected before we observe a terminal state.
        if workload and workload.state == WorkloadStatusStateEnum.RUNNING:
            self._maybe_snapshot_logs(benchmark)
            self._maybe_sync_partial_metrics(benchmark)

        if self._should_skip_workload(benchmark, workload):
            return

        if self._is_workload_completed(workload):
            self._handle_benchmark_completion(benchmark)
            return

        if self._is_workload_failed(workload):
            if workload is None and self._runner_left_terminal_artifact(benchmark):
                # Gone because it finished, not because it died. The container
                # can be reaped between the runner returning and this 3-second
                # poll, and on that path every point was already measured and
                # written — reporting ERROR there put a red badge on a complete
                # curve. The runner's terminal sidecar is what tells the two
                # apart; completion re-reads the point files, so nothing is lost.
                logger.info(
                    f"Benchmark {benchmark.name}(id={benchmark.id}) workload is gone "
                    "but the runner wrote its terminal artifact; treating as finished."
                )
                self._handle_benchmark_completion(benchmark)
                return
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

    def _runner_left_terminal_artifact(self, benchmark: Benchmark) -> bool:
        """True when the runner got far enough to write its end-of-run sidecar.

        One per load shape, each written only as the runner returns: `__curve`
        for manual stages, `__ramp` for an adaptive ramp, and the single report
        for a one-rate run. Only presence is checked — the file's contents are a
        diagnostic, while the measurement lives in the point/stage reports that
        the completion path re-reads.
        """
        d = self._benchmark_dir
        for path in (
            artifacts.curve_outcome_path(d, benchmark.id),
            artifacts.ramp_facts_path(d, benchmark.id),
            artifacts.single_report_path(d, benchmark.id),
        ):
            try:
                if os.path.exists(path):
                    return True
            except OSError:
                continue
        return False

    def _handle_benchmark_timeout(self, benchmark: Benchmark):
        """Handle benchmark timeout.

        State before analysis, for the reason spelled out in
        `_handle_benchmark_failure`: this runs from the RUNNING-only state poll, so
        leaving the row RUNNING while the analysis is written invites the next tick
        into a second teardown.
        """
        patch_dict = {
            "state": BenchmarkStateEnum.ERROR,
            "state_message": "Benchmark timed out.",
        }
        self._update_benchmark_state_sync(benchmark.id, **patch_dict)
        self._finalize_partial_analysis(benchmark)
        self._dump_benchmark_logs_to_file(benchmark)
        self._stop_benchmark(benchmark)

    def _is_target_gone(self, benchmark: Benchmark) -> bool:
        """Whether this run's target has been missing long enough to give up on.

        Only `instance` mode is judged here. That mode pins one member at create
        time -- `_resolve_target_endpoint` requires it to be RUNNING and records
        its name -- so the target leaving RUNNING, or being replaced by a
        differently-named one (what a rescheduled disaggregated group does to its
        router), means the thing being measured is not there any more. `route`
        mode is deliberately exempt: its whole point is that the instances behind
        it come and go, so the same observation is not a failure there.

        Throttled, and grace-period'd rather than acted on at first sight -- see
        the two interval constants for why each is set where it is.
        """
        # Read defensively. The caller catches per row, so a missing field
        # cannot take the round down with it — but it would still spend the
        # run's whole lifetime logging one traceback every three seconds, and a
        # run whose row cannot answer "which instance?" is exactly the one this
        # check has nothing to say about. Absent means "not judged", which is
        # the safe way to be wrong here.
        if getattr(benchmark, "target_mode", None) != BenchmarkTargetModeEnum.INSTANCE:
            return False

        name = (getattr(benchmark, "model_instance_name", None) or "").strip()
        if not name:
            # Nothing to check against. Pre-`model_instance_name` rows and any
            # future mode that does not pin a member land here; the run is left
            # to the checks below rather than failed on a missing field.
            return False

        now = time.time()
        last_checked = self._last_target_check_at.get(benchmark.id, 0.0)
        if now - last_checked < BENCHMARK_TARGET_CHECK_INTERVAL_SECONDS:
            return False
        self._last_target_check_at[benchmark.id] = now

        alive = self._is_target_instance_alive(name)
        if alive is None:
            # Could not tell. Not the same as "gone": see
            # `_is_target_instance_alive`.
            return False

        if alive:
            self._target_missing_since.pop(benchmark.id, None)
            return False

        missing_since = self._target_missing_since.setdefault(benchmark.id, now)
        missing_for = now - missing_since
        if missing_for < BENCHMARK_TARGET_GONE_GRACE_SECONDS:
            logger.debug(
                f"Benchmark {benchmark.name}(id={benchmark.id}) target instance "
                f"'{name}' has been missing for {missing_for:.0f}s; waiting out "
                f"the {BENCHMARK_TARGET_GONE_GRACE_SECONDS}s grace period."
            )
            return False

        logger.warning(
            f"Benchmark {benchmark.name}(id={benchmark.id}) target instance "
            f"'{name}' has been missing for {missing_for:.0f}s. Failing the run."
        )
        return True

    def _is_target_instance_alive(self, name: str) -> Optional[bool]:
        """True / False / None for "cannot tell", for the instance called `name`.

        Reads through `ClientSet.model_instances`, which ServeManager's watch
        keeps populated for the WHOLE cluster (`awatch` there passes no filter)
        and which this manager shares -- the worker builds one ClientSet for
        both. So the normal path is a dict scan, not a request, which is what
        makes a 15-second check affordable. When the watch is down the client
        falls back to an unpaginated GET by itself.

        The empty-result guard is the important part: a reconnecting watch
        clears its cache before replaying, so there is a window where the cache
        is considered authoritative and holds nothing. Reading that as "the
        target is gone" would fail every running benchmark in the cluster at
        once, on nothing but a dropped stream. A cluster that genuinely has zero
        instances also has no benchmark running against one, so treating the
        empty set as unknown costs nothing real.
        """
        try:
            instances = self._clientset.model_instances.list()
        except Exception as e:
            logger.debug(f"Could not check benchmark target instance '{name}': {e}")
            return None

        items = getattr(instances, "items", None) or []
        if not items:
            return None

        target = next((i for i in items if i.name == name), None)
        if target is None:
            return False
        return target.state == ModelInstanceStateEnum.RUNNING

    def _handle_target_gone(self, benchmark: Benchmark):
        """Fail a run whose target is gone, keeping what it already measured.

        Same shape and same ordering as `_handle_benchmark_timeout`, for the
        reason given there: this runs from the RUNNING-only state poll, so the
        row has to leave RUNNING before the analysis is written, or the next tick
        walks into a second teardown.
        """
        name = benchmark.model_instance_name or "?"
        patch_dict = {
            "state": BenchmarkStateEnum.ERROR,
            "state_message": self._truncate_state_message(
                f"Target model instance '{name}' is no longer running. "
                "The benchmark was stopped because there was nothing left to measure."
            ),
        }
        self._update_benchmark_state_sync(benchmark.id, **patch_dict)
        self._finalize_partial_analysis(benchmark)
        self._dump_benchmark_logs_to_file(benchmark)
        self._stop_benchmark(benchmark)

    def _handle_benchmark_completion(self, benchmark: Benchmark):
        """Handle successful benchmark completion.

        Order matters: the FINAL analysis is uploaded BEFORE the row flips to
        COMPLETED. The other way round left a window in which the state already
        read "completed" while `validity` was still the last partial snapshot —
        carrying `in_progress` and a coverage verdict computed from a subset of the
        points (e.g. "the curve never turned over, raise the upper bound"), which
        anything keying off state alone would render as the run's conclusion. The
        window is short when the upload succeeds and PERMANENT when it doesn't.

        Neither the sync nor the state patch may skip the teardown below: the
        workload has already gone INACTIVE and sync_benchmark_state only polls
        RUNNING rows, so a benchmark left un-stopped here is never revisited — its
        container workload leaks and its process/dict entries are never cleaned up.
        Hence state patch + stop both in the finally.
        """
        logger.info(f"Benchmark {benchmark.name} finished.")
        self._dump_benchmark_logs_to_file(benchmark)

        sync_failure = None
        try:
            self._sync_benchmark_metrics(benchmark)
        except Exception as e:
            logger.error(
                "Failed to sync results for "
                f"{benchmark.name}(id={benchmark.id}): {e}"
            )
            # Surface a persistent failure on state_message rather than only in the
            # worker log. Carried in the same patch as the state below instead of a
            # second round-trip. "sync" rather than "upload" because this covers both
            # halves — the result files being unreadable and the POST not landing —
            # and the exception text says which.
            sync_failure = self._truncate_state_message(
                f"Result sync failed: {e}. See worker logs for details."
            )
        finally:
            patch_dict = {"state": BenchmarkStateEnum.COMPLETED}
            if sync_failure is not None:
                patch_dict["state_message"] = sync_failure
            try:
                self._update_benchmark_state_sync(benchmark.id, **patch_dict)
            except Exception as e:
                # Leaves the row RUNNING with its workload gone; the next poll
                # sees no workload, takes the failure branch and marks it ERROR.
                # Log and still tear down rather than leaking the workload.
                logger.error(
                    "Failed to mark benchmark completed for "
                    f"{benchmark.name}(id={benchmark.id}): {e}"
                )
            self._stop_benchmark(benchmark)

    def _handle_benchmark_failure(self, benchmark: Benchmark):
        """Handle benchmark failure.

        State FIRST here, then the analysis — the opposite of
        `_handle_benchmark_completion`, and deliberately so. That method orders the
        upload first because the conclusion IS its deliverable: a row reading
        COMPLETED while `validity` still holds a provisional snapshot is a wrong
        answer on screen. This path has no such deliverable to protect (the analysis
        it writes is explicitly best-effort, see `_finalize_partial_analysis`) and it
        has something else to protect instead: this handler is reached FROM the state
        poll, which only scans RUNNING rows. Flipping the row out of RUNNING first is
        what stops the next tick — one second later — from re-entering the same
        handler and running a second finalize and a second teardown against a
        workload already being torn down. Both orderings also self-heal if the patch
        fails: the row stays RUNNING and the poll retries.
        """
        patch_dict = {
            "state": BenchmarkStateEnum.ERROR,
            "state_message": "Benchmark exited or unhealthy.",
        }
        self._update_benchmark_state_sync(benchmark.id, **patch_dict)
        self._finalize_partial_analysis(benchmark)
        self._dump_benchmark_logs_to_file(benchmark)
        self._stop_benchmark(benchmark)

    def _finalize_partial_analysis(self, benchmark: Benchmark):
        """Re-run the analysis without the `in_progress` tag on a terminal exit.

        Only `_handle_benchmark_completion` gets a full terminal sync. The other
        three ways a run ends — ERROR, timeout, and a user stop — must not simply
        patch the state and tear down, leaving whatever the last partial sync
        wrote. For a multi-point run that leaves an analysis still flagged
        `in_progress`, i.e. a snapshot labelled provisional on a row that will
        never change again: the UI hides the coverage banner and shows Coverage
        "-" forever, while the peak/recommended cards from that same snapshot
        stay on screen and read as final.

        The points that DID finish are real measurements and stay published; only
        the "still firming up" label is wrong once the run is over. So the analysis
        is recomputed over whatever was collected and written once more as final.

        Best-effort by construction: this runs on the terminal path of a failing
        run, so it must not raise (the caller still has to tear the workload down)
        and it does not retry — a row that keeps the stale flag is a cosmetic
        regression, a leaked container is not.

        NOT gated on a partial sync having landed. `_partial_synced_count` is only
        written after a SUCCESSFUL partial sync and those run with `attempts=1`, so
        keying off it meant a run that measured eight points against a briefly flaky
        API published none of them: every partial sync had failed, so this method
        returned immediately and the eight point files were left on disk. Clearing a
        stale flag and publishing points that never made it are the same operation
        here — re-read the disk and post what is there — and the disk is the only
        record either way. The `SINGLE` gate above stays: a single-point run never
        partial-syncs by design, and its terminal path is the completion handler.
        """
        if benchmark_load_mode(benchmark) is BenchmarkLoadModeEnum.SINGLE:
            # Never partially synced, so there is no provisional flag to clear.
            return
        try:
            collected = self._collect_results(benchmark)
            if not collected.results:
                return
            self._write_best_points_and_validity(
                benchmark, collected.results, attempts=1
            )
            # The grid too, and AFTER the conclusion. Points that finished since the
            # last partial sync feed the analysis above, so publishing only the
            # conclusion leaves a peak / recommended rate with no row behind it in
            # the results table — a card naming a load the stage table does not show.
            # Second because _write_best_points_and_validity cannot raise while this
            # can: dropping the provisional flag is the one thing this method exists
            # for and must not be blocked by an upload blip. attempts=1 — the caller
            # still has to tear the workload down.
            if collected.metrics:
                self._post_metrics_and_results(
                    benchmark, collected.metrics, collected.results, attempts=1
                )
        except Exception as e:
            logger.warning(
                "Could not finalize the partial analysis of benchmark "
                f"{benchmark.name}(id={benchmark.id}): {e}"
            )

    def _aggregate_points(self, benchmark, paths: List[str]) -> "CollectedResults":
        """Aggregate a list of per-point / per-stage report files into one curve.

        Shared by the ramp and the manual-stage branch: both write one report file
        per measured point, each holding a single benchmark whose `run_index` is
        always 0, so the point number has to come from the file order here —
        `sequence` is the probe order the results API sorts by.

        A file that cannot be read (or cannot be converted) is SKIPPED, not fatal:
        one bad point must not drop the other eleven, and during a partial sync the
        newest file is routinely still being written. The count of skipped files is
        returned so the terminal sync — where a skip means permanent data loss
        rather than a mid-write race — can say so.
        """
        results: list = []
        best = None
        report = None
        worst_errs = -1
        loaded = 0
        skipped = 0
        skipped_reason = None

        for path in paths:
            try:
                rep = GenerativeBenchmarksReport.load_file(path)
                # Convert INSIDE the try: a file that loads but whose
                # to_results()/to_metrics() raises (schema drift, or a point still
                # mid-write) must be skipped like an unreadable one.
                point_results = rep.to_results(
                    input_tokens=benchmark.dataset_input_tokens,
                    sequence_start=len(results),
                )
                m = rep.to_metrics()
            except Exception as e:
                skipped += 1
                # The FIRST reason, not the last: the files are read in point
                # order, so the earliest failure is the one that explains the
                # run. A later point failing differently is usually a
                # consequence of it.
                if skipped_reason is None:
                    skipped_reason = f"{os.path.basename(path)}: {e}"
                logger.warning(
                    f"Skipping result file {path} of benchmark "
                    f"{benchmark.name}(id={benchmark.id}); unavailable: {e}"
                )
                continue
            results.extend(point_results)
            loaded += 1
            errs = 0
            if m:
                errs = (m.request_errored or 0) + (m.request_incomplete or 0)
                if best is None or (m.tokens_per_second_mean or 0) > (
                    best.tokens_per_second_mean or 0
                ):
                    best = m
            # Keep the report with the MOST failed requests for error-sample
            # extraction: failures concentrate at the high-load points, so the
            # first (lowest-load) point's samples are usually empty.
            if report is None or errs > worst_errs:
                report = rep
                worst_errs = errs

        return CollectedResults(results, best, report, loaded, skipped, skipped_reason)

    def _collect_results(self, benchmark) -> "CollectedResults":
        """Load this benchmark's result files into one aggregated curve.

        Shared by the terminal sync and the in-progress partial sync so both
        aggregate points the same way — the only difference between a partial and a
        final read is how many point files happen to exist on disk at the time.
        `metrics` is the representative (throughput-peak) point; `report` is the one
        used for error samples.
        """
        mode = benchmark_load_mode(benchmark)
        try:
            if mode is BenchmarkLoadModeEnum.AUTO_TUNE:
                # Adaptive ramp: one file per measured point, the count decided at
                # runtime, so the set is discovered rather than derived.
                collected = self._aggregate_points(
                    benchmark,
                    [
                        f"{self._benchmark_dir}/{name}"
                        for name in artifacts.list_point_files(
                            self._benchmark_dir, benchmark.id
                        )
                    ],
                )
                self._append_saturation_probe(benchmark, collected.results)
                return collected

            if mode is BenchmarkLoadModeEnum.STAGES:
                # One single-rate run per user-specified stage.
                return self._aggregate_points(
                    benchmark,
                    [
                        artifacts.stage_report_path(
                            self._benchmark_dir, benchmark.id, i
                        )
                        for i in range(len(benchmark.stages))
                    ],
                )

            # Single run: one report written at the end.
            path = artifacts.single_report_path(self._benchmark_dir, benchmark.id)
            report = GenerativeBenchmarksReport.load_file(path)
            metrics = report.to_metrics()
            results = report.to_results(input_tokens=benchmark.dataset_input_tokens)
            return CollectedResults(results, metrics, report, 1 if metrics else 0, 0)
        except Exception as e:
            logger.error(
                f"Failed to load metrics for benchmark {benchmark.name}(id={benchmark.id}): {e}"
            )
            return CollectedResults([], None, None, 0, 0)

    def _append_saturation_probe(self, benchmark, results: list) -> None:
        """Append the saturation probe as a trailing result row, if it ran.

        The probe is the ceiling measurement that soft-capped a rate-axis ramp, so
        the user should see it — but it is NOT a measured ramp point: its throughput
        profile yields rate=None, which already excludes it from the peak /
        recommendation / validity (all of which require a rate). It is deliberately
        not fed into the representative metrics or the error-sample report either.
        Appended last => highest sequence => shown at the end of the stages table.
        """
        probe_path = artifacts.saturation_probe_path(self._benchmark_dir, benchmark.id)
        if not os.path.exists(probe_path):
            return
        try:
            prep = GenerativeBenchmarksReport.load_file(probe_path)
            results.extend(
                prep.to_results(
                    input_tokens=benchmark.dataset_input_tokens,
                    sequence_start=len(results),
                )
            )
        except Exception as e:
            logger.warning(
                "Skipping saturation probe of benchmark "
                f"{benchmark.name}(id={benchmark.id}): {e}"
            )

    def _sync_benchmark_metrics(self, benchmark):  # noqa: C901
        """Synchronize a finished benchmark's metrics and per-point results.

        Terminal (definitive) counterpart of _maybe_sync_partial_metrics: the
        same aggregation, but it additionally logs request failures and writes
        the state_message, and marks the analysis final (no in_progress flag).
        """
        collected = self._collect_results(benchmark)
        results, metrics, report = (
            collected.results,
            collected.metrics,
            collected.report,
        )

        if not metrics:
            # Raise rather than return: a silent return let the row flip to
            # COMPLETED with every metric null and nothing saying why. The caller
            # turns this into a state_message, so the reason has to be accurate —
            # "upload failed" would be wrong, nothing was ever uploaded.
            logger.error(
                f"No metrics found for benchmark {benchmark.name}(id={benchmark.id})."
            )
            if collected.skipped:
                # Carrying the first file's actual error, because the count on
                # its own answers nothing: "unreadable" is a truncated write, a
                # schema the installed guidellm does not know, and a run that
                # died mid-point all at once, and they are fixed differently.
                # "See worker logs" is a poor referral here — the log page next
                # to this message shows the container's output, not the agent's.
                raise RuntimeError(
                    f"all {collected.skipped} result file(s) were unreadable"
                    + (
                        f" ({collected.skipped_reason})"
                        if collected.skipped_reason
                        else ""
                    )
                )
            raise RuntimeError("the run produced no result file")

        # An unreadable file means something different here than in a partial sync.
        # There it is routine (the newest point is still being written and the next
        # tick picks it up); here the run is over, so the point is gone for good and
        # the curve the user is about to read is missing it. Say so on the row rather
        # than only in the worker log.
        lost_points_message = None
        if collected.skipped:
            lost_points_message = (
                f"{collected.skipped} of {collected.loaded + collected.skipped} "
                "measured point(s) could not be read and are missing from the "
                "results"
                + (f": {collected.skipped_reason}" if collected.skipped_reason else "")
                + ". See worker logs for the rest."
            )
            logger.error(
                f"Benchmark {benchmark.name}(id={benchmark.id}) finished with "
                f"{collected.skipped} unreadable result file(s)."
            )

        # Failure counts aggregate across ALL stages — a failure in any stage should
        # surface, not only the representative peak point. (For a single-run
        # benchmark `results` has one stage, so this matches the old behavior.)
        # The saturation probe and legacy bound passes are excluded by
        # measured_stages: their requests are instrument readings, not part of the
        # load the user asked to be run, so they must not move the success rate.
        measured = analysis.measured_stages(results)
        if measured:
            total = sum(r.get("request_total") or 0 for r in measured)
            successful = sum(r.get("request_successful") or 0 for r in measured)
            errored = sum(r.get("request_errored") or 0 for r in measured)
            incomplete = sum(r.get("request_incomplete") or 0 for r in measured)
        else:
            total = metrics.request_total or 0
            successful = metrics.request_successful or 0
            errored = metrics.request_errored or 0
            incomplete = metrics.request_incomplete or 0

        try:
            errored_samples, incomplete_samples = self._load_request_samples(
                report, limit=None
            )
        except Exception as e:
            logger.error(
                "Failed to read request error samples for benchmark "
                f"{benchmark.name}(id={benchmark.id}): {e}"
            )
            errored_samples, incomplete_samples = [], []

        self._log_request_failures_if_any(
            benchmark=benchmark,
            total=total,
            successful=successful,
            errored=errored,
            incomplete=incomplete,
            errored_samples=errored_samples,
            incomplete_samples=incomplete_samples,
        )

        partial_failure_message = self._build_partial_failure_state_message(
            errored=errored,
            incomplete=incomplete,
            errored_samples=errored_samples,
            incomplete_samples=incomplete_samples,
        )

        self._post_metrics_and_results(benchmark, metrics, results)
        label = f"{benchmark.name}(id={benchmark.id})"
        analysis_failure_message = self._write_best_points_and_validity(
            benchmark, results
        )

        # Surface partial failures (errored / incomplete requests) on the
        # benchmark's state_message so the UI shows why a run partly failed, plus
        # the lost-point and analysis-writeback failures when there were any. All go
        # in a single patch so none overwrites another.
        message = " ".join(
            m
            for m in (
                partial_failure_message,
                lost_points_message,
                analysis_failure_message,
            )
            if m
        )
        if message:
            try:
                self._retry_sync(
                    lambda: self._update_benchmark_state_sync(
                        benchmark.id,
                        state_message=self._truncate_state_message(message),
                    ),
                    what=f"state message for {label}",
                )
            except Exception as e:
                logger.error(
                    "Failed to update state message for "
                    f"{benchmark.name}(id={benchmark.id}): {e}"
                )

    def _post_metrics_and_results(self, benchmark, metrics, results, attempts: int = 3):
        """POST the representative metrics and the per-point result grid.

        The results POST is a full idempotent replace (the route deletes the
        benchmark's existing rows first), so calling this repeatedly during a
        run — once per batch of newly finished points — cleanly overwrites with
        the growing set rather than accumulating duplicates.

        BOTH posts are retried and both propagate on final failure. The benchmark's
        container has already exited by the terminal sync, so a single API blip would
        otherwise strand a finished run — and the two must be treated alike: losing
        the grid is the more deceptive of the two, because the parent row still shows
        a representative point and the list page looks normal while the detail page's
        curve is simply empty. Letting it raise turns that into a stated failure on
        `state_message` instead. Partial syncs pass `attempts=1` and swallow the
        error one level up, so a blip just retries on the next poll.
        """
        label = f"{benchmark.name}(id={benchmark.id})"
        client = self._clientset.http_client.get_httpx_client()

        def _post(path: str, payload):
            def _do():
                raise_if_response_error(
                    client.post(f"/benchmarks/{benchmark.id}/{path}", json=payload)
                )

            return _do

        self._retry_sync(
            _post("metrics", metrics.model_dump()),
            what=f"metrics for {label}",
            attempts=attempts,
        )
        # Per-point results (one row per (input_tokens, rate) grid cell); the parent
        # metrics above hold the representative (throughput-peak) point.
        self._retry_sync(
            _post("results", results),
            what=f"results for {label}",
            attempts=attempts,
        )

    def _write_best_points_and_validity(
        self, benchmark, results, in_progress: bool = False, attempts: int = 3
    ) -> Optional[str]:
        """Compute and persist the best operating points and coverage validity.

        Returns a human-readable failure message when the write did not land
        (surfaced on state_message at the end of a terminal sync), else None.
        `in_progress=True` tags the validity so the UI can show the conclusion is
        still firming up as points arrive; the terminal sync leaves it off.
        `attempts=1` for partial syncs — a transient blip just retries on the next
        poll instead of blocking it with backoff.
        """
        # Best operating points: peak throughput / max rate meeting the SLO.
        # Computed from the per-point grid and persisted on the parent row for
        # the detail page's "Best Operating Points" cards.
        best_points = analysis.compute_best_points(benchmark, results)
        # Test-coverage validity: whether the sweep explored enough to trust
        # the result (single source of truth on the parent; the UI just renders
        # the warning codes). See analysis.compute_validity.
        #
        # The ramp facts are read fresh here rather than cached: on a partial sync
        # the sidecar does not exist yet (it is written when the ramp returns), and
        # its appearance is precisely the signal that the search has ended.
        validity = analysis.compute_validity(
            benchmark, results, best_points, self._read_ramp_facts(benchmark)
        )
        if any(
            w.get("code") == "saturated_at_lower_bound"
            for w in validity.get("warnings", [])
        ):
            # The whole search range sits above saturation, so the single measured
            # "peak" is just the offered floor knob — which the banner explicitly
            # calls NOT the optimum ("lower the range"). Showing it as
            # peak/recommended contradicts that advice, so clear the cards; the
            # sustained ceiling the user needs is carried in the warning's
            # params.ceiling instead. Explicit Nones so a prior partial sync's
            # values are overwritten, not left stale.
            best_points = {
                "peak_rate": None,
                "slo_met_rate": None,
                "recommended_rate": None,
            }
        if in_progress:
            # Not a warning code — a transient flag marking this analysis as a
            # SNAPSHOT taken after N points, not the run's verdict. Dropped by the
            # terminal sync, which is the definitive read.
            #
            # The analysis itself is published in full, coverage codes included:
            # "as of 4 points the curve has not turned over" is a true statement
            # about what has been measured so far, and withholding it would delete
            # real data to compensate for a presentation problem. It is the UI that
            # must not render a snapshot as advice — a run at 55% was showing
            # "raise the upper bound and re-run", i.e. telling the user to abandon a
            # run that was about to answer the question. So the whole banner is
            # hidden while this flag is set (and the list's Coverage column shows
            # "-"), which also fails SAFE: a validity code added later is hidden
            # mid-run by default instead of having to be added to a suppress-list.
            validity = {**validity, "in_progress": True}
        # Always send all three conclusion fields, explicitly None when the grid
        # does not yield them. Omitting a key leaves whatever an earlier partial
        # sync wrote on the row, so a conclusion that no longer holds would survive
        # as a stale number next to a validity that contradicts it.
        patch = {
            "peak_rate": best_points.get("peak_rate"),
            "slo_met_rate": best_points.get("slo_met_rate"),
            "recommended_rate": best_points.get("recommended_rate"),
            "validity": validity,
        }
        # This write carries the whole conclusion of the run (peak / SLO
        # capacity / coverage). Losing it leaves a benchmark that reads as
        # "completed" with every conclusion field null and no hint why, so retry a
        # few times and, if it still fails, say so on state_message rather than
        # only in the worker log.
        label = f"{benchmark.name}(id={benchmark.id})"
        try:
            self._retry_sync(
                lambda: self._update_benchmark_state_sync(benchmark.id, **patch),
                what=f"best operating points / validity for {label}",
                attempts=attempts,
            )
            return None
        except Exception as e:
            logger.error(
                "Failed to update best operating points / validity for "
                f"{benchmark.name}(id={benchmark.id}): {e}"
            )
            return (
                "Result analysis was not saved (best operating points / test "
                f"coverage): {e}. The per-stage results are still available."
            )

    def _maybe_sync_partial_metrics(self, benchmark: Benchmark):
        """Stream finished points to the API while the benchmark is still running.

        Only multi-point runs (auto-tune ramp / manual stages) have anything to
        stream — a single fixed run writes its one file at the very end. The
        runner writes each point's result file as soon as that point finishes, so
        the detail page can show the curve growing instead of staying empty until
        the whole ramp completes.

        Throttled two ways: at most once per
        BENCHMARK_PARTIAL_SYNC_INTERVAL_SECONDS, and only when the number of
        finished point files has grown since the last successful sync. The growth
        check compares against the count actually LOADED last time (not the count
        on disk): a point still mid-write fails to parse and is skipped, so the
        on-disk count stays ahead of the synced count and the next tick retries it
        rather than stranding it until the run ends.

        The clock is stamped as soon as an attempt is made, not only when one
        succeeds — it is a throttle, not a success record. A point that is on disk
        but still being written keeps the on-disk count ahead of the synced count,
        so without this the attempt (and a full re-parse of every point already
        synced) would repeat on every 3s poll for as long as the write takes.
        """
        if benchmark_load_mode(benchmark) is BenchmarkLoadModeEnum.SINGLE:
            return
        now = time.time()
        last_at = self._last_partial_sync_at.get(benchmark.id, 0.0)
        if now - last_at < BENCHMARK_PARTIAL_SYNC_INTERVAL_SECONDS:
            return
        already = self._partial_synced_count.get(benchmark.id, 0)
        if self._count_ready_point_files(benchmark) <= already:
            return
        self._last_partial_sync_at[benchmark.id] = now
        try:
            collected = self._collect_results(benchmark)
            if not collected.metrics or collected.loaded <= already:
                return
            loaded, results = collected.loaded, collected.results
            # attempts=1: a transient blip just retries on the next poll rather
            # than blocking this one with backoff (the terminal sync retries).
            self._post_metrics_and_results(
                benchmark, collected.metrics, _without_raw_metrics(results), attempts=1
            )
            self._write_best_points_and_validity(
                benchmark, results, in_progress=True, attempts=1
            )
            self._partial_synced_count[benchmark.id] = loaded
            logger.debug(
                f"Partial sync of benchmark {benchmark.name}(id={benchmark.id}): "
                f"{loaded} point(s) uploaded."
            )
        except Exception as e:
            # Never let an in-progress sync break the state poll; the terminal
            # sync at completion is the backstop.
            logger.warning(
                "Partial metrics sync for benchmark "
                f"{benchmark.name}(id={benchmark.id}) failed: {e}"
            )

    def _count_ready_point_files(self, benchmark: Benchmark) -> int:
        """Number of finished per-point / per-stage result files on disk.

        The cheap gate in front of a partial sync: counting names costs a listdir,
        while actually loading them parses every point again.
        """
        mode = benchmark_load_mode(benchmark)
        try:
            if mode is BenchmarkLoadModeEnum.AUTO_TUNE:
                return len(
                    artifacts.list_point_files(self._benchmark_dir, benchmark.id)
                )
            if mode is BenchmarkLoadModeEnum.STAGES:
                return sum(
                    1
                    for i in range(len(benchmark.stages))
                    if os.path.exists(
                        artifacts.stage_report_path(
                            self._benchmark_dir, benchmark.id, i
                        )
                    )
                )
        except Exception:
            return 0
        return 0

    @staticmethod
    def _retry_sync(fn: Callable[[], None], what: str, attempts: int = 3):
        """Run `fn`, retrying transient failures with a linear backoff.

        The benchmark's own container has already exited by the time these writes
        happen, so a blip talking to the API server would otherwise silently drop
        the result for good. Re-raises the last exception when every attempt fails.
        """
        for attempt in range(1, attempts + 1):
            try:
                fn()
                return
            except Exception as e:
                if attempt == attempts:
                    raise
                logger.warning(
                    f"Failed to update {what} (attempt {attempt}/{attempts}): {e}. "
                    "Retrying."
                )
                time.sleep(attempt)

    def _read_ramp_facts(self, benchmark) -> Optional[dict]:
        """Ramp facts for this benchmark, read from its sidecar.

        Thin wrapper over :func:`analysis.read_ramp_facts`: the analysis is pure,
        this side is the one that knows where the result files live.
        """
        return analysis.read_ramp_facts(self._benchmark_dir, benchmark)

    def _log_request_failures_if_any(
        self,
        benchmark: Benchmark,
        total: int,
        successful: int,
        errored: int,
        incomplete: int,
        errored_samples: List[GenerativeRequestStats],
        incomplete_samples: List[GenerativeRequestStats],
        limit: int = 5,
    ) -> None:
        if errored <= 0 and incomplete <= 0:
            return

        errored_samples_to_show = errored_samples[:limit]
        incomplete_samples_to_show = incomplete_samples[:limit]

        if not errored_samples_to_show and not incomplete_samples_to_show:
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

        if errored_samples_to_show:
            lines.append("")
            lines.append(f"---- ERRORED REQUESTS (SHOWING UP TO {limit}) ----")
            lines.extend(self._format_request_samples(errored_samples_to_show))

        if incomplete_samples_to_show:
            lines.append("")
            lines.append(f"---- INCOMPLETE REQUESTS (SHOWING UP TO {limit}) ----")
            lines.extend(self._format_request_samples(incomplete_samples_to_show))

        message = "\n".join(lines)
        self._append_benchmark_log(benchmark, message)

    def _load_request_samples(
        self, report: GenerativeBenchmarksReport, limit: Optional[int] = 5
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

        if limit is None:
            return errored, incomplete

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

    def _build_partial_failure_state_message(
        self,
        errored: int,
        incomplete: int,
        errored_samples: List[GenerativeRequestStats],
        incomplete_samples: List[GenerativeRequestStats],
        top_n: int = 3,
    ) -> Optional[str]:
        if errored <= 0 and incomplete <= 0:
            return None

        summary = (
            "Completed with partial success: "
            f"errored={errored}, incomplete={incomplete}."
        )

        errored_reasons = self._collect_failure_reasons(
            errored_samples, fallback="Errored"
        )
        incomplete_reasons = self._collect_failure_reasons(
            incomplete_samples, fallback="Incomplete"
        )

        reason_parts: List[str] = []
        if errored_reasons:
            top_errored = ", ".join(
                f"{reason} (x{count})"
                for reason, count in errored_reasons.most_common(top_n)
            )
            reason_parts.append(f"Top errored reasons: {top_errored}")

        if incomplete_reasons:
            top_incomplete = ", ".join(
                f"{reason} (x{count})"
                for reason, count in incomplete_reasons.most_common(top_n)
            )
            reason_parts.append(f"Top incomplete reasons: {top_incomplete}")

        if reason_parts:
            summary = f"{summary} {'; '.join(reason_parts)}"
        else:
            summary = f"{summary} See benchmark logs for details."

        return self._truncate_state_message(summary)

    def _collect_failure_reasons(
        self, samples: List[GenerativeRequestStats], fallback: str
    ) -> Counter[str]:
        reasons: Counter[str] = Counter()
        for sample in samples:
            error = sample.info.error
            if error:
                reason = self._normalize_error_message(error)
            else:
                status = sample.info.status or "unknown"
                reason = f"{fallback} request (status={status})"
            reasons[reason] += 1
        return reasons

    def _normalize_error_message(self, error: str) -> str:
        stripped = error.strip()
        if not stripped:
            return "Unknown error"

        first_line = stripped.splitlines()[0]
        match = HTTP_ERROR_PATTERN.match(first_line)
        if not match:
            return first_line

        status = match.group("status")
        msg = " ".join(match.group("msg").split())
        error_type = match.group("type").strip()
        code = match.group("code").strip()

        if code and code.lower() != "none":
            normalized = f"HTTP {status} {error_type}/{code}: {msg}"
        else:
            normalized = f"HTTP {status} {error_type}: {msg}"

        return self._truncate_with_ellipsis(
            normalized, BENCHMARK_FAILURE_REASON_MAX_LEN
        )

    def _truncate_state_message(self, message: str) -> str:
        return self._truncate_with_ellipsis(message, BENCHMARK_STATE_MESSAGE_MAX_LEN)

    def _truncate_with_ellipsis(self, text: str, max_len: int) -> str:
        if len(text) <= max_len:
            return text
        if max_len <= len(TRUNCATION_SUFFIX):
            return TRUNCATION_SUFFIX[:max_len]
        return text[: max_len - len(TRUNCATION_SUFFIX)] + TRUNCATION_SUFFIX

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

    def _record_teardown(self, benchmark_id: int):
        """Remember that this benchmark's teardown has run.

        Plain dict mutation on purpose: this is reached from the sync thread as
        well as the event loop, and the module's other shared state takes the
        same approach (see `_stop_benchmark` on why `_queue_lock` cannot cover
        the thread). A lost race here costs one extra teardown, which is the
        behaviour we already had, never a missed one.
        """
        self._torn_down_ids.pop(benchmark_id, None)
        self._torn_down_ids[benchmark_id] = None
        while len(self._torn_down_ids) > _TEARDOWN_MEMO_LIMIT:
            self._torn_down_ids.popitem(last=False)

    def _is_torn_down(self, benchmark_id: int) -> bool:
        return benchmark_id in self._torn_down_ids

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

    def _maybe_snapshot_logs(self, benchmark: Benchmark):
        """Throttled log snapshot for a running benchmark (see
        BENCHMARK_LOG_SNAPSHOT_INTERVAL_SECONDS)."""
        last = self._last_log_snapshot_at.get(benchmark.id, 0.0)
        now = time.time()
        if now - last < BENCHMARK_LOG_SNAPSHOT_INTERVAL_SECONDS:
            return
        self._last_log_snapshot_at[benchmark.id] = now
        self._dump_benchmark_logs_to_file(benchmark)

    def _dump_benchmark_logs_to_file(
        self,
        benchmark: Benchmark,
    ):
        """Write the container's (full) logs to the benchmark log file.

        The provisioning subprocess already wrote its own logs to the same file;
        the container logs are (re)written after that boundary. `logs_workload`
        returns the full log each call, so we truncate back to the recorded
        boundary and rewrite — making repeated snapshots idempotent while
        preserving the provisioning logs.
        """
        try:
            logs = logs_workload(name=benchmark.name)
        except Exception as e:
            logger.error(
                f"Failed to fetch workload logs for benchmark {benchmark.name}(id={benchmark.id}): {e}"
            )
            return
        if logs is None:
            return

        log_str = logs
        if isinstance(log_str, (bytes, bytearray)):
            log_str = log_str.decode("utf-8", errors="replace")
        log_str = str(log_str)

        log_file_path = f"{self._benchmark_log_dir}/{benchmark.id}.log"
        try:
            size = (
                os.path.getsize(log_file_path) if os.path.exists(log_file_path) else 0
            )
            # Boundary = end of the provisioning logs, captured on first snapshot.
            offset = self._container_log_offset.get(benchmark.id)
            if offset is None:
                offset = size
                self._container_log_offset[benchmark.id] = offset
            offset = min(offset, size)  # guard against a shrunk/recreated file

            mode = "r+" if os.path.exists(log_file_path) else "w"
            with open(log_file_path, mode, encoding="utf-8") as f:
                f.seek(offset)
                f.truncate()
                if offset > 0:
                    f.write("\n---- Benchmark container logs ----\n")
                f.write(log_str)
                if not log_str.endswith("\n"):
                    f.write("\n")
        except Exception as e:
            logger.error(
                f"Failed to write workload logs for benchmark {benchmark.name}(id={benchmark.id}): {e}"
            )
