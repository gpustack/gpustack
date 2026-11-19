import asyncio
import json
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from gpustack.worker import benchmark_manager as manager
from gpustack.worker.benchmark.runner import BenchmarkRunner


@pytest.mark.parametrize('run_id', ['execution-a', 'execution &x=#?+/%', None])
def test_result_posts_are_bound_to_their_execution(run_id):
    requests = []

    def record(request):
        requests.append(request)
        return httpx.Response(200)

    with httpx.Client(
        base_url='http://server/v2', transport=httpx.MockTransport(record)
    ) as client:
        instance = object.__new__(manager.BenchmarkManager)
        instance._clientset_getter = lambda: SimpleNamespace(
            http_client=SimpleNamespace(get_httpx_client=lambda: client)
        )
        benchmark = SimpleNamespace(id=7, name='bench', run_id=run_id)
        instance._post_metrics_and_results(
            benchmark, SimpleNamespace(model_dump=lambda: {}), []
        )

    assert [request.url.path for request in requests] == [
        '/v2/benchmarks/7/metrics',
        '/v2/benchmarks/7/results',
    ]
    expected = [('run_id', run_id)] if run_id is not None else []
    assert all(request.url.params.multi_items() == expected for request in requests)


def test_runner_failure_patch_includes_execution_identity():
    client = Mock()
    runner = object.__new__(BenchmarkRunner)
    runner._benchmark = SimpleNamespace(run_id='execution-a')
    runner._clientset = SimpleNamespace(
        http_client=SimpleNamespace(get_httpx_client=lambda: client)
    )
    runner._update_benchmark_state(7, state='error')
    assert client.patch.call_args.kwargs['json']['run_id'] == 'execution-a'


@pytest.mark.parametrize(
    'code', ['benchmark_run_changed', 'benchmark_already_executed']
)
def test_retired_writer_ack_releases_local_execution(code):
    response = Mock(status_code=409)
    response.json.return_value = {'details': {'code': code}}
    assert manager.BenchmarkManager._state_patch_landed(7, response) is False
    response.raise_for_status.assert_not_called()


@pytest.mark.parametrize(
    'body',
    [
        {},
        {'details': {'code': 'benchmark_target_changed'}},
        {'details': {'code': 'unrelated_conflict'}},
        {'details': None},
    ],
)
def test_conflicts_that_do_not_retire_an_execution_still_raise(body):
    response = httpx.Response(
        409,
        json=body,
        request=httpx.Request('PATCH', 'http://server/benchmarks/7/state'),
    )
    with pytest.raises(httpx.HTTPStatusError):
        manager.BenchmarkManager._state_patch_landed(7, response)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'code', ['benchmark_run_changed', 'benchmark_already_executed']
)
async def test_retired_enqueue_is_skipped_and_the_next_execution_can_start(code):
    patches = []

    def patch(request):
        patches.append(request)
        if request.url.path == '/benchmarks/7/state':
            return httpx.Response(409, json={'details': {'code': code}})
        return httpx.Response(200)

    async with httpx.AsyncClient(
        base_url='http://server', transport=httpx.MockTransport(patch)
    ) as client:
        instance = object.__new__(manager.BenchmarkManager)
        instance._clientset_getter = lambda: SimpleNamespace(
            http_client=SimpleNamespace(get_async_httpx_client=lambda: client)
        )
        instance._benchmark_queue = deque()
        instance._queue_lock = asyncio.Lock()
        instance._canceled_ids = set()
        instance._torn_down_ids = {}
        instance._active_benchmark_id = None
        started = []

        async def start(benchmark):
            started.append(benchmark.id)
            raise asyncio.CancelledError

        instance._start_benchmark = start
        for id in (7, 8):
            await instance._enqueue_benchmark(
                SimpleNamespace(id=id, name=f'bench-{id}', run_id=f'run-{id}')
            )
        with pytest.raises(asyncio.CancelledError):
            await instance._benchmark_queue_worker()

    assert started == [8]
    assert not instance._benchmark_queue
    assert len(patches) == 2
    assert [request.url.path for request in patches] == [
        '/benchmarks/7/state',
        '/benchmarks/8/state',
    ]


@pytest.mark.asyncio
async def test_start_rejected_for_a_terminal_run_stops_without_posting_error(
    tmp_path, monkeypatch
):
    updates = []

    def patch(request):
        update = json.loads(request.content)
        updates.append(update)
        if update['state'] == 'running':
            return httpx.Response(
                409, json={'details': {'code': 'benchmark_already_executed'}}
            )
        return httpx.Response(200)

    process = Mock(pid=123)
    process.is_alive.return_value = True
    terminate = Mock()
    delete_workload = Mock()
    monkeypatch.setattr(manager.multiprocessing, 'Process', Mock(return_value=process))
    monkeypatch.setattr(
        manager.registration, 'determine_default_registry', lambda _: None
    )
    monkeypatch.setattr(manager, 'terminate_multiprocessing_process', terminate)
    monkeypatch.setattr(manager, 'delete_workload', delete_workload)
    async with httpx.AsyncClient(
        base_url='http://server', transport=httpx.MockTransport(patch)
    ) as client:
        instance = object.__new__(manager.BenchmarkManager)
        instance._clientset_getter = lambda: SimpleNamespace(
            headers={},
            http_client=SimpleNamespace(get_async_httpx_client=lambda: client),
        )
        instance._benchmark_dir = instance._benchmark_log_dir = str(tmp_path)
        instance._config = SimpleNamespace(system_default_container_registry=None)
        instance._provisioning_processes = {}
        instance._benchmark_by_id = {}
        instance._container_log_offset = {}
        instance._last_log_snapshot_at = {}
        instance._partial_synced_count = {}
        instance._last_partial_sync_at = {}
        instance._last_target_check_at = {}
        instance._target_missing_since = {}
        instance._torn_down_ids = {}
        benchmark = SimpleNamespace(id=7, name='bench', run_id='execution-a')

        await instance._start_benchmark(benchmark)

    process.start.assert_called_once()
    terminate.assert_called_once_with(process)
    delete_workload.assert_called_once_with(benchmark.name)
    assert not instance._provisioning_processes
    assert instance._active_benchmark_id is None
    assert updates == [{'run_id': 'execution-a', 'state': 'running', 'pid': 123}]
