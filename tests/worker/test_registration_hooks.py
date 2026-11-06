"""Worker registration waits for process-local plugin prerequisites."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import tenacity

from gpustack import extension
from gpustack.extension import Plugin
from gpustack.worker.worker import Worker


def worker_with_plugins(monkeypatch, plugins):
    monkeypatch.setattr(extension, 'iter_plugin_classes', lambda: iter(plugins))
    worker = Worker.__new__(Worker)
    worker._config = SimpleNamespace()
    row = SimpleNamespace(id=1, cluster_id=2, name='node', worker_uuid='uuid')
    client = SimpleNamespace(
        workers=SimpleNamespace(list=Mock(return_value=SimpleNamespace(items=[row])))
    )
    worker._worker_manager = SimpleNamespace(
        register_with_server=AsyncMock(return_value=(client, None))
    )
    worker.worker_name = lambda: 'node'
    return worker


@pytest.mark.asyncio
async def test_registration_awaits_hooks_in_entry_point_order(monkeypatch):
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = []

    class First(Plugin):
        @classmethod
        async def before_worker_registration(cls, cfg):
            calls.append(('first', cfg))
            entered.set()
            await release.wait()

    class Second(Plugin):
        @classmethod
        async def before_worker_registration(cls, cfg):
            calls.append(('second', cfg))

    worker = worker_with_plugins(monkeypatch, [('first', First), ('second', Second)])
    task = asyncio.create_task(Worker._register.__wrapped__(worker))
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        worker._worker_manager.register_with_server.assert_not_awaited()
        assert calls == [('first', worker._config)]
        release.set()
        await task
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert calls == [('first', worker._config), ('second', worker._config)]
    assert worker.worker_id() == 1


@pytest.mark.asyncio
async def test_hook_error_prevents_registration(monkeypatch):
    class Reject(Plugin):
        @classmethod
        async def before_worker_registration(cls, cfg):
            raise RuntimeError('prerequisite failed')

    worker = worker_with_plugins(monkeypatch, [('reject', Reject)])
    with pytest.raises(RuntimeError, match='prerequisite failed'):
        await Worker._register.__wrapped__(worker)
    worker._worker_manager.register_with_server.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancelling_wait_does_not_register_or_retry(monkeypatch):
    entered = asyncio.Event()
    calls = []

    class Wait(Plugin):
        @classmethod
        async def before_worker_registration(cls, cfg):
            calls.append(cfg)
            entered.set()
            await asyncio.Event().wait()

    worker = worker_with_plugins(monkeypatch, [('wait', Wait)])
    task = asyncio.create_task(worker._register())
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled()
    assert calls == [worker._config]
    worker._worker_manager.register_with_server.assert_not_awaited()


@pytest.mark.asyncio
async def test_registration_rechecks_prerequisites_on_retry(monkeypatch):
    hook = AsyncMock()

    class Probe(Plugin):
        before_worker_registration = classmethod(hook)

    worker = worker_with_plugins(monkeypatch, [('probe', Probe)])
    manager = worker._worker_manager
    success = manager.register_with_server.return_value
    manager.register_with_server.side_effect = [RuntimeError('try again'), success]
    await Worker._register.retry_with(wait=tenacity.wait_none())(worker)
    assert hook.await_count == 2
    assert manager.register_with_server.await_count == 2


@pytest.mark.asyncio
async def test_default_and_legacy_plugins_preserve_registration(monkeypatch):
    class Legacy(Plugin):
        def __init__(self, app, cfg):
            raise AssertionError('worker hooks cannot construct an app plugin')

        @classmethod
        async def before_worker_registration(cls, cfg):
            raise AssertionError('new hooks are static-only')

    worker = worker_with_plugins(
        monkeypatch, [('default', Plugin), ('legacy', Legacy), ('unrelated', object)]
    )
    await Worker._register.__wrapped__(worker)
    worker._worker_manager.register_with_server.assert_awaited_once()
    assert worker.worker_id() == 1


def _register_in_spawned_worker(queue, cfg):
    """Discover a worker plugin without constructing any server-side context."""
    from importlib import metadata

    class WorkerPlugin(Plugin):
        @classmethod
        def setup(cls, ctx):
            raise AssertionError('server setup cannot run in a worker')

        @classmethod
        async def before_worker_registration(cls, config):
            queue.put(('hook', config.marker))

    metadata.entry_points = lambda **kwargs: [
        SimpleNamespace(name='probe', load=lambda: WorkerPlugin)
    ]
    worker = worker_with_plugins(SimpleNamespace(setattr=lambda *args: None), [])
    worker._config = cfg
    asyncio.run(Worker._register.__wrapped__(worker))
    queue.put(('registered', worker.worker_id()))


def test_spawned_worker_discovers_plugin_hooks_without_server_setup():
    import multiprocessing

    ctx = multiprocessing.get_context('spawn')
    queue = ctx.Queue()
    process = ctx.Process(
        target=_register_in_spawned_worker,
        args=(queue, SimpleNamespace(marker='worker-config')),
    )
    process.start()
    try:
        assert queue.get(timeout=15) == ('hook', 'worker-config')
        assert queue.get(timeout=15) == ('registered', 1)
        process.join(timeout=15)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
        queue.close()
        queue.join_thread()
