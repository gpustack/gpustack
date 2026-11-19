"""Independent route regressions for execution restart and archive failure isolation."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from datetime import datetime, timezone

from fastapi.testclient import TestClient
import pytest

from gpustack.api.exceptions import ServiceUnavailableException
from gpustack.routes import benchmarks as routes
from gpustack.schemas.benchmark import BenchmarkStateEnum
from gpustack.schemas.benchmark_monitoring import BenchmarkRunEvent
from gpustack.schemas.principals import OrgRole, PrincipalType
from gpustack.server.deps import get_session
from tests.routes.test_benchmark_monitoring import api as monitoring_api

api = monitoring_api


@pytest.fixture
def restart(api, monkeypatch):
    api.benchmark.run_id = 'previous-run'
    api.benchmark.state = BenchmarkStateEnum.COMPLETED
    api.benchmark.progress = 100
    api.benchmark.cluster_id = 1
    api.benchmark.model_instance_name = 'saved-instance'
    api.benchmark.created_at = api.benchmark.updated_at = datetime.now(timezone.utc)
    trace = []
    session = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock())
    api.app.dependency_overrides[get_session] = lambda: session

    async def archive(_session, benchmark):
        trace.append(
            (
                'archive',
                benchmark.run_id,
                benchmark.state,
                benchmark.snapshot.monitoring.confirmed_at,
            )
        )

    async def remove(_session, *, auto_commit):
        assert auto_commit is False
        trace.append(('remove-results',))

    async def update(benchmark, _session, source, *, auto_commit):
        assert auto_commit is False
        trace.append(('write-execution', source.get('run_id', benchmark.run_id)))
        for key, value in source.items():
            setattr(benchmark, key, value)
        return benchmark

    async def commit():
        trace.append(('commit',))

    session.commit.side_effect = commit
    ensure = AsyncMock(side_effect=archive)
    write = AsyncMock(side_effect=update)
    monkeypatch.setattr(routes, 'update_archive_snapshot', ensure)

    # A real method binding is needed so the write acts on the selected row.
    async def update_bound(benchmark, _session, source, *, auto_commit):
        return await write(benchmark, _session, source, auto_commit=auto_commit)

    monkeypatch.setattr(routes.Benchmark, 'update', update_bound)
    monkeypatch.setattr(
        routes.BenchmarkResult,
        'all_by_field',
        AsyncMock(return_value=[SimpleNamespace(delete=AsyncMock(side_effect=remove))]),
    )
    verify = AsyncMock(return_value=api.benchmark.snapshot.model_copy(deep=True))
    monkeypatch.setattr(routes, 'verify_benchmark_run', verify)
    return SimpleNamespace(
        api=api, session=session, trace=trace, ensure=ensure, write=write, verify=verify
    )


@pytest.mark.parametrize('state', ['queued', 'running'])
def test_worker_state_reports_observe_changed_generation_without_restarting(
    restart, state
):
    restart.api.ctx.user.kind = PrincipalType.SYSTEM
    restart.api.ctx.scoped_cluster_id = 1
    restart.api.ctx.scoped_cluster_owner_id = 4
    restart.api.benchmark.state = (
        BenchmarkStateEnum.PENDING if state == 'queued' else BenchmarkStateEnum.QUEUED
    )
    snapshot = restart.api.benchmark.snapshot.model_copy(deep=True)
    snapshot.monitoring.events.append(
        BenchmarkRunEvent(kind='generation_changed', at=1050, model_instance_id=12)
    )
    restart.verify.return_value = snapshot

    with TestClient(restart.api.app) as client:
        result = client.patch(
            '/v2/benchmarks/7/state',
            json={'state': state, 'run_id': 'previous-run'},
        )

    assert result.status_code == 200, result.text
    assert result.json()['run_id'] == 'previous-run'
    assert result.json()['state'] == state
    assert 'run_id' not in restart.write.await_args.args[2]
    assert restart.api.benchmark.snapshot.monitoring.events[-1].kind == (
        'generation_changed'
    )
    assert restart.ensure.await_count == 1
    restart.session.commit.assert_awaited_once()
    restart.session.rollback.assert_not_awaited()


def test_stale_worker_state_report_is_fenced_before_restart_work(restart):
    restart.api.ctx.user.kind = PrincipalType.SYSTEM
    restart.api.ctx.scoped_cluster_id = 1
    restart.api.ctx.scoped_cluster_owner_id = 4
    with TestClient(restart.api.app) as client:
        result = client.patch(
            '/v2/benchmarks/7/state',
            json={'state': 'queued', 'run_id': 'retired-run'},
        )

    assert result.status_code == 409
    assert result.json()['details'] == {'code': 'benchmark_run_changed'}
    restart.write.assert_not_awaited()
    restart.ensure.assert_not_awaited()
    restart.verify.assert_not_awaited()
    restart.session.commit.assert_not_awaited()


@pytest.mark.parametrize('denial', ['foreign_owner', 'foreign_cluster', 'member'])
def test_state_update_authorizes_before_archiving_or_mutating(restart, denial):
    if denial == 'foreign_owner':
        restart.api.benchmark.owner_principal_id = 99
    elif denial == 'foreign_cluster':
        restart.api.ctx.user.kind = PrincipalType.SYSTEM
        restart.api.ctx.scoped_cluster_id = 99
        restart.api.ctx.scoped_cluster_owner_id = 4
    else:
        restart.api.ctx.org_role = OrgRole.MEMBER
    with TestClient(restart.api.app) as client:
        result = client.patch(
            '/v2/benchmarks/7/state',
            json={'state': 'pending', 'run_id': 'previous-run'},
        )

    assert result.status_code == (403 if denial == 'member' else 404)
    restart.write.assert_not_awaited()
    restart.ensure.assert_not_awaited()
    restart.verify.assert_not_awaited()
    restart.session.commit.assert_not_awaited()


@pytest.mark.parametrize(
    'path,reader',
    [
        ('metrics', 'read_archive_metrics'),
        ('resource-metrics', 'read_archive_resources'),
        ('phase-observations', 'read_archive_phase_observations'),
    ],
)
def test_pinned_archive_storage_failure_never_uses_current_deployment(
    api, monkeypatch, path, reader
):
    api.benchmark.state = BenchmarkStateEnum.COMPLETED
    read = AsyncMock(
        side_effect=ServiceUnavailableException(
            message='Archive temporarily unreadable'
        )
    )
    monkeypatch.setattr(routes, reader, read)
    live = AsyncMock()
    monkeypatch.setattr(routes, 'load_benchmark_history', live)
    with TestClient(api.app) as client:
        result = client.get(
            f'/v2/benchmarks/7/{path}',
            params={'run_id': 'previous-run', 'content_version': '3'},
        )
    assert result.status_code == 503
    read.assert_awaited_once()
    assert read.await_args.kwargs['run_id'] == 'previous-run'
    assert read.await_args.kwargs['content_version'] == '3'
    live.assert_not_awaited()


def test_retry_body_execution_token_is_forwarded_and_conflicts_do_not_enqueue(
    api, monkeypatch
):
    retry = AsyncMock(
        return_value={
            'state': 'capturing',
            'run_id': 'saved-run',
            'content_version': '2',
        }
    )
    monkeypatch.setattr(routes, 'retry_archive', retry)
    with TestClient(api.app) as client:
        success = client.post(
            '/v2/benchmarks/7/monitoring-archive/retry', json={'run_id': 'saved-run'}
        )
        conflict = client.post(
            '/v2/benchmarks/7/monitoring-archive/retry?run_id=current-run',
            json={'run_id': 'saved-run'},
        )
    assert success.status_code == 200
    assert success.json()['state'] == 'capturing'
    assert conflict.status_code == 400
    retry.assert_awaited_once_with(None, api.ctx, api.benchmark, run_id='saved-run')


@pytest.mark.parametrize(
    'path', ['monitoring-archive', 'phase-observations', 'monitoring-archive/retry']
)
@pytest.mark.parametrize('denial', ['foreign', 'member'])
def test_archive_control_routes_authorize_before_read_or_retry(
    api, monkeypatch, path, denial
):
    if denial == 'foreign':
        api.benchmark.owner_principal_id = 99
    else:
        api.ctx.org_role = OrgRole.MEMBER
    calls = []
    for name in ('archive_status', 'read_archive_phase_observations', 'retry_archive'):
        operation = AsyncMock()
        monkeypatch.setattr(routes, name, operation)
        calls.append(operation)
    with TestClient(api.app) as client:
        result = (
            client.post(f'/v2/benchmarks/7/{path}', json={'run_id': 'saved-run'})
            if path.endswith('/retry')
            else client.get(f'/v2/benchmarks/7/{path}')
        )
    assert result.status_code == (404 if denial == 'foreign' else 403)
    for operation in calls:
        operation.assert_not_awaited()
