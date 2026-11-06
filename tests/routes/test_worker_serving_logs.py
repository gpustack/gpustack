"""Route behavior for paged worker serving logs."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.api.exceptions import BadRequestException, NotFoundException
from gpustack.routes.model_instances import _exact_length_chunks
from gpustack.routes.worker.logs import get_serve_logs
from gpustack.worker import logs as worker_logs
from gpustack.worker.logs import LogOptions
from gpustack.worker.log_sources import (
    main_log_path,
    container_log_path,
    sidecar_container_log_path,
)

MODULE = "gpustack.routes.worker.logs"


@pytest.mark.asyncio
@pytest.mark.parametrize('paged', [False, True])
async def test_explicit_round_does_not_fall_back_to_follower_local_latest(
    tmp_path, paged
):
    directory = tmp_path / 'serve'
    main = main_log_path(directory, 'model', 7, 7)
    main.parent.mkdir(parents=True)
    main.write_text('old startup\n')
    container_log_path(directory, 'model', 7, 7).write_text('old engine\n')
    options = LogOptions(offset=0 if paged else None)
    with pytest.raises(NotFoundException):
        await get_serve_logs(
            _request(tmp_path),
            7,
            options,
            current_restart_count=8,
            container_name=None,
            model_file_id=None,
        )
    options.previous = True
    response = await get_serve_logs(
        _request(tmp_path),
        7,
        options,
        current_restart_count=8,
        container_name=None,
        model_file_id=None,
    )
    body = b''.join(
        [
            chunk if isinstance(chunk, bytes) else chunk.encode()
            async for chunk in response.body_iterator
        ]
    )
    assert b'old startup' in body
    assert b'old engine' in body


@pytest.mark.asyncio
@pytest.mark.parametrize('paged', [False, True])
@pytest.mark.parametrize('container', [None, 'default'])
async def test_explicit_round_keeps_download_logs_before_preparation_exists(
    tmp_path, paged, container
):
    directory = tmp_path / 'serve'
    directory.mkdir()
    (directory / 'model_file_9.download.log').write_text('preparing model\n')
    response = await get_serve_logs(
        _request(tmp_path),
        7,
        LogOptions(offset=0 if paged else None),
        current_restart_count=8,
        container_name=container,
        model_file_id=9,
    )
    body = b''.join(
        [
            chunk if isinstance(chunk, bytes) else chunk.encode()
            async for chunk in response.body_iterator
        ]
    )
    assert b'preparing model' in body


@pytest.mark.asyncio
@pytest.mark.parametrize('paged', [False, True])
@pytest.mark.parametrize('previous', [False, True])
@pytest.mark.parametrize(
    'existing,requested',
    [
        ('router', None),
        ('router', 'default'),
        ('main', 'ray-head'),
        ('router', 'ray-head'),
    ],
)
async def test_explicit_round_requires_the_requested_stream(
    tmp_path, paged, previous, existing, requested
):
    directory = tmp_path / 'serve'
    path = (
        main_log_path(directory, 'model', 7, 8 - int(previous))
        if existing == 'main'
        else sidecar_container_log_path(
            directory, 'model', 7, existing, 8 - int(previous)
        )
    )
    path.parent.mkdir(parents=True)
    path.write_text('unrelated output\n')
    (directory / 'model_file_9.download.log').write_text('download output\n')

    with pytest.raises(NotFoundException):
        await get_serve_logs(
            _request(tmp_path),
            7,
            LogOptions(offset=0 if paged else None, previous=previous),
            current_restart_count=8,
            container_name=requested,
            model_file_id=9 if requested == 'ray-head' else None,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize('paged', [False, True])
@pytest.mark.parametrize('previous', [False, True])
async def test_explicit_round_streams_only_the_requested_sidecar(
    tmp_path, paged, previous
):
    directory = tmp_path / 'serve'
    restart = 8 - int(previous)
    main = main_log_path(directory, 'model', 7, restart)
    main.parent.mkdir(parents=True)
    main.write_text('startup output\n')
    (directory / 'model_file_9.download.log').write_text('download output\n')
    sidecar_container_log_path(directory, 'model', 7, 'ray-head', restart).write_text(
        'ray output\n'
    )
    response = await get_serve_logs(
        _request(tmp_path),
        7,
        LogOptions(offset=0 if paged else None, previous=previous),
        current_restart_count=8,
        container_name='ray-head',
        model_file_id=9,
    )
    body = b''.join(
        [
            chunk if isinstance(chunk, bytes) else chunk.encode()
            async for chunk in response.body_iterator
        ]
    )
    assert body == b'ray output\n'


def _request(log_dir):
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(config=SimpleNamespace(log_dir=log_dir))
        )
    )


@pytest.mark.asyncio
async def test_a_range_page_reports_the_merged_stream_total(tmp_path):
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    (serve_dir / "model_file_9.download.log").write_bytes(b"download\n")
    head = serve_dir / "main.log"
    head.write_bytes(b"h1\nh2\n")
    marker = serve_dir / "main.log.truncated"
    marker.write_bytes(b"... omitted ...\n")
    shard = serve_dir / "main.log.1"
    shard.write_bytes(b"s1\ns2\n")

    with (
        patch(f"{MODULE}.resolve_restart_count", AsyncMock(return_value=0)),
        patch(
            f"{MODULE}.serve_log_paths",
            AsyncMock(return_value=[head, marker, shard]),
        ),
    ):
        response = await get_serve_logs(
            _request(tmp_path),
            7,
            LogOptions(offset=2, limit=3),
            model_file_id=9,
            container_name="default",
        )
        body = b"".join([chunk async for chunk in response.body_iterator])

    assert body == b"h2\n... omitted ...\ns1\n"
    assert response.headers["x-log-offset"] == "2"
    assert response.headers["x-log-line-count"] == "3"
    assert response.headers["x-log-total-lines"] == "6"
    assert "x-log-total-bytes" not in response.headers


@pytest.mark.asyncio
async def test_a_whole_range_reports_its_exact_byte_length(tmp_path):
    head = tmp_path / "main.log"
    head.write_bytes(b"one\ntwo\n")

    with (
        patch(f"{MODULE}.resolve_restart_count", AsyncMock(return_value=0)),
        patch(f"{MODULE}.serve_log_paths", AsyncMock(return_value=[head])),
    ):
        response = await get_serve_logs(
            _request(tmp_path), 7, LogOptions(offset=0, limit=100)
        )
        body = b"".join([chunk async for chunk in response.body_iterator])

    assert body == b"one\ntwo\n"
    assert response.headers["x-log-total-lines"] == "2"
    assert response.headers["x-log-total-bytes"] == str(len(body))


@pytest.mark.asyncio
@pytest.mark.parametrize("option", [{"tail": 1}, {"follow": True}])
async def test_a_range_rejects_tail_and_follow(tmp_path, option):
    with pytest.raises(BadRequestException) as raised:
        await get_serve_logs(_request(tmp_path), 7, LogOptions(offset=0, **option))

    assert raised.value.status_code == 400
    assert raised.value.message == "offset cannot be combined with tail or follow"


@pytest.mark.asyncio
async def test_a_measured_download_fails_when_its_file_cannot_open(tmp_path):
    head = tmp_path / "main.log"
    head.write_bytes(b"important failure details\n")

    with (
        patch(f"{MODULE}.resolve_restart_count", AsyncMock(return_value=0)),
        patch(f"{MODULE}.serve_log_paths", AsyncMock(return_value=[head])),
    ):
        response = await get_serve_logs(
            _request(tmp_path), 7, LogOptions(offset=0, limit=100)
        )
    expected = int(response.headers["x-log-total-bytes"])
    chunks = []

    with (
        patch.object(
            worker_logs, "open", side_effect=OSError("disk error"), create=True
        ),
        pytest.raises(OSError, match="disk error"),
    ):
        async for chunk in _exact_length_chunks(response.body_iterator, expected):
            chunks.append(chunk)

    assert chunks == []
    assert expected == len(head.read_bytes())


@pytest.mark.asyncio
async def test_a_measured_download_fails_when_its_file_stops_reading(tmp_path):
    head = tmp_path / "main.log"
    head.write_bytes(b"one\ntwo\nthree\n")

    with (
        patch(f"{MODULE}.resolve_restart_count", AsyncMock(return_value=0)),
        patch(f"{MODULE}.serve_log_paths", AsyncMock(return_value=[head])),
    ):
        response = await get_serve_logs(
            _request(tmp_path), 7, LogOptions(offset=0, limit=100)
        )
    expected = int(response.headers["x-log-total-bytes"])

    class FailingReader:
        def __init__(self):
            self.file = head.open("rb")
            self.lines_read = 0

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

        def seek(self, *args):
            return self.file.seek(*args)

        def readline(self, *args):
            self.lines_read += 1
            if self.lines_read == 3:
                raise OSError("disk error")
            return self.file.readline(*args)

        def close(self):
            self.file.close()

    chunks = []
    with (
        patch.object(worker_logs, "_open_reads", return_value=[FailingReader()]),
        patch.object(worker_logs, "_SLICE_CHUNK_BYTES", 4),
        pytest.raises(OSError, match="disk error"),
    ):
        async for chunk in _exact_length_chunks(response.body_iterator, expected):
            chunks.append(chunk)

    assert chunks == [b"one\n", b"two\n"]


@pytest.mark.asyncio
async def test_explicit_rank_source_reads_the_same_filtered_window_and_length(tmp_path):
    from tests.worker.test_logs import _rank_source_archive
    from gpustack.routes.worker.logs import get_serve_log_options

    _, _, sources = _rank_source_archive(tmp_path / 'serve')
    request = _request(tmp_path)
    request.app.state.serve_manager = SimpleNamespace(_worker_id=7)
    source = next(
        s.descriptor
        for s in sources
        if s.descriptor.kind == 'rank' and s.descriptor.dp_rank == 0
    )
    with patch(f'{MODULE}.instance_log_files') as legacy_walk:
        options = await get_serve_log_options(request, 42, current_restart_count=3)
    legacy_walk.assert_not_called()
    assert all(entry.started_at is not None for entry in options.restarts)
    assert any(s.id == source.id for entry in options.restarts for s in entry.sources)
    response = await get_serve_logs(
        request,
        42,
        LogOptions(offset=0, limit=100),
        container_name=None,
        model_file_id=None,
        source=source.id,
        current_restart_count=3,
    )
    data = b''.join([chunk async for chunk in response.body_iterator])
    assert data == b'[dp0-node0] first\n[dp0-node0] last\n'
    assert response.headers['x-log-total-bytes'] == str(len(data))
    assert response.headers['x-log-total-lines'] == '2'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'restart,container,previous,message',
    [
        (None, None, False, 'source requires current_restart_count'),
        (None, 'default', False, 'source requires current_restart_count'),
        (3, 'default', False, 'A source cannot be combined with container or previous'),
        (3, None, True, 'A source cannot be combined with container or previous'),
    ],
)
async def test_source_selector_reports_the_invalid_parameter(
    tmp_path, restart, container, previous, message
):
    with pytest.raises(BadRequestException) as raised:
        await get_serve_logs(
            _request(tmp_path),
            7,
            LogOptions(previous=previous),
            source='opaque',
            current_restart_count=restart,
            container_name=container,
            model_file_id=None,
        )
    assert raised.value.status_code == 400
    assert raised.value.message == message


@pytest.mark.asyncio
async def test_missing_previous_source_never_reads_current(tmp_path):
    from tests.worker.test_logs import _rank_source_archive
    from gpustack.api.exceptions import NotFoundException

    _rank_source_archive(tmp_path / 'serve')
    request = _request(tmp_path)
    request.app.state.serve_manager = SimpleNamespace(_worker_id=7)
    with pytest.raises(NotFoundException):
        await get_serve_logs(
            request,
            42,
            LogOptions(previous=True),
            source=None,
            model_file_id=None,
            container_name=None,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'failure',
    [
        FileNotFoundError('missing'),
        OSError('Log source changed while reading'),
        OSError('Log source was truncated while reading'),
    ],
)
async def test_source_removed_during_planning_reports_expiration(tmp_path, failure):
    from gpustack.api.exceptions import NotFoundException
    from gpustack.routes.worker.logs import _serve_source

    request = _request(tmp_path)
    request.app.state.serve_manager = SimpleNamespace(_worker_id=7)
    missing = tmp_path / 'main.log.1'
    source = SimpleNamespace(paths=[missing], descriptor=SimpleNamespace(kind='rank'))
    with (
        patch(f'{MODULE}.resolve_log_source', return_value=source),
        patch(f'{MODULE}.plan_source_window', side_effect=failure) as plan,
    ):
        with pytest.raises(NotFoundException) as error:
            await _serve_source(request, tmp_path, 42, 'rank', 3, LogOptions(offset=0))
    assert error.value.status_code == 404
    assert 'expired' in error.value.message
    assert plan.call_args.args[-1].is_set()
