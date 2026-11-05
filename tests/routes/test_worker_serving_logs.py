"""Route behavior for paged worker serving logs."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gpustack.api.exceptions import BadRequestException
from gpustack.routes.model_instances import _exact_length_chunks
from gpustack.routes.worker.logs import get_serve_logs
from gpustack.worker import logs as worker_logs
from gpustack.worker.logs import LogOptions

MODULE = "gpustack.routes.worker.logs"


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
