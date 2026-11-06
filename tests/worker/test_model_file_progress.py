from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import asyncio
import hashlib
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import httpx

from gpustack.api.exceptions import NotFoundException
from gpustack.schemas.model_files import ModelFile, ModelFileStateEnum
from gpustack.schemas.models import SourceEnum
from gpustack.worker import model_file_manager
from gpustack.utils.hub import FileEntry
from gpustack.worker import downloaders


def test_model_file_info_keeps_hub_file_sizes(monkeypatch):
    from huggingface_hub import hf_api
    from huggingface_hub.utils import httpx

    response = httpx.Response(
        200,
        json={
            'id': 'test/model',
            'sha': 'a' * 40,
            'siblings': [
                {'rfilename': 'sub/weights.safetensors', 'size': 100},
                {'rfilename': 'config.json', 'size': 20},
            ],
        },
        request=httpx.Request('GET', 'https://huggingface.co/api/models/test/model'),
    )
    session = MagicMock()
    session.get.return_value = response
    monkeypatch.setattr(hf_api, 'get_session', lambda: session)

    files = downloaders.HfDownloader.get_model_file_info(
        ModelFile(source=SourceEnum.HUGGING_FACE, huggingface_repo_id='test/model'),
        None,
    )

    assert [(file.rfilename, file.size) for file in files] == [
        ('sub/weights.safetensors', 100),
        ('config.json', 20),
    ]


@pytest.fixture
def task_factory(monkeypatch, config):
    hijack_tqdm = model_file_manager.ModelFileDownloadTask.hijack_tqdm_progress
    monkeypatch.setattr(model_file_manager, "setup_logging", MagicMock())
    monkeypatch.setattr(model_file_manager, "read_worker_token", lambda _: "test-token")

    def clientset(**kwargs):
        client = MagicMock()
        client.http_client.get_httpx_client.return_value.timeout = httpx.Timeout(
            kwargs.get('timeout', 60.0)
        )
        return client

    monkeypatch.setattr(model_file_manager, "ClientSet", clientset)
    for name in (
        "_ensure_model_file_size_and_paths",
        "_setup_instance_log_files",
        "hijack_tqdm_progress",
    ):
        monkeypatch.setattr(model_file_manager.ModelFileDownloadTask, name, MagicMock())

    def create_task(track_progress=False):
        model_file = ModelFile(
            id=1,
            worker_id=1,
            source=SourceEnum.HUGGING_FACE,
            huggingface_repo_id="test/model",
            state=ModelFileStateEnum.DOWNLOADING,
            size=100,
        )
        task = model_file_manager.ModelFileDownloadTask(
            model_file, config, threading.Event()
        )
        task.prerun()
        task._clientset.model_files.get.return_value = model_file
        task._progress_clientset.model_files = task._clientset.model_files
        if track_progress:
            from tqdm import tqdm

            for name, original in (
                ('__init__', tqdm.__init__),
                ('update', tqdm.update),
                ('_original_init', tqdm.__init__),
                ('_original_update', tqdm.update),
            ):
                monkeypatch.setattr(tqdm, name, original, raising=False)
            hijack_tqdm(task)
            task._log_update_interval = 0
        return task

    return create_task


@pytest.mark.parametrize('filename', [None, 'sub/*.safetensors'])
@pytest.mark.parametrize(
    'outcome', ['complete', 'error', 'cancel', 'cache', 'stale_cache']
)
def test_huggingface_xet_progress_counts_file_bytes(
    task_factory, tmp_path, monkeypatch, filename, outcome
):
    from huggingface_hub import file_download
    from huggingface_hub.utils import _xet

    task = task_factory(track_progress=True)
    task._model_file.local_dir = str(tmp_path)
    task._model_file.huggingface_filename = filename
    if outcome in ('cache', 'stale_cache'):
        (tmp_path / 'sub').mkdir()
        (tmp_path / 'sub/weights.safetensors').write_bytes(
            (b'x' if outcome == 'cache' else b'y') * 100
        )
    monkeypatch.setattr(task, 'prerun', lambda: None)
    monkeypatch.setattr(task, '_resolve_lora_state', lambda _: {})
    monkeypatch.setattr(
        downloaders,
        'match_hugging_face_files',
        lambda *args: ['sub/weights.safetensors'],
    )
    monkeypatch.setattr(file_download, 'is_xet_available', lambda: True)
    monkeypatch.setattr(
        file_download,
        'get_hf_file_metadata',
        lambda **kwargs: file_download.HfFileMetadata(
            commit_hash='a' * 40,
            etag=hashlib.sha256(b'x' * 100).hexdigest(),
            location='https://huggingface.co/test/model/resolve/main/sub/weights.safetensors',
            size=100,
            xet_file_data=_xet.XetFileData(
                'c' * 64, 'https://huggingface.co/xet-token'
            ),
        ),
    )
    monkeypatch.setattr(
        _xet,
        'refresh_xet_connection_info',
        lambda **kwargs: _xet.XetConnectionInfo(
            'test-token', 9999999999, 'https://cas.test'
        ),
    )
    samples = []
    native_abort = threading.Event()
    monkeypatch.setattr(_xet, 'abort_xet_session', native_abort.set)

    def new_download_group(**kwargs):
        def start_download_file(info, path):
            if outcome == 'cancel':
                task._cancel_flag.set()
            for reconstructed, transferred in (
                (0, 200),
                (25, 250),
                (50, 300),
                (100, 400),
            ):
                Path(path).write_bytes(b'x' * reconstructed)
                kwargs['progress_callback'](
                    SimpleNamespace(
                        total_bytes_completed=reconstructed,
                        total_transfer_bytes_completed=transferred,
                        total_bytes_completion_rate=1,
                        total_transfer_bytes_completion_rate=1,
                        total_bytes=100,
                    ),
                    {},
                )
                if outcome == 'cancel':
                    assert native_abort.wait(5)
                    raise KeyboardInterrupt('Native operation cancelled')
                samples.append(task._model_downloaded_size)
                if outcome == 'error' and reconstructed == 50:
                    raise RuntimeError('Xet download failed')

        return nullcontext(SimpleNamespace(start_download_file=start_download_file))

    monkeypatch.setattr(
        _xet,
        'get_xet_session',
        lambda: SimpleNamespace(new_file_download_group=new_download_group),
    )

    task.run()

    updates = task._clientset.model_files.update.call_args_list
    if outcome in ('error', 'cancel'):
        assert samples == ([0, 25, 50] if outcome == 'error' else [])
        assert not task._download_completed
        assert not (tmp_path / 'sub/weights.safetensors').exists()
        assert all(
            call.kwargs['model_update'].download_progress != 100 for call in updates
        )
        if outcome == 'error':
            assert updates[-1].kwargs['model_update'].state == ModelFileStateEnum.ERROR
        return

    assert samples == ([] if outcome == 'cache' else [0, 25, 50, 100])
    assert task._model_downloaded_size == 100
    assert (tmp_path / 'sub/weights.safetensors').read_bytes() == b'x' * 100
    reports = [
        call.kwargs['model_update'].download_progress
        for call in updates
        if call.kwargs['model_update'].download_progress
    ]
    assert reports[-2:] == [99.0, 100]
    assert all(progress <= 99 for progress in reports[:-1])
    assert updates[-1].kwargs['model_update'].state == ModelFileStateEnum.READY


def test_untracked_tqdm_keeps_its_counter(task_factory):
    from tqdm import tqdm

    task = task_factory(track_progress=True)
    with tqdm(total=2, unit='file') as bar:
        bar.update(1)
        assert bar.n == 1
    assert task._model_downloaded_size == 0


def test_huggingface_http_progress_counts_file_bytes(
    task_factory, tmp_path, monkeypatch
):
    from huggingface_hub import file_download

    task = task_factory(track_progress=True)
    task._model_file.local_dir = str(tmp_path)
    monkeypatch.setattr(task, '_resolve_lora_state', lambda _: {})
    monkeypatch.setattr(
        downloaders, 'match_hugging_face_files', lambda *args: ['weights.safetensors']
    )
    url = 'https://huggingface.co/test/model/resolve/main/weights.safetensors'
    monkeypatch.setattr(
        file_download,
        'get_hf_file_metadata',
        lambda **kwargs: file_download.HfFileMetadata(
            commit_hash='a' * 40,
            etag='b' * 64,
            location=url,
            size=100,
            xet_file_data=None,
        ),
    )
    http = getattr(file_download, 'httpx2', None) or file_download.httpx
    response = http.Response(200, content=b'x' * 100, request=http.Request('GET', url))
    monkeypatch.setattr(
        file_download, 'http_stream_backoff', lambda **kwargs: nullcontext(response)
    )

    task._download_model_file()

    assert task._model_downloaded_size == 100
    assert (tmp_path / 'weights.safetensors').read_bytes() == b'x' * 100
    updates = task._clientset.model_files.update.call_args_list
    reports = [call.kwargs['model_update'].download_progress for call in updates]
    assert reports[-2:] == [99, 100]
    assert all(progress <= 99 for progress in reports[:-1])
    assert updates[-1].kwargs['model_update'].state == ModelFileStateEnum.READY


@pytest.mark.parametrize('transport', ['http', 'xet'])
def test_huggingface_parallel_failure_removes_only_failed_file_progress(
    task_factory, tmp_path, monkeypatch, transport
):
    from huggingface_hub import file_download
    from huggingface_hub.utils import _xet

    unit = 1024
    task = task_factory(track_progress=True)
    task._model_file.size = task._model_file_size = 100 * unit
    task._model_file.local_dir = str(tmp_path)
    monkeypatch.setattr(task, 'prerun', lambda: None)
    failed_started = threading.Event()
    survivor_started = threading.Event()
    failed_finished = threading.Event()
    samples = []
    files = ['a/weights.safetensors', 'b/weights.safetensors']
    monkeypatch.setattr(downloaders, 'match_hugging_face_files', lambda *args: files)
    monkeypatch.setattr(file_download, 'is_xet_available', lambda: True)
    monkeypatch.setattr(file_download.constants, 'DOWNLOAD_CHUNK_SIZE', 10 * unit)

    def metadata(**kwargs):
        filename = kwargs['url'].split('/resolve/main/')[1]
        return file_download.HfFileMetadata(
            commit_hash='a' * 40,
            etag=('c' if filename == files[0] else 'd') * 64,
            location=kwargs['url'],
            size=50 * unit,
            xet_file_data=(
                _xet.XetFileData('e' * 64, f'https://test/xet/{filename[0]}')
                if transport == 'xet'
                else None
            ),
        )

    monkeypatch.setattr(file_download, 'get_hf_file_metadata', metadata)

    def transfer(filename):
        if filename == files[0]:
            yield b'a' * (20 * unit)
            failed_started.set()
            assert survivor_started.wait(5)
            raise RuntimeError('Shard download failed')
        assert failed_started.wait(5)
        yield b'b' * (30 * unit)
        survivor_started.set()
        assert failed_finished.wait(5)
        samples.append(task._model_downloaded_size / unit)
        incomplete = list(tmp_path.rglob('*.incomplete'))
        assert sum(path.stat().st_size for path in incomplete) == 30 * unit
        yield b'b' * (20 * unit)

    if transport == 'http':
        http = getattr(file_download, 'httpx2', None) or file_download.httpx

        class Stream(http.SyncByteStream):
            def __init__(self, filename):
                self.filename = filename

            def __iter__(self):
                return transfer(self.filename)

        def response(**kwargs):
            url = kwargs['url']
            return http.Response(
                200,
                headers={'content-length': str(50 * unit)},
                stream=Stream(url.split('/resolve/main/')[1]),
                request=http.Request('GET', url),
            )

        monkeypatch.setattr(
            file_download,
            'http_stream_backoff',
            lambda **kwargs: nullcontext(response(**kwargs)),
        )
    else:
        monkeypatch.setattr(
            _xet,
            'refresh_xet_connection_info',
            lambda **kwargs: _xet.XetConnectionInfo(
                'test', 9999999999, 'https://cas.test'
            ),
        )

        def group(**kwargs):
            filename = (
                files[0] if kwargs['token_refresh_url'].endswith('/a') else files[1]
            )

            def start(info, path):
                total = 0
                for chunk in transfer(filename):
                    with Path(path).open('ab') as stream:
                        stream.write(chunk)
                    total += len(chunk)
                    kwargs['progress_callback'](
                        SimpleNamespace(
                            total_bytes_completed=total,
                            total_transfer_bytes_completed=total * 3,
                            total_bytes_completion_rate=1,
                            total_transfer_bytes_completion_rate=1,
                            total_bytes=50 * unit,
                        ),
                        {},
                    )

            return nullcontext(SimpleNamespace(start_download_file=start))

        monkeypatch.setattr(
            _xet,
            'get_xet_session',
            lambda: SimpleNamespace(new_file_download_group=group),
        )

    original_map = downloaders.thread_map

    def thread_map(function, items, **kwargs):
        def tracked(filename):
            try:
                return function(filename)
            finally:
                if filename == files[0]:
                    failed_finished.set()

        return original_map(tracked, items, **kwargs)

    monkeypatch.setattr(downloaders, 'thread_map', thread_map)
    task.run()

    assert samples == [30.0]
    assert task._last_reported_progress == 50.0
    updates = task._clientset.model_files.update.call_args_list
    assert updates[-1].kwargs['model_update'].state == ModelFileStateEnum.ERROR
    assert not task._download_completed
    assert (tmp_path / files[1]).stat().st_size == 50 * unit
    assert not list(tmp_path.rglob('*.incomplete'))


@pytest.mark.parametrize('resumed', [True, False])
def test_huggingface_http_retry_replaces_file_progress(
    task_factory, tmp_path, monkeypatch, resumed
):
    from huggingface_hub import file_download

    http = getattr(file_download, 'httpx2', None) or file_download.httpx
    task = task_factory(track_progress=True)
    task._model_file.local_dir = str(tmp_path)
    task._model_file.size = task._model_file_size = 100 * 1024
    monkeypatch.setattr(task, '_resolve_lora_state', lambda _: {})
    monkeypatch.setattr(
        downloaders, 'match_hugging_face_files', lambda *args: ['weights.safetensors']
    )
    monkeypatch.setattr(file_download.constants, 'DOWNLOAD_CHUNK_SIZE', 10 * 1024)
    monkeypatch.setattr(file_download.time, 'sleep', lambda _: None)
    url = 'https://huggingface.co/test/model/resolve/main/weights.safetensors'
    monkeypatch.setattr(
        file_download,
        'get_hf_file_metadata',
        lambda **kwargs: file_download.HfFileMetadata(
            commit_hash='a' * 40,
            etag='b' * 64,
            location=url,
            size=100 * 1024,
            xet_file_data=None,
        ),
    )
    calls = []
    counts = []
    report_progress = task._report_huggingface_progress

    def record_progress(filename, downloaded):
        report_progress(filename, downloaded)
        counts.append(task._model_downloaded_size)

    monkeypatch.setattr(task, '_report_huggingface_progress', record_progress)

    class Stream(http.SyncByteStream):
        def __iter__(self):
            yield b'x' * (50 * 1024)
            raise http.RemoteProtocolError('Connection interrupted')

    def response(**kwargs):
        calls.append(dict(kwargs['headers']))
        if len(calls) == 1:
            return http.Response(
                200,
                headers={'content-length': str(100 * 1024)},
                stream=Stream(),
                request=http.Request('GET', url),
            )
        return http.Response(
            206 if resumed else 200,
            content=b'x' * ((50 if resumed else 100) * 1024),
            request=http.Request('GET', url),
        )

    monkeypatch.setattr(
        file_download,
        'http_stream_backoff',
        lambda **kwargs: nullcontext(response(**kwargs)),
    )

    task._download_model_file()

    assert calls[1]['Range'] == f'bytes={50 * 1024}-'
    assert task._model_downloaded_size == 100 * 1024
    assert (tmp_path / 'weights.safetensors').read_bytes() == b'x' * (100 * 1024)
    reports = [
        call.kwargs['model_update'].download_progress
        for call in task._clientset.model_files.update.call_args_list
    ]
    assert reports[-2:] == [99.0, 100]
    assert max(counts) == 100 * 1024
    if not resumed:
        assert counts[counts.index(50 * 1024) + 1] == 0


def test_huggingface_discarded_bytes_report_without_waiting_for_throttle(task_factory):
    task = task_factory()
    task._log_update_interval = 100
    task._report_huggingface_progress('a/weights.safetensors', 50)
    task._flush_huggingface_progress()
    task._report_huggingface_progress('b/weights.safetensors', 10)
    task._flush_huggingface_progress()
    task._report_huggingface_progress('a/weights.safetensors', 0)
    task._flush_huggingface_progress()

    assert task._model_downloaded_size == 10
    assert [
        call.kwargs['model_update'].download_progress
        for call in task._clientset.model_files.update.call_args_list
    ] == [50.0, 10.0]


@pytest.mark.parametrize('blocked_at', ['get', 'update'])
def test_huggingface_slow_api_does_not_block_shards_or_overwrite_rollback(
    task_factory, monkeypatch, blocked_at
):
    task = task_factory(track_progress=True)
    task._log_update_interval = 100
    entered = threading.Event()
    release = threading.Event()
    rolled_back = threading.Event()
    saved = []
    request_threads = []

    def block_request():
        entered.set()
        assert release.wait(5)

    def get(**kwargs):
        if blocked_at == 'get' and not saved:
            block_request()
        return task._model_file

    def update(id, model_update):
        if blocked_at == 'update' and not saved:
            block_request()
        saved.append(model_update.download_progress)
        request_threads.append(threading.get_ident())
        if model_update.download_progress == 40:
            rolled_back.set()

    def initialize_bar():
        with downloaders.HfDownloadProgress(
            total=100, unit='B', name='huggingface_hub.xet_get'
        ):
            pass

    task._clientset.model_files.get.side_effect = get
    task._clientset.model_files.update.side_effect = update
    stop = threading.Event()
    monitor = threading.Thread(target=task._poll_huggingface_progress, args=(stop,))
    monitor.start()
    with ThreadPoolExecutor(max_workers=3) as pool:
        try:
            first = pool.submit(task._report_huggingface_progress, 'a.bin', 60)
            assert entered.wait(3)
            first.result(timeout=0.5)
            callbacks = [
                pool.submit(task._report_huggingface_progress, 'a.bin', 0),
                pool.submit(task._report_huggingface_progress, 'b.bin', 40),
                pool.submit(initialize_bar),
            ]
            for callback in callbacks:
                callback.result(timeout=0.5)
            assert task._model_downloaded_size == 40
            assert not saved
            release.set()
            assert rolled_back.wait(3)
        finally:
            release.set()
            stop.set()
            task._huggingface_progress_event.set()
            monitor.join(3)
    assert not monitor.is_alive()
    assert saved == [60.0, 40.0]
    assert set(request_threads) == {monitor.ident}


@pytest.mark.parametrize('reporting_failure', [False, True])
def test_huggingface_throttled_progress_flushes_without_new_callback(
    task_factory, monkeypatch, reporting_failure
):
    task = task_factory()
    now = [1000.0]
    monkeypatch.setattr(model_file_manager.time, 'time', lambda: now[0])
    task._report_huggingface_progress('weights.safetensors', 50)
    task._flush_huggingface_progress()
    now[0] += 0.5
    task._report_huggingface_progress('weights.safetensors', 60)
    task._flush_huggingface_progress()
    assert task._last_reported_progress == 50
    if reporting_failure:
        now[0] = 1003.0
        task._clientset.model_files.update.side_effect = [
            RuntimeError('Progress API unavailable'),
            None,
        ]
        task._flush_huggingface_progress()
        assert task._last_reported_progress == 50
    task._huggingface_progress_event.clear()

    stop = threading.Event()

    def wait(_):
        if now[0] < 1006.0:
            now[0] = 1006.0
            return False
        stop.set()
        return False

    monkeypatch.setattr(task._huggingface_progress_event, 'wait', wait)
    task._poll_huggingface_progress(stop)

    assert [
        call.kwargs['model_update'].download_progress
        for call in task._clientset.model_files.update.call_args_list
    ] == ([50.0, 60.0, 60.0] if reporting_failure else [50.0, 60.0])
    assert task._huggingface_file_progress == {'weights.safetensors': 60}


@pytest.mark.parametrize('outcome', ['complete', 'error', 'cancel'])
def test_huggingface_progress_monitor_stops_with_download(
    task_factory, monkeypatch, outcome
):
    task = task_factory()
    started = threading.Event()
    stopped = threading.Event()

    def monitor(stop):
        started.set()
        if stop.wait(5):
            stopped.set()

    def download(*args, **kwargs):
        assert started.wait(5)
        if outcome == 'error':
            raise RuntimeError('Download failed')
        if outcome == 'cancel':
            raise InterruptedError('Download cancelled')
        return ['/model']

    monkeypatch.setattr(task, '_poll_huggingface_progress', monitor, raising=False)
    monkeypatch.setattr(downloaders, 'download_model', download)
    monkeypatch.setattr(task, '_resolve_lora_state', lambda _: {})
    if outcome == 'complete':
        task._download_model_file()
    else:
        error = RuntimeError if outcome == 'error' else InterruptedError
        with pytest.raises(error):
            task._download_model_file()

    assert stopped.is_set()
    reports = [
        call.kwargs['model_update'].download_progress
        for call in task._clientset.model_files.update.call_args_list
    ]
    assert (100 in reports) == (outcome == 'complete')


def test_huggingface_reporting_failure_does_not_fail_download(task_factory):
    task = task_factory()
    task._clientset.model_files.update.side_effect = [
        RuntimeError('Server unavailable'),
        None,
    ]

    task._report_huggingface_progress('weights.safetensors', 50)
    task._flush_huggingface_progress()
    task._report_huggingface_progress('weights.safetensors', 60)
    task._flush_huggingface_progress()

    assert task._model_downloaded_size == 60
    assert task._last_reported_progress == 60


@pytest.mark.parametrize('missing_at', ['get', 'update'])
def test_huggingface_missing_model_cancels_without_repeated_reports(
    task_factory, caplog, missing_at
):
    task = task_factory()
    getattr(task._clientset.model_files, missing_at).side_effect = NotFoundException(
        'Model file 1 not found'
    )
    task._report_huggingface_progress('weights.safetensors', 50)
    task._flush_huggingface_progress()
    assert task._cancel_flag.is_set()
    reads = task._clientset.model_files.get.call_count
    writes = task._clientset.model_files.update.call_count
    task._report_huggingface_progress('weights.safetensors', 60)
    task._flush_huggingface_progress()
    assert task._clientset.model_files.get.call_count == reads
    assert task._clientset.model_files.update.call_count == writes
    assert 'Failed to report' not in caplog.text


@pytest.mark.parametrize('cancel_before_start', [True, False])
def test_huggingface_cancelled_files_do_not_start_queued_downloads(
    task_factory, monkeypatch, tmp_path, cancel_before_start
):
    task = task_factory()
    files = ['a.bin', 'b.bin', 'c.bin']
    monkeypatch.setattr(downloaders, 'match_hugging_face_files', lambda *a: files)

    def download(**kwargs):
        path = tmp_path / kwargs['filename']
        path.write_bytes(b'x')
        task._cancel_flag.set()
        return str(path)

    native = MagicMock(side_effect=download)
    monkeypatch.setattr(downloaders, 'hf_hub_download', native)
    if cancel_before_start:
        task._cancel_flag.set()
    with pytest.raises(asyncio.CancelledError):
        downloaders.HfDownloader.download_file(
            'test/model',
            '*',
            local_dir=tmp_path,
            max_workers=1,
            cancel_check=task._check_cancelled,
        )
    assert native.call_count == (0 if cancel_before_start else 1)


@pytest.mark.parametrize('transport', ['xet_get', 'http_get'])
def test_huggingface_callback_cancellation_uses_native_abort_for_xet(
    task_factory, transport
):
    task = task_factory(track_progress=True)
    with downloaders.HfDownloadProgress(
        total=100, unit='B', name=f'huggingface_hub.{transport}'
    ) as bar:
        task._cancel_flag.set()
        if transport == 'xet_get':
            bar.update(25)
        else:
            with pytest.raises(asyncio.CancelledError):
                bar.update(25)
    task._clientset.model_files.update.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize('blocked_at', [None, 'get', 'update'])
async def test_deleting_stalled_huggingface_download_aborts_before_cleanup(
    task_factory, monkeypatch, caplog, tmp_path, blocked_at
):
    task = task_factory()
    task._model_file.cleanup_on_delete = True
    model_dir = tmp_path / 'model'
    model_dir.mkdir()
    (model_dir / 'weights.safetensors.incomplete').write_bytes(b'x')
    task._model_file.resolved_paths = [str(model_dir)]
    monkeypatch.setattr(task, 'prerun', lambda: None)
    entered = threading.Event()
    aborted = threading.Event()
    rpc_entered = threading.Event()
    release_rpc = threading.Event()
    rpc_finished = threading.Event()
    native_finished = threading.Event()
    abort_while_rpc_pending = []
    monitors = []
    original_thread = threading.Thread

    def monitor_thread(*args, **kwargs):
        thread = original_thread(*args, **kwargs)
        if getattr(kwargs.get('target'), '__self__', None) is task:
            monitors.append(thread)
        return thread

    monkeypatch.setattr(model_file_manager.threading, 'Thread', monitor_thread)
    from huggingface_hub.utils import _xet

    def abort_native():
        abort_while_rpc_pending.append(not rpc_finished.is_set())
        aborted.set()

    abort = MagicMock(side_effect=abort_native)
    monkeypatch.setattr(_xet, 'abort_xet_session', abort)

    def blocked_request(**kwargs):
        rpc_entered.set()
        try:
            client = task._progress_clientset.http_client.get_httpx_client()
            release_rpc.wait(client.timeout.read)
            raise httpx.ReadTimeout('Progress API did not respond')
        finally:
            rpc_finished.set()

    if blocked_at:
        getattr(task._clientset.model_files, blocked_at).side_effect = blocked_request

    def stalled_download(*args, **kwargs):
        entered.set()
        try:
            if blocked_at:
                kwargs['huggingface_progress']('weights.safetensors', 50)
            assert aborted.wait(8), 'Native download was not aborted'
            raise RuntimeError('Native operation cancelled')
        finally:
            native_finished.set()

    monkeypatch.setattr(downloaders, 'download_model', stalled_download)
    manager = model_file_manager.ModelFileManager(1, MagicMock(), task._config)
    delete_model_file = manager._delete_model_file
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(task.run)

        async def cleanup(model_file):
            assert future.done()
            assert aborted.is_set()
            assert native_finished.is_set()
            assert len(monitors) == 2
            assert all(not monitor.is_alive() for monitor in monitors)
            if blocked_at:
                assert rpc_finished.is_set()
                assert not release_rpc.is_set()
                assert abort_while_rpc_pending == [True]
            await delete_model_file(model_file)
            assert not model_dir.exists()

        manager._delete_model_file = AsyncMock(side_effect=cleanup)
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            if blocked_at:
                assert await asyncio.to_thread(rpc_entered.wait, 3)
            manager._active_downloads[1] = (future, task._cancel_flag)
            await asyncio.wait_for(manager._handle_deletion(task._model_file), 4)
        finally:
            task._cancel_flag.set()
            release_rpc.set()
            aborted.set()
        assert future.exception() is None
        assert pool.submit(lambda: 'available').result(timeout=1) == 'available'

    abort.assert_called_once()
    # Closed eagerly when the download stops and again by the task teardown;
    # HTTPClient.close is idempotent, so both calls are fine.
    task._progress_clientset.http_client.close.assert_called()
    manager._delete_model_file.assert_awaited_once_with(task._model_file)
    assert 1 not in manager._active_downloads
    assert task._clientset.model_files.update.call_count == int(blocked_at == 'update')
    assert 'Failed to report' not in caplog.text


def test_huggingface_cancel_discards_session_for_next_download(
    task_factory, monkeypatch
):
    import hf_xet
    from huggingface_hub.utils import _xet

    sessions = [MagicMock(), MagicMock()]
    factory = MagicMock(side_effect=sessions)
    monkeypatch.setattr(hf_xet, 'XetSession', factory)
    monkeypatch.setattr(_xet, '_GLOBAL_XET_HOLDER', _xet.XetSessionHolder())
    assert _xet.get_xet_session() is sessions[0]

    task = task_factory()
    task._cancel_flag.set()
    stop = MagicMock()
    stop.wait.return_value = False
    task._watch_huggingface_cancellation(stop)

    sessions[0].sigint_abort.assert_called_once()
    assert _xet.get_xet_session() is sessions[1]
    assert factory.call_count == 2
    sessions[1].sigint_abort.assert_not_called()


@pytest.mark.parametrize(
    "reports, expected",
    [([96, 91, 99], [96, 99]), ([0, 0, 50, 50, 49, 99], [0, 50, 99])],
)
def test_progress_reports_never_decrease(task_factory, reports, expected):
    task = task_factory()
    for progress in reports:
        task._update_model_file_progress(1, progress)

    assert [
        call.kwargs["model_update"].download_progress
        for call in task._clientset.model_files.update.call_args_list
    ] == expected


def test_progress_requests_are_serialized(task_factory):
    task = task_factory()
    first_started = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()
    second_request = threading.Event()
    saved = []

    def update(id, model_update):
        progress = model_update.download_progress
        if progress == 91:
            first_started.set()
            assert release_first.wait(5)
        else:
            second_request.set()
        saved.append(progress)

    def report_second():
        second_started.set()
        task._update_model_file_progress(1, 96)

    task._clientset.model_files.update.side_effect = update
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(task._update_model_file_progress, 1, 91)
        try:
            assert first_started.wait(5)
            second = pool.submit(report_second)
            assert second_started.wait(5)
            assert not second_request.wait(0.1)
        finally:
            release_first.set()
        first.result(timeout=5)
        second.result(timeout=5)

    assert saved == [91, 96]


def test_failed_progress_report_can_be_retried(task_factory):
    task = task_factory()
    task._clientset.model_files.update.side_effect = [RuntimeError("unavailable"), None]
    with pytest.raises(RuntimeError, match="unavailable"):
        task._update_model_file_progress(1, 96)
    task._update_model_file_progress(1, 96)

    assert task._clientset.model_files.update.call_count == 2


def test_new_download_task_can_restart_progress(task_factory):
    task = task_factory()
    task._update_model_file_progress(1, 96)

    retry = task_factory()
    retry._update_model_file_progress(1, 0)
    update = retry._clientset.model_files.update.call_args.kwargs["model_update"]
    assert update.download_progress == 0


def test_download_completion_reports_ready_at_100(task_factory, monkeypatch):
    task = task_factory()
    monkeypatch.setattr(
        model_file_manager.downloaders, "download_model", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(task, "_resolve_lora_state", lambda _: {})
    task._update_model_file_progress(1, 99)
    task._download_model_file()

    update = task._clientset.model_files.update.call_args.kwargs["model_update"]
    assert update.state == ModelFileStateEnum.READY
    assert update.download_progress == 100


def _modelscope_task(tmp_path, config):
    model_file = ModelFile(
        id=1,
        worker_id=1,
        source=SourceEnum.MODEL_SCOPE,
        model_scope_model_id='test/model',
        state=ModelFileStateEnum.DOWNLOADING,
        size=100,
        local_dir=str(tmp_path),
    )
    task = model_file_manager.ModelFileDownloadTask(
        model_file, config, threading.Event()
    )
    task._clientset = MagicMock()
    task._clientset.model_files.get.return_value = model_file
    task._speed_lock = threading.Lock()
    task._progress_lock = threading.Lock()
    task._last_reported_progress = -1.0
    task._model_file_size = model_file.size
    task._instance_download_log_file = None
    return task


def _reported_progress(task):
    return [
        call.kwargs['model_update'].download_progress
        for call in task._clientset.model_files.update.call_args_list
    ]


@pytest.mark.parametrize(
    'sdk_version, uses_legacy_temp', [('1.37.0', True), ('1.38.0', False)]
)
def test_modelscope_temp_layout_follows_sdk_version(
    tmp_path, config, monkeypatch, sdk_version, uses_legacy_temp
):
    monkeypatch.setattr(model_file_manager.modelscope, '__version__', sdk_version)

    task = _modelscope_task(tmp_path, config)

    assert task._modelscope_uses_legacy_temp is uses_legacy_temp


def test_modelscope_progress_uses_disk_bytes_across_retry_and_truncation(
    tmp_path, config
):
    task = _modelscope_task(tmp_path, config)
    unit = 1024 * 1024
    path = tmp_path / 'weights.safetensors'
    incomplete = path.with_suffix('.safetensors.incomplete')
    task._modelscope_manifest = {path: 100 * unit}
    task._model_file_size = 100 * unit

    with incomplete.open('wb') as stream:
        stream.truncate(50 * unit)
    task._report_modelscope_progress()
    task._report_modelscope_progress()

    with incomplete.open('r+b') as stream:
        stream.truncate(51 * unit)
    task._report_modelscope_progress()

    with incomplete.open('r+b') as stream:
        stream.truncate(20 * unit)
    task._report_modelscope_progress()

    assert _reported_progress(task) == [50.0, 51.0, 20.0]


def test_modelscope_progress_counts_cached_and_same_named_files_once(tmp_path, config):
    task = _modelscope_task(tmp_path, config)
    first = tmp_path / 'a' / 'weights.safetensors'
    second = tmp_path / 'b' / 'weights.safetensors'
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b'a' * 40)
    second.with_suffix('.safetensors.incomplete').write_bytes(b'b' * 30)
    task._modelscope_manifest = {first: 40, second: 60}
    task._model_file_size = 100

    task._report_modelscope_progress()

    assert _reported_progress(task) == [70.0]


def test_modelscope_progress_counts_parallel_parts_during_assembly(tmp_path, config):
    task = _modelscope_task(tmp_path, config)
    target = tmp_path / 'weights.safetensors'
    task._modelscope_manifest = {target: 100}
    task._model_file_size = 100
    (tmp_path / 'weights.safetensors_0_39').write_bytes(b'a' * 40)
    (tmp_path / 'weights.safetensors_40_99').write_bytes(b'b' * 20)

    task._report_modelscope_progress()
    target.with_suffix('.safetensors.parallel_tmp').write_bytes(b'a' * 40)
    (tmp_path / 'weights.safetensors_0_39').unlink()
    task._report_modelscope_progress()

    assert _reported_progress(task) == [60.0]


@pytest.mark.parametrize(
    'file_size, expected_state',
    [
        (None, ModelFileStateEnum.ERROR),
        (50, ModelFileStateEnum.ERROR),
        (100, ModelFileStateEnum.READY),
    ],
)
def test_modelscope_completion_checks_every_file(
    tmp_path, config, monkeypatch, file_size, expected_state
):
    task = _modelscope_task(tmp_path, config)
    target = tmp_path / 'weights.safetensors'
    task._modelscope_manifest = {target: 100}
    task._modelscope_selected_files = ['weights.safetensors']
    if file_size is not None:
        target.write_bytes(b'x' * file_size)
    monkeypatch.setattr(
        downloaders, 'download_model', lambda *args, **kwargs: [str(tmp_path)]
    )
    monkeypatch.setattr(task, 'prerun', lambda: None)
    monkeypatch.setattr(task, '_resolve_lora_state', lambda paths: {})

    task.run()

    updates = task._clientset.model_files.update.call_args_list
    assert updates[-1].kwargs['model_update'].state == expected_state
    if expected_state == ModelFileStateEnum.READY:
        assert updates[-1].kwargs['model_update'].download_progress == 100
    else:
        assert all(
            call.kwargs['model_update'].download_progress != 100 for call in updates
        )
    assert (task._download_completed is True) == (
        expected_state == ModelFileStateEnum.READY
    )


def test_modelscope_manifest_refreshes_size_and_selects_download_files(
    tmp_path, config, monkeypatch
):
    task = _modelscope_task(tmp_path, config)
    task._model_file.model_scope_file_path = 'sub/*.safetensors'
    task._model_file.size = 999
    files = [
        FileEntry('sub/weights.safetensors', 100),
        FileEntry('other/weights.safetensors', 200),
        FileEntry('sub/config.json', 20),
    ]
    monkeypatch.setattr(
        downloaders, 'get_model_file_info', lambda *args, **kwargs: files
    )

    task._ensure_model_file_size_and_paths()

    assert task._model_file.size == 100
    assert task._modelscope_selected_files == ['sub/weights.safetensors']
    assert task._modelscope_manifest == {tmp_path / 'sub/weights.safetensors': 100}


def test_modelscope_manifest_skips_directories_but_keeps_empty_files(
    tmp_path, config, monkeypatch
):
    from modelscope.hub.api import HubApi
    from modelscope_hub._legacy_api import LegacyClient

    task = _modelscope_task(tmp_path, config)
    repo_files = [
        {'Path': 'transformer', 'Type': 'tree', 'Size': 0},
        {'Path': 'transformer/weights.safetensors', 'Type': 'blob', 'Size': 100},
        {'Path': 'empty.txt', 'Type': 'blob', 'Size': 0},
    ]
    monkeypatch.setattr(
        LegacyClient,
        '_request',
        lambda *args, **kwargs: SimpleNamespace(
            json=lambda: {'Data': {'Files': repo_files}}
        ),
    )

    assert HubApi().get_model_files('test/model', recursive=True)[0] == {
        'Path': 'transformer',
        'Size': 0,
    }
    task._ensure_model_file_size_and_paths()

    assert task._model_file.size == 100
    assert task._modelscope_selected_files == [
        'empty.txt',
        'transformer/weights.safetensors',
    ]
    (tmp_path / 'transformer').mkdir()
    (tmp_path / 'transformer/weights.safetensors').write_bytes(b'w' * 100)
    (tmp_path / 'empty.txt').write_bytes(b'')
    monkeypatch.setattr(
        downloaders, 'download_model', lambda *args, **kwargs: [str(tmp_path)]
    )
    monkeypatch.setattr(task, 'prerun', lambda: None)
    monkeypatch.setattr(task, '_resolve_lora_state', lambda paths: {})

    task.run()

    update = task._clientset.model_files.update.call_args.kwargs['model_update']
    assert update.state == ModelFileStateEnum.READY
    assert update.download_progress == 100


def test_modelscope_manifest_counts_valid_cache_from_sdk_metadata(
    tmp_path, config, monkeypatch
):
    from modelscope_hub._legacy_api import LegacyClient

    task = _modelscope_task(tmp_path, config)
    cached = tmp_path / 'cached.safetensors'
    cached.write_bytes(b'c' * 90)
    repo_files = [
        {
            'Path': 'cached.safetensors',
            'Type': 'blob',
            'Size': 90,
            'BlobId': 'not-a-sha256-digest',
            'Sha256': hashlib.sha256(b'c' * 90).hexdigest(),
        },
        {'Path': 'pending.safetensors', 'Type': 'blob', 'Size': 10},
    ]
    monkeypatch.setattr(
        LegacyClient,
        '_request',
        lambda *args, **kwargs: SimpleNamespace(
            json=lambda: {'Data': {'Files': repo_files}}
        ),
    )

    task._ensure_model_file_size_and_paths()
    task._report_modelscope_progress()

    assert task._model_file.size == 100
    assert [
        progress for progress in _reported_progress(task) if progress is not None
    ] == [90.0]


def test_modelscope_legacy_sdk_keeps_file_metadata(tmp_path, config, monkeypatch):
    task = _modelscope_task(tmp_path, config)
    monkeypatch.setattr(downloaders.modelscope, '__version__', '1.37.0')
    api = MagicMock()
    api.get_model_files.return_value = [
        {'Path': 'weights.safetensors', 'Type': 'blob', 'Size': 100, 'Sha256': 'abc'}
    ]
    monkeypatch.setattr(downloaders, 'HubApi', lambda: api)

    files = downloaders.ModelScopeDownloader.get_model_file_info(task._model_file)

    assert [
        (file.rfilename, file.size, file.file_type, file.sha256) for file in files
    ] == [('weights.safetensors', 100, 'blob', 'abc')]


def test_modelscope_manifest_requires_main_file_when_mmproj_exists(
    tmp_path, config, monkeypatch
):
    task = _modelscope_task(tmp_path, config)
    task._model_file.model_scope_file_path = 'missing.gguf'
    monkeypatch.setattr(
        downloaders,
        'get_model_file_info',
        lambda *args, **kwargs: [FileEntry('mmproj-F32.gguf', 10)],
    )

    with pytest.raises(ValueError, match='No ModelScope files match'):
        task._ensure_model_file_size_and_paths()


def test_modelscope_unknown_size_has_no_numeric_progress(tmp_path, config):
    task = _modelscope_task(tmp_path, config)
    target = tmp_path / 'weights.safetensors'
    target.write_bytes(b'data')
    task._modelscope_manifest = {target: None}
    task._model_file_size = None

    task._report_modelscope_progress()
    task._verify_modelscope_download()

    assert _reported_progress(task) == []


def test_modelscope_tqdm_retry_does_not_add_initial_bytes(tmp_path, config):
    task = _modelscope_task(tmp_path, config)
    task._model_downloaded_size = 50
    bar = SimpleNamespace()

    def initialize(instance, **kwargs):
        instance.unit = kwargs['unit']
        instance.n = kwargs['initial']
        instance.total = kwargs['total']
        instance.desc = kwargs['desc']

    task._handle_tqdm_init(
        bar,
        initialize,
        unit='B',
        initial=50,
        total=100,
        desc='weights.safetensors',
    )
    task._handle_tqdm_update(
        bar, lambda instance, n: setattr(instance, 'n', instance.n + n), 1
    )

    assert task._model_downloaded_size == 50
    assert bar.n == 51


def test_modelscope_legacy_sdk_ignores_current_temp(tmp_path, config):
    task = _modelscope_task(tmp_path, config)
    task._modelscope_uses_legacy_temp = True
    target = tmp_path / 'weights.safetensors'
    legacy = tmp_path / model_file_manager.TEMPORARY_FOLDER_NAME
    legacy.mkdir()
    (legacy / target.name).write_bytes(b'a' * 40)
    target.with_suffix('.safetensors.incomplete').write_bytes(b'b' * 50)
    task._modelscope_manifest = {target: 100}
    task._model_file_size = 100

    task._report_modelscope_progress()

    assert _reported_progress(task) == [40.0]


def test_modelscope_progress_counts_legacy_subdirectory_and_parts(tmp_path, config):
    task = _modelscope_task(tmp_path, config)
    task._modelscope_uses_legacy_temp = True
    target = tmp_path / 'sub' / 'weights.safetensors'
    legacy = tmp_path / model_file_manager.TEMPORARY_FOLDER_NAME / 'sub'
    legacy.mkdir(parents=True)
    task._modelscope_manifest = {target: 100}
    task._model_file_size = 100

    (legacy / 'weights.safetensors').write_bytes(b'a' * 40)
    task._report_modelscope_progress()
    (legacy / 'weights.safetensors').unlink()
    (legacy / 'weights.safetensors_0_39').write_bytes(b'a' * 40)
    (legacy / 'weights.safetensors_40_99').write_bytes(b'b' * 20)
    task._report_modelscope_progress()
    (legacy / 'weights.safetensors').write_bytes(b'a' * 40)
    task._report_modelscope_progress()

    assert _reported_progress(task) == [40.0, 60.0]


@pytest.mark.parametrize(
    'relative_path', ['weights.safetensors', 'sub/weights.safetensors']
)
def test_modelscope_current_sdk_ignores_leftover_legacy_temp(
    tmp_path, config, relative_path
):
    task = _modelscope_task(tmp_path, config)
    task._modelscope_uses_legacy_temp = False
    target = tmp_path / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    legacy = tmp_path / model_file_manager.TEMPORARY_FOLDER_NAME / relative_path
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_bytes(b'o' * 90)
    target.with_suffix('.safetensors.incomplete').write_bytes(b'n' * 20)
    task._modelscope_manifest = {target: 100}
    task._model_file_size = 100

    task._report_modelscope_progress()

    assert _reported_progress(task) == [20.0]


def test_modelscope_progress_ignores_stale_final_during_redownload(
    tmp_path, config, monkeypatch
):
    task = _modelscope_task(tmp_path, config)
    target = tmp_path / 'weights.safetensors'
    target.write_bytes(b'o' * 100)
    digest = hashlib.sha256(b'n' * 100).hexdigest()
    monkeypatch.setattr(
        downloaders,
        'get_model_file_info',
        lambda *args, **kwargs: [FileEntry('weights.safetensors', 100, sha256=digest)],
    )

    task._ensure_model_file_size_and_paths()
    task._report_modelscope_progress()
    (tmp_path / 'weights.safetensors.incomplete').write_bytes(b'n' * 20)
    task._report_modelscope_progress()
    (tmp_path / 'weights.safetensors.incomplete').unlink()
    target.write_bytes(b'n' * 100)
    task._report_modelscope_progress()

    assert [
        progress for progress in _reported_progress(task) if progress is not None
    ] == [
        0.0,
        20.0,
        99.0,
    ]


def test_modelscope_progress_counts_verified_cached_file(tmp_path, config, monkeypatch):
    task = _modelscope_task(tmp_path, config)
    target = tmp_path / 'weights.safetensors'
    target.write_bytes(b'n' * 100)
    digest = hashlib.sha256(b'n' * 100).hexdigest()
    monkeypatch.setattr(
        downloaders,
        'get_model_file_info',
        lambda *args, **kwargs: [FileEntry('weights.safetensors', 100, sha256=digest)],
    )

    task._ensure_model_file_size_and_paths()
    task._report_modelscope_progress()

    assert [
        progress for progress in _reported_progress(task) if progress is not None
    ] == [99.0]


def test_modelscope_unknown_metadata_clears_stale_progress(
    tmp_path, config, monkeypatch
):
    task = _modelscope_task(tmp_path, config)
    task._model_file.download_progress = 99
    monkeypatch.setattr(
        downloaders,
        'get_model_file_info',
        lambda *args, **kwargs: [FileEntry('weights.safetensors', None)],
    )

    task._ensure_model_file_size_and_paths()

    assert task._model_file.size is None
    assert task._model_file.download_progress is None
    update = task._clientset.model_files.update.call_args.kwargs['model_update']
    assert update.download_progress is None


def test_modelscope_download_uses_the_selected_manifest(tmp_path, monkeypatch):
    snapshot = MagicMock(return_value=str(tmp_path))
    monkeypatch.setattr(downloaders, 'modelscope_snapshot_download', snapshot)
    monkeypatch.setattr(
        downloaders, 'HeartbeatSoftFileLock', lambda *args, **kwargs: nullcontext()
    )

    paths = downloaders.ModelScopeDownloader.download(
        model_id='test/model',
        file_path=None,
        extra_file_path=None,
        local_dir=str(tmp_path),
        cache_dir=str(tmp_path),
        matching_files=['a.safetensors', 'b/config.json'],
    )

    assert paths == [str(tmp_path)]
    assert snapshot.call_args.kwargs['allow_patterns'] == [
        'a.safetensors',
        'b/config.json',
    ]


def test_modelscope_download_reports_disk_progress_through_sdk_boundary(
    tmp_path, config, monkeypatch
):
    task = _modelscope_task(tmp_path, config)
    task._model_file.model_scope_file_path = 'weights.safetensors'
    task._log_update_interval = 0.01
    unit = 1024 * 1024
    target = tmp_path / 'weights.safetensors'
    incomplete = target.with_suffix('.safetensors.incomplete')
    seen_50 = threading.Event()
    seen_51 = threading.Event()

    def record_update(id, model_update):
        if model_update.download_progress == 50:
            seen_50.set()
        elif model_update.download_progress == 51:
            seen_51.set()

    def snapshot_download(model_id, local_dir, allow_patterns):
        assert model_id == 'test/model'
        assert allow_patterns == ['weights.safetensors']
        with incomplete.open('wb') as stream:
            stream.truncate(50 * unit)
        assert seen_50.wait(5)
        with incomplete.open('r+b') as stream:
            stream.truncate(51 * unit)
        assert seen_51.wait(5)
        with incomplete.open('r+b') as stream:
            stream.truncate(100 * unit)
        incomplete.replace(target)
        return local_dir

    task._clientset.model_files.update.side_effect = record_update
    monkeypatch.setattr(
        downloaders,
        'get_model_file_info',
        lambda *args, **kwargs: [FileEntry('weights.safetensors', 100 * unit)],
    )
    monkeypatch.setattr(downloaders, 'modelscope_snapshot_download', snapshot_download)
    monkeypatch.setattr(
        downloaders, 'HeartbeatSoftFileLock', lambda *args, **kwargs: nullcontext()
    )

    def prerun():
        task._ensure_model_file_size_and_paths()
        task._model_file_size = task._model_file.size

    monkeypatch.setattr(task, 'prerun', prerun)
    monkeypatch.setattr(task, '_resolve_lora_state', lambda paths: {})

    task.run()

    reports = [
        progress for progress in _reported_progress(task) if progress is not None
    ]
    assert reports.index(50) < reports.index(51) < reports.index(99)
    assert reports[-1] == 100
    assert (
        task._clientset.model_files.update.call_args.kwargs['model_update'].state
        == ModelFileStateEnum.READY
    )


@pytest.mark.parametrize(
    'complete, expected_state',
    [
        (False, ModelFileStateEnum.ERROR),
        (True, ModelFileStateEnum.READY),
    ],
)
def test_modelscope_progress_report_failure_does_not_skip_file_verification(
    tmp_path, config, monkeypatch, complete, expected_state
):
    task = _modelscope_task(tmp_path, config)
    target = tmp_path / 'weights.safetensors'
    task._modelscope_manifest = {target: 100}
    task._modelscope_selected_files = ['weights.safetensors']
    report_attempted = threading.Event()

    def fail_report():
        report_attempted.set()
        raise RuntimeError('progress unavailable')

    def download(*args, **kwargs):
        assert report_attempted.wait(5)
        if complete:
            target.write_bytes(b'x' * 100)
        return [str(tmp_path)]

    monkeypatch.setattr(task, '_report_modelscope_progress', fail_report)
    monkeypatch.setattr(downloaders, 'download_model', download)
    monkeypatch.setattr(task, 'prerun', lambda: None)
    monkeypatch.setattr(task, '_resolve_lora_state', lambda paths: {})

    task.run()

    update = task._clientset.model_files.update.call_args.kwargs['model_update']
    assert update.state == expected_state
    assert (update.download_progress == 100) == complete
