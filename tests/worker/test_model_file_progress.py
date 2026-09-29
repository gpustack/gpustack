from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gpustack.schemas.model_files import ModelFile, ModelFileStateEnum
from gpustack.schemas.models import SourceEnum
from gpustack.worker import model_file_manager
from gpustack.utils.hub import FileEntry
from gpustack.worker import downloaders


@pytest.fixture
def task_factory(monkeypatch, config):
    monkeypatch.setattr(model_file_manager, "setup_logging", MagicMock())
    monkeypatch.setattr(model_file_manager, "read_worker_token", lambda _: "test-token")
    monkeypatch.setattr(model_file_manager, "ClientSet", lambda **kwargs: MagicMock())
    for name in (
        "_ensure_model_file_size_and_paths",
        "_setup_instance_log_files",
        "hijack_tqdm_progress",
    ):
        monkeypatch.setattr(model_file_manager.ModelFileDownloadTask, name, MagicMock())

    def create_task():
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
        return task

    return create_task


@pytest.mark.parametrize(
    "reports, expected",
    [([96, 91, 99], [96, 99]), ([0, 0, 50, 50, 49, 99], [0, 50, 99])],
)
def test_progress_reports_never_decrease(task_factory, reports, expected):
    task = task_factory()
    for progress in reports:
        task._update_progress_func(progress)

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
        task._update_progress_func(96)

    task._clientset.model_files.update.side_effect = update
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(task._update_progress_func, 91)
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
        task._update_progress_func(96)
    task._update_progress_func(96)

    assert task._clientset.model_files.update.call_count == 2


def test_new_download_task_can_restart_progress(task_factory):
    task = task_factory()
    task._update_progress_func(96)

    retry = task_factory()
    retry._update_progress_func(0)
    update = retry._clientset.model_files.update.call_args.kwargs["model_update"]
    assert update.download_progress == 0


def test_download_completion_reports_ready_at_100(task_factory, monkeypatch):
    task = task_factory()
    monkeypatch.setattr(
        model_file_manager.downloaders, "download_model", lambda *args, **kwargs: []
    )
    monkeypatch.setattr(task, "_resolve_lora_state", lambda _: {})
    task._update_progress_func(99)
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
