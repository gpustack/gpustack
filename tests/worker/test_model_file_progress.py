from concurrent.futures import ThreadPoolExecutor
import threading
from unittest.mock import MagicMock

import pytest

from gpustack.schemas.model_files import ModelFile, ModelFileStateEnum
from gpustack.schemas.models import SourceEnum
from gpustack.worker import model_file_manager


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
