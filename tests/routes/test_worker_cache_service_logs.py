import asyncio
from types import SimpleNamespace

import pytest

from gpustack.api.exceptions import BadRequestException, NotFoundException
from gpustack.routes.worker.logs import (
    get_cache_service_instance_logs,
    get_serve_logs,
)
from gpustack.worker.log_sources import CappedLogWriter, main_log_path
from gpustack.worker.logs import LogOptions


@pytest.fixture(params=["cache_services", "serve"])
def log_kind(request):
    return request.param


def startup(log_dir, log_kind, restart=0):
    path = main_log_path(log_dir / log_kind, "test-instance", 11, restart)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


async def response_for(log_dir, log_kind, options):
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(config=SimpleNamespace(log_dir=str(log_dir)))
        )
    )
    if log_kind == "cache_services":
        return await get_cache_service_instance_logs(
            request, 11, options, cache_service_id=5
        )
    return await get_serve_logs(
        request,
        11,
        options,
        model_instance_name="test-instance",
        model_file_id=None,
        container_name=None,
    )


async def text_of(stream):
    return "".join(
        [part.decode() if isinstance(part, bytes) else part async for part in stream]
    )


@pytest.mark.asyncio
async def test_previous_and_pagination_use_the_shared_archive_layout(
    tmp_path, log_kind
):
    first = startup(tmp_path, log_kind)
    first.write_text("old startup\n")
    first.with_name("container.log").write_text("old container\n")
    startup(tmp_path, log_kind, 1).write_text("new startup\n")
    response = await response_for(
        tmp_path, log_kind, LogOptions(previous=True, offset=1, limit=1)
    )
    assert await text_of(response.body_iterator) == "old container\n"
    assert response.headers["X-Log-Offset"] == "1"
    assert response.headers["X-Log-Total-Lines"] == "2"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options", [LogOptions(offset=0, tail=1), LogOptions(offset=0, follow=True)]
)
async def test_log_range_rejects_tail_and_follow(tmp_path, log_kind, options):
    with pytest.raises(BadRequestException):
        await response_for(tmp_path, log_kind, options)


@pytest.mark.asyncio
@pytest.mark.parametrize("previous", [False, True])
async def test_log_route_selects_launch_archives(tmp_path, log_kind, previous):
    for restart, label in enumerate(["old", "new"]):
        path = startup(tmp_path, log_kind, restart)
        path.write_text(f"{label} startup\n")
        path.with_name("container.log").write_text(f"{label} container\n")
    response = await response_for(tmp_path, log_kind, LogOptions(previous=previous))
    label = "old" if previous else "new"
    assert (
        await text_of(response.body_iterator) == f"{label} startup\n{label} container\n"
    )


@pytest.mark.asyncio
async def test_follow_reads_image_pull_progress_then_container_output(
    tmp_path, log_kind
):
    path = startup(tmp_path, log_kind)
    path.write_text("Pulling image: 10%\n")
    response = await response_for(tmp_path, log_kind, LogOptions(follow=True))
    stream = response.body_iterator
    try:
        assert await asyncio.wait_for(anext(stream), 3) == "Pulling image: 10%\n"
        with path.open("a") as output:
            output.write("Pull complete\n")
        assert await asyncio.wait_for(anext(stream), 3) == "Pull complete\n"
        archive = path.with_name("container.log")
        archive.write_text("Ready\n")
        assert await asyncio.wait_for(anext(stream), 3) == "Ready\n"
        with archive.open("a") as output:
            output.write("Serving\n")
        assert await asyncio.wait_for(anext(stream), 3) == "Serving\n"
    finally:
        await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("previous", [False, True])
async def test_follow_uses_the_selected_restart(tmp_path, log_kind, previous):
    paths = []
    for restart in range(2):
        path = startup(tmp_path, log_kind, restart)
        path.write_text(f"startup-{restart}\n")
        path.with_name("container.log").write_text(f"container-{restart}\n")
        paths.append(path)
    response = await response_for(
        tmp_path, log_kind, LogOptions(previous=previous, follow=True)
    )
    stream = response.body_iterator
    selected = 0 if previous else 1
    try:
        assert await asyncio.wait_for(anext(stream), 3) == f"startup-{selected}\n"
        assert await asyncio.wait_for(anext(stream), 3) == f"container-{selected}\n"
        with paths[selected].with_name("container.log").open("a") as output:
            output.write("more output\n")
        assert await asyncio.wait_for(anext(stream), 3) == "more output\n"
    finally:
        await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("tail", [-1, 2])
async def test_tail_and_shards_follow_model_log_semantics(tmp_path, log_kind, tail):
    path = startup(tmp_path, log_kind)
    lines = [f"line-{number:02d}\n" for number in range(8)]
    with CappedLogWriter(path, max_bytes=64, head_bytes=16) as output:
        output.writelines(lines)
    container_lines = ["Ready\n", "Serving\n", "Done\n"]
    path.with_name("container.log").write_text("".join(container_lines))
    response = await response_for(tmp_path, log_kind, LogOptions(tail=tail))
    expected = lines + (container_lines[-tail:] if tail > 0 else container_lines)
    assert await text_of(response.body_iterator) == "".join(expected)


@pytest.mark.asyncio
async def test_capped_archive_exposes_marker_and_latest_output(tmp_path, log_kind):
    path = startup(tmp_path, log_kind)
    with CappedLogWriter(path, max_bytes=64, head_bytes=16) as output:
        output.writelines(f"line-{number:02d}\n" for number in range(40))
    response = await response_for(tmp_path, log_kind, LogOptions())
    text = await text_of(response.body_iterator)
    assert "line-00\n" in text
    assert "line-39\n" in text
    assert "omitted" in text
    assert "line-10\n" not in text


@pytest.mark.asyncio
async def test_missing_log_archive_reports_not_found(tmp_path, log_kind):
    response = await response_for(tmp_path, log_kind, LogOptions())
    with pytest.raises(NotFoundException):
        await text_of(response.body_iterator)
