import asyncio
from types import SimpleNamespace
from typing import List, Union
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gpustack.worker.logs import LogOptions, log_generator

# Importing gpustack.worker.logs above initialises gpustack.worker, which pulls
# in gpustack.routes.worker.logs; importing the router before it would hit a
# circular import.
from gpustack.routes.worker.logs import (  # noqa: E402
    resolve_restart_count,
    restart_entries_from_main_log_files,
    router as worker_logs_router,
)


@pytest.fixture
def sample_log_file(tmp_path):
    log_content = "line1\nline2\nline3\nline4\nline5\n"
    log_file = tmp_path / "test.log"
    log_file.write_text(log_content)
    return log_file


@pytest.fixture
def large_log_file(tmp_path):
    # Create a log file with 2KB in two lines
    log_content = "line" * 256 + "\n" + "line" * 256 + "\n"
    log_file = tmp_path / "large_test.log"
    log_file.write_text(log_content)
    return log_file


def normalize_newlines(data: Union[str, List[str]]) -> Union[str, List[str]]:
    if isinstance(data, str):
        return data.replace("\r\n", "\n")
    elif isinstance(data, list):
        return [line.replace("\r\n", "\n") for line in data]


@pytest.mark.asyncio
async def test_log_generator_default(sample_log_file):
    options = LogOptions()
    log_path = str(sample_log_file)

    result = normalize_newlines(
        [line async for line in log_generator(log_path, options)]
    )
    assert result == [
        "line1\n",
        "line2\n",
        "line3\n",
        "line4\n",
        "line5\n",
    ]


@pytest.mark.asyncio
async def test_log_generator_tail(sample_log_file):
    options = LogOptions(tail=2)
    log_path = str(sample_log_file)

    result = normalize_newlines(
        [line async for line in log_generator(log_path, options)]
    )
    assert result == ["line4\n", "line5\n"]


@pytest.mark.asyncio
async def test_log_generator_follow(sample_log_file):
    options = LogOptions(follow=True)
    log_path = str(sample_log_file)

    generator = log_generator(log_path, options)
    result = []
    async for line in generator:
        result.append(line)
        if len(result) == 5:
            break
    assert normalize_newlines(result) == [
        "line1\n",
        "line2\n",
        "line3\n",
        "line4\n",
        "line5\n",
    ]

    # Append a new line to the log file
    with open(log_path, "a") as file:
        file.write("line6\n")
    try:
        line6 = await asyncio.wait_for(generator.__anext__(), timeout=1)
        assert normalize_newlines(line6) == "line6\n"
    except StopAsyncIteration:
        pytest.fail("Expected a new line in the log file")


@pytest.mark.asyncio
async def test_log_generator_empty_file(tmp_path):
    empty_file = tmp_path / "empty.log"
    empty_file.touch()
    options = LogOptions(tail=0)

    result = [line async for line in log_generator(empty_file, options)]
    assert result == []


@pytest.mark.asyncio
async def test_log_generator_tail_larger_than_file(sample_log_file):
    options = LogOptions(tail=10)
    log_path = str(sample_log_file)

    result = normalize_newlines(
        [line async for line in log_generator(log_path, options)]
    )
    assert result == ["line1\n", "line2\n", "line3\n", "line4\n", "line5\n"]


@pytest.mark.asyncio
async def test_log_generator_tail_large_file(large_log_file):
    options = LogOptions(tail=1)
    log_path = str(large_log_file)

    result = normalize_newlines(
        [line async for line in log_generator(log_path, options)]
    )
    assert result == ["line" * 256 + "\n"]


@pytest.mark.asyncio
async def test_log_generator_tail_larger_than_large_file(large_log_file):
    options = LogOptions(tail=3)
    log_path = str(large_log_file)

    result = normalize_newlines(
        [line async for line in log_generator(log_path, options)]
    )
    assert result == ["line" * 256 + "\n", "line" * 256 + "\n"]


def test_log_options_url_encode_carries_restart_count():
    """The server proxies to the worker through this encoding, so an addressed
    restart must survive the hop; unset stays absent rather than 'None'."""
    assert "restart_count" not in LogOptions(tail=5).url_encode()
    assert "restart_count=0" in LogOptions(tail=5, restart_count=0).url_encode()
    assert "restart_count=198" in LogOptions(restart_count=198).url_encode()


@pytest.mark.asyncio
async def test_resolve_restart_count_addresses_a_specific_restart(tmp_path):
    """With the first-failure log pinned, on-disk counts are not contiguous.
    'previous' can only name the second newest, so restart 0 is reachable only
    by asking for it directly."""
    mid = 2945
    for rc in (0, 198, 199):
        (tmp_path / f"{mid}.{rc}.log").write_text("x", encoding="utf-8")

    async def resolve(previous=False, restart_count=None):
        return await resolve_restart_count(tmp_path, mid, previous, restart_count)

    assert await resolve() == 199
    assert await resolve(previous=True) == 198
    assert await resolve(restart_count=0) == 0
    # An explicit restart_count wins over previous.
    assert await resolve(previous=True, restart_count=0) == 0
    # A count that is no longer on disk falls back instead of streaming nothing.
    assert await resolve(restart_count=42) == 199
    assert await resolve(previous=True, restart_count=42) == 198


@pytest.mark.asyncio
async def test_resolve_restart_count_without_files(tmp_path):
    """No log files on disk yet resolves to None, even when one is addressed."""
    assert await resolve_restart_count(tmp_path, 1, False) is None
    assert await resolve_restart_count(tmp_path, 1, False, 0) is None


def test_restart_entries_expose_their_restart_count(tmp_path):
    """The log-options list must name each retained session so a client can
    address the pinned first-failure log."""
    mid = 42
    files = [tmp_path / f"{mid}.{rc}.log" for rc in (0, 198, 199)]
    for f in files:
        f.write_text("x", encoding="utf-8")

    entries = restart_entries_from_main_log_files(files)

    assert [(e.restart_count, e.previous) for e in entries] == [
        (199, False),
        (198, True),
        (0, True),
    ]


def _serve_log_client(tmp_path, model_instance_id: int, contents: dict) -> TestClient:
    """A worker app over a serve log dir holding {restart_count: text}."""
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir(parents=True, exist_ok=True)
    for rc, text in contents.items():
        (serve_dir / f"{model_instance_id}.{rc}.log").write_text(text, encoding="utf-8")

    app = FastAPI()
    app.include_router(worker_logs_router)
    app.state.config = SimpleNamespace(log_dir=str(tmp_path))
    return TestClient(app)


def test_serve_log_options_lists_two_by_default(tmp_path):
    """Retention keeps more than two sessions, but 'previous' addresses only the
    two newest. Listing the extras unconditionally would offer an existing
    client a session it fetches as the wrong log, so they stay opt-in."""
    mid = 116
    client = _serve_log_client(
        tmp_path, mid, {0: "THE STALL\n", 198: "restart 198\n", 199: "restart 199\n"}
    )

    listed = client.get(f"/serveLogOptions/{mid}").json()["restarts"]
    assert [(e["restart_count"], e["previous"]) for e in listed] == [
        (199, False),
        (198, True),
    ]

    everything = client.get(
        f"/serveLogOptions/{mid}", params={"all_restarts": True}
    ).json()["restarts"]
    assert [(e["restart_count"], e["previous"]) for e in everything] == [
        (199, False),
        (198, True),
        (0, True),
    ]


def test_every_listed_restart_is_fetchable_by_restart_count(tmp_path):
    """Whatever all_restarts lists must be readable, and read back as itself."""
    mid = 116
    contents = {0: "THE STALL\n", 198: "restart 198\n", 199: "restart 199\n"}
    client = _serve_log_client(tmp_path, mid, contents)

    listed = client.get(
        f"/serveLogOptions/{mid}", params={"all_restarts": True}
    ).json()["restarts"]

    for entry in listed:
        rc = entry["restart_count"]
        resp = client.get(f"/serveLogs/{mid}", params={"restart_count": rc})
        assert resp.status_code == 200
        assert resp.text.strip() == contents[rc].strip()


def test_serve_logs_query_precedence(tmp_path):
    """restart_count wins over previous; an unset one keeps today's behaviour;
    a count no longer on disk falls back instead of returning an empty view."""
    mid = 116
    client = _serve_log_client(
        tmp_path, mid, {0: "THE STALL\n", 198: "restart 198\n", 199: "restart 199\n"}
    )

    def body(**params):
        return client.get(f"/serveLogs/{mid}", params=params).text.strip()

    assert body() == "restart 199"
    assert body(previous=True) == "restart 198"
    assert body(restart_count=0) == "THE STALL"
    assert body(restart_count=0, previous=True) == "THE STALL"
    assert body(restart_count=9999) == "restart 199"
    assert body(restart_count=9999, previous=True) == "restart 198"


@pytest.mark.parametrize(
    "on_disk",
    [
        [0],
        [0, 1],
        [0, 1, 2],
        [0, 198, 199],
        [0, 7, 8, 9, 10, 11],
    ],
)
def test_default_listing_is_always_addressable_by_previous(tmp_path, on_disk):
    """The invariant the two-entry cap exists to hold: every session listed by
    default resolves, from its own 'previous' flag alone, to itself. Without the
    cap a third entry is listed whose flag resolves to a different log, which is
    a silently wrong log view rather than a missing one."""
    mid = 116
    contents = {rc: f"restart {rc}\n" for rc in on_disk}
    client = _serve_log_client(tmp_path, mid, contents)

    listed = client.get(f"/serveLogOptions/{mid}").json()["restarts"]
    assert listed, "at least the current session must be listed"

    for entry in listed:
        by_flag = client.get(
            f"/serveLogs/{mid}", params={"previous": entry["previous"]}
        )
        assert by_flag.text.strip() == contents[entry["restart_count"]].strip(), (
            f"entry rc={entry['restart_count']} previous={entry['previous']} "
            f"served the wrong log"
        )
