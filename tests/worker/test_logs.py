import asyncio
from pathlib import Path
import sys
import threading
from typing import List, Union
from unittest.mock import patch
import pytest

from gpustack.worker import logs as worker_logs
from gpustack.worker.logs import (
    LineWindow,
    LogOptions,
    checkpoint_at_or_before,
    index_log_file,
    line_window_generator,
    log_generator,
    plan_line_window,
    read_line_slice,
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


@pytest.fixture(scope="module")
def million_line_log(tmp_path_factory):
    """More lines than the index keeps checkpoints for, by three orders of
    magnitude: the arithmetic between checkpoints is what a page read rests
    on, and a file small enough to fit in one never exercises it."""
    log_file = tmp_path_factory.mktemp("range") / "million.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.writelines(f"line-{i:07d}\n" for i in range(1_001_000))
    return log_file


def _window_text(paths, offset, limit):
    window = plan_line_window(paths, offset, limit)
    body = b"".join(chunk for read in window.reads for chunk in read_line_slice(*read))
    return body.decode("utf-8"), window


def test_a_line_window_lands_exactly_on_the_lines_it_names(million_line_log: Path):
    """The whole point of the index: a page deep into the log is the page the
    offset names, not the one a rounding error lands on."""
    text, window = _window_text([million_line_log], 1_000_000, 1000)

    lines = text.splitlines()
    assert lines[0] == "line-1000000"
    assert lines[-1] == "line-1000999"
    assert (len(lines), window.line_count, window.total_lines) == (
        1000,
        1000,
        1_001_000,
    )


def test_a_deep_window_starts_reading_near_the_line_it_wants(million_line_log: Path):
    """Cost has to be flat in the offset. Starting every read at the head of
    the file gives the right lines and gets slower the further in they are --
    which is the shape the index exists to remove."""
    index = index_log_file(million_line_log)

    _offset, at_line = checkpoint_at_or_before(index, 1_000_000)

    assert 0 < 1_000_000 - at_line < 10_000


def test_a_second_window_read_does_not_rescan_the_file(million_line_log: Path):
    """A page read that rescans the file is a page read that gets slower the
    longer the log is -- which is the shape this replaces."""
    index_log_file(million_line_log)

    with patch.object(
        worker_logs, "_extend_index", side_effect=AssertionError("rescanned")
    ):
        text, _ = _window_text([million_line_log], 500_000, 3)

    assert text == "line-0500000\nline-0500001\nline-0500002\n"


def test_a_growing_log_reports_more_lines_without_moving_the_earlier_ones(
    tmp_path: Path,
):
    """Offsets count from the start of the file, so appending may only ever add
    to the total. A page the viewer already holds must not shift under it."""
    log_file = tmp_path / "growing.log"
    log_file.write_text("a\nb\nc\n", encoding="utf-8")
    before, first = _window_text([log_file], 1, 2)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("d\ne\n")
    after, second = _window_text([log_file], 1, 2)

    assert (before, after) == ("b\nc\n", "b\nc\n")
    assert (first.total_lines, second.total_lines) == (3, 5)


def test_a_line_without_a_newline_still_counts_as_one(tmp_path: Path):
    """A stream cut mid-line leaves the archive ending without a newline. Not
    counting it would hide the last line the viewer can reach."""
    log_file = tmp_path / "partial.log"
    log_file.write_text("a\nb\nhalf", encoding="utf-8")

    text, window = _window_text([log_file], 2, 5)

    assert (text, window.total_lines, window.line_count) == ("half", 3, 1)


def test_an_offset_past_the_end_is_empty_and_still_reports_the_total(tmp_path: Path):
    """Paging past the last page is an ordinary request, not an error: the
    viewer needs the total back to work out where the end actually is."""
    log_file = tmp_path / "short.log"
    log_file.write_text("a\nb\n", encoding="utf-8")

    text, window = _window_text([log_file], 500, 10)

    assert (text, window.line_count, window.total_lines) == ("", 0, 2)


def test_a_whole_log_range_is_handed_over_in_bounded_pieces(million_line_log: Path):
    """A measured download asks for every line in one range. Answering it with
    one string puts a copy of the whole log in the worker's memory."""
    window = plan_line_window([million_line_log], 0, 1_001_000)
    chunks = list(read_line_slice(*window.reads[0]))

    assert len(chunks) > 1
    # A piece is cut at the first line end past _SLICE_CHUNK_BYTES, so it holds
    # at most a chunk plus one line; this bound needs every line shorter than a
    # chunk, as the fixture's are. A longer line leaves as a piece of its size.
    assert max(len(chunk) for chunk in chunks) < 2 * worker_logs._SLICE_CHUNK_BYTES
    assert sum(len(chunk) for chunk in chunks) == million_line_log.stat().st_size


def test_a_range_hands_over_the_bytes_the_file_holds(tmp_path: Path):
    """A download promises a Content-Length measured on the file. Decoding the
    range would spend three bytes on every byte no encoding claims, leaving the
    body longer than the promise and the tail cut off to fit it."""
    log_file = tmp_path / "main.log"
    log_file.write_bytes(b"clean\n\xff\xfe raw bytes\n\xe6 truncated char\n")

    window = plan_line_window([log_file], 0, 3)
    body = b"".join(read_line_slice(*window.reads[0]))

    assert body == log_file.read_bytes()


@pytest.mark.parametrize(
    "line_format, lines",
    [(b"b%04d padded wider\n", 200), (b"b%04d\n", 120)],
    ids=["longer", "shorter"],
)
def test_a_log_rewritten_at_the_same_path_is_indexed_afresh(
    tmp_path: Path, line_format: bytes, lines: int
):
    """A restart truncates main.log and writes a new log behind the same name.
    Extending the old counts over it reports lines that never existed, and a
    download would promise the old log's length."""
    log_file = tmp_path / "main.log"
    log_file.write_bytes(b"".join(b"a%04d padded\n" % n for n in range(200)))
    assert index_log_file(log_file).line_count == 200

    with open(log_file, "r+b") as f:
        f.truncate(0)
        f.write(b"".join(line_format % n for n in range(lines)))
    text, window = _window_text([log_file], 100, 1)

    assert window.total_lines == lines
    assert text == (line_format % 100).decode()
    whole = plan_line_window([log_file], 0, sys.maxsize)
    assert whole.byte_count == log_file.stat().st_size


def test_a_short_log_growing_under_its_first_count_keeps_its_lines(
    tmp_path: Path, monkeypatch
):
    """A head or a fresh shard holds a line or two when a reader first counts
    it, and the writer does not wait for the count to finish."""
    log_file = tmp_path / "main.log.2"
    log_file.write_bytes(b"a\n")
    real_open = open

    class GrowsAtItsEnd:
        def __init__(self, file):
            self._file = file

        def read(self, size=-1):
            data = self._file.read(size)
            if not data:
                with real_open(log_file, "ab") as writer:
                    writer.write(b"b\n")
                monkeypatch.undo()
            return data

        def __getattr__(self, name):
            return getattr(self._file, name)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self._file.close()

    monkeypatch.setattr(
        worker_logs,
        "open",
        lambda *args: GrowsAtItsEnd(real_open(*args)),
        raising=False,
    )
    assert index_log_file(log_file).line_count == 1

    assert index_log_file(log_file).line_count == 2


def test_a_window_spans_the_files_of_one_stream_in_order(tmp_path: Path):
    """A capped log is a head, a marker and its shards. A line number has to
    address the concatenation, or the page numbers stop at the head."""
    head = tmp_path / "main.log"
    head.write_text("h1\nh2\n", encoding="utf-8")
    marker = tmp_path / "main.log.truncated"
    marker.write_text("... omitted ...\n", encoding="utf-8")
    shard = tmp_path / "main.log.1"
    shard.write_text("s1\ns2\ns3\n", encoding="utf-8")

    text, window = _window_text([head, marker, shard], 1, 4)

    assert text == "h2\n... omitted ...\ns1\ns2\n"
    assert window.total_lines == 6


@pytest.mark.asyncio
async def test_a_window_read_cancelled_mid_chunk_stays_cancelled(
    monkeypatch, tmp_path: Path
):
    """A viewer paging away aborts the range request while a chunk is still
    being read. The read is abandoned, not stopped, so the close that follows
    lands on a reader still inside it -- and the file it reads is left to it,
    as closing that would wait out the read on the event loop."""
    reading = threading.Event()
    release = threading.Event()
    handed = []

    def blocking_slice(path, first, count, end=None, file=None, fail_on_error=False):
        handed.append(file)
        with file:
            reading.set()
            release.wait(5)
            yield b"line\n"

    monkeypatch.setattr(worker_logs, "read_line_slice", blocking_slice)
    main = tmp_path / "main.log"
    main.write_text("line\n")
    window = LineWindow(offset=0, line_count=1, total_lines=1, reads=((main, 0, 1),))

    async def consume():
        async for _chunk in line_window_generator(window):
            pass

    task = asyncio.create_task(consume())
    try:
        await asyncio.to_thread(reading.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not handed[0].closed
    finally:
        release.set()


@pytest.mark.asyncio
async def test_a_window_reads_the_stream_as_it_was_planned(tmp_path: Path):
    """A download states its length before its body, so the body has to be the
    stream as it was measured: a line completed after planning, in a file the
    reads have not finished, belongs to the next read and not to this one, and
    a file the cap removes while the body is on its way is read all the same."""
    main, container = tmp_path / "main.log", tmp_path / "container.log"
    main.write_bytes(b"a\nhal")
    container.write_bytes(b"c\n")

    window = plan_line_window([main, container], 0, sys.maxsize)
    with open(main, "ab") as f:
        f.write(b"f done\nmore\n")
    chunks = line_window_generator(window)
    body = await anext(chunks)
    container.unlink()
    body += b"".join([chunk async for chunk in chunks])

    assert body == b"a\nhalc\n"
    assert window.byte_count == len(body)
    assert plan_line_window([main, container], 1, 1).byte_count is None


def test_a_slow_scan_does_not_hold_up_another_log(tmp_path: Path):
    """An uncapped log takes seconds to index the first time. Every other
    log's pages and listings must not queue behind it."""
    slow, quick = tmp_path / "slow.log", tmp_path / "quick.log"
    slow.write_text("a\n", encoding="utf-8")
    quick.write_text("b\n", encoding="utf-8")
    scanning, release, indexed = threading.Event(), threading.Event(), threading.Event()
    extend = worker_logs._extend_index

    def slow_extend(path, index):
        if path == slow:
            scanning.set()
            release.wait(5)
        extend(path, index)

    with patch.object(worker_logs, "_extend_index", side_effect=slow_extend):
        scanner = threading.Thread(target=index_log_file, args=(slow,))
        other = threading.Thread(target=lambda: (index_log_file(quick), indexed.set()))
        scanner.start()
        try:
            assert scanning.wait(5)
            other.start()
            assert indexed.wait(2)
        finally:
            release.set()
            scanner.join(5)
            if other.ident is not None:
                other.join(5)


def test_component_log_file_has_a_distinct_source_namespace(tmp_path):
    from gpustack.worker.log_sources import parse_serve_log_path

    directory = tmp_path / 'model.42' / '3'
    component = parse_serve_log_path(directory / 'component.sglang_smg.log')
    sidecar = parse_serve_log_path(directory / 'container.sglang_smg.log')
    assert component is not None
    assert component.kind != sidecar.kind


def _rank_source_archive(tmp_path, multi_port_external=False):
    from tests.utils.model import new_model_instance
    from gpustack.schemas.models import DistributedServers, ModelInstanceWorkload
    from gpustack.worker.log_sources import (
        write_log_sources,
        restart_log_dir,
        list_log_sources,
    )

    instance = new_model_instance(42, 'model', model_id=1, worker_id=7)
    instance.restart_count = 3
    instance.distributed_servers = DistributedServers(
        workloads=[
            ModelInstanceWorkload(
                name='rank-workload',
                worker_id=7,
                state='running',
                dp_rank_placements=[
                    dict(dp_rank=0, node_rank=0, gpu_indexes=[0], api_port=8000),
                    dict(dp_rank=1, node_rank=1, gpu_indexes=[1]),
                ],
            )
        ]
    )
    write_log_sources(tmp_path, instance, 7, multi_port_external)
    directory = restart_log_dir(tmp_path, instance.name, 42, 3)
    (directory / 'container.log').write_bytes(
        b'[dp0-node0] first\ncommon\n[dp1-node1] headless\n[dp0-node0] last\n'
    )
    return instance, directory, list_log_sources(tmp_path, 42, 7, 3)


def test_archived_rank_views_use_trusted_generation_metadata(tmp_path):
    from gpustack.worker.log_sources import list_log_sources, resolve_log_source

    instance, directory, sources = _rank_source_archive(tmp_path)
    ranks = [s for s in sources if s.descriptor.kind == 'rank']
    assert [(s.descriptor.dp_rank, s.descriptor.main) for s in ranks] == [
        (0, True),
        (1, False),
    ]
    assert ranks[0].accepts(b'[dp0-node0] first\n')
    assert not ranks[0].accepts(b'continuation\n')
    assert next(s for s in sources if s.descriptor.kind == 'public').accepts(
        b'[dp99-node0] unknown\n'
    )
    assert (
        resolve_log_source(tmp_path, 42, 7, 3, ranks[0].descriptor.id).descriptor
        == ranks[0].descriptor
    )
    with pytest.raises(ValueError):
        resolve_log_source(tmp_path, 42, 8, 3, ranks[0].descriptor.id)
    assert not any(
        s.descriptor.kind == 'rank' for s in list_log_sources(tmp_path, 42, 7, 6)
    )
    (directory / 'sources.json').unlink()
    assert not any(
        s.descriptor.kind == 'rank' for s in list_log_sources(tmp_path, 42, 7, 3)
    )


def test_engine_source_reports_empty_archive_without_hiding_readable_source(tmp_path):
    from gpustack.worker.log_sources import list_log_sources, resolve_log_source

    _, directory, _ = _rank_source_archive(tmp_path)
    (directory / 'container.log').write_bytes(b'')
    engine = next(
        s
        for s in list_log_sources(tmp_path, 42, 7, 3)
        if s.descriptor.kind == 'workload'
    )
    assert engine.descriptor.available
    assert resolve_log_source(tmp_path, 42, 7, 3, engine.descriptor.id) == engine


def test_multi_port_external_logs_only_attribute_explicit_api_prefix(tmp_path):
    _, _, sources = _rank_source_archive(tmp_path, multi_port_external=True)
    rank = next(
        s for s in sources if s.descriptor.kind == 'rank' and s.descriptor.dp_rank == 0
    )
    common = next(s for s in sources if s.descriptor.kind == 'public')
    assert rank.accepts(b'\x1b[0;36m(APIServer_DP0 pid=123)\x1b[0m INFO hello\n')
    for line in (
        b'(EngineCore_DP0 pid=123) hello\n',
        b'raw traceback\n',
        b'(APIServer_DP99 pid=123) bad\n',
    ):
        assert not rank.accepts(line)
        assert common.accepts(line)


@pytest.mark.asyncio
async def test_source_filter_precedes_offset_tail_and_download_length(tmp_path):
    from gpustack.worker.logs import (
        plan_source_window,
        source_window_generator,
        LogOptions,
    )

    _, _, sources = _rank_source_archive(tmp_path)
    source = next(
        s for s in sources if s.descriptor.kind == 'rank' and s.descriptor.dp_rank == 0
    )
    window = plan_source_window(source, 1, 1)
    assert window.total_lines == 2
    data = b''.join(
        [
            chunk
            async for chunk in source_window_generator(
                window, LogOptions(), lambda: source
            )
        ]
    )
    assert data == b'[dp0-node0] last\n'
    tail = plan_source_window(source, tail=1)
    assert tail.offset == 1
    whole = plan_source_window(source, limit=100)
    data = b''.join(
        [
            chunk
            async for chunk in source_window_generator(
                whole, LogOptions(), lambda: source
            )
        ]
    )
    assert whole.byte_count == len(data)
    assert b'common' not in data and b'headless' not in data


@pytest.mark.asyncio
async def test_rank_follow_preserves_line_attribution_and_closes_on_cancel(
    tmp_path, monkeypatch
):
    import asyncio
    from gpustack.worker.logs import (
        plan_source_window,
        source_window_generator,
        LogOptions,
    )
    from gpustack.worker.log_sources import list_log_sources

    _, directory, sources = _rank_source_archive(tmp_path)
    head = directory / 'container.log'
    head.write_bytes(b'[dp0-node0] partial')
    source = next(
        s for s in sources if s.descriptor.kind == 'rank' and s.descriptor.dp_rank == 0
    )
    window = plan_source_window(source, limit=100)
    opened = []
    real_open = Path.open

    def track_open(path, mode='r', *args, **kwargs):
        stream = real_open(path, mode, *args, **kwargs)
        if mode == 'rb':
            opened.append(stream)
        return stream

    monkeypatch.setattr(Path, 'open', track_open)

    async def append(_):
        with head.open('ab') as stream:
            stream.write(b' continuation\ncommon\n[dp0-node0] new\n')

    monkeypatch.setattr(asyncio, 'sleep', append)
    generator = source_window_generator(
        window,
        LogOptions(follow=True),
        lambda: next(
            s
            for s in list_log_sources(tmp_path, 42, 7, 3)
            if s.descriptor.id == source.descriptor.id
        ),
    )
    assert await anext(generator) == b'[dp0-node0] partial'
    assert await anext(generator) == b' continuation\n'
    assert await anext(generator) == b'[dp0-node0] new\n'
    await generator.aclose()
    assert window.cancelled.is_set()
    assert opened and all(stream.closed for stream in opened)


def test_component_source_shares_bounded_cap_shards_and_generation_window(tmp_path):
    from gpustack.worker.log_sources import (
        CappedLogWriter,
        list_log_sources,
        parse_serve_log_path,
    )

    directory = tmp_path / 'model.42' / '3'
    directory.mkdir(parents=True)
    head = directory / 'component.sglang_smg.log'
    with CappedLogWriter(head, max_bytes=128, head_bytes=32) as writer:
        for index in range(100):
            writer.write(f'line {index}\n')
    files = [p for p in directory.iterdir() if parse_serve_log_path(p)]
    assert len(files) < 66
    assert all(parse_serve_log_path(p).kind.value == 'component' for p in files)
    sources = list_log_sources(tmp_path, 42, 7, 4)
    assert len(sources) == 1 and sources[0].descriptor.previous
    assert sources[0].descriptor.kind == 'workload'
    assert set(sources[0].paths) == set(files)
    assert list_log_sources(tmp_path, 42, 7, 5) == []


@pytest.mark.parametrize(
    'damage',
    [
        'null',
        'list',
        'string',
        'version',
        'identity',
        'placements_string',
        'placement_null',
        'placement_negative',
    ],
)
def test_damaged_source_metadata_only_offers_unattributed_physical_logs(
    tmp_path, damage
):
    import json
    from gpustack.worker.log_sources import list_log_sources

    _, directory, _ = _rank_source_archive(tmp_path)
    path = directory / 'sources.json'
    payload = json.loads(path.read_text())
    if damage in ('null', 'list', 'string'):
        payload = {'null': None, 'list': [], 'string': 'broken'}[damage]
    elif damage == 'version':
        payload['version'] = 99
    elif damage == 'identity':
        payload['instance_id'] = 999
    else:
        record = next(s for s in payload['streams'] if s['key'] == 'engine')
        record['placements'] = {
            'placements_string': 'broken',
            'placement_null': [None],
            'placement_negative': [dict(dp_rank=-1, node_rank=0)],
        }[damage]
    path.write_text(json.dumps(payload))
    sources = list_log_sources(tmp_path, 42, 7, 3)
    assert not any(s.descriptor.kind == 'rank' for s in sources)
    assert any(
        s.descriptor.kind == 'workload' and s.descriptor.available for s in sources
    )


@pytest.mark.parametrize('damage', ['json', 'shape', 'record'])
def test_invalid_source_manifest_has_a_debug_diagnostic(tmp_path, caplog, damage):
    import json
    from gpustack.worker.log_sources import read_log_sources

    _, directory, _ = _rank_source_archive(tmp_path)
    path = directory / 'sources.json'
    payload = json.loads(path.read_text())
    if damage == 'json':
        path.write_text('{')
    elif damage == 'shape':
        path.write_text('null')
    else:
        payload['streams'][0]['physical_kind'] = 'untrusted'
        path.write_text(json.dumps(payload))
    with caplog.at_level('DEBUG', logger='gpustack.worker.log_sources'):
        records = read_log_sources(path, 42, 7, 3)
    assert len(caplog.records) == 1
    assert str(path) in caplog.text
    if damage == 'record':
        assert len(records) == len(payload['streams']) - 1
    else:
        assert records == []


@pytest.mark.asyncio
async def test_follow_open_seek_and_read_run_off_the_event_loop(tmp_path, monkeypatch):
    from types import SimpleNamespace

    path = tmp_path / 'follow.log'
    path.write_bytes(b'old\nnew\n')
    main_thread = threading.get_ident()
    calls = []
    opened = []
    real_open = Path.open

    class Reader:
        def __init__(self):
            calls.append(('open', threading.get_ident()))
            self.stream = real_open(path, 'rb')
            opened.append(self.stream)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def seek(self, offset):
            calls.append(('seek', threading.get_ident()))
            return self.stream.seek(offset)

        def readline(self, size):
            calls.append(('read', threading.get_ident()))
            return self.stream.readline(size)

    monkeypatch.setattr(Path, 'open', lambda *args, **kwargs: Reader())
    source = SimpleNamespace(accepts=lambda block: True)
    data = b''.join(
        [
            block
            async for block in worker_logs._follow_source_path(path, source, 4, 8, {})
        ]
    )
    assert data == b'new\n'
    assert {name for name, _ in calls} == {'open', 'seek', 'read'}
    assert all(thread != main_thread for _, thread in calls)
    assert all(stream.closed for stream in opened)


@pytest.mark.asyncio
async def test_cancelled_follow_open_closes_file_when_background_work_finishes(
    tmp_path,
    monkeypatch,
):
    from types import SimpleNamespace

    path = tmp_path / 'follow.log'
    path.write_bytes(b'line\n')
    real_open = Path.open
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    class Reader:
        def __init__(self):
            self.stream = real_open(path, 'rb')

        def __enter__(self):
            entered.set()
            release.wait(2)
            return self

        def __exit__(self, *args):
            self.stream.close()
            closed.set()

        def seek(self, offset):
            self.stream.seek(offset)

        def readline(self, size):
            return self.stream.readline(size)

    monkeypatch.setattr(Path, 'open', lambda *args, **kwargs: Reader())
    source = SimpleNamespace(accepts=lambda block: True)
    reader = worker_logs._follow_source_path(path, source, 0, 5, {})
    task = asyncio.create_task(anext(reader))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        await reader.aclose()
    assert await asyncio.to_thread(closed.wait, 2)


def test_physical_selector_survives_source_manifest_publication(tmp_path):
    from gpustack.worker.log_sources import list_log_sources

    _, directory, sources = _rank_source_archive(tmp_path)
    known = {
        s.descriptor.kind: s.descriptor.id
        for s in sources
        if s.descriptor.kind in ('workload', 'preparation')
    }
    (directory / 'sources.json').unlink()
    fallback = list_log_sources(tmp_path, 42, 7, 3)
    for source in fallback:
        assert source.descriptor.id == known[source.descriptor.kind]


@pytest.mark.asyncio
@pytest.mark.parametrize('unavailable', [False, True])
async def test_source_follow_does_not_rescan_the_archive_for_every_poll(
    monkeypatch, unavailable
):
    from types import SimpleNamespace
    from unittest.mock import Mock

    clock = [0.0]
    stop = asyncio.Event()
    source = SimpleNamespace(paths=[], descriptor=SimpleNamespace(previous=False))
    window = worker_logs.plan_source_window(source)
    refresh = Mock(
        return_value=source,
        side_effect=ValueError('Log source is unavailable') if unavailable else None,
    )

    async def advance(_):
        clock[0] += 0.2
        if clock[0] >= 1:
            stop.set()

    monkeypatch.setattr(worker_logs, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(worker_logs.asyncio, 'sleep', advance)
    chunks = [
        chunk
        async for chunk in worker_logs.source_window_generator(
            window, LogOptions(follow=True, stop_event=stop), refresh
        )
    ]
    assert chunks == []
    assert refresh.call_count == 1


@pytest.mark.asyncio
async def test_closing_source_stream_closes_the_snapshot_reader(monkeypatch):
    from types import SimpleNamespace
    import threading

    closed = []

    def read(window):
        try:
            yield b'first line\n'
            yield b'second line\n'
        finally:
            closed.append(True)

    monkeypatch.setattr(worker_logs, '_read_source_window', read)
    window = SimpleNamespace(cancelled=threading.Event())
    stream = worker_logs.source_window_generator(window, LogOptions(), lambda: None)
    assert await anext(stream) == b'first line\n'
    await stream.aclose()
    assert window.cancelled.is_set()
    assert closed == [True]


@pytest.mark.asyncio
async def test_cancelling_snapshot_read_tolerates_an_executing_generator(monkeypatch):
    from types import SimpleNamespace

    entered, release, finished = (threading.Event() for _ in range(3))
    window = SimpleNamespace(cancelled=threading.Event())

    def read(window):
        try:
            entered.set()
            assert release.wait(timeout=3)
            if not window.cancelled.is_set():
                yield b'line\n'
        finally:
            finished.set()

    monkeypatch.setattr(worker_logs, '_read_source_window', read)
    reader = worker_logs._read_source_window_chunks(window)
    task = asyncio.create_task(anext(reader))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert window.cancelled.is_set()
    finally:
        release.set()
        assert await asyncio.to_thread(finished.wait, 2)
        await reader.aclose()


@pytest.mark.parametrize('worker_id', [1, 2, 3, 4])
def test_hybrid_logs_preserve_current_and_previous_worker_task_without_rank_views(
    tmp_path, worker_id
):
    from tests.routes.test_worker_proxy import rank_instance
    from tests.worker.test_runtime_metrics import _hybrid_tasks
    from gpustack.schemas.models import ModelInstanceWorkload
    from gpustack.worker.log_sources import (
        write_log_sources,
        restart_log_dir,
        list_log_sources,
        resolve_log_source,
    )

    instance = rank_instance.__wrapped__()
    instance.distributed_servers.workloads = [
        ModelInstanceWorkload.model_validate(t) for t in _hybrid_tasks()
    ]
    instance.distributed_servers.orchestration_mode = "managed"
    write_log_sources(tmp_path, instance, worker_id)
    directory = restart_log_dir(tmp_path, instance.name, instance.id, 3)
    (directory / 'container.log').write_bytes(b'[dp0-node1] compute\ncommon\n')
    for current in (3, 4):
        sources = list_log_sources(tmp_path, instance.id, worker_id, current)
        workload = next(s for s in sources if s.descriptor.kind == 'workload')
        assert workload.descriptor.component_kind == 'vllm_hybrid'
        assert workload.descriptor.workload_name == f'hybrid-{worker_id}'
        assert workload.descriptor.main == (worker_id in (1, 3))
        assert workload.descriptor.previous == (current == 4)
        assert not any(s.descriptor.kind == 'rank' for s in sources)
        assert (
            resolve_log_source(
                tmp_path, instance.id, worker_id, current, workload.descriptor.id
            ).paths
            == workload.paths
        )
    assert list_log_sources(tmp_path, instance.id, worker_id, 5) == []
