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

    def blocking_slice(path, first, count, file=None):
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
