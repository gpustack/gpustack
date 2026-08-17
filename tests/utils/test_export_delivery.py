"""Response-level tests for how a sized export becomes a file.

These cover what the route tests cannot see: which resources the archive
holds, and for how long. A split export feeds every part from ONE row stream,
so releasing that stream is the writer's job — and it is a leak nothing in a
response body reveals.
"""

import io
import zipfile

import pytest

from gpustack.utils.export_delivery import ExportSheetPlan, split_export_response


def _rows_factory(total: int, state: dict):
    """A row source that records when it is closed."""

    def factory():
        async def rows():
            state["open"] = True
            try:
                for index in range(total):
                    yield [index]
            finally:
                state["closed"] = True

        return rows()

    return factory


def _plan(total: int, state: dict) -> ExportSheetPlan:
    return ExportSheetPlan(
        key="user",
        name="Users",
        columns=["N"],
        total=total,
        rows=_rows_factory(total, state),
    )


async def _collect(response) -> bytes:
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk if isinstance(chunk, bytes) else chunk.encode("utf-8"))
    return b"".join(chunks)


@pytest.mark.asyncio
async def test_split_releases_the_shared_stream_when_the_last_part_fills_it():
    """The exact-multiple case is the one that strands a cursor.

    Each part stops as soon as it has its quota, so when the row count divides
    evenly the last part never reaches the end of the stream: the generator
    stays suspended at a ``yield``, holding its cursor (and on the resource
    side a second session) until the event loop finalizes it.
    """
    state = {}
    response = split_export_response(
        [_plan(20, state)],
        request=_Request(),
        prefix="usage",
        context="scope=all organization=*",
        limit=10,
    )
    archive = zipfile.ZipFile(io.BytesIO(await _collect(response)))

    assert len(archive.namelist()) == 2
    assert archive.testzip() is None
    assert state["closed"]


@pytest.mark.asyncio
async def test_split_releases_the_shared_stream_when_the_client_disconnects():
    """Half a download is the common case, not an exotic one."""
    state = {}
    response = split_export_response(
        [_plan(50, state)],
        request=_Request(),
        prefix="usage",
        context="scope=all organization=*",
        limit=10,
    )
    body = response.body_iterator
    while not state.get("open"):
        await body.__anext__()
    await body.aclose()

    assert state["closed"]


@pytest.mark.asyncio
async def test_split_parts_are_consecutive_slices_of_one_pass():
    """Parts cut the row stream, so no row can land in two files or none."""
    state = {}
    response = split_export_response(
        [_plan(25, state)],
        request=_Request(),
        prefix="usage",
        context="scope=all organization=*",
        limit=10,
    )
    archive = zipfile.ZipFile(io.BytesIO(await _collect(response)))

    values = []
    for name in archive.namelist():
        text = archive.read(name).decode("utf-8-sig")
        # Header first, ``#`` trailer last.
        values.extend(
            line.split(",")[0]
            for line in text.splitlines()[1:]
            if not line.startswith("#")
        )
    assert values == [str(index) for index in range(25)]
    assert archive.namelist()[-1].endswith("part-3-of-3.csv")


class _Request:
    """Only the date range is read out of the request, for the file name."""

    start_date = "2026-04-01"
    end_date = "2026-04-03"
