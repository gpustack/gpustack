"""Writer-level tests for the export file formats.

These cover what only the writer can get wrong: what a value becomes once it
is a cell. The route tests above them check which values are chosen.
"""

import io
import time
import re
import zipfile
from datetime import date, datetime

import pytest

from gpustack.utils.tabular_export import ExportStageTimer, build_xlsx, stream_csv


async def _rows(rows):
    for row in rows:
        yield row


def _sheet_xml(payload: bytes, index: int = 1) -> str:
    return (
        zipfile.ZipFile(io.BytesIO(payload))
        .read(f"xl/worksheets/sheet{index}.xml")
        .decode()
    )


def _workbook_names(payload: bytes) -> list:
    xml = zipfile.ZipFile(io.BytesIO(payload)).read("xl/workbook.xml").decode()
    return re.findall(r'<sheet name="([^"]+)"', xml)


@pytest.mark.asyncio
async def test_a_name_that_looks_like_a_formula_is_written_as_text():
    """Entity names are not character-restricted, so they must not be code.

    ``xlsxwriter.write()`` type-guesses, and one of its guesses is that a
    string starting with ``=`` is a formula. API key and volume names reach
    these cells verbatim, so that guess turns "name your key
    ``=HYPERLINK(...)``" into code that runs when an operator opens the
    workbook.
    """
    payload = await build_xlsx(
        [
            (
                "s",
                ["Name"],
                _rows([['=HYPERLINK("http://evil.example","x")'], ["=1+1"]]),
            )
        ]
    )
    xml = _sheet_xml(payload)

    assert "<f>" not in xml  # no formula cell anywhere
    assert 'HYPERLINK("http://evil.example","x")' in xml


@pytest.mark.asyncio
async def test_cells_keep_the_type_they_came_with():
    payload = await build_xlsx(
        [
            (
                "s",
                ["Date", "When", "Flag", "Count", "Ratio", "Text"],
                _rows(
                    [
                        [
                            date(2026, 8, 2),
                            datetime(2026, 8, 2, 13, 30),
                            True,
                            42,
                            1.5,
                            "007",
                        ]
                    ]
                ),
            )
        ]
    )
    xml = _sheet_xml(payload)
    row = re.search(r'<row r="2".*?</row>', xml).group(0)

    assert 't="b"' in row  # bool stayed a bool, not the number 1
    assert "<v>42</v>" in row and "<v>1.5</v>" in row
    # A numeric-looking id keeps its leading zero instead of becoming 7.
    assert "<t>007</t>" in row


@pytest.mark.asyncio
async def test_duplicate_sheet_names_do_not_abort_the_workbook():
    """Sheet names are localized display strings; two can be the same.

    xlsxwriter rejects a duplicate (case-insensitively), which would turn a
    request the server already accepted into a 500 halfway through writing.
    The data is unaffected — only the label needs to differ.
    """
    payload = await build_xlsx(
        [
            ("用量", ["A"], _rows([["x"]])),
            ("用量", ["A"], _rows([["y"]])),
            ("用量", ["A"], _rows([["z"]])),
        ]
    )

    names = _workbook_names(payload)
    assert len(names) == 3 and len(set(names)) == 3
    assert names[0] == "用量"
    # Every sheet still carries its own rows.
    assert "<t>y</t>" in _sheet_xml(payload, 2)


@pytest.mark.asyncio
async def test_long_sheet_names_stay_within_excels_limit_after_deduping():
    long_name = "a" * 40
    payload = await build_xlsx(
        [(long_name, ["A"], _rows([["x"]])), (long_name, ["A"], _rows([["y"]]))]
    )

    names = _workbook_names(payload)
    assert all(len(name) <= 31 for name in names)
    assert len(set(names)) == 2


@pytest.mark.asyncio
async def test_xlsx_dates_carry_a_number_format():
    """A date cell without a number format opens as "46246" in Excel.

    That serial IS what the file stores — how it reads is a property of the
    cell, not of the value — so every date has to be written with a format
    attached. Regression: the whole Date and Last Active columns arrived as
    five-digit numbers once xlsx became the default format again.
    """

    data = await build_xlsx(
        [
            (
                "s",
                ["Date", "Last Active", "Total"],
                _rows([[date(2026, 8, 12), datetime(2026, 8, 12, 10, 30), 42]]),
            )
        ]
    )

    archive = zipfile.ZipFile(io.BytesIO(data))
    styles = archive.read("xl/styles.xml").decode()
    sheet = _sheet_xml(data)
    # Day precision for a date; a rollup instant keeps its time of day.
    assert 'formatCode="yyyy-mm-dd"' in styles
    assert 'formatCode="yyyy-mm-dd hh:mm:ss"' in styles
    # Style 0 is General — the state this test exists to catch. A number keeps
    # it, and must: reformatting metrics would break their summability.
    assert re.search(r'<c r="A2" s="[1-9]', sheet)
    assert re.search(r'<c r="B2" s="[1-9]', sheet)
    assert '<c r="C2"><v>42</v></c>' in sheet
    # A formatted date is useless in a column too narrow for it: Excel refuses
    # to clip a number the way it clips text and renders "########" instead.
    # So the date columns are widened to fit, which needs the widths set
    # before the first cell is written.
    widths = [float(w) for w in re.findall(r'<col [^>]*width="([\d.]+)"', sheet)]
    assert widths[0] >= 12  # yyyy-mm-dd
    assert widths[1] >= 21  # yyyy-mm-dd hh:mm:ss
    # The peeked-at row must still reach the sheet — header plus both rows.
    assert sheet.count('<row r=') == 2


def test_stage_timer_charges_unmeasured_time_to_downstream():
    """Time nobody claimed must surface, not vanish.

    This is the whole point of the timer: an export generator idles at
    ``yield`` while the CSV writer and the client's socket do the work, and a
    naive total charges that idling to the query. ``downstream`` is derived
    from the total rather than measured, so unattributed time is visible by
    construction.
    """
    timer = ExportStageTimer()
    with timer.stage("aggregate"):
        time.sleep(0.05)
    with timer.stage("build"):
        time.sleep(0.01)
    with timer.stage("build"):  # repeat calls accumulate into one number
        time.sleep(0.01)
    time.sleep(0.05)  # unmeasured: stands in for suspension at yield

    stages = timer.breakdown()

    assert stages["aggregate"] >= 0.05
    assert stages["build"] >= 0.02  # both blocks, summed into one number
    assert stages["downstream"] >= 0.04
    assert stages["total"] >= stages["aggregate"] + stages["build"]
    # Rendered for humans, so the log line is checked too — one decimal, and
    # the remainder named rather than left to be inferred from a subtraction.
    assert "downstream=" in timer.summary() and "aggregate=0.1s" in timer.summary()


def test_stage_timer_separates_waiting_for_cpu_from_using_it():
    """Sleeping burns wall time; spinning burns CPU. The log must tell them
    apart, because "slow" means opposite things in the two cases: a stage with
    cpu ≈ wall is executing too much work, while cpu ≪ wall means the process
    never got the CPU and no amount of code tuning would move it.
    """
    timer = ExportStageTimer()
    with timer.stage("waiting"):
        time.sleep(0.2)
    assert timer.breakdown()["cpu"] < 0.1

    timer = ExportStageTimer()
    with timer.stage("working"):
        deadline = time.monotonic() + 0.2
        while time.monotonic() < deadline:
            pass
    working = timer.breakdown()
    assert working["cpu"] >= 0.5 * working["working"]


@pytest.mark.asyncio
async def test_csv_writes_a_bom_a_header_and_a_trailer():
    chunks = [
        chunk
        async for chunk in stream_csv(["A", "B"], _rows([[1, None]]), trailer="rows=1")
    ]
    text = b"".join(chunks).decode("utf-8-sig")

    assert text.splitlines() == ["A,B", "1,", "# rows=1"]
