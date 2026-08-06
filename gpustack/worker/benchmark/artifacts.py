"""Result-file naming protocol shared with benchmark-runner.

benchmark-runner writes its output as files under the benchmark directory, and
the file NAME is the only thing that says what a given file is: a measured ramp
point, a per-stage run, the saturation probe, or the ramp's diagnostic sidecar.
That makes the naming an interface between two repositories, so it lives in one
module here instead of being re-spelled at each call site — the collection pass,
the ready-file count, the sidecar read and the index parsing used to carry four
independent copies of the same string surgery.

The counterpart is benchmark-runner's output layout (``auto_tune.py`` writes
``{base}__p{index}.dual_json`` and ``{base}__satprobe.dual_json``, ``main.py``
writes ``{base}__stage{i}`` and ``RAMP_OUTCOME_SUFFIX``). ``{base}`` is the
benchmark id. A mismatch here does not raise: the collection pass skips files it
cannot find, so the symptom is a curve that silently loses points. Keep the two
sides together.

Every ``*.dual_json`` output produces a pair — a trimmed ``.json`` summary and a
``.full.json`` companion holding the untrimmed record. Only the summary is read;
:func:`is_point_file` exists mainly to keep the companion (and the probe, whose
name also starts with the id) out of the point set.
"""

import os
import re
from typing import List

# ``{id}__p{index}.json`` — one measured ramp point. The index is the probe order,
# which is also the only record of how the curve was walked (the ramp doubles,
# then bisects), so it is parsed back out for ordering.
_POINT_INDEX_RE = re.compile(r"__p(\d+)\.json$")

_FULL_SUFFIX = ".full.json"


def single_report_path(benchmark_dir: str, benchmark_id: int) -> str:
    """The one report a non-ramp, non-stage run writes at the end."""
    return f"{benchmark_dir}/{benchmark_id}.json"


def point_prefix(benchmark_id: int) -> str:
    return f"{benchmark_id}__p"


def is_point_file(name: str, benchmark_id: int) -> bool:
    """True for ``{id}__p{index}.json``, excluding the ``.full.json`` companion.

    The ``__p`` in the prefix is what keeps ``{id}__satprobe.json`` out, and the
    id is part of the prefix so a sibling benchmark's files are never counted.
    """
    return (
        name.startswith(point_prefix(benchmark_id))
        and name.endswith(".json")
        and not name.endswith(_FULL_SUFFIX)
    )


def point_file_index(name: str) -> int:
    """The probe index encoded in a point file name; 0 when absent."""
    m = _POINT_INDEX_RE.search(name)
    return int(m.group(1)) if m else 0


def list_point_files(benchmark_dir: str, benchmark_id: int) -> List[str]:
    """Point file names for this benchmark, in probe order.

    An unreadable directory yields an empty list: "no points ready yet" is a
    normal state (the ramp has not finished its first point), not an error.
    """
    try:
        names = [n for n in os.listdir(benchmark_dir) if is_point_file(n, benchmark_id)]
    except OSError:
        return []
    return sorted(names, key=point_file_index)


def stage_report_path(benchmark_dir: str, benchmark_id: int, stage_index: int) -> str:
    """The report for one manual stage (one single-rate run)."""
    return f"{benchmark_dir}/{benchmark_id}__stage{stage_index}.json"


def saturation_probe_path(benchmark_dir: str, benchmark_id: int) -> str:
    """The throughput probe that soft-caps a rate-axis ramp.

    Not a measured point: its throughput profile yields no rate, so it is
    reported as a trailing row and excluded from peak / recommendation / validity.
    """
    return f"{benchmark_dir}/{benchmark_id}__satprobe.json"


def ramp_facts_path(benchmark_dir: str, benchmark_id: int) -> str:
    """The ramp's diagnostic sidecar: WHY the search stopped.

    Written when the ramp returns, so its absence is also a signal — during a
    partial sync the search has not ended yet.
    """
    return f"{benchmark_dir}/{benchmark_id}__ramp.json"
