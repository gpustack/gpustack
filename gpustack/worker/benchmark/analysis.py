"""Turn a measured load curve into a conclusion.

Pure derivation over the per-point result grid, split out of ``BenchmarkManager``:
that class owns processes, containers and API round-trips, while everything here
is a function of (benchmark config, measured points, ramp facts) with no IO
besides reading the ramp sidecar. The split follows how the code was already
used — the tests call these as plain functions — and keeps the product judgement
(what to recommend, what to warn about) in one readable place.

Two kinds of input, deliberately kept apart:

* **facts** — benchmark-runner's ``{id}__ramp.json`` sidecar says WHY the search
  stopped. Several terminations leave identical grids (``capacity_plateau`` and a
  threshold breaking at the top both end with "the highest knob met the SLO";
  ``budget_seconds`` and a self-directed stop both end with "fewer points than the
  range allows"), so this cannot be re-derived downstream in general.
* **the grid** — what the remaining verdicts are about: is the answer
  trustworthy, and what should the user change. These cannot move into the runner
  because they depend on ``recommended_rate``, which is this side's product
  decision, and they must also work for runs that have no ramp at all (manual
  stages, legacy single-rate rows, and mid-run partial syncs).
"""

import json
import logging
import math
from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gpustack.schemas.benchmark import (
    SLO_THRESHOLDS,
    BenchmarkBase,
    BenchmarkLoadTypeEnum,
    BenchmarkLoadModeEnum,
    SLOThreshold,
    benchmark_load_mode,
)
from gpustack.worker.benchmark.artifacts import ramp_facts_path

logger = logging.getLogger(__name__)

# A sampled point's success rate below this = overloaded / not trustworthy.
MIN_SUCCESS_RATE = 0.95

# A throughput gain below this on the last measured step = the curve flattened
# (the ramp engine stops on the same <5% plateau). Used to tell a sweep that ran
# out of RANGE while still climbing (raise the bound) from one that simply found
# saturation at the top (don't).
PEAK_PLATEAU_TOLERANCE = 0.05

# Achieved rps below this fraction of the OFFERED rate = the server plainly cannot
# keep up. Deliberately loose: a finite max_requests leaves achieved a few percent
# under offered even on an idle server (the drain tail), so ~6x that bias only
# fires on unmistakable saturation (offering 64 req/s to a server delivering 23.6).
MIN_KEEPUP_RATIO = 0.8

# ── Ramp stop reasons (benchmark-runner's RampOutcome.bracket_reason) ──────────
RAMP_STOP_CAPACITY_PLATEAU = "capacity_plateau"
RAMP_STOP_THROUGHPUT_DECLINE = "throughput_decline"
RAMP_STOP_THROUGHPUT_GAIN_CONVERGED = "throughput_gain_converged"
RAMP_STOP_UPPER_BOUND = "upper_bound"
RAMP_STOP_BUDGET_POINTS = "budget_points"
RAMP_STOP_BUDGET_SECONDS = "budget_seconds"


class _ThroughputDeclinePoint(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    rate: float = Field(gt=0, allow_inf_nan=False)
    total_tps: float = Field(ge=0, allow_inf_nan=False)
    drop_percent: float = Field(ge=0, le=100, allow_inf_nan=False)


class _ThroughputDeclineEvidence(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    threshold_percent: float = Field(gt=0, le=100, allow_inf_nan=False)
    consecutive_points: int = Field(gt=0)
    peak_rate: float = Field(gt=0, allow_inf_nan=False)
    peak_total_tps: float = Field(gt=0, allow_inf_nan=False)
    points: list[_ThroughputDeclinePoint] = Field(min_length=1)


class _ThroughputGainPoint(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    rate: float = Field(gt=0, allow_inf_nan=False)
    total_tps: float = Field(gt=0, allow_inf_nan=False)
    best_total_tps: float = Field(gt=0, allow_inf_nan=False)


class _ThroughputGainEvidence(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    threshold_percent: float = Field(gt=0, le=100, allow_inf_nan=False)
    window_points: int = Field(ge=3, le=3)
    baseline_rate: float = Field(gt=0, allow_inf_nan=False)
    baseline_total_tps: float = Field(gt=0, allow_inf_nan=False)
    best_rate: float = Field(gt=0, allow_inf_nan=False)
    best_total_tps: float = Field(gt=0, allow_inf_nan=False)
    gain_percent: float = Field(ge=0, allow_inf_nan=False)
    lower_rate: float = Field(gt=0, allow_inf_nan=False)
    upper_rate: float = Field(gt=0, allow_inf_nan=False)
    points: list[_ThroughputGainPoint] = Field(min_length=3, max_length=3)


def read_ramp_facts(benchmark_dir: str, benchmark) -> Optional[dict]:
    """Load ``{id}__ramp.json``, or None when there is none to read.

    Only auto-tune runs have a ramp. A missing / unreadable file is normal, not an
    error: it means "no facts available, judge from the grid". In particular it is
    absent during a partial sync — the sidecar is written when the ramp returns,
    so its appearance is itself the signal that the search has ended.
    """
    if not getattr(benchmark, "auto_tune", False):
        return None
    path = ramp_facts_path(benchmark_dir, benchmark.id)
    try:
        with open(path, "r", encoding="utf-8") as f:
            facts = json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning(
            f"Ignoring unreadable ramp outcome for benchmark "
            f"{benchmark.name}(id={benchmark.id}): {e}"
        )
        return None
    return facts if isinstance(facts, dict) else None


# ── SLO evaluation ────────────────────────────────────────────────────────────
# The thresholds themselves live in SLO_THRESHOLDS (one row per metric x
# aggregation) so the runner's CLI forwarding and the checks below cannot drift.


def measured_stages(results: list) -> list:
    """The rows that are stages of the sweep — everything with a load value.

    A benchmark_results row without a `rate` is NOT a stage. Two kinds exist:
    the auto-tune saturation probe (its `throughput` profile has no rate by
    construction — that is what makes it a probe) and, on legacy sweep records,
    the synchronous / throughput bound passes.

    They are measurements, and they are kept in the table for that reason, but
    they are not points on the load curve: they have no x coordinate to plot, no
    configured load to compare an achieved one against, and their numbers come
    from a burst rather than a steady state (a probe measured 1.71s against a
    1.65s mean latency, i.e. one batch — its prompt-token rate came back 3.4x the
    true value and its TTFT 6x the steady-state value at the same concurrency).

    Counting them as stages inflates every aggregate: "N stages", the request
    total, and the success rate. This exists as ONE function because the rule
    spelled out at each call site drifts into slightly different predicates —
    which is how the next call site gets it wrong.
    """
    return [r for r in results if r.get("rate") is not None]


def has_slo(benchmark) -> bool:
    """True when any latency threshold is set on this benchmark."""
    return any(getattr(benchmark, t.attr, None) for t in SLO_THRESHOLDS)


def _slo_value(r: dict, t: SLOThreshold) -> Optional[float]:
    """The point's value for one SLO threshold, or None when it was not measured.

    Non-positive counts as not measured, not as "0 ms, passes everything". No
    latency here can really be zero, and one of them reaches zero routinely: the
    decode-only TPOT needs two token timestamps, so a response that arrives as a
    single chunk (the whole output at once — common at low load, and the only shape
    a single-token output can have) collapses it to 0.0. Treating that as a pass
    would let such a run climb to the upper bound and report the ceiling as
    SLO-approved.

    `t.fallback` covers exactly that case for the TPOT rows: total-time-over-tokens
    is the only per-token number a non-incremental response leaves behind. Failing
    the point instead would bracket the ramp wherever the server batched its
    stream.
    """
    for metric in (t.metric, t.fallback):
        if not metric:
            continue
        val = r.get(metric)
        if val is not None and val > 0:
            return val
    return None


def meets_slo(benchmark, r: dict) -> bool:
    """True iff every SET SLO threshold holds for this point (AND).

    A point that does not carry the metric a threshold is set on never meets it —
    rows written before that percentile was collected have nothing to compare.
    """
    for t in SLO_THRESHOLDS:
        thr = getattr(benchmark, t.attr, None)
        if thr is None:
            continue
        val = _slo_value(r, t)
        if val is None or val * t.scale > thr:
            return False
    return True


def slo_utilization(benchmark, r: dict) -> Optional[float]:
    """How close this point came to its TIGHTEST set SLO threshold.

    1.0 = sitting exactly on a threshold; 0.3 = using 30% of the strictest budget.
    None when no threshold is set, or the point carries none of the metrics one is
    set on. Tells an SLO that genuinely binds at the throughput peak from one that
    never mattered — the rates alone cannot: both end up with
    slo_met_rate == peak_rate == the top knob measured.
    """
    worst = None
    for t in SLO_THRESHOLDS:
        thr = getattr(benchmark, t.attr, None)
        if thr is None or thr <= 0:
            continue
        val = _slo_value(r, t)
        if val is None:
            continue
        used = val * t.scale / thr
        worst = used if worst is None else max(worst, used)
    return worst


def success_ok(r: dict) -> bool:
    total = r.get("request_total") or 0
    if total <= 0:
        return False
    return (r.get("request_successful") or 0) / total >= MIN_SUCCESS_RATE


def slo_boundary_located(
    benchmark, rate_points: list, best_points: dict, ramp: Optional[dict]
) -> bool:
    """Is ``slo_met_rate`` an EDGE, or merely a FLOOR?

    The number itself cannot say. "The max measured load meeting the SLO" is true
    either way, but it means two different things:

    * EDGE — a load just above it was measured BREACHING the SLO, so this is the
      boundary: "257 breaks it".
    * FLOOR — nothing above it was ever measured failing, because the search ended
      first (capacity plateau, budget, range ceiling). The honest reading is
      ">= 256", and treating it as an edge invents a limit nobody measured.
      Capacity planning off a fabricated ceiling is the failure this prevents.

    With ramp facts this is a lookup: ``slo_bracket = (last_pass, first_fail)`` and
    ``first_fail is None`` IS "no boundary located". Without them, the same
    evidence is read off the grid — a measured point above the answer that fails it
    — using the same pass definition :func:`compute_best_points` used to pick the
    answer (SLO thresholds AND the success floor), so the two cannot drift.
    """
    slo_rate = best_points.get("slo_met_rate")
    if slo_rate is None:
        return False

    bracket = (ramp or {}).get("slo_bracket")
    if isinstance(bracket, (list, tuple)) and len(bracket) == 2:
        return bracket[1] is not None

    # Reaching the top of the range with everything passing is deliberately NOT a
    # located boundary: the range ran out, so the edge is somewhere above it.
    return any(
        r["rate"] > slo_rate and not (meets_slo(benchmark, r) and success_ok(r))
        for r in measured_stages(rate_points)
    )


# ── Best operating points ─────────────────────────────────────────────────────


def total_throughput(point: dict) -> Optional[float]:
    """Read the measured total token throughput without substituting output TPS.

    Args:
        point: One measured stage's flat result fields.

    Returns:
        Total tokens per second, or None when it was not measured.
    """
    return point.get("tokens_per_second_mean")


def select_result_points(
    benchmark: BenchmarkBase, results: list
) -> Tuple[Optional[dict], Optional[dict]]:
    """Select the primary result and the independently observed total-TPS peak.

    SLO tests answer the highest measured load satisfying all targets and the
    request-success floor. Other tests summarize the total-throughput peak.
    Equal peaks prefer the lower load, then the earlier measured stage.

    Args:
        benchmark: Workload configuration and optional SLO targets.
        results: Measured stages and any auxiliary probe results.

    Returns:
        Primary and peak stage dictionaries, each optional.
    """
    points = sorted(
        measured_stages(results),
        key=lambda point: (
            point["rate"],
            point.get("sequence") or 0,
            point.get("input_tokens") or 0,
        ),
    )
    measured = [point for point in points if total_throughput(point) is not None]
    peak = max(measured, key=total_throughput, default=None)
    if not has_slo(benchmark):
        return peak, peak
    passing = [
        point for point in points if meets_slo(benchmark, point) and success_ok(point)
    ]
    primary = max(
        passing,
        key=lambda point: (point["rate"], total_throughput(point) or 0),
        default=None,
    )
    return primary, peak


def point_coordinates(point: Optional[dict]) -> Optional[dict]:
    """Identify a stage without relying on result rows that uploads replace.

    Args:
        point: Selected stage, or None when no stage applies.

    Returns:
        Stable input-length, sequence and load coordinates, or None.
    """
    if point is None:
        return None
    return {
        "input_tokens": point.get("input_tokens"),
        "sequence": point.get("sequence") or 0,
        "rate": point["rate"],
    }


def compute_best_points(benchmark, results: list) -> dict:
    """Derive the primary result load and independently observed total-TPS peak.

    `recommended_rate` is the primary result's configured concurrency or request
    rate. With SLO targets it equals `slo_met_rate`, the highest measured passing
    load, even when a lower load delivered more total throughput. Without SLO
    targets it equals `peak_rate`. No passing SLO stage means no primary result.
    """
    primary, peak = select_result_points(benchmark, results)
    out = {}
    if peak is not None:
        out["peak_rate"] = float(peak["rate"])
    if primary is not None:
        out["recommended_rate"] = float(primary["rate"])
        if has_slo(benchmark):
            out["slo_met_rate"] = float(primary["rate"])
    return out


# ── Coverage validity ─────────────────────────────────────────────────────────


def compute_validity(
    benchmark, results: list, best_points: dict, ramp: Optional[dict] = None
) -> dict:
    """Judge whether the sweep explored enough to trust the result.

    Returns ``coverage_applicable`` and ``warnings``. Search runs also report
    ``sufficient`` plus, when available, ``stop_reason`` / ``stopped_at`` /
    ``slo_boundary_located`` and the saturation probe's ``probe_ceiling`` /
    ``probe_bound`` / ``probe_relaxed``. Every run also carries primary_point and
    peak_point coordinates from the same selection used for parent metrics.
    Non-search runs include those selections and measured findings without a
    search coverage verdict. A protective throughput-decline stop can also carry
    validated trigger-time evidence recorded by the runner.

    Codes (rendered/localized by the UI):
    - ``slo_never_met``: SLO targets set but no measured point meets them -> the
      server is too slow for this SLO; no usable capacity.
    - ``not_saturated``: the best point is the highest one measured and the user's
      own search range is what stopped the sweep -> raise upper_bound.
    - ``budget_exhausted``: same situation, but the measurement budget ran out
      first -> raise max_points / max_total_seconds (raising the range would not
      help; the sweep never got to use the range it had).
    - ``saturated_at_lower_bound``: the sweep's LOWEST point already outran the
      server (achieved rps << offered), so the search range starts above
      saturation and cannot contain the optimum -> lower lower_bound. Carries the
      measured sustained rps as ``params.ceiling`` so the advice is a number, not
      a direction. Supersedes the top-edge codes, whose advice ("raise the upper
      bound") would point the opposite way.
    - ``slo_not_binding``: a runner stopped an SLO search on throughput plateau
      without locating a latency boundary. The highest passing load remains a
      measured lower bound; the throughput peak is only an independent fact.
    - ``throughput_decline``: the runner explicitly stopped an SLO search after
      sustained total-throughput decline. The last passing load remains the
      primary result, but the latency boundary has not been located.
    - ``peak_at_floor``: the best point is the LOWEST one measured -> the optimum
      may be below the search range; lower lower_bound.
    - ``point_high_error``: some point's success rate < 95% -> overloaded /
      unreliable at that load (already flagged red in the table).
    - ``few_points``: too few measured points (< 3, no SLO) to trust the curve.

    ``not_saturated`` and ``budget_exhausted`` are the same observation ("we never
    saw the curve turn over") split by WHAT limited the sweep, because the two need
    opposite advice. One code saying "raise the bound / budget" is actively
    wrong whenever the ramp stopped below the user's range of its own accord:
    telling someone who asked for 4..1024 and got points up to 38 to raise the
    bound sends them to change the one number that was never the constraint.
    """
    warnings: list = []
    primary, peak = select_result_points(benchmark, results)
    selection = {
        "primary_point": point_coordinates(primary),
        "peak_point": point_coordinates(peak),
    }

    rate_points = measured_stages(results)
    slo_set = has_slo(benchmark)

    # Any measured point that overloaded (low success rate).
    overloaded_any = False
    worst_ok = None
    for r in rate_points:
        total = r.get("request_total") or 0
        if total <= 0:
            continue
        ok = (r.get("request_successful") or 0) / total
        worst_ok = ok if worst_ok is None else min(worst_ok, ok)
        if ok < MIN_SUCCESS_RATE:
            overloaded_any = True
    if overloaded_any and worst_ok is not None:
        # `rate` here is the worst point's SUCCESS PERCENTAGE, not a load — the odd
        # name is the wire contract the UI already renders ("{rate}% succeeded").
        warnings.append(
            {"code": "point_high_error", "params": {"rate": round(worst_ok * 100)}}
        )

    if slo_set and best_points.get("slo_met_rate") is None:
        # SLO set but nothing met it — even the lowest load is too slow.
        warnings.append({"code": "slo_never_met", "params": {}})

    if benchmark_load_mode(benchmark) is not BenchmarkLoadModeEnum.AUTO_TUNE:
        # Fixed loads do not claim to locate a curve maximum or boundary.
        # Point failures and unmet SLO targets remain useful measurements.
        return {"coverage_applicable": False, "warnings": warnings, **selection}

    warnings.extend(
        _edge_warnings(
            benchmark, rate_points, best_points, overloaded_any, slo_set, ramp
        )
    )

    # `few_points` is the weakest signal, so only surface it when there is nothing
    # more specific to say. A run that already reports e.g.
    # `saturated_at_lower_bound` or `peak_at_floor` doesn't need "and also, few
    # points" tacked on — it dilutes the primary cause the user should act on
    # (observed alongside those codes in rounds 5/6).
    if not slo_set and 0 < len(rate_points) < 3 and not warnings:
        warnings.append({"code": "few_points", "params": {}})

    out = {
        "coverage_applicable": True,
        "sufficient": len(warnings) == 0,
        "warnings": warnings,
        **selection,
    }
    if slo_set and best_points.get("slo_met_rate") is not None:
        # Whether `slo_met_rate` is a measured boundary or a floor. Rides in
        # `validity` for the same two reasons as the stop reason: no migration
        # (JSON column), and it IS a coverage statement — "did we actually observe
        # where the SLO breaks?".
        out["slo_boundary_located"] = slo_boundary_located(
            benchmark, rate_points, best_points, ramp
        )
    out.update(_ramp_facts_for_ui(ramp))
    out.update(_throughput_decline_facts_for_ui(ramp, warnings))
    out.update(_throughput_gain_facts_for_ui(benchmark, results, ramp))
    return out


def _throughput_decline_evidence(ramp: Optional[dict]) -> Optional[dict]:
    """Validate the runner's recorded evidence without recalculating its values."""
    evidence = (ramp or {}).get("throughput_decline")
    if not isinstance(evidence, dict):
        return None
    try:
        validated = _ThroughputDeclineEvidence.model_validate(evidence)
    except ValidationError:
        return None
    if len(validated.points) != validated.consecutive_points:
        return None
    return validated.model_dump()


def _throughput_decline_warning(ramp: Optional[dict]) -> dict:
    evidence = _throughput_decline_evidence(ramp)
    params = (
        {
            "points": evidence["consecutive_points"],
            "threshold": evidence["threshold_percent"],
        }
        if evidence is not None
        else {}
    )
    return {"code": "throughput_decline", "params": params}


def _throughput_decline_facts_for_ui(ramp: Optional[dict], warnings: list) -> dict:
    if not any(warning["code"] == "throughput_decline" for warning in warnings):
        return {}
    evidence = _throughput_decline_evidence(ramp)
    return {"throughput_decline": evidence} if evidence is not None else {}


def _positive_finite(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _valid_gain_sample(point: Optional[dict]) -> bool:
    return (
        point is not None
        and _positive_finite(total_throughput(point))
        and success_ok(point)
    )


def _gain_ordered_points(results: list) -> Optional[list]:
    points = measured_stages(results)
    sequences = [point.get("sequence") for point in points]
    if (
        len(points) <= 3
        or any(
            not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 0
            for sequence in sequences
        )
        or len(set(sequences)) != len(sequences)
        or any(not _positive_finite(point["rate"]) for point in points)
    ):
        return None
    return sorted(points, key=lambda point: point["sequence"])


def _gain_window_matches(window: list, evidence: _ThroughputGainEvidence) -> bool:
    best = evidence.baseline_total_tps
    for row, recorded in zip(window, evidence.points):
        if not _valid_gain_sample(row):
            return False
        best = max(best, total_throughput(row))
        if (
            row["rate"] != recorded.rate
            or total_throughput(row) != recorded.total_tps
            or best != recorded.best_total_tps
        ):
            return False
    return True


def _gain_coverage_matches(points: list, evidence: _ThroughputGainEvidence) -> bool:
    valid = [point for point in points if _valid_gain_sample(point)]
    lower = max(
        (point["rate"] for point in valid if point["rate"] < evidence.best_rate),
        default=None,
    )
    upper = min(
        (point["rate"] for point in valid if point["rate"] > evidence.best_rate),
        default=None,
    )
    return lower == evidence.lower_rate and upper == evidence.upper_rate


def _gain_matches_measurements(
    benchmark, results: list, evidence: _ThroughputGainEvidence
) -> bool:
    points = _gain_ordered_points(results)
    if points is None:
        return False
    before, window = points[:-3], points[-3:]
    _primary, baseline = select_result_points(benchmark, before)
    _primary, best = select_result_points(benchmark, points)
    if not _valid_gain_sample(baseline) or not _valid_gain_sample(best):
        return False
    if (
        baseline["rate"] != evidence.baseline_rate
        or total_throughput(baseline) != evidence.baseline_total_tps
        or best["rate"] != evidence.best_rate
        or total_throughput(best) != evidence.best_total_tps
    ):
        return False
    gain = 100 * (evidence.best_total_tps / evidence.baseline_total_tps - 1)
    return (
        evidence.gain_percent < evidence.threshold_percent
        and 0 <= gain < evidence.threshold_percent
        and math.isclose(gain, evidence.gain_percent, rel_tol=0, abs_tol=5.00001e-7)
        and _gain_window_matches(window, evidence)
        and _gain_coverage_matches(points, evidence)
    )


def _throughput_gain_facts_for_ui(
    benchmark, results: list, ramp: Optional[dict]
) -> dict:
    if (
        has_slo(benchmark)
        or (ramp or {}).get("stop_reason") != RAMP_STOP_THROUGHPUT_GAIN_CONVERGED
    ):
        return {}
    raw = ramp.get("throughput_gain_evidence")
    if not isinstance(raw, dict):
        return {}
    try:
        evidence = _ThroughputGainEvidence.model_validate(raw)
    except ValidationError:
        return {}
    if not _gain_matches_measurements(benchmark, results, evidence):
        return {}
    return {"throughput_gain_evidence": evidence.model_dump()}


def _ramp_facts_for_ui(ramp: Optional[dict]) -> dict:
    """The ramp's own account of the search, forwarded verbatim.

    Rides inside the existing `validity` JSON column: no migration, and it IS the
    evidence behind the verdict sitting next to it.

    Forwarded rather than re-derived. `ceil(ceiling * 1.2)`, the Phase-1/2 split
    and the clamp rule all live in benchmark-runner, so recomputing what the
    probe's cap did would mean keeping three formulas in sync with a file in
    another repository — the failure this whole layer exists to stop.

    Probe keys are omitted, not zeroed, when no probe ran (the concurrency axis
    never probes): `probe_relaxed: 0` would read as "the cap held", which is a
    claim about a cap that never existed.

    The sidecar carries TWO reasons and they are not interchangeable, so both are
    forwarded under their own names:

    * ``stop_reason`` — why the ramp AS A WHOLE ended, Phase 2 included. This is
      the "where the search stopped" the detail page reports.
    * ``bracket_reason`` — why the Phase-1 geometric bracket ended, i.e. WHAT
      bounded the answer. This is what the coverage verdicts above key off, and
      it is kept here as the evidence behind them.

    A run that brackets on ``capacity_plateau`` and then converges reports
    ``capacity_plateau`` / ``converged``: the search completed normally AND the
    thing that limited it was capacity. Feeding one into the other's slot turns
    "finished normally" into "cut short by capacity". Both are declared
    non-optional on benchmark-runner's ``RampOutcome``, so the ``or`` below is a
    guard for a legacy sidecar, not a choice between them.
    """
    if not ramp:
        return {}
    facts = {
        "stop_reason": ramp.get("stop_reason") or ramp.get("bracket_reason"),
        "bracket_reason": ramp.get("bracket_reason"),
        "stopped_at": ramp.get("stopped_at"),
    }
    for key in ("probe_ceiling", "probe_bound", "probe_relaxed", "slo_bracket"):
        if ramp.get(key) is not None:
            facts[key] = ramp[key]
    return facts


def _edge_warnings(
    benchmark,
    rate_points: list,
    best_points: dict,
    overloaded_any: bool,
    slo_set: bool,
    ramp: Optional[dict] = None,
) -> list:
    """Warnings for a best point sitting at an EDGE of the measured range.

    Same observation at either end — "we never saw the curve turn over on this
    side" — but each end (and, at the top, each possible cause) needs different
    advice, so they are distinct codes. See :func:`compute_validity`.
    """
    out: list = []
    rec = best_points.get("recommended_rate")
    if rec is None or not rate_points:
        return out

    final_stop = (ramp or {}).get("stop_reason")
    if final_stop == RAMP_STOP_THROUGHPUT_GAIN_CONVERGED and slo_set:
        if not slo_boundary_located(benchmark, rate_points, best_points, ramp):
            return [{"code": "not_saturated", "params": {}}]
        return []

    if slo_set:
        facts = ramp or {}
        stop_reason = facts.get("stop_reason") or facts.get("bracket_reason")
        if stop_reason == RAMP_STOP_THROUGHPUT_DECLINE and not slo_boundary_located(
            benchmark, rate_points, best_points, ramp
        ):
            return [_throughput_decline_warning(ramp)]
        limits = _limit_warnings(
            benchmark,
            rate_points,
            rec,
            max(r["rate"] for r in rate_points),
            overloaded_any,
            ramp,
        )
        if limits:
            return limits
        if (ramp or {}).get(
            "bracket_reason"
        ) == RAMP_STOP_CAPACITY_PLATEAU and not slo_boundary_located(
            benchmark, rate_points, best_points, ramp
        ):
            # A throughput plateau does not establish where latency targets fail.
            top_point = max(rate_points, key=lambda point: point["rate"])
            used = slo_utilization(benchmark, top_point)
            return [
                {
                    "code": "slo_not_binding",
                    "params": {
                        "rate": best_points.get("peak_rate"),
                        "used": round(used * 100) if used is not None else None,
                    },
                }
            ]
        return []

    floor_ceiling = saturation_floor_ceiling(benchmark, rate_points)
    if floor_ceiling is not None:
        return [
            {"code": "saturated_at_lower_bound", "params": {"ceiling": floor_ceiling}}
        ]

    top_rate = max(r["rate"] for r in rate_points)
    out.extend(
        _limit_warnings(benchmark, rate_points, rec, top_rate, overloaded_any, ramp)
    )

    # Bottom edge: the best point sits at the lowest knob measured, so the optimum
    # may be lower still — lower the search range and re-run. On the concurrency
    # axis this is the only signal available (the knob counts streams, so there is
    # no achieved-vs-offered comparison to make).
    if (
        not slo_set
        and len(rate_points) > 1
        and rec == min(r["rate"] for r in rate_points)
    ):
        out.append({"code": "peak_at_floor", "params": {}})

    return out


def _limit_warnings(
    benchmark,
    rate_points: list,
    rec,
    top_rate,
    overloaded_any: bool,
    ramp: Optional[dict],
) -> list:
    """Which limit ended the sweep — from the ramp's own word when we have it.

    Two codes, opposite advice: `not_saturated` means the user's search RANGE
    stopped us (raise it), `budget_exhausted` means the measurement budget did
    (raise that instead; the range was never the constraint).

    The final stop reason includes Phase 2 refinement. Its point or time budget
    can expire after Phase 1 has already found a bracket, leaving an incomplete
    answer even when the recommended load is below the highest measured point.
    Without ramp facts, a time cap cannot be distinguished from a completed run
    by looking at the measured grid alone.

    A THIRD bound exists and is not the user's: the saturation probe's soft cap.
    `not_saturated` must never be reported for it — its advice is "raise
    upper_bound", and raising it changes nothing, because the probe re-derives the
    cap from a fresh measurement on every run. Observed: a run configured 4..1024
    stopped at 31 (cap = ceil(25.6 * 1.2)) and was told to raise the 1024.
    """
    stop = (ramp or {}).get("stop_reason")
    if stop == RAMP_STOP_THROUGHPUT_GAIN_CONVERGED:
        return []
    if stop in (RAMP_STOP_BUDGET_POINTS, RAMP_STOP_BUDGET_SECONDS):
        which = "seconds" if stop == RAMP_STOP_BUDGET_SECONDS else "points"
        return [{"code": "budget_exhausted", "params": {"which": which}}]

    bracket = (ramp or {}).get("bracket_reason")
    if bracket is not None:
        if bracket == RAMP_STOP_UPPER_BOUND and not overloaded_any:
            if _stopped_on_the_probes_cap(benchmark, ramp):
                return []
            return [{"code": "not_saturated", "params": {}}]
        if bracket in (RAMP_STOP_BUDGET_POINTS, RAMP_STOP_BUDGET_SECONDS):
            # `which` names the cap, so the message can point at the right knob.
            which = "seconds" if bracket == RAMP_STOP_BUDGET_SECONDS else "points"
            return [{"code": "budget_exhausted", "params": {"which": which}}]
        # Any other reason means either the search had its answer
        # (capacity_plateau, slo_failed, ...) or the limit was one the user cannot
        # act on: `probe_bound` is the saturation probe's own soft cap, and raising
        # upper_bound does not move a cap the probe re-derives from a fresh
        # measurement on every run. Either way: no limit to report.
        return []

    # ── No facts (stage / legacy / pre-sidecar run): judge from the grid ──
    # A throughput search can finish on a plateau. An SLO search still needs a
    # higher failed point even if throughput already flattened or declined.
    if (
        overloaded_any
        or rec != top_rate
        or (not has_slo(benchmark) and _plateaued_at_top(rate_points))
    ):
        return []
    # Point budget checked first: when it ran out, the search range was never the
    # binding constraint no matter where the top point landed. The TIME cap is
    # invisible from here — that gap is what the facts path above closes.
    max_points = getattr(benchmark, "max_points", None)
    if max_points is not None and len(rate_points) >= max_points:
        return [{"code": "budget_exhausted", "params": {"which": "points"}}]
    return [{"code": "not_saturated", "params": {}}]


def _stopped_on_the_probes_cap(benchmark, ramp: Optional[dict]) -> bool:
    """Did the SOFT cap end this run, while reporting itself as `upper_bound`?

    Runners that name the two bounds apart report `probe_bound` and never reach
    this. For one that does not, the sidecar still carries enough to tell: the run
    ended AT OR ABOVE the probe's cap, and that cap was below the range the user
    asked for. Those two facts are jointly decisive, because the ramp only breaks
    on a bound it has reached (`knob >= bound`, `bound = min(upper_bound, cap)`):

      * the cap ended it  -> bound is the cap, so stopped_at >= probe_bound, and
        the cap sits under the user's range.
      * the range ended it -> bound is upper_bound, which means the cap was absent
        or already at/above it, so `probe_bound < upper` is false.

    `>=` rather than `==` because the run does not have to LAND on the cap. Points
    are clamped to the bound at the end of an iteration, so the FIRST point is
    never clamped: a range starting above the cap (lower_bound 32, cap 31) measures
    32, stops, and reports 32. That is the one case the current runner can still
    produce — it is exactly when the cap is allowed to end a search — so requiring
    equality would miss it on every pre-`probe_bound` image.

    Every one of the three facts is required: the cap (`probe_bound`), where the
    run ended (`stopped_at`), and the range to compare the cap against
    (`benchmark.upper_bound`, absent on runs that never had a search range). Any
    of them missing and the verdict is left alone. That asymmetry is deliberate —
    this only ever SUPPRESSES advice, so a false negative merely restores the old
    (wrong) message, while a false positive would hide a real range limit.
    """
    facts = ramp or {}
    probe_bound = facts.get("probe_bound")
    stopped_at = facts.get("stopped_at")
    upper = getattr(benchmark, "upper_bound", None)
    if probe_bound is None or stopped_at is None or upper is None:
        return False
    return stopped_at >= probe_bound and probe_bound < upper


def _plateaued_at_top(rate_points: list) -> bool:
    """True when the last measured step barely grew throughput (or dropped).

    Compares the top two points by rate: a gain below PEAK_PLATEAU_TOLERANCE means
    the curve flattened at the top, so the sweep found saturation rather than
    running out of range. Fewer than two points => not a plateau.

    A top point with no throughput reading is NOT treated as a plateau: "we cannot
    tell" must not silently become "saturation found", which would suppress the
    range-limit advice.
    """
    pts = sorted(measured_stages(rate_points), key=lambda r: r["rate"])
    if len(pts) < 2:
        return False
    prev_tps = total_throughput(pts[-2]) or 0
    top_tps = total_throughput(pts[-1])
    if prev_tps <= 0 or top_tps is None:
        return False
    return (top_tps / prev_tps - 1.0) < PEAK_PLATEAU_TOLERANCE


def saturation_floor_ceiling(
    benchmark: BenchmarkBase, results: List[dict]
) -> Optional[float]:
    """Read the sustained request rate when a throughput search starts saturated.

    Only a fixed-rate throughput search compares delivered and offered request
    rates. A shortfall does not determine whether latency SLO targets bind.

    Args:
        benchmark: The search configuration.
        results: Measured stages and any auxiliary probe readings.

    Returns:
        Sustained requests per second, or None without an invalid throughput floor.
    """
    if (
        benchmark_load_mode(benchmark) is not BenchmarkLoadModeEnum.AUTO_TUNE
        or has_slo(benchmark)
        or getattr(benchmark, "load_type", None) != BenchmarkLoadTypeEnum.FIXED_RATE
    ):
        return None
    rate_points = measured_stages(results)
    if not rate_points:
        return None
    lowest = min(rate_points, key=lambda r: r["rate"])
    offered = lowest.get("rate") or 0
    achieved = lowest.get("requests_per_second_mean")
    if offered <= 0 or achieved is None:
        return None
    if achieved >= offered * MIN_KEEPUP_RATIO:
        return None
    return round(achieved, 2)
