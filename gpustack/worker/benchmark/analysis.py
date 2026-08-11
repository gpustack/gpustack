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
  threshold breaking at the top both end with "the highest knob met the SLA";
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
from typing import Optional

from gpustack.schemas.benchmark import (
    SLA_THRESHOLDS,
    BenchmarkLoadTypeEnum,
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

# Fraction of its tightest SLA budget a point may consume and still count as "the
# thresholds never came into play". A sweep that ended on 31% of its TTFT budget
# was not stopped by latency, whatever the SLA-capacity number says. Used to
# phrase the message, and as the fallback discriminator when no ramp facts exist.
SLA_HEADROOM_RATIO = 0.7

# Achieved rps below this fraction of the OFFERED rate = the server plainly cannot
# keep up. Deliberately loose: a finite max_requests leaves achieved a few percent
# under offered even on an idle server (the drain tail), so ~6x that bias only
# fires on unmistakable saturation (offering 64 req/s to a server delivering 23.6).
MIN_KEEPUP_RATIO = 0.8

# ── Ramp stop reasons (benchmark-runner's RampOutcome.bracket_reason) ──────────
RAMP_STOP_CAPACITY_PLATEAU = "capacity_plateau"
RAMP_STOP_UPPER_BOUND = "upper_bound"
RAMP_STOP_BUDGET_POINTS = "budget_points"
RAMP_STOP_BUDGET_SECONDS = "budget_seconds"


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


# ── SLA evaluation ────────────────────────────────────────────────────────────
# The thresholds themselves live in SLA_THRESHOLDS (one row per metric x
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
    total, and the success rate. This exists as ONE function because the rule was
    previously spelled out at each call site with slightly different predicates —
    which is how the next call site gets it wrong.
    """
    return [r for r in results if r.get("rate") is not None]


def has_sla(benchmark) -> bool:
    """True when any latency threshold is set on this benchmark."""
    return any(getattr(benchmark, t.attr, None) for t in SLA_THRESHOLDS)


def _sla_value(r: dict, t) -> Optional[float]:
    """The point's value for one SLA threshold, or None when it was not measured.

    Non-positive counts as not measured, not as "0 ms, passes everything". No
    latency here can really be zero, and one of them reaches zero routinely: the
    decode-only TPOT needs two token timestamps, so a response that arrives as a
    single chunk (the whole output at once — common at low load, and the only shape
    a single-token output can have) collapses it to 0.0. Treating that as a pass
    would let such a run climb to the upper bound and report the ceiling as
    SLA-approved.

    `t.fallback` covers exactly that case for the TPOT rows: total-time-over-tokens
    is the only per-token number a non-incremental response leaves behind. Failing
    the point instead would bracket the ramp wherever the server batched its
    stream.
    """
    for metric in (t.metric, getattr(t, "fallback", None)):
        if not metric:
            continue
        val = r.get(metric)
        if val is not None and val > 0:
            return val
    return None


def meets_sla(benchmark, r: dict) -> bool:
    """True iff every SET SLA threshold holds for this point (AND).

    A point that does not carry the metric a threshold is set on never meets it —
    rows written before that percentile was collected have nothing to compare.
    """
    for t in SLA_THRESHOLDS:
        thr = getattr(benchmark, t.attr, None)
        if thr is None:
            continue
        val = _sla_value(r, t)
        if val is None or val * t.scale > thr:
            return False
    return True


def sla_utilization(benchmark, r: dict) -> Optional[float]:
    """How close this point came to its TIGHTEST set SLA threshold.

    1.0 = sitting exactly on a threshold; 0.3 = using 30% of the strictest budget.
    None when no threshold is set, or the point carries none of the metrics one is
    set on. Tells an SLA that genuinely binds at the throughput peak from one that
    never mattered — the rates alone cannot: both end up with
    sla_met_rate == peak_rate == the top knob measured.
    """
    worst = None
    for t in SLA_THRESHOLDS:
        thr = getattr(benchmark, t.attr, None)
        if thr is None or thr <= 0:
            continue
        val = _sla_value(r, t)
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


def sla_boundary_located(
    benchmark, rate_points: list, best_points: dict, ramp: Optional[dict]
) -> bool:
    """Is ``sla_met_rate`` an EDGE, or merely a FLOOR?

    The number itself cannot say. "The max measured load meeting the SLA" is true
    either way, but it means two different things:

    * EDGE — a load just above it was measured BREACHING the SLA, so this is the
      boundary: "257 breaks it".
    * FLOOR — nothing above it was ever measured failing, because the search ended
      first (capacity plateau, budget, range ceiling). The honest reading is
      ">= 256", and treating it as an edge invents a limit nobody measured.
      Capacity planning off a fabricated ceiling is the failure this prevents.

    With ramp facts this is a lookup: ``sla_bracket = (last_pass, first_fail)`` and
    ``first_fail is None`` IS "no boundary located". Without them, the same
    evidence is read off the grid — a measured point above the answer that fails it
    — using the same pass definition :func:`compute_best_points` used to pick the
    answer (SLA thresholds AND the success floor), so the two cannot drift.
    """
    sla_rate = best_points.get("sla_met_rate")
    if sla_rate is None:
        return False

    bracket = (ramp or {}).get("sla_bracket")
    if isinstance(bracket, (list, tuple)) and len(bracket) == 2:
        return bracket[1] is not None

    # Reaching the top of the range with everything passing is deliberately NOT a
    # located boundary: the range ran out, so the edge is somewhere above it.
    return any(
        r["rate"] > sla_rate and not (meets_sla(benchmark, r) and success_ok(r))
        for r in measured_stages(rate_points)
    )


# ── Best operating points ─────────────────────────────────────────────────────


def compute_best_points(benchmark, results: list) -> dict:
    """Derive best operating points from the per-rate result grid.

    - peak_rate: the true throughput argmax — the rate with the highest measured
      throughput (ties resolve to the lowest rate). Exactly what the name says.
    - sla_met_rate: when SLA targets are set, the max rate whose latency metrics
      all stay within the SLA thresholds (ms). The literal answer to "how much
      load still meets the SLA", reported as measured.
    - recommended_rate: the load to actually RUN at.
      * No SLA => the throughput peak (the user asked to maximise throughput).
      * SLA    => min(sla_met_rate, peak_rate). Capped at the PEAK, not below: up
        to the peak more load buys more throughput, so if it meets the SLA
        recommend it (bm93: a real 500ms TTFT target held at concurrency 256 — the
        throughput peak — at 155ms, so 256 is the answer, NOT some lower-latency
        point below it). The cap only bites past the peak, where throughput
        DECLINES and a higher SLA-meeting rate is strictly worse (bm74: the loose
        10s target held to 1024, which delivered LESS throughput than 256 — so
        recommend 256, and _edge_warnings flags the gap via `sla_not_binding`).

      An earlier version capped recommended at a "knee" (lowest rate within 5% of
      peak throughput). That under-delivered on real SLAs: it treated a healthy
      near-peak point (bm93's 256) the same as an over-saturated one and
      recommended a lower load than the SLA actually allowed.
    """
    points = [
        r
        for r in measured_stages(results)
        if r.get("tokens_per_second_mean") is not None
    ]
    if not points:
        return {}
    points = sorted(points, key=lambda r: r["rate"])
    out: dict = {}

    # `points` is sorted ascending by rate; max() returns the first maximal
    # element, so ties resolve to the lowest rate.
    peak = max(points, key=lambda r: r.get("tokens_per_second_mean") or 0)
    peak_rate = float(peak["rate"])
    out["peak_rate"] = peak_rate

    if has_sla(benchmark):
        met = [r for r in points if meets_sla(benchmark, r) and success_ok(r)]
        if met:
            sla_rate = float(max(met, key=lambda r: r["rate"])["rate"])
            out["sla_met_rate"] = sla_rate
            out["recommended_rate"] = min(sla_rate, peak_rate)
    else:
        out["recommended_rate"] = peak_rate

    return out


# ── Coverage validity ─────────────────────────────────────────────────────────


def compute_validity(
    benchmark, results: list, best_points: dict, ramp: Optional[dict] = None
) -> dict:
    """Judge whether the sweep explored enough to trust the result.

    Returns ``{"sufficient": bool, "warnings": [...]}`` plus, when the facts are
    available, ``stop_reason`` / ``stopped_at`` / ``sla_boundary_located`` and the
    saturation probe's ``probe_ceiling`` / ``probe_bound`` / ``probe_relaxed``.

    Codes (rendered/localized by the UI):
    - ``sla_never_met``: SLA targets set but no measured point meets them -> the
      server is too slow for this SLA; no usable capacity.
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
    - ``sla_not_binding``: SLA targets set, but they held at the HIGHEST load
      measured while throughput had already turned over -> capacity, not the
      latency budget, is what limited this run; the thresholds are too loose to
      locate a boundary. Carries the throughput peak as ``params.rate``.
    - ``peak_at_floor``: the best point is the LOWEST one measured -> the optimum
      may be below the search range; lower lower_bound.
    - ``point_high_error``: some point's success rate < 95% -> overloaded /
      unreliable at that load (already flagged red in the table).
    - ``few_points``: too few measured points (< 3, no SLA) to trust the curve.

    ``not_saturated`` and ``budget_exhausted`` are the same observation ("we never
    saw the curve turn over") split by WHAT limited the sweep, because the two need
    opposite advice. They used to be one code whose message said "raise the bound /
    budget", which was actively wrong whenever the ramp stopped below the user's
    range of its own accord: telling someone who asked for 4..1024 and got points
    up to 38 to raise the bound sends them to change the one number that was never
    the constraint.
    """
    warnings: list = []

    rate_points = measured_stages(results)
    sla_set = has_sla(benchmark)

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

    if sla_set and best_points.get("sla_met_rate") is None:
        # SLA set but nothing met it — even the lowest load is too slow.
        warnings.append({"code": "sla_never_met", "params": {}})

    warnings.extend(
        _edge_warnings(
            benchmark, rate_points, best_points, overloaded_any, sla_set, ramp
        )
    )

    # `few_points` is the weakest signal, so only surface it when there is nothing
    # more specific to say. A run that already reports e.g.
    # `saturated_at_lower_bound` or `peak_at_floor` doesn't need "and also, few
    # points" tacked on — it dilutes the primary cause the user should act on
    # (observed alongside those codes in rounds 5/6).
    if not sla_set and 0 < len(rate_points) < 3 and not warnings:
        warnings.append({"code": "few_points", "params": {}})

    out = {"sufficient": len(warnings) == 0, "warnings": warnings}
    if sla_set and best_points.get("sla_met_rate") is not None:
        # Whether `sla_met_rate` is a measured boundary or a floor. Rides in
        # `validity` for the same two reasons as the stop reason: no migration
        # (JSON column), and it IS a coverage statement — "did we actually observe
        # where the SLA breaks?".
        out["sla_boundary_located"] = sla_boundary_located(
            benchmark, rate_points, best_points, ramp
        )
    out.update(_ramp_facts_for_ui(ramp))
    return out


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
    for key in ("probe_ceiling", "probe_bound", "probe_relaxed"):
        if ramp.get(key) is not None:
            facts[key] = ramp[key]
    return facts


def _edge_warnings(
    benchmark,
    rate_points: list,
    best_points: dict,
    overloaded_any: bool,
    sla_set: bool,
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

    floor_ceiling = _saturated_at_floor_ceiling(benchmark, rate_points)
    if floor_ceiling is not None:
        # The search range STARTS above what the server can sustain, so nothing
        # inside it can be the optimum. Reported instead of searching below the
        # range (see the ramp's Phase-1 bracket): [lower_bound, upper_bound] is the
        # user's range and the sweep never leaves it.
        #
        # This supersedes the top-edge codes: the best point is necessarily also
        # the highest one measured here, and "raise the upper bound" would send the
        # user in exactly the wrong direction.
        out.append(
            {"code": "saturated_at_lower_bound", "params": {"ceiling": floor_ceiling}}
        )
        return out

    top_point = max(rate_points, key=lambda r: r["rate"])
    top_rate = top_point["rate"]
    sla_rate = best_points.get("sla_met_rate")
    peak_rate = best_points.get("peak_rate")
    if (
        sla_set
        and sla_rate is not None
        and peak_rate is not None
        and sla_rate >= top_rate
    ):
        # The SLA held at the highest load measured. Two different runs land here,
        # and only the second one has a real latency boundary:
        #
        # (a) throughput PEAKED strictly below the top — the thresholds have
        #     headroom the throughput does not.
        # (b) the sweep STOPPED on the throughput plateau with the thresholds
        #     barely touched. The ramp's SLA branch breaks out on capacity
        #     saturation (past it, more load buys only queueing), so the sweep ends
        #     mid-range with peak == sla == top and every rate comparison comes out
        #     equal. bm102: stopped at concurrency 256 of a 4..1024 range after 7 of
        #     12 points, reporting "max within SLA = 256" — while TTFT there was
        #     156ms of a 500ms budget (31%) and TPOT 1.2ms of 50ms. Judging this on
        #     rates alone said "sufficient, no warnings": a capacity ceiling wearing
        #     an SLA label, with nothing to tell the user why the run ended there.
        #
        # Either way the advice is "tighten the thresholds", NOT the top-edge codes'
        # "raise the upper bound", which would only buy points past the peak. A
        # threshold actually pressed against at the peak (bm93) is a binding SLA and
        # still reports nothing: its recommendation IS the peak.
        used = sla_utilization(benchmark, top_point)
        bracket = (ramp or {}).get("bracket_reason")
        if bracket is not None:
            # FACT: the ramp says capacity is what ended the bracket.
            capacity_bound = bracket == RAMP_STOP_CAPACITY_PLATEAU
        else:
            # No facts (stage / legacy / pre-sidecar run): fall back to the grid.
            # `used <= ratio` + a flattened top step is the closest observable
            # proxy, and the reason this cannot be the primary signal is that a
            # binding SLA at the peak produces the same rates.
            capacity_bound = (
                used is not None
                and used <= SLA_HEADROOM_RATIO
                and _plateaued_at_top(rate_points)
            )
        if peak_rate < sla_rate or capacity_bound:
            out.append(
                {
                    "code": "sla_not_binding",
                    "params": {
                        "rate": peak_rate,
                        # Percent of the strictest budget the top point used, so the
                        # message can say HOW loose the thresholds were instead of
                        # only asserting that they were.
                        "used": round(used * 100) if used is not None else None,
                    },
                }
            )
            return out

    out.extend(
        _limit_warnings(benchmark, rate_points, rec, top_rate, overloaded_any, ramp)
    )

    # Bottom edge: the best point sits at the lowest knob measured, so the optimum
    # may be lower still — lower the search range and re-run. On the concurrency
    # axis this is the only signal available (the knob counts streams, so there is
    # no achieved-vs-offered comparison to make).
    if (
        not sla_set
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

    Telling them apart from the grid is not merely imprecise, it is impossible for
    one case: a run ended by `max_total_seconds` leaves exactly the same curve as
    one that stopped of its own accord, so the inference path below reported "raise
    the upper bound" for a run the CLOCK ended. Hence the facts.

    A THIRD bound exists and is not the user's: the saturation probe's soft cap.
    `not_saturated` must never be reported for it — its advice is "raise
    upper_bound", and raising it changes nothing, because the probe re-derives the
    cap from a fresh measurement on every run. Observed: a run configured 4..1024
    stopped at 31 (cap = ceil(25.6 * 1.2)) and was told to raise the 1024.
    """
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
        # (capacity_plateau, sla_failed, ...) or the limit was one the user cannot
        # act on: `probe_bound` is the saturation probe's own soft cap, and raising
        # upper_bound does not move a cap the probe re-derives from a fresh
        # measurement on every run. Either way: no limit to report.
        return []

    # ── No facts (stage / legacy / pre-sidecar run): judge from the grid ──
    # "Could go higher" only makes sense when nothing overloaded AND the curve was
    # still CLIMBING at the top. If throughput already flattened on the last step
    # the sweep found saturation, so "raise the bound" is wrong advice — more range
    # would only add plateau points (bm86/bm93: top step gained <5%).
    if overloaded_any or rec != top_rate or _plateaued_at_top(rate_points):
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
    prev_tps = pts[-2].get("tokens_per_second_mean") or 0
    top_tps = pts[-1].get("tokens_per_second_mean")
    if prev_tps <= 0 or top_tps is None:
        return False
    return (top_tps / prev_tps - 1.0) < PEAK_PLATEAU_TOLERANCE


def _saturated_at_floor_ceiling(benchmark, rate_points: list) -> Optional[float]:
    """Sustained rps when the sweep's LOWEST point already outran the server.

    Returns None unless this is a fixed-rate run whose lowest measured point shows
    achieved << offered. Only the rate axis can be judged this way: there the knob
    IS an offered request rate, so `requests_per_second_mean` is directly
    comparable to it. On the concurrency axis the knob counts in-flight streams and
    no such comparison exists.
    """
    if getattr(benchmark, "load_type", None) != BenchmarkLoadTypeEnum.FIXED_RATE:
        return None
    lowest = min(rate_points, key=lambda r: r["rate"])
    offered = lowest.get("rate") or 0
    achieved = lowest.get("requests_per_second_mean")
    if offered <= 0 or achieved is None:
        return None
    if achieved >= offered * MIN_KEEPUP_RATIO:
        return None
    return round(achieved, 2)
