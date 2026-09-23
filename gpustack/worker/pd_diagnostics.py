"""Turning a disaggregated member's failure into something actionable.

Two problems this exists for, both measured rather than anticipated.

**The last exception is not the cause.** vLLM's `_handle_failed_transfer` raises
`IndexError: list index out of range` while the real reason —
`NIXL_ERR_BACKEND`, a compatibility hash mismatch, an address already in use —
was logged earlier. Reporting the last exception therefore reports the symptom
of the symptom. It was first assumed this only happened on a tensor-parallelism
mismatch; it happens on any handshake failure. So the log is *scanned* for known
signatures and the earliest one wins, because the earliest is the one that
caused the rest.

**A crash loop looks like progress.** Every port-level failure in a
disaggregated deployment produces the same shape: the container binds, fails,
exits, is restarted, and the instance sits at `starting` forever. Nothing ever
marks it failed, so nothing surfaces it and nothing stops it. A member that has
restarted repeatedly without ever serving has failed, and saying so is the whole
point of `RestartTracker`.
"""

import logging
import re
from collections import deque
from datetime import datetime, timedelta
from typing import Deque, Dict, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Diagnosis(BaseModel):
    """A recognised failure and what to do about it."""

    signature: str
    """The pattern that matched, so a report can be traced back to the log."""
    line: str
    """The log line it matched on, trimmed. Kept because the actionable text
    below is a generalisation and an operator often needs the specific."""
    summary: str
    """What the user should check. Written as an instruction, not a
    restatement of the error."""


# Ordered by nothing but readability: the scan reports the EARLIEST match in the
# log, not the first pattern in this list, so the order here carries no
# precedence. Each pattern is a signature observed in a real failure, and each
# summary names the setting to look at rather than describing the error again.
_SIGNATURES: Tuple[Tuple[str, str, str], ...] = (
    (
        "kv transfer failed",
        r"Mooncake transfer failed, ret: -?\d+",
        "A KV transfer between prefill and decode failed. The request that "
        "needed it did NOT fail with it: decode went on to generate from "
        "blocks it never received, the engine still counted an external "
        "prefix-cache hit, and the caller got a 200 carrying nonsense. "
        "Measured on 910B2 across two hosts, where the reply was the token "
        "'ee' repeated to the length limit. Mooncake exports no Prometheus "
        "counter, so this log line is the only place the failure is visible "
        "at all — which is why it is matched here rather than left to the "
        "transfer metrics. The usual cause is a fabric the transport cannot "
        "use between those two workers: check that RDMA is present and "
        "reachable, or place the group's members on one host.",
    ),
    (
        "NIXL_ERR_BACKEND",
        r"NIXL_ERR_BACKEND",
        "The KV transport could not reach its peer. This is almost always the "
        "network interface: check that kv_ifname names the interface carrying "
        "KV traffic on both workers, and that the address the engine "
        "advertised is one the peer can route to. Left to itself the transport "
        "picks up a container bridge and advertises an unroutable address.",
    ),
    (
        "compatibility hash mismatch",
        r"compatibility hash mismatch|hash mismatch.*kv|kv.*hash mismatch",
        "Prefill and decode were configured differently in a way the "
        "connector refuses. Compare the two roles' data type, KV cache data "
        "type, block size, KV cache layout and attention backend — the engine "
        "hashes all of them and rejects a pair that disagrees.",
    ),
    (
        "address already in use",
        r"[Aa]ddress already in use|EADDRINUSE",
        "A port this member needs was already taken. Two members of one role "
        "on one host collide unless their connector ports are allocated as "
        "separate bands; if the port below is outside the service port range, "
        "it is one the engine chose itself and GPUStack cannot reserve it. "
        "A member can also collide with itself: Mooncake offsets each rank's "
        "handshake port by the data-parallel rank, and vLLM zeroes that rank "
        "for a model that is not a mixture of experts (its ranks are "
        "independent, so it collapses them to DP=1), which makes every rank "
        "compute the same port. Data parallelism above one needs an MoE model "
        "on this connector.",
    ),
    (
        "kv connector dp size mismatch",
        r"conflicting data parallel size",
        "The data-parallel size in the connector's extra config does not match "
        "the one the engine resolved. It has to equal --data-parallel-size. If "
        "the engine says it expected 1 while the role declares more, the model "
        "is not a mixture of experts and vLLM has collapsed its data-parallel "
        "ranks to independent DP=1 engines — this connector cannot be used "
        "with data parallelism on such a model.",
    ),
    (
        "kv transport library missing",
        r"ascend_transport\.so: cannot open shared object file|"
        r"libmooncake[^\s]*\.so: cannot open shared object file",
        "The KV transport's own shared library is not on the loader's path in "
        "this image. Mooncake's engine.so has RPATH $ORIGIN and so looks for "
        "ascend_transport.so beside itself; a copy elsewhere on the filesystem "
        "is not found, and ldconfig will not cache it either because the name "
        "has no lib prefix. This is an image packaging fault, not a "
        "deployment one.",
    ),
    (
        "rdma unavailable",
        r"rdma_create_event_channel failed|rdma_create_id failed|"
        r"No RDMA devices found|ibv_open_device failed",
        "RDMA is not usable in this container. Check that the host has an HCA "
        "and that the container has IPC_LOCK and can see the RDMA devices. "
        "Without it the transport falls back to TCP if the connector allows "
        "one, and fails outright if it does not.",
    ),
    (
        "ucx no device",
        r"No such device.*tcp://|UCX_NET_DEVICES.*not found|"
        r"ZMQError: No such device",
        "The interface name the engine was given does not exist on this "
        "worker. If it looks like an unresolved template placeholder, the "
        "value could not be derived — set kv_ifname on this worker.",
    ),
    (
        "unresolved placeholder",
        r"\{\{[A-Za-z_][A-Za-z0-9_.]*\}\}",
        "A configuration value reached the engine unrendered. The placeholder "
        "below had no value at launch: a named port that was not allocated, or "
        "a network interface that could not be derived.",
    ),
    (
        "command not in image",
        r"exec .*failed: No such file or directory|"
        r"executable file not found|"
        r"[Cc]ommand not found|"
        r"No such file or directory: '[^']*'",
        "The image does not contain the command this member was started "
        "with, so it can never start rather than having failed to. A managed "
        "router is the usual case: the mode's recipe launches a router binary "
        "that the engine's runner image does not ship, and the fix is to give "
        "the router role an image of its own that carries it.",
    ),
    (
        "accelerator out of memory",
        r"torch\.(cuda\.)?OutOfMemoryError|torch_npu.*OutOfMemoryError|"
        r"CUDA out of memory|NPU out of memory|"
        r"RuntimeError: NPU error, error code is 507899|"
        r"out of memory\. Tried to allocate",
        "The accelerator ran out of memory while this member was starting. "
        "Read the group's total, not this member's: a disaggregated "
        "deployment puts three to five processes on the cards a single "
        "deployment would put one on, and each role's "
        "--gpu-memory-utilization is a fraction of the WHOLE card, not of "
        "what is left. Two roles both left at the default 0.9 means the one "
        "that starts first takes almost everything and the second gets this. "
        "Give each role an explicit fraction that sums to under 1.0 across "
        "the members sharing a card, or place the roles on separate cards.",
    ),
    (
        "no memory left for kv cache",
        r"Loaded weights leave no (GPU|NPU) memory for the KV cache|"
        r"No available memory for the cache blocks|"
        r"Free memory on device .* is less than desired GPU memory "
        r"utilization",
        "The weights loaded but nothing was left for the KV cache, so the "
        "engine cannot serve a single request. This is the same shortage as "
        "an out-of-memory kill, caught one step earlier and reported as a "
        "ValueError rather than a crash — which is why it reads as a "
        "configuration error and is in fact a sizing one. Either this role's "
        "share of the card is too small, or the members sharing the card "
        "over-committed it between them.",
    ),
    (
        "kv connector conflict",
        r"kv[-_]transfer[-_]config.*(specified|duplicate|already)|"
        r"multiple.*kv_connector",
        # Disaggregation and an extended KV cache DO coexist:
        # `worker/kv_transfer.py` folds the two descriptors into one
        # MultiConnector and vLLM creates the composite with both children. So
        # the cause is never "you asked for both". It is whatever got past the
        # fold, and the three ways that happens are what the summary sends the
        # operator to look for — in the order that tells them apart with one
        # grep each.
        #
        # Deliberately says nothing about which role asks which connector
        # first, nor about which role stands down from taking a cache: both
        # are decided elsewhere (`worker/kv_transfer.py` and the catalog's
        # `composed_cache`), and a second prose copy here drifts the moment
        # either changes.
        "More than one KV connector configuration reached the engine in a "
        "flag that carries one. A disaggregated role CAN also enable an "
        "extended KV cache: the disaggregation connector and the cache's are "
        "folded into a single MultiConnector, and that composite has been "
        "measured being instantiated by the engine. So this is the fold "
        "having been bypassed, and the worker's own log says which way. "
        "Search it for 'Composed N KV connectors into a MultiConnector'. If "
        "that line is missing, search for 'Leaving N --kv-transfer-config "
        "arguments unmerged': the fold declines when one of the values is not "
        "a JSON document — usually a descriptor edited by hand in this role's "
        "engine parameters, still carrying quotes that only a shell would "
        "have stripped — and it then leaves both flags in place on purpose, "
        "so the engine reports a duplicate rather than one connector silently "
        "going missing. If neither line is present, this member's command was "
        "not assembled by the vLLM path, which is the only one that folds: "
        "another engine, or a custom backend version, passes on every flag it "
        "was handed. And if the composition line IS present, then one flag is "
        "all the engine got and the duplication is inside it — read the "
        "connectors listed in kv_connector_extra_config.",
    ),
)

_COMPILED = tuple(
    (name, re.compile(pattern), summary) for name, pattern, summary in _SIGNATURES
)

_PORT_IN_LINE = re.compile(r"(?<!\d)(\d{4,5})(?!\d)")

_MAX_LINE = 400
"""Log lines from an engine can be enormous (a full config dump). The line is
carried for context, not for archival."""


def diagnose(
    log_text: Optional[str],
    named_ports: Optional[Mapping[str, object]] = None,
) -> Optional[Diagnosis]:
    """The earliest recognised failure in `log_text`, or None.

    Earliest, not last, and not "highest priority". A handshake failure is
    followed by a cascade of derived errors — the transport reports it, the
    scheduler mishandles the empty result, and the exception that escapes is
    an `IndexError` about a list. Only the first line in that sequence names
    something a user can act on.

    `named_ports` lets an "address already in use" be attributed to the band it
    belongs to, which turns "port 40031 is taken" into "the kv_side_channel
    band is taken" — the difference between a number and a thing to fix.
    """
    if not log_text:
        return None

    lines = log_text.splitlines()
    warned: Optional[Diagnosis] = None

    for raw_line in lines:
        for name, pattern, summary in _COMPILED:
            match = pattern.search(raw_line)
            if match is None:
                continue
            line = raw_line.strip()[:_MAX_LINE]
            detail = summary
            if name == "address already in use":
                band = _attribute_port(line, named_ports)
                if band:
                    detail = f"{summary} The port belongs to the '{band}' band."
            elif name == "unresolved placeholder":
                detail = f"{summary} Unrendered: {match.group(0)}."
            found = Diagnosis(signature=name, line=line, summary=detail)
            if not _is_warning(raw_line):
                return found
            # A warning is held back rather than returned. A startup warning
            # such as `No RDMA devices found` is benign on a host with no HCA —
            # the transport falls back to TCP and keeps moving KV — yet it can
            # sit seconds in front of the real death, e.g. `ValueError: Loaded
            # weights leave no GPU memory for the KV cache`. "Earliest match
            # wins" is right *within* a cascade of errors; a warning is not part
            # of that cascade, it just happens to come first, and returning it
            # would rename a memory-sizing failure as an RDMA problem.
            if warned is None:
                warned = found
            break

    if warned is not None and not _has_unrecognised_fatal(lines):
        # No fatal error we failed to recognise, so the warning is the best
        # account of the failure available and is better than silence.
        return warned
    return None


# Severity markers, in the two shapes these engines emit: glog's `W0828 ...`
# / `E0828 ...` prefix (Mooncake, NIXL, the transfer engines) and Python's
# `WARNING:` / `ERROR:` (uvicorn, SGLang, vLLM).
_GLOG_WARNING = re.compile(r"^\s*W\d{4}\s")
_TEXT_WARNING = re.compile(r"\b(WARNING|WARN)\b\s*:?", re.IGNORECASE)
# What an unrecognised fatal looks like. Deliberately broad: the cost of a
# false positive here is only that a warning-level diagnosis is withheld and
# the caller reports the engine's own last error instead, which is never
# actively misleading.
_FATAL_MARKERS = re.compile(
    r"^\s*(Traceback \(most recent call last\)|"
    r"[A-Za-z_][A-Za-z0-9_.]*(Error|Exception)\s*:|"
    r"E\d{4}\s)"
)


def _is_warning(line: str) -> bool:
    if _GLOG_WARNING.search(line):
        return True
    # An `ERROR` on the same line wins: some loggers print both a level and a
    # message that happens to contain the word "warning".
    if re.search(r"^\s*E\d{4}\s|\bERROR\b", line):
        return False
    return bool(_TEXT_WARNING.search(line))


def _has_unrecognised_fatal(lines: Sequence[str]) -> bool:
    """Whether the log holds a fatal error no signature matched.

    The question a withheld warning turns on: if the engine died of something
    this module cannot summarise, saying nothing lets the caller report the
    engine's own words, which are at least about the right failure.
    """
    return any(_FATAL_MARKERS.search(line) for line in lines)


def _attribute_port(
    line: str, named_ports: Optional[Mapping[str, object]]
) -> Optional[str]:
    """Which declared band a port mentioned in `line` falls inside.

    Bands, not points: a connector derives several ports from one base, so the
    number in the log is often base+n rather than the base itself, and matching
    only the base would attribute nothing in exactly the cases where two
    members collided on a derived port.
    """
    if not named_ports:
        return None
    numbers = {int(n) for n in _PORT_IN_LINE.findall(line)}
    if not numbers:
        return None
    for name, band in named_ports.items():
        base = getattr(band, "base", None)
        count = getattr(band, "count", 1) or 1
        if base is None:
            continue
        if any(base <= number < base + count for number in numbers):
            return name
    return None


class RestartTracker:
    """Recognises a member that keeps restarting without ever serving.

    Stateful and per worker process, deliberately: the signal is a *rate*, and
    the instance row records only a cumulative count and the time of the last
    restart, which cannot distinguish "restarted twice in a year" from
    "restarting every four seconds".

    The condition is two facts together, and both are needed. Restarts alone
    are normal — a member may legitimately be replaced. Never having served
    alone is normal too, briefly. Restarting repeatedly while never having
    served is the shape every port-level failure takes, and it is the shape
    that otherwise reports itself as `starting` indefinitely.
    """

    def __init__(self, threshold: int = 3, window: timedelta = timedelta(minutes=5)):
        self._threshold = threshold
        self._window = window
        self._restarts: Dict[int, Deque[datetime]] = {}
        self._served: Dict[int, bool] = {}
        self._last_count: Dict[int, int] = {}

    def observe_running(self, instance_id: int) -> None:
        """Record that this member served at least once.

        Latching rather than clearing the history: a member that served and
        then began crash-looping is a different failure (it ran, so its
        configuration is not the problem) and must not be reported as one
        that never started.
        """
        self._served[instance_id] = True

    def observe_restart_count(
        self, instance_id: int, restart_count: int, now: datetime
    ) -> bool:
        """Feed the container's cumulative restart count. Returns True the
        moment this member should be considered failed rather than starting.

        Takes the cumulative count rather than an event because that is what a
        polling loop can actually observe; the increments between polls are
        what get timestamped. A count that went down (the workload was
        recreated) resets rather than underflowing.
        """
        previous = self._last_count.get(instance_id)
        self._last_count[instance_id] = restart_count

        if previous is None or restart_count < previous:
            # First observation, or a fresh workload. Neither is a restart we
            # witnessed, and counting it would attribute the previous
            # workload's history to this one.
            self._restarts.pop(instance_id, None)
            return False
        if restart_count == previous:
            return self._is_looping(instance_id, now)

        stamps = self._restarts.setdefault(instance_id, deque())
        for _ in range(restart_count - previous):
            stamps.append(now)
        return self._is_looping(instance_id, now)

    def _is_looping(self, instance_id: int, now: datetime) -> bool:
        if self._served.get(instance_id):
            return False
        stamps = self._restarts.get(instance_id)
        if not stamps:
            return False
        cutoff = now - self._window
        while stamps and stamps[0] < cutoff:
            stamps.popleft()
        return len(stamps) >= self._threshold

    def forget(self, instance_id: int) -> None:
        """Drop a member's history. Called when it is deleted, so a recreated
        instance reusing the id does not inherit a verdict."""
        self._restarts.pop(instance_id, None)
        self._served.pop(instance_id, None)
        self._last_count.pop(instance_id, None)
