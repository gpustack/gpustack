"""Turning a disaggregated member's failure into something actionable.

Two measured problems shape this, and both tests below are written against
them rather than against the code.

`_handle_failed_transfer` raises `IndexError: list index out of range` while
the real reason was logged earlier, so reporting the last exception reports the
symptom of the symptom. It was first assumed this only happened on a
tensor-parallelism mismatch; it happens on any handshake failure. Hence
earliest-match-wins, which is the single property most of these tests exist to
pin.

And every port-level failure produces the same shape — bind, fail, exit,
restart, `starting` forever — so a member restarting repeatedly without ever
serving has failed, whatever its state column says.
"""

from datetime import datetime, timedelta, timezone

from gpustack.schemas.models import PortBand
from gpustack.worker.pd_diagnostics import RestartTracker, diagnose

T0 = datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc)


# --- earliest match wins --------------------------------------------------- #


def test_the_root_cause_beats_the_exception_that_escaped():
    """The measured case. `IndexError` is what propagates out of vLLM, and it
    says nothing a user can act on."""
    log = "\n".join(
        [
            "INFO loading model",
            "ERROR NIXL_ERR_BACKEND: failed to load remote metadata",
            "ERROR Traceback (most recent call last):",
            "ERROR   File nixl_connector.py, in _handle_failed_transfer",
            "ERROR IndexError: list index out of range",
        ]
    )
    found = diagnose(log)

    assert found.signature == "NIXL_ERR_BACKEND"
    assert "kv_ifname" in found.summary


def test_the_earliest_signature_wins_over_a_later_one():
    """Not a priority order — the first line in a cascade is the one that
    caused the rest."""
    log = "\n".join(
        [
            "ERROR Address already in use: 0.0.0.0:40031",
            "ERROR NIXL_ERR_BACKEND: handshake failed",
        ]
    )
    assert diagnose(log).signature == "address already in use"

    reordered = "\n".join(
        [
            "ERROR NIXL_ERR_BACKEND: handshake failed",
            "ERROR Address already in use: 0.0.0.0:40031",
        ]
    )
    assert diagnose(reordered).signature == "NIXL_ERR_BACKEND"


def test_a_clean_log_diagnoses_nothing():
    assert diagnose("INFO started\nINFO listening on 40027") is None
    assert diagnose("") is None
    assert diagnose(None) is None


# --- the signatures -------------------------------------------------------- #


def test_a_hash_mismatch_points_at_the_factors_that_are_hashed():
    found = diagnose("ERROR kv compatibility hash mismatch with remote engine")

    assert found.signature == "compatibility hash mismatch"
    assert "block size" in found.summary


def test_rdma_unavailability_points_at_the_container_capability():
    found = diagnose("ERROR rdma_create_event_channel failed: No such device")

    assert found.signature == "rdma unavailable"
    assert "IPC_LOCK" in found.summary


def test_the_measured_zmq_error_is_recognised():
    """The M0 failure verbatim: a template placeholder that reached the engine
    unrendered, reported as a device error."""
    found = diagnose("ZMQError: No such device (addr='tcp://{{worker_ip}}:5600')")

    assert found is not None
    assert "kv_ifname" in found.summary or "placeholder" in found.summary


def test_an_unrendered_placeholder_is_named():
    found = diagnose("INFO UCX_NET_DEVICES={{net_device}}")

    assert found.signature == "unresolved placeholder"
    assert "{{net_device}}" in found.summary


def test_two_kv_transfer_configs_are_recognised():
    """The engine accepts exactly one, and GPUStack folds the two it knows
    about into a MultiConnector — so two arriving here means the fold was
    bypassed, and the summary has to name each way that happens rather than
    stopping at the first.
    """
    found = diagnose("ERROR --kv-transfer-config already specified")

    assert found.signature == "kv connector conflict"
    assert "MultiConnector" in found.summary
    # The two log lines that tell the causes apart, quoted closely enough to
    # be grep-able, and the third case where one flag is all the engine got.
    assert "Composed N KV connectors into a MultiConnector" in found.summary
    assert "Leaving N --kv-transfer-config arguments unmerged" in found.summary
    assert "kv_connector_extra_config" in found.summary


def test_the_kv_connector_conflict_does_not_call_the_combination_impossible():
    """Disaggregation and a shared cache are a supported combination, not one
    the user has to give up: `worker/kv_transfer.py` composes the two and a
    prefill runs the composite, so no phrasing that refuses the pair may appear
    in the text the operator reads."""
    found = diagnose("ERROR --kv-transfer-config already specified")

    assert "cannot also enable" not in found.summary
    assert "does not assemble" not in found.summary
    assert "CAN also enable an extended KV cache" in found.summary


# --- attributing a port to its band ---------------------------------------- #


def test_a_taken_port_is_attributed_to_its_band():
    """ "Port 40031 is taken" is a number; "the kv_side_channel band is taken"
    is something to fix."""
    found = diagnose(
        "ERROR Address already in use: 192.168.50.15:40031",
        named_ports={"kv_side_channel": PortBand(base=40031, count=1)},
    )

    assert "kv_side_channel" in found.summary


def test_a_derived_port_is_attributed_to_the_band_it_falls_inside():
    """The number in the log is usually base+n, not the base: a connector
    derives several ports from one. Matching only the base would attribute
    nothing in exactly the cases where two members collided."""
    found = diagnose(
        "ERROR Address already in use: 41107",
        named_ports={"kv_port": PortBand(base=41100, count=8)},
    )

    assert "kv_port" in found.summary


def test_a_port_outside_every_band_is_not_attributed():
    """An engine-chosen port GPUStack cannot reserve. Naming a band it does not
    belong to would send the reader to the wrong setting."""
    found = diagnose(
        "ERROR Address already in use: 15051",
        named_ports={"kv_port": PortBand(base=41100, count=8)},
    )

    assert found.signature == "address already in use"
    assert "kv_port" not in found.summary


# --- crash-loop detection -------------------------------------------------- #


def test_repeated_restarts_without_serving_are_a_failure():
    tracker = RestartTracker(threshold=3, window=timedelta(minutes=5))

    assert tracker.observe_restart_count(1, 0, T0) is False
    assert tracker.observe_restart_count(1, 1, T0 + timedelta(seconds=10)) is False
    assert tracker.observe_restart_count(1, 2, T0 + timedelta(seconds=20)) is False
    assert tracker.observe_restart_count(1, 3, T0 + timedelta(seconds=30)) is True


def test_a_member_that_served_is_never_reported_as_never_started():
    """A member that ran and then began crash-looping is a different failure:
    it ran, so its configuration is not the problem."""
    tracker = RestartTracker(threshold=2, window=timedelta(minutes=5))
    tracker.observe_restart_count(1, 0, T0)
    tracker.observe_running(1)

    assert tracker.observe_restart_count(1, 1, T0 + timedelta(seconds=5)) is False
    assert tracker.observe_restart_count(1, 9, T0 + timedelta(seconds=10)) is False


def test_restarts_spread_out_are_not_a_loop():
    """Restarts alone are normal — a member may legitimately be replaced. The
    signal is a rate, which is why a cumulative count on the row cannot carry
    it."""
    tracker = RestartTracker(threshold=3, window=timedelta(minutes=5))
    tracker.observe_restart_count(1, 0, T0)

    assert tracker.observe_restart_count(1, 1, T0 + timedelta(minutes=10)) is False
    assert tracker.observe_restart_count(1, 2, T0 + timedelta(minutes=20)) is False
    assert tracker.observe_restart_count(1, 3, T0 + timedelta(minutes=30)) is False


def test_the_first_observation_is_not_counted_as_a_restart():
    """A worker restarting mid-loop inherits a non-zero count it did not
    witness, and counting it would attribute the previous history to now."""
    tracker = RestartTracker(threshold=1, window=timedelta(minutes=5))

    assert tracker.observe_restart_count(1, 7, T0) is False


def test_a_recreated_workload_resets_rather_than_underflowing():
    tracker = RestartTracker(threshold=2, window=timedelta(minutes=5))
    tracker.observe_restart_count(1, 5, T0)
    tracker.observe_restart_count(1, 6, T0 + timedelta(seconds=5))

    # Workload recreated: the count starts over.
    assert tracker.observe_restart_count(1, 0, T0 + timedelta(seconds=10)) is False
    assert tracker.observe_restart_count(1, 1, T0 + timedelta(seconds=15)) is False


def test_an_unchanged_count_does_not_add_a_restart():
    """A polling loop observes the same count many times between restarts."""
    tracker = RestartTracker(threshold=2, window=timedelta(minutes=5))
    tracker.observe_restart_count(1, 0, T0)
    tracker.observe_restart_count(1, 1, T0 + timedelta(seconds=5))
    for i in range(20):
        looping = tracker.observe_restart_count(1, 1, T0 + timedelta(seconds=10 + i))
    assert looping is False


def test_forgetting_a_member_clears_its_verdict():
    """A recreated instance reusing the id must not inherit one."""
    tracker = RestartTracker(threshold=2, window=timedelta(minutes=5))
    tracker.observe_restart_count(1, 0, T0)
    tracker.observe_restart_count(1, 1, T0 + timedelta(seconds=5))
    assert tracker.observe_restart_count(1, 2, T0 + timedelta(seconds=10)) is True

    tracker.forget(1)
    assert tracker.observe_restart_count(1, 0, T0 + timedelta(seconds=15)) is False


def test_members_are_tracked_independently():
    tracker = RestartTracker(threshold=2, window=timedelta(minutes=5))
    for instance_id in (1, 2):
        tracker.observe_restart_count(instance_id, 0, T0)
    tracker.observe_restart_count(1, 1, T0 + timedelta(seconds=5))

    assert tracker.observe_restart_count(1, 2, T0 + timedelta(seconds=10)) is True
    assert tracker.observe_restart_count(2, 1, T0 + timedelta(seconds=10)) is False


def test_a_command_the_image_lacks_is_named_as_such():
    """`exit code 127` is the symptom; "the image has no such binary" is the
    cause, and it is one the operator can act on. Observed verbatim from a
    managed router launched on an engine runner image that does not ship
    `vllm-router` — the single biggest thing standing between vLLM PD and
    working out of the box."""
    log = "[FATAL tini (65)] exec vllm-router failed: No such file or directory"

    diagnosis = diagnose(log, None)

    assert diagnosis is not None
    assert diagnosis.signature == "command not in image"
    assert "image of its own" in diagnosis.summary


def test_the_other_shells_wording_is_caught_too():
    """Not every runtime uses tini, and the phrasing differs per runtime."""
    for log in (
        "bash: vllm-router: command not found",
        'exec: "vllm-router": executable file not found in $PATH',
    ):
        assert diagnose(log, None) is not None, log


def test_a_members_ranks_colliding_with_themselves_is_explained():
    """Measured twice, on our runner image and on the vendor's: a dense model
    with DP>1 has every rank compute the same Mooncake handshake port because
    vLLM zeroes data_parallel_rank for non-MoE models. On vllm-ascend v0.23.0
    the process does not even exit -- it sits there with no serving port, which
    is the `starting` forever that this whole diagnostic path exists for."""
    diagnosis = diagnose(
        "ERROR mooncake_connector.py:269 Mooncake KVCacheSendingThread "
        "encountered exception. Thread: tp_rank=0, pp_rank=0, "
        "listening_path=tcp://192.168.13.3:20001. "
        "Error: Address already in use (addr='tcp://192.168.13.3:20001')"
    )

    assert diagnosis is not None
    assert diagnosis.signature == "address already in use"
    assert "mixture of experts" in diagnosis.summary


def test_a_conflicting_dp_size_in_the_connector_config_is_recognised():
    diagnosis = diagnose(
        "ValueError: KV transfer 'prefill' config has a conflicting data "
        "parallel size. Expected 1, but got 2."
    )

    assert diagnosis is not None
    assert diagnosis.signature == "kv connector dp size mismatch"
    assert "--data-parallel-size" in diagnosis.summary


def test_a_missing_transport_library_is_named_as_a_packaging_fault():
    """Hit on our own runner image: the file ships, but at /usr/local/lib
    instead of beside mooncake's engine.so, and engine.so's RPATH is $ORIGIN."""
    diagnosis = diagnose(
        "RuntimeError: Worker failed with error 'ascend_transport.so: "
        "cannot open shared object file: No such file or directory'"
    )

    assert diagnosis is not None
    assert diagnosis.signature == "kv transport library missing"
    assert "packaging" in diagnosis.summary


def test_a_failed_kv_transfer_is_recognised():
    """The worst failure this module exists for, because nothing else sees it.

    When a cross-host transfer raises, decode generates from blocks it never
    received, the engine reports an external prefix-cache hit rate of 100%, and
    the caller gets a 200 whose content is a single token repeated to the length
    limit. Mooncake exports no Prometheus counter, so the transfer metrics
    cannot see it either — this log line is the only evidence that exists."""
    log = (
        "(EngineCore pid=136) ERROR 08-27 01:01:36 [mooncake_hybrid_connector.py:463] "
        "RuntimeError: Mooncake transfer failed, ret: -1"
    )

    diagnosis = diagnose(log)

    assert diagnosis is not None
    assert diagnosis.signature == "kv transfer failed"
    assert "200" in diagnosis.summary


def test_a_healthy_log_is_not_diagnosed_as_a_failed_transfer():
    log = (
        "(EngineCore) [mooncake_hybrid_connector.py:813] KV cache transfer took 2.02 ms"
    )

    assert diagnose(log) is None


# ---------------------------------------------------------------------------
# Severity: "earliest match wins" is right within a cascade, not across one.
# ---------------------------------------------------------------------------

_REAL_SGLANG_PREFILL_LOG = """\
E0828 01:41:15.002852    73 transfer_metadata.cpp:877] Local segment descriptor not found
W0828 01:41:15.002805    73 topology.cpp:156] No RDMA devices found, check your device installation
I0828 01:41:15.002866    73 tcp_transport.cpp:553] TcpTransport: listen on port 16792
[2026-08-28 01:41:17] Load weight end. elapsed=0.60 s
[2026-08-28 01:41:18] Scheduler hit an exception: Traceback (most recent call last):
  File "/sgl-workspace/sglang/python/sglang/srt/managers/scheduler.py", line 4325
    raise ValueError(
ValueError: Loaded weights leave no GPU memory for the KV cache under --mem-fraction-static=0.5.
"""


def test_a_warning_does_not_get_reported_as_the_cause_of_a_fatal_error():
    """The regression this exists for, captured verbatim from a live run.

    The engine logged `No RDMA devices found` — benign on a host with no HCA,
    where the transport falls back to TCP and had already moved 276 KV
    transfers successfully — and died three seconds later of a memory-sizing
    ValueError. "Earliest recognised failure" returned the warning, so the
    instance's `state_message` blamed RDMA for an out-of-memory failure and
    sent whoever read it to check for an HCA.

    The rule stands *within* a cascade of derived errors, which is what it was
    written for. A warning is not part of that cascade; it merely comes first.

    The memory-sizing failure now has a signature of its own, so this log
    resolves to it rather than to silence. What the test still pins is the
    property the regression was about: whatever comes back, it is not the RDMA
    warning.
    """
    result = diagnose(_REAL_SGLANG_PREFILL_LOG)
    assert result is not None
    assert result.signature == "no memory left for kv cache"
    assert "--mem-fraction-static" in result.line


def test_a_warning_is_still_reported_when_nothing_fatal_is_hiding_behind_it():
    """Withholding it unconditionally would throw away a real diagnosis. The
    warning is the best account available when the log holds no fatal error
    this module failed to recognise."""
    log = (
        "W0828 01:41:15.002805 73 topology.cpp:156] No RDMA devices found\n"
        "[2026-08-28 01:41:20] Application startup complete.\n"
    )
    result = diagnose(log)
    assert result is not None
    assert result.signature == "rdma unavailable"


def test_an_error_level_match_still_wins_and_still_wins_earliest():
    """The original behaviour, unchanged: among error-level lines the first
    one is the actionable one, because everything after it is derived."""
    log = (
        "E0828 01:41:15 nixl] NIXL_ERR_BACKEND handshake failed\n"
        "Traceback (most recent call last):\n"
        "IndexError: list index out of range\n"
    )
    result = diagnose(log)
    assert result is not None
    assert result.signature == "NIXL_ERR_BACKEND"


def test_the_word_error_on_a_line_outranks_the_word_warning():
    """Some loggers put both on one line. Treating such a line as a warning
    would reintroduce the masking this fix removes."""
    from gpustack.worker.pd_diagnostics import _is_warning

    assert _is_warning("W0828 01:41:15 topology.cpp:156] No RDMA devices found")
    assert _is_warning("WARNING: rdma_create_event_channel failed")
    assert not _is_warning("ERROR: warning threshold exceeded, rdma_create_id failed")
    assert not _is_warning("E0828 01:41:15 nixl] NIXL_ERR_BACKEND")


# --- running out of accelerator memory ------------------------------------- #


def test_an_oom_exit_is_explained_instead_of_reported_as_exit_code_1():
    """A member that dies of OOM otherwise carries a `state_message` of
    `Error (exit code 1)` with the reason only in the log, while a router's
    `exit 127` beside it carries a paragraph of actionable text. OOM is the
    commonest deployment failure there is, so it must be the one explained
    best."""
    log = (
        "INFO 09-01 11:20:03 [gpu_worker.py:298] Starting to load model...\n"
        "torch.OutOfMemoryError: GPU 0 has a total capacity of 47.37 GiB of "
        "which 147.50 MiB is free.\n"
    )
    result = diagnose(log)
    assert result is not None
    assert result.signature == "accelerator out of memory"
    assert "147.50 MiB is free" in result.line


def test_the_oom_summary_names_what_pd_does_to_the_arithmetic():
    """Not a restatement of the error. The reason this failure keeps happening
    on PD specifically is that each role's --gpu-memory-utilization is a
    fraction of the whole card, so two roles both left at the default 0.9
    over-commit it — and a group puts three to five processes where a plain
    deployment puts one."""
    result = diagnose("torch.OutOfMemoryError: CUDA out of memory.\n")
    assert "gpu-memory-utilization" in result.summary


def test_the_npu_form_is_recognised_too():
    """The Ascend deployments are where a group is most likely to be packed
    onto shared cards, and the engine words it differently there."""
    result = diagnose("RuntimeError: NPU out of memory. Tried to allocate 2.00 GiB\n")
    assert result is not None
    assert result.signature == "accelerator out of memory"


def test_weights_that_leave_no_room_for_kv_is_the_same_shortage_one_step_earlier():
    """It surfaces as a ValueError, which reads like a configuration mistake
    and is in fact a sizing one — so it gets its own summary rather than being
    folded into the OOM text."""
    result = diagnose(
        "ValueError: No available memory for the cache blocks. Try increasing "
        "gpu_memory_utilization when initializing the engine.\n"
    )
    assert result is not None
    assert result.signature == "no memory left for kv cache"


def test_a_handshake_failure_still_outranks_a_later_oom():
    """Earliest-match-wins is not weakened by adding these. A connector that
    failed to hand shake and then died of memory is a handshake failure; the
    OOM is downstream of it."""
    log = (
        "ERROR 09-01 11:20:03 NIXL_ERR_BACKEND creating backend\n"
        "torch.OutOfMemoryError: CUDA out of memory.\n"
    )
    assert diagnose(log).signature == "NIXL_ERR_BACKEND"
