"""Unit conversion in the metrics normalization layer.

`runtime_mapping` renames an engine's metric to the platform's name. Renaming
alone is only half a normalization: two engines can report the same concept in
different units, and one name over mixed units is worse than two names —
downstream can no longer tell which it has.

The case that forces this: SGLang reports KV transfer volume in MEGABYTES and
duration in MILLISECONDS where vLLM reports bytes and seconds.
"""

from prometheus_client.parser import text_string_to_metric_families

from gpustack.utils.metrics import get_builtin_metrics_config
from gpustack.worker.runtime_metrics_aggregator import (
    _parse_mapping_entry,
    get_unified_metric_family_name,
    scale_sample,
)

# An SGLang prefill's own exposition: 83 transfers, 189.0 MB total.
_REAL_SGLANG_HISTOGRAM = """\
# HELP sglang:kv_transfer_total_mb Histogram of KV transfer volume in MB.
# TYPE sglang:kv_transfer_total_mb histogram
sglang:kv_transfer_total_mb_bucket{le="1.0"} 4.0
sglang:kv_transfer_total_mb_bucket{le="4.0"} 60.0
sglang:kv_transfer_total_mb_bucket{le="+Inf"} 83.0
sglang:kv_transfer_total_mb_count 83.0
sglang:kv_transfer_total_mb_sum 189.0
"""

MB = 1048576


def test_the_existing_string_form_still_means_scale_one():
    """Every mapping in the shipped config is `raw: unified`. If the new
    object form changed what a bare string means, every metric in the platform
    would silently change value."""
    assert _parse_mapping_entry("gpustack:x") == ("gpustack:x", 1.0)
    assert _parse_mapping_entry({"name": "gpustack:x"}) == ("gpustack:x", 1.0)
    assert _parse_mapping_entry({"name": "gpustack:x", "scale": 0.001}) == (
        "gpustack:x",
        0.001,
    )
    assert _parse_mapping_entry(None) is None
    # A malformed entry degrades to no scaling rather than crashing the whole
    # aggregation pass for every other metric on the worker.
    assert _parse_mapping_entry({"name": "gpustack:x", "scale": "fast"}) == (
        "gpustack:x",
        1.0,
    )


def test_a_histogram_is_not_multiplied_through():
    """The trap this function exists for.

    A histogram's samples do not share a dimension: `_sum` is in the observed
    unit, `_count` and the bucket values are counts, and `le` is a boundary in
    the observed unit. The dangerous one is `le` — it is a *label*, so any
    implementation that scales "the value" misses it, nothing errors, and the
    histogram ends up with its sum in bytes and its boundaries in megabytes.
    Every `histogram_quantile` over it is then wrong by the scale factor.
    """
    scaled = {}
    for family in text_string_to_metric_families(_REAL_SGLANG_HISTOGRAM):
        for sample in family.samples:
            value, labels = scale_sample(
                sample.name,
                family.name,
                family.type,
                dict(sample.labels),
                sample.value,
                MB,
            )
            scaled[(sample.name, labels.get("le"))] = value

    # sum: megabytes -> bytes
    assert scaled[("sglang:kv_transfer_total_mb_sum", None)] == 189.0 * MB
    # count: an observation count, dimensionless
    assert scaled[("sglang:kv_transfer_total_mb_count", None)] == 83.0
    # bucket values: also counts
    assert scaled[("sglang:kv_transfer_total_mb_bucket", "1048576.0")] == 4.0
    assert scaled[("sglang:kv_transfer_total_mb_bucket", "4194304.0")] == 60.0
    # +Inf has no unit to convert
    assert scaled[("sglang:kv_transfer_total_mb_bucket", "+Inf")] == 83.0


def test_the_converted_histogram_is_internally_consistent():
    """The property that catches a half-applied conversion: mean transfer size
    computed from the converted sum and count must equal the mean computed
    from the engine's own numbers."""
    total_bytes = 189.0 * MB
    count = 83.0
    assert (total_bytes / count) / MB == 189.0 / 83.0


def test_the_shipped_config_declares_the_conversions_it_needs():
    """SGLang's PD histograms come in non-base units on every version it has
    used. A missing scale is silent — it produces a plausible number 1048576x
    too small — so the declaration is asserted rather than trusted."""
    config = get_builtin_metrics_config()

    # The same four families on every version SGLang has shipped them on, so a
    # version floor cannot quietly reintroduce the raw unit on one of them.
    expected = {
        "sglang:kv_transfer_total_mb": ("gpustack:pd_kv_transfer_bytes", MB),
        "sglang:kv_transfer_latency_ms": (
            "gpustack:pd_kv_transfer_latency_seconds",
            0.001,
        ),
        "sglang:kv_transfer_bootstrap_ms": ("gpustack:pd_kv_transfer_seconds", 0.001),
        "sglang:kv_transfer_alloc_ms": (
            "gpustack:pd_kv_transfer_alloc_seconds",
            0.001,
        ),
    }
    for engine_version in ("0.5.12.post1", "0.5.15.post1"):
        for raw, want in expected.items():
            got = get_unified_metric_family_name(config, raw, "SGLang", engine_version)
            assert got == want, f"{raw} on {engine_version}"

    # vLLM's are already bytes and seconds, so they must NOT carry a scale.
    for raw, unified in (
        ("vllm:nixl_bytes_transferred", "gpustack:pd_kv_transfer_bytes"),
        ("vllm:nixl_xfer_time_seconds", "gpustack:pd_kv_transfer_seconds"),
    ):
        name, scale = get_unified_metric_family_name(config, raw, "vLLM", None)
        assert (name, scale) == (unified, 1.0), raw


def test_no_unit_bearing_name_is_mapped_without_a_scale():
    """The whole class of bug, not just today's two: a raw name that announces
    a non-base unit (`_mb`, `_ms`, ...) mapped at scale 1 is the silent
    failure. This is what stops the next connector from reintroducing it."""
    config = get_builtin_metrics_config()
    suspicious = ("_mb", "_ms", "_us", "_kb", "_gb", "_gb_s")
    offenders = []
    for runtime, versions in config.get("runtime_mapping", {}).items():
        for _version, mapping in (versions or {}).items():
            for raw, entry in (mapping or {}).items():
                parsed = _parse_mapping_entry(entry)
                if parsed is None:
                    continue
                _name, scale = parsed
                if any(raw.endswith(s) or f"{s}_" in raw for s in suspicious):
                    if scale == 1.0:
                        offenders.append(f"{runtime}:{raw}")
    assert not offenders, f"mapped without a unit conversion: {offenders}"


def test_sglang_kv_transfer_histograms_stay_in_separate_unified_families():
    """SGLang exports its KV-transfer histograms *at once*, and they measure
    different things: the bootstrap handshake, the receive-buffer wait, the
    wire time and the volume. All of them are registered on every version from
    0.5.12 through current builds — none replaced any other.

    `kv_transfer_speed_gb_s` is deliberately left unmapped: it is the volume
    divided by the wire time, both of which are mapped, and it is present only
    on the connectors that report a size, which is exactly when the two
    operands are present too.

    Two of them mapped onto one unified name is not a cosmetic duplication:
    the aggregator appends samples to whichever family the name resolves to,
    so the exposition would carry the same series twice with different values
    and every count over it would be doubled.
    """
    config = get_builtin_metrics_config()
    raw_names = (
        "sglang:kv_transfer_bootstrap_ms",
        "sglang:kv_transfer_alloc_ms",
        "sglang:kv_transfer_latency_ms",
        "sglang:kv_transfer_total_mb",
    )
    for engine_version in ("0.5.12.post1", "0.5.15.post1"):
        unified = [
            get_unified_metric_family_name(config, raw, "SGLang", engine_version)[0]
            for raw in raw_names
        ]
        assert len(set(unified)) == len(raw_names), f"{engine_version}: {unified}"


def test_the_sglang_pd_signals_that_survive_a_silent_connector_are_mapped():
    """The queue depths and failure counters come from the scheduler, not from
    the connector, so they are the only per-stage PD evidence a deployment
    whose transport publishes no byte count has at all — which is every
    Mooncake group, the transport the Ascend recipes use.
    """
    config = get_builtin_metrics_config()
    expected = {
        "sglang:num_prefill_bootstrap_queue_reqs": (
            "gpustack:pd_prefill_bootstrap_queue_requests"
        ),
        "sglang:num_prefill_inflight_queue_reqs": (
            "gpustack:pd_prefill_inflight_queue_requests"
        ),
        "sglang:num_decode_prealloc_queue_reqs": (
            "gpustack:pd_decode_prealloc_queue_requests"
        ),
        "sglang:num_decode_transfer_queue_reqs": (
            "gpustack:pd_decode_transfer_queue_requests"
        ),
        "sglang:pending_prealloc_token_usage": (
            "gpustack:pd_decode_pending_prealloc_token_usage"
        ),
        # Counters: the engine registers them with a `_total` suffix, and the
        # aggregator keys on the parsed family name, which drops it.
        "sglang:num_transfer_failed_reqs": "gpustack:pd_kv_transfer_failed",
        "sglang:num_bootstrap_failed_reqs": "gpustack:pd_bootstrap_failed",
        "sglang:num_prefill_retries": "gpustack:pd_prefill_retries",
    }
    for raw, unified in expected.items():
        got = get_unified_metric_family_name(config, raw, "SGLang", "0.5.15.post1")
        # None of these bear a unit in their name, so they convert 1:1.
        assert got == (unified, 1.0), raw


def test_both_sglang_router_wheels_reach_the_same_unified_names():
    """`sglang_router.launch_router` resolves inside the model's own runner
    image, and the wheel there is one of two programs: images built on SGLang
    below 0.5.8 carry `sgl-router` (`sgl_router_*`), 0.5.8 and above carry
    `sgl-model-gateway` (`smg_*`). Only one answers on any one group, so both
    are mapped; the older one is the only one with per-worker PD counters.
    """
    config = get_builtin_metrics_config()
    expected = {
        "sgl_router_pd_requests": "gpustack:pd_router_requests",
        "sgl_router_pd_prefill_requests": "gpustack:pd_router_prefill_requests",
        "sgl_router_pd_decode_requests": "gpustack:pd_router_decode_requests",
        "sgl_router_pd_prefill_errors": "gpustack:pd_router_prefill_errors",
        "sgl_router_pd_decode_errors": "gpustack:pd_router_decode_errors",
        "smg_router_requests": "gpustack:pd_router_requests",
        "smg_worker_selection": "gpustack:pd_router_worker_selections",
        "smg_worker_requests_active": "gpustack:pd_router_worker_requests_active",
    }
    for raw, unified in expected.items():
        got = get_unified_metric_family_name(config, raw, "SGLang", "0.5.15.post1")
        assert got == (unified, 1.0), raw


def test_the_current_preemption_gauge_is_the_one_read_on_current_engines():
    """SGLang carries both `num_retracted_reqs` and `num_retracted_requests`
    on current builds. Only the first is mapped at `"*"`; the second stays
    behind a version floor, so the two never merge into one family.
    """
    config = get_builtin_metrics_config()
    assert get_unified_metric_family_name(
        config, "sglang:num_retracted_reqs", "SGLang", "0.5.15.post1"
    ) == ("gpustack:request_preemptions", 1.0)
    assert (
        get_unified_metric_family_name(
            config, "sglang:num_retracted_requests", "SGLang", "0.5.15.post1"
        )
        is None
    )


def test_sglang_uncached_prompt_tokens_is_not_folded_into_the_vllm_counter():
    """`sglang:uncached_prompt_tokens_histogram` is prompt tokens minus this
    engine's own prefix-cache hits. The vLLM family it would join,
    `request_prefill_kv_computed_tokens`, is read on the receiving role, where
    vLLM also subtracts the tokens that arrived over the connector. Mapping
    them together would make a healthy SGLang decode report its full prompt
    length as recomputed, and the p95/p99 degradation band would fire on every
    correct deployment.
    """
    config = get_builtin_metrics_config()
    assert (
        get_unified_metric_family_name(
            config,
            "sglang:uncached_prompt_tokens_histogram",
            "SGLang",
            "0.5.15.post1",
        )
        is None
    )
