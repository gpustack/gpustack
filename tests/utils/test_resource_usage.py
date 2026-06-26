"""Unit tests for the dependency-free resource-usage helpers."""

from datetime import datetime

from gpustack.utils.resource_usage import (
    instance_resource_type,
    instance_sku,
    is_metered_phase,
    volume_sku,
    iter_utc_day_segments,
    iter_utc_hour_segments,
    parse_accelerator_count,
    parse_gpu_descriptor,
    parse_gpu_type,
    parse_gpu_vram_mib,
    parse_quantity_to_mib,
    parse_quantity_to_millicores,
    split_delta_across_utc_midnight,
)


def test_is_metered_phase():
    # Up / coming up / degraded / unknown → metered (reservation held).
    assert is_metered_phase("Pending")
    assert is_metered_phase("Ready")
    assert is_metered_phase("NotReady")
    assert is_metered_phase("Starting")  # reacquiring the GPU after a stop
    assert is_metered_phase("Unknown")
    assert is_metered_phase("Updating")
    # Brand-new (reconciler hasn't run) and failures → not metered.
    assert not is_metered_phase(None)
    assert not is_metered_phase("CreateFailed")
    assert not is_metered_phase("PersistentVolumeCreateFailed")
    assert not is_metered_phase("InitializeFailed")
    # Stop / delete lifecycle → not metered (accelerator released; the user is
    # not charged while stopped).
    assert not is_metered_phase("Stopping")
    assert not is_metered_phase("Stopped")
    assert not is_metered_phase("Deleting")


# Every GPUInstancePhase must be listed in exactly one of these. New phases
# land in neither, so test_every_phase_is_classified fails until someone makes
# a deliberate metering decision — that's the whole point (no silent drift to
# the "metered" default in is_metered_phase). When you add a phase to the
# schema, add it here too: METERED if it holds the accelerator, NON_METERED if
# it has released it.
_EXPECTED_METERED_PHASES = frozenset(
    {
        "Ready",  # up
        "NotReady",  # up, degraded
        "Starting",  # coming up — reacquiring the reservation
        "Unknown",  # status unreadable → assume still allocated (conservative)
    }
)


def test_non_metered_phases_match_schema():
    """The non-metered phase literals are copies of GPUInstancePhase values
    (kept literal to keep utils import-light). Lock them so a rename on the
    schema side can't silently drift the metering predicate."""
    from gpustack.schemas.gpu_instances import GPUInstancePhase
    from gpustack.utils.resource_usage import _NON_METERED_PHASES

    valid = {
        v
        for k, v in vars(GPUInstancePhase).items()
        if not k.startswith("_") and isinstance(v, str)
    }
    assert _NON_METERED_PHASES <= valid, _NON_METERED_PHASES - valid


def test_every_phase_is_classified():
    """Every GPUInstancePhase must be explicitly classified as metered or
    non-metered. A phase added to the schema but to neither set fails here —
    forcing a deliberate metering decision instead of silently defaulting to
    "metered" (is_metered_phase meters anything not in _NON_METERED_PHASES).

    Also locks the two sets disjoint, and verifies is_metered_phase agrees with
    the expected classification for every real schema phase."""
    from gpustack.schemas.gpu_instances import GPUInstancePhase
    from gpustack.utils.resource_usage import _NON_METERED_PHASES

    all_phases = {
        v
        for k, v in vars(GPUInstancePhase).items()
        if not k.startswith("_") and isinstance(v, str)
    }

    unclassified = all_phases - _NON_METERED_PHASES - _EXPECTED_METERED_PHASES
    assert not unclassified, (
        f"new GPUInstancePhase value(s) need a metering decision — add to "
        f"_EXPECTED_METERED_PHASES (holds accelerator) or _NON_METERED_PHASES "
        f"(released): {sorted(unclassified)}"
    )

    # The two classifications must not overlap.
    overlap = _NON_METERED_PHASES & _EXPECTED_METERED_PHASES
    assert (
        not overlap
    ), f"phase(s) classified as both metered and not: {sorted(overlap)}"

    # And the predicate must agree with the classification for every phase.
    for phase in all_phases:
        assert is_metered_phase(phase) == (phase in _EXPECTED_METERED_PHASES), phase


def test_parse_gpu_vram_mib():
    # JSON-string descriptor (as stored on GPUInstance.description)
    assert parse_gpu_vram_mib('{"spec": {"memory": "48Gi"}}') == 48 * 1024
    # already-parsed dict
    assert parse_gpu_vram_mib({"spec": {"memory": "80Gi"}}) == 80 * 1024
    # missing / malformed / CPU (no descriptor) → 0
    assert parse_gpu_vram_mib(None) == 0
    assert parse_gpu_vram_mib("not json") == 0
    assert parse_gpu_vram_mib('{"spec": {}}') == 0
    assert parse_gpu_vram_mib("{}") == 0


def test_parse_gpu_type():
    assert parse_gpu_type("gpustack-nvidia-geforce-rtx-4090-c9bjn") == (
        "nvidia-geforce-rtx-4090",
        "nvidia",
    )
    assert parse_gpu_type("gpustack-amd-mi300x-ab12c") == ("amd-mi300x", "amd")
    assert parse_gpu_type(None) == ("unknown", None)


def test_parse_quantity_to_mib():
    assert parse_quantity_to_mib("100Gi") == 100 * 1024
    assert parse_quantity_to_mib("2048Mi") == 2048
    assert parse_quantity_to_mib("512Ki") == 0  # < 1 MiB rounds down
    assert parse_quantity_to_mib(None) == 0
    assert parse_quantity_to_mib("") == 0


def test_parse_quantity_to_millicores():
    assert parse_quantity_to_millicores("2") == 2000
    assert parse_quantity_to_millicores("500m") == 500
    assert parse_quantity_to_millicores(None) == 0


def test_parse_accelerator_count():
    assert parse_accelerator_count("2") == 2
    assert parse_accelerator_count(None) == 0
    assert parse_accelerator_count("") == 0


def test_instance_resource_type():
    assert instance_resource_type(2) == "gpu_instance"
    assert instance_resource_type(0) == "cpu_instance"
    assert instance_resource_type(None) == "cpu_instance"


def test_instance_sku():
    # sku = the instance spec's ``type`` verbatim, GPU or CPU alike.
    assert (
        instance_sku("gpustack-nvidia-h100-ab12c", "nvidia-h100", 2, 8000, 128000)
        == "gpustack-nvidia-h100-ab12c"
    )
    assert (
        instance_sku("gpustack--generic-31-ln-a64-1c-2g", None, 0, 1000, 2048)
        == "gpustack--generic-31-ln-a64-1c-2g"
    )
    # Fallbacks for snapshots missing ``type``:
    # GPU instances → gpu_type (count carried separately)
    assert instance_sku(None, "nvidia-h100", 2, 8000, 128000) == "nvidia-h100"
    assert instance_sku(None, "nvidia-h100", 1, 4000, 64000) == "nvidia-h100"
    # CPU instances → derived cpu flavor
    assert instance_sku(None, None, 0, 2000, 8192) == "cpu-2vcpu-8g"


def test_volume_sku():
    # volume--<category>--<type_name> — all volumes of a type share one sku, so
    # "by type" breakdown still aggregates per storage type (issue #5716).
    assert volume_sku("nfs", "aws") == "volume--nfs--aws"
    assert volume_sku("s3", "minio") == "volume--s3--minio"


def test_split_delta_across_utc_midnight():
    segs = split_delta_across_utc_midnight(
        datetime(2026, 5, 28, 23, 59, 30), datetime(2026, 5, 29, 0, 0, 30)
    )
    assert segs == [
        (datetime(2026, 5, 28).date(), 30),
        (datetime(2026, 5, 29).date(), 30),
    ]
    # non-positive window → empty
    assert (
        split_delta_across_utc_midnight(
            datetime(2026, 5, 28, 10), datetime(2026, 5, 28, 10)
        )
        == []
    )


def test_iter_utc_day_segments_bounds():
    segs = iter_utc_day_segments(
        datetime(2026, 5, 28, 23, 59, 30), datetime(2026, 5, 29, 0, 0, 30)
    )
    assert segs == [
        (
            datetime(2026, 5, 28).date(),
            datetime(2026, 5, 28, 23, 59, 30),
            datetime(2026, 5, 29, 0, 0, 0),
        ),
        (
            datetime(2026, 5, 29).date(),
            datetime(2026, 5, 29, 0, 0, 0),
            datetime(2026, 5, 29, 0, 0, 30),
        ),
    ]
    # total seconds preserved
    total = sum(int((e - s).total_seconds()) for _, s, e in segs)
    assert total == 60


def test_iter_utc_hour_segments_bounds():
    segs = iter_utc_hour_segments(
        datetime(2026, 5, 26, 10, 59, 30), datetime(2026, 5, 26, 11, 0, 30)
    )
    assert segs == [
        (
            datetime(2026, 5, 26, 10, 0, 0),
            datetime(2026, 5, 26, 10, 59, 30),
            datetime(2026, 5, 26, 11, 0, 0),
        ),
        (
            datetime(2026, 5, 26, 11, 0, 0),
            datetime(2026, 5, 26, 11, 0, 0),
            datetime(2026, 5, 26, 11, 0, 30),
        ),
    ]
    # bucket_start is hour-truncated; total seconds preserved
    assert sum(int((e - s).total_seconds()) for _, s, e in segs) == 60
    # within one hour → single bucket
    one = iter_utc_hour_segments(
        datetime(2026, 5, 26, 10, 0, 0), datetime(2026, 5, 26, 10, 30, 0)
    )
    assert len(one) == 1 and one[0][0] == datetime(2026, 5, 26, 10, 0, 0)


def test_local_resource_type_constants_match_schema():
    """The utils keep private copies of the resource-type constants to stay
    import-light (no schema dependency). Lock them to the schema's so a future
    rename on one side can't silently drift the two apart."""
    from gpustack.schemas.metered_usage import (
        RESOURCE_TYPE_CPU_INSTANCE,
        RESOURCE_TYPE_GPU_INSTANCE,
    )
    from gpustack.utils.resource_usage import (
        _RESOURCE_TYPE_CPU_INSTANCE,
        _RESOURCE_TYPE_GPU_INSTANCE,
    )

    assert _RESOURCE_TYPE_GPU_INSTANCE == RESOURCE_TYPE_GPU_INSTANCE
    assert _RESOURCE_TYPE_CPU_INSTANCE == RESOURCE_TYPE_CPU_INSTANCE


def test_parse_gpu_descriptor():
    desc = (
        '{"name":"gpustack-nvidia-geforce-rtx-5090-d-18c-54gi",'
        '"spec":{"product":"NVIDIA-GeForce-RTX-5090-D","memory":"32607Mi",'
        '"unitResourcesParsed":{"cpu":{"cores":18},'
        '"ram":{"value":54,"unit":"Gi"}}}}'
    )
    out = parse_gpu_descriptor(desc)
    assert out["product"] == "NVIDIA-GeForce-RTX-5090-D"
    assert out["unit_cpu_milli"] == 18000
    assert out["unit_memory_mib"] == 54 * 1024
    assert out["vram_mib"] == 32607  # per-card VRAM from spec.memory

    # Raw ``unitResources`` strings are preferred (unambiguous k8s quantities).
    raw = parse_gpu_descriptor(
        '{"spec":{"unitResources":{"cpu":"8000m","ram":"24576Mi"}}}'
    )
    assert raw["unit_cpu_milli"] == 8000
    assert raw["unit_memory_mib"] == 24576

    # Inconsistent ``unitResourcesParsed.ram`` (value/unit disagree with num,
    # e.g. value=24 + unit="Mi" for a 24Gi card) — trust ``num`` over ``value``.
    inconsistent = parse_gpu_descriptor(
        '{"spec":{"unitResourcesParsed":'
        '{"cpu":{"cores":8},"ram":{"value":24,"unit":"Mi","num":24576}}}}'
    )
    assert inconsistent["unit_cpu_milli"] == 8000
    assert inconsistent["unit_memory_mib"] == 24576

    # accepts an already-parsed dict too
    assert parse_gpu_descriptor({"spec": {"product": "X"}}) == {"product": "X"}
    # missing / unparseable → empty (e.g. CPU instances with no descriptor)
    assert parse_gpu_descriptor(None) == {}
    assert parse_gpu_descriptor("not json") == {}
    assert parse_gpu_descriptor({}) == {}
