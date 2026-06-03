"""Unit tests for the dependency-free resource-usage helpers."""

from datetime import datetime

from gpustack.utils.resource_usage import (
    instance_resource_type,
    instance_sku,
    is_metered_phase,
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
    assert is_metered_phase("Pending")
    assert is_metered_phase("Ready")
    assert is_metered_phase("Updating")
    assert not is_metered_phase(None)
    assert not is_metered_phase("CreateFailed")
    assert not is_metered_phase("PersistentVolumeCreateFailed")


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
    # GPU instances price per card → sku = gpu_type (count carried separately)
    assert instance_sku("nvidia-h100", 2, 8000, 128000) == "nvidia-h100"
    assert instance_sku("nvidia-h100", 1, 4000, 64000) == "nvidia-h100"
    # CPU instances price whole-machine → sku = cpu flavor
    assert instance_sku(None, 0, 2000, 8192) == "cpu-2vcpu-8g"


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

    # accepts an already-parsed dict too
    assert parse_gpu_descriptor({"spec": {"product": "X"}}) == {"product": "X"}
    # missing / unparseable → empty (e.g. CPU instances with no descriptor)
    assert parse_gpu_descriptor(None) == {}
    assert parse_gpu_descriptor("not json") == {}
    assert parse_gpu_descriptor({}) == {}
