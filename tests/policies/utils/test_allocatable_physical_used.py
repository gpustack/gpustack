from gpustack.policies.utils import get_worker_allocatable_resource
from tests.fixtures.workers.fixtures import linux_nvidia_13_A100_80gx8
from tests.utils.model import new_model_instance
from gpustack.schemas.models import ComputedResourceClaim


def _set_gpu_used(worker, index, used_bytes):
    """Simulate external (non-GPUStack) processes occupying a device's VRAM."""
    for gpu in worker.status.gpu_devices:
        if gpu.index == index:
            gpu.memory.used = used_bytes
            if gpu.memory.total:
                gpu.memory.utilization_rate = used_bytes / gpu.memory.total * 100
            return


def test_allocatable_accounts_for_physical_used_without_instances():
    """Repro: with no GPUStack-managed instances, a GPU saturated by an
    external process used to be reported as fully allocatable. The fix makes
    the scheduler account for the physical `used` VRAM (i.e. does NOT ignore
    it) so the saturated device is no longer considered available."""
    worker = linux_nvidia_13_A100_80gx8()
    total = worker.status.gpu_devices[0].memory.total

    # GPU0 is ~fully used by an external process (e.g. independent container).
    _set_gpu_used(worker, 0, total - 1 * 1024**3)

    allocatable = get_worker_allocatable_resource([], worker)
    gpu0_alloc = allocatable.vram.get(0, 0)
    # We left exactly 1 GiB free on GPU0; anything above that means the
    # external usage was ignored.
    assert gpu0_alloc <= 1 * 1024**3, (
        f"GPU0 has ~all VRAM consumed externally but reported "
        f"{gpu0_alloc / 1024**3:.1f} GiB allocatable"
    )


def test_allocatable_free_gpu_keeps_full_capacity():
    worker = linux_nvidia_13_A100_80gx8()
    total = worker.status.gpu_devices[0].memory.total
    allocatable = get_worker_allocatable_resource([], worker)
    assert allocatable.vram.get(4, 0) >= total * 0.9


def test_allocatable_uses_max_of_claimed_and_physical_used():
    """An instance GPUStack already scheduled claims VRAM even before the
    engine has actually allocated it (physical used may lag), while an
    external process that holds more than the claim must still be respected."""
    worker = linux_nvidia_13_A100_80gx8()
    total = worker.status.gpu_devices[0].memory.total

    # Physical externally-used is 60 GiB on GPU0...
    _set_gpu_used(worker, 0, int(60 * 1024**3))
    # ...while a GPUStack instance only claims 40 GiB on the same device.
    mi = new_model_instance(
        1,
        name="test-instance",
        model_id=1,
        worker_id=worker.id,
        gpu_indexes=[0],
        computed_resource_claim=ComputedResourceClaim(
            vram={0: int(40 * 1024**3)}, ram=0
        ),
    )

    allocatable = get_worker_allocatable_resource([mi], worker)
    # Should reflect the bigger of the two (physical 60 GiB), not double-count.
    assert allocatable.vram.get(0, 0) == total - int(60 * 1024**3)
