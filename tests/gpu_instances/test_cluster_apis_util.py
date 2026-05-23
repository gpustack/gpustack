import pytest

from gpustack.gpu_instances.cluster_apis_util import (
    SUFFIX,
    spec_persistent_volume_type,
)
from gpustack.schemas.gpu_instance_persistent_volume_types import (
    GPUInstancePersistentVolumeNFS,
    GPUInstancePersistentVolumeType,
    GPUInstancePersistentVolumeTypeSpec,
    GPUInstancePersistentVolumeS3,
)


def _build_nfs_pvt(sub_directory):
    return GPUInstancePersistentVolumeType(
        name="test-pvt",
        spec=GPUInstancePersistentVolumeTypeSpec(
            nfs=GPUInstancePersistentVolumeNFS(
                server="10.0.0.1",
                share="/data",
                sub_directory=sub_directory,
            )
        ),
    )


def _build_s3_pvt(bucket):
    return GPUInstancePersistentVolumeType(
        name="test-pvt",
        spec=GPUInstancePersistentVolumeTypeSpec(
            s3=GPUInstancePersistentVolumeS3(
                endpoint="https://my-bucket.s3.us-east-1.amazonaws.com",
                region="us-east-1",
                bucket=bucket,
            ),
        ),
    )


@pytest.mark.parametrize(
    "pvt, principal_name, expected",
    [
        # No subDirectory specified, should default to "default/${pvc.metadata.name}"
        (
            _build_nfs_pvt(None),
            "default",
            f"default/{SUFFIX}",
        ),
        # User-specified subDirectory should be preserved and suffixed with "default/${pvc.metadata.name}"
        (
            _build_nfs_pvt("tenant-a"),
            "default",
            f"tenant-a/default/{SUFFIX}",
        ),
        # User-specified subDirectory with trailing slash should be normalized and suffixed with "default/${pvc.metadata.name}"
        (
            _build_nfs_pvt("tenant-a/"),
            "default",
            f"tenant-a/default/{SUFFIX}",
        ),
        # No bucket specified, should default to "default" with prefix "default/${pvc.metadata.name}"
        (
            _build_s3_pvt(None),
            "default",
            SUFFIX,
        ),
        # User-specified bucket should be preserved with prefix "default/${pvc.metadata.name}"
        (
            _build_s3_pvt("tenant-a"),
            "default",
            f"default/{SUFFIX}",
        ),
    ],
)
@pytest.mark.asyncio
async def test_spec_persistent_volume_type_inject(pvt, principal_name, expected):
    spec = spec_persistent_volume_type(pvt, principal_name)

    if pvt.spec.nfs:
        assert spec["nfs"]["subDirectory"] == expected
    else:
        assert spec["s3"]["prefix"] == expected
