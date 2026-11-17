"""``gpu_devices_view`` rows load with typed telemetry.

The view projects each ``gpu_devices`` entry out of the ``workers.status``
JSON with ``json_extract``, so the ``core``/``memory``/``network`` columns
reach SQLAlchemy as raw JSON. The ``GPUDevice`` mapper declares those columns
with ``pydantic_column_type``, which validates each row once at the load
boundary. Without that declaration the loaded row holds plain dicts, and every
``model_dump()`` of the row — which is what the ``/v2/gpu-devices`` list and
watch responses do — emits a ``PydanticSerializationUnexpectedValue`` warning
per dict field.
"""

import warnings

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack.routes.gpu_devices import to_gpu_device_public
from gpustack.schemas.gpu_devices import GPUDevice
from gpustack.schemas.stmt import worker_after_create_view_stmt_sqlite
from gpustack.schemas.workers import GPUCoreInfo, GPUNetworkInfo, MemoryInfo, Worker

DEVICE_STATUS = {
    "name": "Ascend910B",
    "vendor": "huawei",
    "type": "cuda",
    "index": 0,
    "device_index": 0,
    "uuid": "uuid-device-1",
    "core": {"total": 24, "utilization_rate": 0.0},
    "memory": {
        "total": 68719476736,
        "used": 65316847616,
        "allocated": 0,
        "is_unified_memory": False,
        "free": 3402629120,
    },
    "network": {"mac": "", "inet": "10.0.0.1", "netmask": "", "gateway": ""},
    "temperature": 41.0,
    "power": 400.0,
}

WORKER_STATUS = {
    "cpu": {"total": 96},
    "memory": {"total": 137438953472, "is_unified_memory": False},
    "gpu_devices": [DEVICE_STATUS],
}


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite://")
    async with engine.begin() as connection:
        await connection.run_sync(Worker.__table__.create)
        await connection.exec_driver_sql(worker_after_create_view_stmt_sqlite)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        session.add(
            Worker.model_validate(
                {
                    "id": 1,
                    "name": "worker-1",
                    "worker_uuid": "uuid-worker-1",
                    "hostname": "host-1",
                    "ip": "10.0.0.1",
                    "ifname": "eth0",
                    "port": 19002,
                    "cluster_id": 1,
                    "status": WORKER_STATUS,
                }
            )
        )
        await session.commit()
        yield session
    await engine.dispose()


async def load_devices(session):
    result = await session.exec(select(GPUDevice))
    return list(result.all())


def serializer_warnings(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [str(w.message) for w in caught if "Pydantic" in str(w.message)]


@pytest.mark.asyncio
async def test_view_rows_load_with_typed_telemetry(session):
    devices = await load_devices(session)

    assert len(devices) == 1
    device = devices[0]
    assert device.id == "worker-1:cuda:0"
    assert type(device.memory) is MemoryInfo
    assert type(device.core) is GPUCoreInfo
    assert type(device.network) is GPUNetworkInfo
    assert device.memory.used == 65316847616


@pytest.mark.asyncio
async def test_route_serialization_emits_no_serializer_warnings(session):
    devices = await load_devices(session)
    device = devices[0]

    caught = serializer_warnings(lambda: to_gpu_device_public(device, {0: 1024}))
    assert caught == []


@pytest.mark.asyncio
async def test_route_serialization_injects_allocated_for_device_index(session):
    devices = await load_devices(session)
    device = devices[0]

    public = to_gpu_device_public(device, {0: 1024})
    assert public.memory.allocated == 1024
