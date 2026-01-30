from typing import List, Optional, Tuple

from gpustack.schemas.benchmark import (
    GPUSnapshot,
    GPUSnapshots,
    ModelInstanceRuntimeInfo,
    ModelInstanceSnapshot,
    WorkerSnapshot,
)
from gpustack.schemas.models import Model, ModelInstance
from gpustack.schemas.workers import Worker
from gpustack.utils.gpu import make_gpu_id


def create_model_instance_snapshot(
    model_instance: ModelInstance, model: Model
) -> ModelInstanceSnapshot:
    """Create a snapshot of the model instance."""

    subordinate_workers_snapshots: Optional[List[ModelInstanceRuntimeInfo]] = None
    if (
        model_instance.distributed_servers
        and model_instance.distributed_servers.subordinate_workers
    ):
        subordinate_workers_snapshots = []
        for subworker in model_instance.distributed_servers.subordinate_workers:
            subordinate_workers_snapshots.append(
                ModelInstanceRuntimeInfo(
                    worker_id=subworker.worker_id,
                    worker_name=subworker.worker_name,
                    worker_ip=subworker.worker_ip,
                    ports=subworker.ports,
                    gpu_type=subworker.gpu_type,
                    gpu_indexes=subworker.gpu_indexes,
                    gpu_ids=[
                        make_gpu_id(subworker.worker_name, subworker.gpu_type, idx)
                        for idx in (subworker.gpu_indexes or [])
                    ],
                    computed_resource_claim=subworker.computed_resource_claim,
                )
            )

    return ModelInstanceSnapshot(
        id=model_instance.id,
        name=model_instance.name,
        state=model_instance.state,
        state_message=model_instance.state_message,
        worker_id=model_instance.worker_id,
        worker_name=model_instance.worker_name,
        worker_ip=model_instance.worker_ip,
        ports=model_instance.ports,
        gpu_type=model_instance.gpu_type,
        gpu_indexes=model_instance.gpu_indexes,
        gpu_ids=[
            make_gpu_id(model_instance.worker_name, model_instance.gpu_type, idx)
            for idx in (model_instance.gpu_indexes or [])
        ],
        computed_resource_claim=model_instance.computed_resource_claim,
        resolved_path=model_instance.resolved_path,
        subordinate_workers=subordinate_workers_snapshots,
        backend=model_instance.backend,
        backend_version=model_instance.backend_version,
        api_detected_backend_version=model_instance.api_detected_backend_version,
        backend_parameters=model.backend_parameters,
        env=model.env,
        image_name=model.image_name,
        run_command=model.run_command,
        extended_kv_cache=model.extended_kv_cache,
        speculative_config=model.speculative_config,
    )


def create_worker_snapshot(
    worker: Worker, gpu_type: str, gpu_indexes: List[int]
) -> Tuple[Optional[WorkerSnapshot], Optional[GPUSnapshots]]:
    worker_snapshot = WorkerSnapshot(
        id=worker.id,
        name=worker.name,
        os=worker.status.os if worker.status and worker.status.os else None,
        cpu_total=(
            worker.status.cpu.total if worker.status and worker.status.cpu else None
        ),
        memory_total=(
            worker.status.memory.total
            if worker.status and worker.status.memory
            else None
        ),
    )

    gpu_snapshots = None
    if worker.status and worker.status.gpu_devices:
        gpu_snapshots = {}
        for gpu_device in worker.status.gpu_devices:
            if gpu_device.type != gpu_type or gpu_device.index not in gpu_indexes:
                continue
            gpu_id = make_gpu_id(worker.name, gpu_device.type, gpu_device.index)
            gpu_snapshot = GPUSnapshot(
                id=gpu_id,
                worker_id=worker.id,
                worker_name=worker.name,
                vendor=gpu_device.vendor,
                type=gpu_device.type,
                index=gpu_device.index,
                device_index=gpu_device.device_index,
                device_chip_index=gpu_device.device_chip_index,
                arch_family=gpu_device.arch_family,
                name=gpu_device.name,
                uuid=gpu_device.uuid,
                driver_version=gpu_device.driver_version,
                runtime_version=gpu_device.runtime_version,
                compute_capability=gpu_device.compute_capability,
                core_total=gpu_device.core.total if gpu_device.core else None,
                memory_total=gpu_device.memory.total if gpu_device.memory else None,
            )
            gpu_snapshots[gpu_id] = gpu_snapshot

    return worker_snapshot, gpu_snapshots
