from dataclasses import dataclass
from typing import Optional

from gpustack.schemas.models import ModelInstance


@dataclass(frozen=True)
class ModelInstanceWorkerMatch:
    is_main_worker: bool = False
    subordinate_worker_indexes: tuple[int, ...] = ()

    @property
    def matched(self) -> bool:
        return self.is_main_worker or bool(self.subordinate_worker_indexes)


def get_model_instance_worker_match(
    instance: ModelInstance,
    *,
    worker_name: Optional[str] = None,
    worker_id: Optional[int] = None,
) -> ModelInstanceWorkerMatch:
    is_main_worker = False
    if worker_name is not None and instance.worker_name == worker_name:
        is_main_worker = True
    if worker_id is not None and instance.worker_id == worker_id:
        is_main_worker = True

    subordinate_worker_indexes = []
    subordinate_workers = (
        instance.distributed_servers.subordinate_workers
        if instance.distributed_servers
        and instance.distributed_servers.subordinate_workers
        else []
    )
    for index, subordinate_worker in enumerate(subordinate_workers):
        if worker_name is not None and subordinate_worker.worker_name == worker_name:
            subordinate_worker_indexes.append(index)
            continue
        if worker_id is not None and subordinate_worker.worker_id == worker_id:
            subordinate_worker_indexes.append(index)

    return ModelInstanceWorkerMatch(
        is_main_worker=is_main_worker,
        subordinate_worker_indexes=tuple(subordinate_worker_indexes),
    )
