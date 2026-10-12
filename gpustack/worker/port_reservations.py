from contextlib import contextmanager
import threading
from typing import Dict, Iterator, Optional, Set, Tuple

from gpustack.client import ClientSet
from gpustack.schemas.models import ModelInstance, member_worker_ids


def model_instance_ports(instance: ModelInstance, worker_id: int) -> Set[int]:
    """Collect the serving ports and complete bands held on one worker."""
    ports = set(instance.ports or [])
    if instance.port:
        ports.add(instance.port)
    for band in (instance.named_ports or {}).values():
        ports.update(range(band.base, band.base + max(band.count, 1)))
    if instance.distributed_servers:
        for subordinate in instance.distributed_servers.subordinate_workers or []:
            if subordinate.worker_id == worker_id:
                ports.update(subordinate.ports or [])
    return ports


class WorkerPortReservations:
    """Share pending port allocations between model and cache workloads.

    Each manager uses its own instance ID namespace within this registry.
    Callers select ports and update their map inside ``allocation``; releases
    use the corresponding release method. These maps cover the interval
    before an allocation reaches the API and before a deleted workload stops.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.model_ports: Dict[int, Set[int]] = {}
        self.cache_ports: Dict[int, Tuple[int, ...]] = {}
        self._generation = 0

    @contextmanager
    def allocation(
        self,
        clientset: ClientSet,
        worker_id: int,
        *,
        model_instance_id: Optional[int] = None,
        cache_instance_id: Optional[int] = None,
    ) -> Iterator[Set[int]]:
        """Yield unavailable ports while holding the lock for selection and registration.

        API reads run outside the lock so a slow server cannot block cleanup.
        A local allocation can be persisted and then released during those
        reads, so combining an old snapshot with current local maps is not
        sufficient. Retry if any allocation or release changed the generation.
        """
        while True:
            with self.lock:
                generation = self._generation
            taken = self._persisted_ports(
                clientset,
                worker_id,
                model_instance_id=model_instance_id,
                cache_instance_id=cache_instance_id,
            )
            with self.lock:
                if generation != self._generation:
                    continue
                for instance_id, ports in self.model_ports.items():
                    if instance_id != model_instance_id:
                        taken.update(ports)
                for instance_id, ports in self.cache_ports.items():
                    if instance_id != cache_instance_id:
                        taken.update(ports)
                try:
                    yield taken
                finally:
                    self._generation += 1
                return

    def release_model(self, instance_id: int) -> None:
        """Release a model's local reservation after its workload stops."""
        with self.lock:
            if self.model_ports.pop(instance_id, None) is not None:
                self._generation += 1

    def release_cache(self, instance_id: int) -> None:
        """Release a cache instance's local reservation after cleanup."""
        with self.lock:
            if self.cache_ports.pop(instance_id, None) is not None:
                self._generation += 1

    @staticmethod
    def _persisted_ports(
        clientset: ClientSet,
        worker_id: int,
        *,
        model_instance_id: Optional[int] = None,
        cache_instance_id: Optional[int] = None,
    ) -> Set[int]:
        """Read other owners' persisted reservations without the allocation lock.

        Unpaginated API reads include instances whose watch events have not
        arrived after startup or reconnect. A failed read must abort allocation:
        an unknown reservation is not a free port. Rows retain their ports while
        starting, backing off or in ERROR, so no state filter applies here.
        """
        # A model's worker_id only identifies its primary; subordinate placement
        # must be checked locally against the complete list.
        models = clientset.model_instances.list(
            params={"page": -1}, use_cache=False
        ).items
        caches = clientset.cache_service_instances.list(
            params={"worker_id": worker_id, "page": -1}, use_cache=False
        ).items

        taken: Set[int] = set()
        for instance in models or []:
            if instance.id != model_instance_id and worker_id in member_worker_ids(
                instance
            ):
                taken.update(model_instance_ports(instance, worker_id))
        for instance in caches or []:
            if instance.id != cache_instance_id and instance.worker_id == worker_id:
                taken.update((instance.ports or {}).values())
                if instance.port:
                    taken.add(instance.port)
        return taken
