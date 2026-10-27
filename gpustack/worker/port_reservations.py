from contextlib import contextmanager
import threading
from typing import Dict, Iterable, Iterator, Optional, Set, Tuple
from uuid import UUID, uuid4

from gpustack.client import ClientSet
from gpustack.schemas.models import ModelInstance, member_worker_ids
from gpustack.schemas.port_reservations import PortClaimResult
from gpustack.utils import network


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


class PortReservationSnapshot(set):
    """Unavailable ports and their owners from one allocation snapshot."""

    def __init__(self):
        super().__init__()
        self._owners: Dict[Tuple[str, int, str], Set[int]] = {}
        self.assigned_model_ids: Set[int] = set()
        self.assigned_cache_ids: Set[int] = set()
        self.pending_cleanup = False

    def reserve(
        self, resource: str, instance_id: int, ports: Iterable[int], source: str
    ) -> None:
        """Record a reservation together with its resource and storage source."""
        ports = set(ports)
        self.update(ports)
        self._owners.setdefault((resource, instance_id, source), set()).update(ports)

    def describe(self, ports: Set[int]) -> str:
        """Describe only owners intersecting the requested ports."""
        descriptions = []
        for (resource, instance_id, source), reserved in sorted(self._owners.items()):
            overlap = ports & reserved
            if overlap:
                descriptions.append(
                    f"{resource} instance {instance_id} ({source}) ports {sorted(overlap)}"
                )
        return "; ".join(descriptions)


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
        self.model_claims: Dict[int, Tuple[str, Set[int]]] = {}
        self._claim_generations: Dict[int, UUID] = {}
        self._claim_versions: Dict[int, UUID] = {}
        self._generation = 0

    @contextmanager
    def allocation(
        self,
        clientset: ClientSet,
        worker_id: int,
        *,
        model_instance_id: Optional[int] = None,
        cache_instance_id: Optional[int] = None,
    ) -> Iterator[PortReservationSnapshot]:
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
                        taken.reserve("model", instance_id, ports, "local")
                        if ports and instance_id not in taken.assigned_model_ids:
                            taken.pending_cleanup = True
                for instance_id, ports in self.cache_ports.items():
                    if instance_id != cache_instance_id:
                        taken.reserve("cache", instance_id, ports, "local")
                        if ports and instance_id not in taken.assigned_cache_ids:
                            taken.pending_cleanup = True
                for instance_id, (_, ports) in self.model_claims.items():
                    if instance_id != model_instance_id:
                        taken.reserve("model", instance_id, ports, "tentative")
                        if ports and instance_id not in taken.assigned_model_ids:
                            taken.pending_cleanup = True
                try:
                    yield taken
                finally:
                    self._generation += 1
                return

    def reserve_model(
        self,
        clientset: ClientSet,
        worker_id: int,
        instance_id: int,
        token: str,
        ports: Set[int],
        port_range: str,
        host: str,
        generation: UUID,
        replace_version: Optional[UUID] = None,
        recoverable: bool = True,
    ) -> PortClaimResult:
        """Fence a complete distributed allocation under the local allocation lock."""
        with self.allocation(
            clientset, worker_id, model_instance_id=instance_id
        ) as taken:
            previous = self.model_claims.get(instance_id)
            same_claim = previous == (token, ports)
            if not same_claim and generation != self._claim_generations.setdefault(
                instance_id, uuid4()
            ):
                # A stale attempt must refresh its snapshot; its ports may
                # still be free and must not be excluded from the next layout.
                return PortClaimResult(accepted=False)
            replacing = previous is not None and not same_claim
            if replacing:
                if (
                    not recoverable
                    or replace_version is None
                    or replace_version != self._claim_versions.get(instance_id)
                    or self.model_ports.get(instance_id)
                ):
                    return PortClaimResult(accepted=False)
            start, end = network.parse_port_range(port_range)
            owned = self.model_ports.get(instance_id, set())
            conflicts = {
                port
                for port in ports
                if port in taken
                or not start <= port <= end
                or (port not in owned and not network.is_port_available(port, host))
            }
            if conflicts:
                return PortClaimResult(
                    accepted=False, unavailable_ports=sorted(conflicts)
                )
            self.model_claims[instance_id] = (token, set(ports))
            if not same_claim:
                self._claim_versions[instance_id] = uuid4()
            if replacing:
                # Retire every in-flight request from the replaced allocation,
                # including requests that arrive after its replacement stops.
                self._claim_generations[instance_id] = uuid4()
            return PortClaimResult(accepted=True)

    def claim_state(self, instance_id: int) -> Tuple[UUID, Optional[UUID]]:
        """Read the request fence and tentative revision from one locked snapshot."""
        with self.lock:
            return (
                self._claim_generations.setdefault(instance_id, uuid4()),
                self._claim_versions.get(instance_id),
            )

    def claim_generation(self, instance_id: int) -> UUID:
        """Return the current request generation without retaining token history."""
        with self.lock:
            return self._claim_generations.setdefault(instance_id, uuid4())

    def release_claim(self, instance_id: int, token: str) -> None:
        """Cancel one attempt without releasing a committed workload's ports."""
        with self.lock:
            current = self.model_claims.get(instance_id)
            if not current or current[0] == token:
                self.model_claims.pop(instance_id, None)
                self._claim_versions.pop(instance_id, None)
                self._claim_generations[instance_id] = uuid4()
            self._generation += 1

    def release_model(self, instance_id: int, forget_generation: bool = False) -> None:
        """Release a model's local reservation after its workload stops."""
        with self.lock:
            self.model_ports.pop(instance_id, None)
            self.model_claims.pop(instance_id, None)
            self._claim_versions.pop(instance_id, None)
            self._claim_generations[instance_id] = uuid4()
            if forget_generation:
                self._claim_generations.pop(instance_id, None)
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
    ) -> PortReservationSnapshot:
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

        taken = PortReservationSnapshot()
        for instance in models or []:
            if worker_id not in member_worker_ids(instance):
                continue
            taken.assigned_model_ids.add(instance.id)
            if instance.id != model_instance_id:
                taken.reserve(
                    "model",
                    instance.id,
                    model_instance_ports(instance, worker_id),
                    "persisted",
                )
        for instance in caches or []:
            if instance.worker_id != worker_id:
                continue
            taken.assigned_cache_ids.add(instance.id)
            if instance.id != cache_instance_id:
                ports = set((instance.ports or {}).values())
                if instance.port:
                    ports.add(instance.port)
                taken.reserve("cache", instance.id, ports, "persisted")
        return taken
