from datetime import datetime, timezone
from decimal import Decimal
from math import ceil
from typing import List, NamedTuple, Optional, Set

from gpustack.schemas.workers import Worker


class CPUFit(NamedTuple):
    workers: List[Worker]
    unknown_ids: Set[int]
    messages: List[str]


def filter_workers_by_cpu_budget(
    workers: List[Worker], cpu_cores: Optional[float]
) -> CPUFit:
    """Filter a Router's candidates using opted-in Kubernetes request budgets."""
    if cpu_cores is None:
        return CPUFit(workers, set(), [])

    now = datetime.now(timezone.utc)
    # The runtime sends the string form of this core quantity to Kubernetes.
    required = ceil(Decimal(str(cpu_cores)) * 1000)
    eligible = []
    unknown = set()
    insufficient = []
    for worker in workers:
        budget = getattr(getattr(worker, "status", None), "kubernetes_cpu_budget", None)
        if budget is None:
            eligible.append(worker)
            continue
        available = budget.available_at(now)
        if available is None:
            unknown.add(worker.id)
        elif available < required:
            insufficient.append(available)
        else:
            eligible.append(worker)

    messages = []
    if insufficient:
        messages.append(
            f"Router requires {required}m CPU; {len(insufficient)} Kubernetes "
            f"worker(s) have insufficient CPU requests available "
            f"(at most {max(insufficient)}m available)."
        )
    if unknown:
        messages.append(
            f"Kubernetes CPU request budget is unavailable or stale on "
            f"{len(unknown)} worker(s); check the worker's CPU budget collector."
        )
    return CPUFit(eligible, unknown, messages)
