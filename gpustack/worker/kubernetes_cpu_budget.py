"""Read CPU request headroom for a Kubernetes Worker node."""

import logging
from contextlib import AbstractContextManager, contextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from math import ceil
from time import monotonic
from typing import Callable, Dict, Iterator, Optional
from urllib.parse import urlsplit

from kubernetes import client, config
from kubernetes.utils.quantity import parse_quantity
from urllib3.exceptions import HTTPError

from gpustack.envs import WORKER_STATUS_SYNC_INTERVAL
from gpustack.schemas.workers import KubernetesCPUBudget

logger = logging.getLogger(__name__)
_REQUEST_TIMEOUT = 5
_COLLECTION_TIMEOUT = 10


def _cpu(resources: Optional[Dict[str, str]]) -> Decimal:
    return parse_quantity((resources or {}).get("cpu", "0"))


def _requests(resources: Optional[client.V1ResourceRequirements]) -> Dict[str, str]:
    return (resources.requests or {}) if resources is not None else {}


def _container_request(
    container: client.V1Container, statuses: Dict[str, client.V1ContainerStatus]
) -> Decimal:
    declared = _cpu(_requests(container.resources))
    status = statuses.get(container.name)
    if status is None:
        return declared
    return max(
        declared,
        _cpu(_requests(status.resources)),
        _cpu(status.allocated_resources),
    )


def _pod_request_millicores(pod: client.V1Pod) -> int:
    """Account for concurrent sidecars, peak init usage, and Pod overhead."""
    spec = pod.spec
    pod_requests = _requests(spec.resources)
    if "cpu" in pod_requests:
        return ceil((_cpu(pod_requests) + _cpu(spec.overhead)) * 1000)

    status = pod.status
    statuses = {
        item.name: item
        for item in (
            (status.container_statuses or []) + (status.init_container_statuses or [])
            if status is not None
            else []
        )
    }
    running = sum(_container_request(c, statuses) for c in spec.containers)
    sidecars = peak_init = Decimal(0)
    for container in spec.init_containers or []:
        request = _container_request(container, statuses)
        peak_init = max(peak_init, sidecars + request)
        if container.restart_policy == "Always":
            sidecars += request

    return ceil((max(running + sidecars, peak_init) + _cpu(spec.overhead)) * 1000)


class KubernetesCPUCollector:
    def __init__(
        self,
        open_api: Callable[[], AbstractContextManager[client.CoreV1Api]],
        node_name: str,
    ):
        self._open_api = open_api
        self._node_name = node_name

    @classmethod
    def from_kubeconfig(
        cls, path: str, node_name: Optional[str]
    ) -> "KubernetesCPUCollector":
        """Use a dedicated credential, not the Worker's proxied ServiceAccount."""
        if not node_name:
            raise ValueError(
                "CPU budget collection requires GPUSTACK_RUNTIME_KUBERNETES_NODE_NAME"
            )

        def settings() -> client.Configuration:
            result = client.Configuration()
            config.load_kube_config(
                config_file=path, client_configuration=result, persist_config=False
            )
            if not result.verify_ssl or urlsplit(result.host).scheme != "https":
                raise ValueError(
                    "CPU budget collection requires a verified HTTPS Kubernetes API"
                )
            result.retries = 0
            return result

        settings()

        @contextmanager
        def open_api() -> Iterator[client.CoreV1Api]:
            # Reload tokenFile and other mounted credentials on each sample.
            with client.ApiClient(settings()) as api_client:
                yield client.CoreV1Api(api_client)

        return cls(open_api, node_name)

    def collect(self) -> KubernetesCPUBudget:
        """Report an unavailable budget when the node or Pod list cannot be read."""
        started = datetime.now(timezone.utc)
        try:
            with self._open_api() as api:
                available = self._available_millicores(api)
        except (
            client.ApiException,
            config.ConfigException,
            HTTPError,
            OSError,
            KeyError,
            ValueError,
            InvalidOperation,
        ) as error:
            # Do not include an API response body; it may contain Pod details.
            logger.warning(
                "Cannot establish Kubernetes CPU request budget (%s, HTTP status %s)",
                type(error).__name__,
                getattr(error, "status", None),
            )
            return KubernetesCPUBudget()
        return KubernetesCPUBudget(
            available_millicores=available,
            expires_at=started
            + timedelta(seconds=max(3 * WORKER_STATUS_SYNC_INTERVAL, 30)),
        )

    def _available_millicores(self, api: client.CoreV1Api) -> int:
        deadline = monotonic() + _COLLECTION_TIMEOUT
        node = api.read_node(self._node_name, _request_timeout=_REQUEST_TIMEOUT)
        if node.status is None or not node.status.allocatable:
            raise ValueError("Kubernetes Node has no allocatable CPU data")
        allocatable = int(parse_quantity(node.status.allocatable["cpu"]) * 1000)
        requested = 0
        cursor = None
        while True:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("CPU budget collection exceeded its deadline")
            page = api.list_pod_for_all_namespaces(
                field_selector=f"spec.nodeName={self._node_name}",
                limit=500,
                _continue=cursor,
                _request_timeout=min(ceil(remaining), _REQUEST_TIMEOUT),
            )
            if page.metadata is None:
                raise ValueError("Kubernetes Pod page has no list metadata")
            requested += sum(
                _pod_request_millicores(pod)
                for pod in page.items
                if pod.spec.node_name == self._node_name
                and (
                    pod.status is None
                    or pod.status.phase not in ("Succeeded", "Failed")
                )
            )
            cursor = page.metadata._continue
            if not cursor:
                return max(0, allocatable - requested)
