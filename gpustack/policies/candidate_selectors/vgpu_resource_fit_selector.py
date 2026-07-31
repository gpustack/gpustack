import logging
import math
import re
from typing import List, Optional

from gpustack.config.config import Config
from gpustack.policies.base import (
    ModelInstanceScheduleCandidate,
)
from gpustack.policies.candidate_selectors.base_candidate_selector import (
    ScheduleCandidatesSelector,
)
from gpustack.policies.utils import (
    estimate_model_vram,
    get_model_ram_claim,
)
from gpustack.utils.unit import byte_to_gib
from gpustack.schemas.gpu_instance_types import (
    GPUInstanceType,
    GPUInstanceTypeDetail,
)
from gpustack.schemas.models import (
    ComputedResourceClaim,
    GPUTypeSelector,
    Model,
    ModelInstance,
    ModelInstanceSubordinateWorker,
)
from gpustack.gpu_instances.cluster_apis import ClusterOps
from gpustack.schemas.clusters import Cluster, ClusterProvider
from gpustack.schemas.principals import PLATFORM_PRINCIPAL_NAME
from gpustack.schemas.workers import Worker
from gpustack.server.db import async_session
from gpustack.utils.resource_usage import parse_quantity_to_mib

logger = logging.getLogger(__name__)

_MIB = 1024 * 1024


def _normalize_device_name(name: str) -> str:
    """
    Normalize a GPU device/product name for comparison, e.g.
    "NVIDIA A100-SXM4-40GB" and "A100-SXM4-40GB" both normalize
    with lowercase alnum tokens separated by single spaces.
    """
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def _device_matches_pool(
    vendor: str, device_name: str, detail: GPUInstanceTypeDetail
) -> bool:
    """
    Check whether a worker GPU device belongs to the InstanceType pool:
    same manufacturer, and the product name matches after normalization
    (worker names carry a vendor prefix the pool product may omit).
    """
    if not detail.manufacturer or not detail.product:
        return False
    if vendor.lower() != detail.manufacturer.lower():
        return False
    device_norm = _normalize_device_name(device_name)
    product_norm = _normalize_device_name(detail.product)
    # Token-boundary suffix: the worker name may carry a vendor prefix the
    # pool product omits ("nvidia h100 80gb hbm3" vs "h100 80gb hbm3"), but a
    # bare endswith would also match a longer product on its tail token
    # ("tesla a2100" vs "100").
    return device_norm == product_norm or device_norm.endswith(" " + product_norm)


def _node_serves_claim(
    devices: Optional[dict],
    detail: GPUInstanceTypeDetail,
    profile: Optional[str],
) -> bool:
    """Whether this node's ``Devices`` can take one claim on the pool.

    Two facts have to line up, per accelerator group belonging to the pool:

    - the mode the claim needs is enabled on the node. ``spec.groups[].
      acceleratorSlicedDetail.logical.count`` says software slicing is on;
      ``.physical.count`` says hardware partitioning is, and the claim's profile
      has to be among ``.physical.profiles``. Enablement is all that is checked
      — on Kubernetes the partitions are cut on demand by the operator's
      device-manager, so a MIG-enabled card carries no instances until a
      workload asks for one, and a profile that is offered is one that can
      still be created.
    - some accelerator in the group has room: ``status.groups[].
      accelerators[].remaining`` above zero. A card already fully claimed
      reports zero and takes no more.

    A node with no Devices at all is not a node this cluster manages devices
    for, so it cannot serve the claim.
    """
    if not devices:
        return False
    status_groups = {
        group.get("id"): group
        for group in ((devices.get("status") or {}).get("groups") or [])
    }
    for group in (devices.get("spec") or {}).get("groups") or []:
        if not _device_matches_pool(
            group.get("manufacturer") or "", group.get("name") or "", detail
        ):
            continue
        sliced = group.get("acceleratorSlicedDetail") or {}
        if profile:
            physical = sliced.get("physical") or {}
            if not physical.get("count"):
                continue
            if not any(
                p.get("name") == profile for p in physical.get("profiles") or []
            ):
                continue
        elif not (sliced.get("logical") or {}).get("count"):
            continue
        status_group = status_groups.get(group.get("id")) or {}
        for accelerator in status_group.get("accelerators") or []:
            if accelerator.get("remaining"):
                return True
    return False


class VGPUResourceFitSelector(ScheduleCandidatesSelector):
    """
    Candidate selector for models with ``gpu_type_selector``: schedule onto
    workers whose GPUs belong to the selected operator InstanceType pool,
    exactly one slice (or whole card / partition) per worker, without
    card-level GPU index selection.
    """

    def __init__(
        self,
        config: Config,
        model: Model,
        model_instances: List[ModelInstance],
    ):
        super().__init__(config, model, model_instances)
        self._messages: List[str] = []
        self._vram_claim = 0
        self._ram_claim = 0
        self._slice_vram = 0

    def get_messages(self) -> List[str]:
        return self._messages

    def _should_check_vision_tp_divisibility(self) -> bool:
        return False

    async def select_candidates(
        self, workers: List[Worker]
    ) -> List[ModelInstanceScheduleCandidate]:
        selector = self._model.gpu_type_selector
        if not selector:
            return []

        detail = await self._get_instance_type_detail(selector)
        if detail is None:
            self._messages = [
                f"InstanceType '{selector.type}' is not available in the model's cluster."
            ]
            return []

        card_vram = self._get_card_vram(detail)
        if card_vram <= 0:
            self._messages = [
                f"InstanceType '{selector.type}' does not report its accelerator memory."
            ]
            return []

        self._slice_vram = self._get_slice_vram(selector, detail, card_vram)
        if self._slice_vram <= 0:
            self._messages = [
                f"InstanceType '{selector.type}' has no profile named "
                f"'{selector.accelerator_partitioned_profile}'."
            ]
            return []

        pool_workers = [w for w in workers if self._worker_matches_pool(w, detail)]
        if not pool_workers:
            self._messages = [
                f"No workers have GPUs matching InstanceType '{selector.type}' "
                f"({detail.manufacturer} {detail.product})."
            ]
            return []

        # A worker's own inventory says which cards it has, not which slicing
        # modes the cluster enabled on them nor what is left — and on Kubernetes
        # this fit only picks the worker, after which the node-side scheduler
        # has the final say. Pick against the cluster's Devices so the pick and
        # that final say agree; otherwise the workload lands on a node that
        # never advertises the resource and sits Pending with nothing to report.
        profile = selector.accelerator_partitioned_profile
        devices_by_node = await self._load_cluster_devices()
        matching_workers = (
            pool_workers
            if devices_by_node is None
            else [
                w
                for w in pool_workers
                if _node_serves_claim(devices_by_node.get(w.name), detail, profile)
            ]
        )
        if not matching_workers:
            wanted = (
                f"hardware partitioning offering profile '{profile}'"
                if profile
                else "software slicing"
            )
            self._messages = [
                f"No node backing InstanceType '{selector.type}' "
                f"({detail.manufacturer} {detail.product}) has {wanted} enabled "
                f"with capacity left."
            ]
            return []

        self._vram_claim = await estimate_model_vram(
            self._model, self._config.huggingface_token, workers
        )
        self._ram_claim = get_model_ram_claim(self._model)

        if self._vram_claim <= self._slice_vram:
            return [
                self._create_candidate(worker, detail) for worker in matching_workers
            ]

        if not self._model.distributed_inference_across_workers:
            self._messages = [
                f"The model requires approximately {byte_to_gib(self._vram_claim)} GiB of VRAM, "
                f"but one slice of InstanceType '{selector.type}' provides "
                f"{byte_to_gib(self._slice_vram)} GiB."
            ]
            return []

        slices_needed = math.ceil(self._vram_claim / self._slice_vram)
        if len(matching_workers) < slices_needed:
            self._messages = [
                f"The model requires approximately {byte_to_gib(self._vram_claim)} GiB of VRAM, "
                f"needs {slices_needed} slices of InstanceType '{selector.type}' "
                f"({byte_to_gib(self._slice_vram)} GiB each), "
                f"but only {len(matching_workers)} matching workers are available."
            ]
            return []

        ordered = sorted(matching_workers, key=lambda w: w.name)
        candidates = []
        for i, worker in enumerate(ordered):
            subordinates = (ordered[:i] + ordered[i + 1 :])[: slices_needed - 1]
            candidates.append(self._create_candidate(worker, detail, subordinates))
        return candidates

    async def _get_instance_type_detail(
        self, selector: GPUTypeSelector
    ) -> Optional[GPUInstanceTypeDetail]:
        """
        Read the local GPUInstanceType projection (synced by the operator
        controller); no Kubernetes reads in the scheduling path.
        """
        fields: dict = {
            "deleted_at": None,
            "name": selector.type,
        }
        # getattr: evaluation-time ModelSpec carries no cluster unless the
        # evaluation route stamped one; then fall back to an unscoped read.
        model_cluster_id = getattr(self._model, "cluster_id", None)
        if model_cluster_id is not None:
            fields["cluster_id"] = model_cluster_id
        async with async_session() as session:
            instance_types = await GPUInstanceType.all_by_fields(
                session,
                fields=fields,
            )
        matched = instance_types[0] if instance_types else None
        if matched is None or matched.status is None or matched.status.detail is None:
            return None
        return matched.status.detail

    def _get_card_vram(self, detail: GPUInstanceTypeDetail) -> int:
        if not detail.memory:
            return 0
        mib = parse_quantity_to_mib(detail.memory)
        return int(mib * _MIB) if mib else 0

    def _get_slice_vram(
        self,
        selector: GPUTypeSelector,
        detail: GPUInstanceTypeDetail,
        card_vram: int,
    ) -> int:
        if selector.accelerator_partitioned_profile:
            sliced_detail = detail.sliced_detail
            physical = sliced_detail.physical if sliced_detail else None
            for profile in (physical.profiles if physical else None) or []:
                if profile.name == selector.accelerator_partitioned_profile:
                    return int((profile.memory_mib or 0) * _MIB)
            return 0

        memory_percentage = selector.accelerator_sliced_memory_percentage or 0
        if memory_percentage == 0:
            # Whole-card exclusive mode.
            return card_vram
        return int(card_vram * memory_percentage / 100)

    def _worker_matches_pool(
        self, worker: Worker, detail: GPUInstanceTypeDetail
    ) -> bool:
        if not worker.status or not worker.status.gpu_devices:
            return False
        return any(
            _device_matches_pool(device.vendor or "", device.name or "", detail)
            for device in worker.status.gpu_devices
        )

    async def _load_cluster_devices(self) -> Optional[dict]:
        """Every node's ``Devices`` in the model's cluster, keyed by node name.

        ``None`` — no cluster on the model (an evaluation that was never scoped
        to one), a cluster that has no such API, or a read that failed — means
        "cannot tell", and the caller keeps every pool worker rather than
        refusing to schedule on a transient cluster-API problem.
        """
        cluster_id = getattr(self._model, "cluster_id", None)
        if cluster_id is None:
            return None
        async with async_session() as session:
            cluster = await Cluster.one_by_id(session, cluster_id)
        if cluster is None or cluster.provider != ClusterProvider.Kubernetes:
            return None
        try:
            # Devices is cluster-scoped, so the owner identifier — which only
            # derives the org namespace of namespaced CRDs — never reaches the
            # wire; the constructor requires one regardless.
            async with ClusterOps(
                server_api_port=self._config.get_api_port(),
                cluster_id=cluster.id,
                cluster_registration_token=cluster.registration_token,
                cluster_owner_principal_identifier=PLATFORM_PRINCIPAL_NAME,
            ) as ops:
                result = await ops.list_devices()
        except Exception as e:
            logger.warning("Failed to read devices of cluster %s: %s", cluster_id, e)
            return None
        by_node = {}
        for item in result.get("items") or []:
            name = (item.get("metadata") or {}).get("name")
            if name:
                by_node[name] = item
        return by_node

    def _get_worker_gpu_type(
        self, worker: Worker, detail: GPUInstanceTypeDetail
    ) -> str:
        for device in worker.status.gpu_devices or []:
            if _device_matches_pool(device.vendor or "", device.name or "", detail):
                return device.type
        return ""

    def _create_slice_claim(self) -> ComputedResourceClaim:
        # The vram key is a placeholder card index: without card-level
        # selection the real index is only known after the device plugin
        # allocates, and is re-keyed on allocation read-back.
        return ComputedResourceClaim(
            vram={0: self._slice_vram},
            ram=self._ram_claim,
        )

    def _create_candidate(
        self,
        worker: Worker,
        detail: GPUInstanceTypeDetail,
        subordinate_workers: Optional[List[Worker]] = None,
    ) -> ModelInstanceScheduleCandidate:
        gpu_type = self._get_worker_gpu_type(worker, detail)
        subordinates = [
            ModelInstanceSubordinateWorker(
                worker_id=w.id,
                worker_name=w.name,
                worker_ip=w.ip,
                worker_ifname=w.ifname,
                total_gpus=len(w.status.gpu_devices or []),
                gpu_type=self._get_worker_gpu_type(w, detail),
                gpu_indexes=None,
                computed_resource_claim=self._create_slice_claim(),
            )
            for w in subordinate_workers or []
        ]
        return ModelInstanceScheduleCandidate(
            worker=worker,
            gpu_indexes=None,
            gpu_type=gpu_type,
            computed_resource_claim=self._create_slice_claim(),
            subordinate_workers=subordinates or None,
        )
