import asyncio
from datetime import datetime, timedelta, timezone
import json
import logging
import os
import queue
from typing import List, Sequence, Tuple, Optional
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.orm import selectinload
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from gpustack.policies.scorers.pairing_affinity_scorer import PairingAffinityScorer
from gpustack.policies.scorers.placement_scorer import PlacementScorer
from gpustack.policies.scorers.topology_proximity_scorer import (
    TopologyProximityScorer,
)
from gpustack.policies.scorers.model_file_locality_scorer import (
    ModelFileLocalityScorer,
)
from gpustack.policies.scorers.score_chain import CandidateScoreChain
from gpustack.config.config import Config, get_global_config
from gpustack.policies.base import (
    ModelInstanceScheduleCandidate,
    ScheduleCandidatesScorer,
    WorkerFilterChain,
)
from gpustack.policies.candidate_selectors import (
    AscendMindIEResourceFitSelector,
    GGUFResourceFitSelector,
    SGLangResourceFitSelector,
    VGPUResourceFitSelector,
    VLLMResourceFitSelector,
)
from gpustack.policies.candidate_selectors.instance_type_whole_card_selector import (
    InstanceTypeWholeCardSelector,
    is_whole_card_claim,
)
from gpustack.policies.candidate_selectors.custom_backend_resource_fit_selector import (
    CustomBackendResourceFitSelector,
)
from gpustack.policies.utils import ListMessageBuilder, should_skip_gpu_count_check
from gpustack.policies.worker_filters.backend_framework_filter import (
    BackendFrameworkFilter,
)
from gpustack.policies.worker_filters.label_matching_filter import LabelMatchingFilter
from gpustack.policies.worker_filters.gpu_matching_filter import GPUMatchingFilter
from gpustack.policies.worker_filters.local_path_filter import LocalPathFilter
from gpustack.policies.worker_filters.pd_mode_filter import PDModeRuntimeFilter
from gpustack.policies.worker_filters.cluster_filter import ClusterFilter
from gpustack.policies.worker_filters.gather_floor_filter import GatherFloorFilter
from gpustack.scheduler.model_registry import detect_model_type
from gpustack.scheduler.meta_registry import get_model_meta
from gpustack.scheduler.queue import AsyncUniqueQueue
from gpustack.policies.worker_filters.status_filter import StatusFilter
from gpustack import envs
from gpustack.schemas.inference_backend import is_built_in_backend
from gpustack.schemas.clusters import Cluster, GatherStrategyEnum
from gpustack.schemas.workers import Worker
from gpustack.schemas.models import (
    BackendEnum,
    CategoryEnum,
    DistributedServers,
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    get_backend,
    is_gguf_model,
    DistributedServerCoordinateModeEnum,
    SourceEnum,
    is_omni_model,
    role_effective_model,
    role_container_resources,
    role_takes_no_accelerator,
)
from gpustack.schemas.model_files import ModelFileStateEnum
from gpustack.server.bus import EventType
from gpustack.server.db import async_session
from gpustack.scheduler.group_schedule import (
    is_group_forming,
    schedule_group,
)
from gpustack.topology.view import TopologyView, build_view
from gpustack.scheduler.calculator import (
    GPUOffloadEnum,
    calculate_gguf_model_resource_claim,
    check_diffusers_model_index_from_workers,
)
from gpustack.server.cache_services import resolve_instance_cache_config_safe
from gpustack.server.services import (
    ModelInstanceService,
    ModelService,
    ModelFileService,
)
from gpustack.utils.command import find_parameter
from gpustack.utils.gpu import group_gpu_ids_by_worker
from gpustack.utils.hub import has_diffusers_model_index
from gpustack.utils.math import largest_power_of_2_leq
from gpustack.utils.model_source import get_draft_model_source
from gpustack.scheduler.calculator import get_pretrained_config_with_workers
from sqlalchemy.orm.attributes import flag_modified

logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, cfg: Config, check_interval: int = 180):
        """
        Init the scheduler with queue and interval.
        """

        self._id = "model-instance-scheduler"
        self._config = cfg
        self._check_interval = check_interval
        self._queue = AsyncUniqueQueue()
        self._cache_dir = None

        if self._config.cache_dir is not None:
            self._cache_dir = os.path.join(self._config.cache_dir, "gguf-parser")
            os.makedirs(self._cache_dir, exist_ok=True)

    async def start(self):
        """
        Start the scheduler.
        """

        try:
            # scheduler queue.
            asyncio.create_task(self._schedule_cycle())

            # scheduler job trigger by time interval.
            trigger = IntervalTrigger(
                seconds=self._check_interval, timezone=timezone.utc
            )
            scheduler = AsyncIOScheduler(timezone=timezone.utc)
            scheduler.add_job(
                self._enqueue_pending_instances,
                trigger=trigger,
                id=self._id,
                max_instances=1,
            )
            scheduler.start()
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")

        logger.info("Scheduler started.")

        # Bootstrap pending state once at startup; replaces the bus replay
        # of every existing instance which would flood the queue (#4794).
        await self._enqueue_pending_instances()

        # Live trigger. event_types/replay_existing keep this subscription
        # cheap so UPDATED/HEARTBEAT bursts don't fill the queue.
        async for event in ModelInstance.subscribe(
            source="scheduler",
            event_types={EventType.CREATED},
            replay_existing=False,
        ):
            # The bus filter only blocks events from publishers; the
            # subscribe() generator still yields HEARTBEAT events on its
            # own to keep the stream alive (active_record.py). Skip those
            # and any other non-CREATED events that may surface in future.
            if event.type != EventType.CREATED:
                continue
            # Single-instance path; the IntervalTrigger above is still the
            # full-scan fallback for anything missed here.
            await self._enqueue_event_instance(event.data)

    async def _enqueue_pending_instances(self):
        """
        Periodic / bootstrap full scan of pending model instances.
        """
        try:
            async with async_session() as session:
                instances = await ModelInstance.all(session)
                tasks = []
                for instance in instances:
                    if self._should_schedule(instance):
                        task = asyncio.create_task(self._evaluate(instance))
                        tasks.append(task)

                await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Failed to enqueue pending model instances: {e}")

    async def _enqueue_event_instance(self, instance: Optional[ModelInstance]):
        """Event-driven single-instance path. ``_evaluate`` re-fetches from
        DB, so the event payload is used only for ``_should_schedule``."""
        if instance is None or instance.id is None:
            return
        try:
            if self._should_schedule(instance):
                await self._evaluate(instance)
        except Exception as e:
            logger.error(f"Failed to evaluate instance {instance.id} from event: {e}")

    async def _evaluate(self, instance: ModelInstance):  # noqa: C901
        """
        Evaluate the model instance's metadata.
        """
        async with async_session() as session:
            try:
                instance = await ModelInstance.one_by_id(session, instance.id)
                # Re-check against the freshly-fetched row: the caller's
                # snapshot may be stale (event payload, last full scan, etc.)
                # and the user may have deleted or transitioned the instance
                # between dispatch and now.
                if instance is None or not self._should_schedule(instance):
                    return

                model = await Model.one_by_id(session, instance.model_id)
                if model is None:
                    raise Exception("Model not found.")

                if instance.state != ModelInstanceStateEnum.ANALYZING:
                    instance.state = ModelInstanceStateEnum.ANALYZING
                    instance.state_message = "Evaluating resource requirements"
                    await ModelInstanceService(session).update(instance)

                # Get available workers for potential remote parsing
                workers = await Worker.all(session)
                sorted_workers = await prioritize_workers_with_model_files(
                    session, model, workers
                )

                should_update_model = False
                try:
                    if is_gguf_model(model):
                        should_update_model = await evaluate_gguf_model(
                            model, sorted_workers
                        )
                        if await self.check_model_distributability(
                            session, model, instance
                        ):
                            return
                    else:
                        should_update_model = await evaluate_pretrained_config(
                            model,
                            workers=sorted_workers,
                            raise_raw=True,
                        )
                except Exception as e:
                    # Even if the evaluation failed, we still want to proceed to deployment.
                    # Cases can be:
                    # 1. Model config is not valid, but is overridable by backend parameters.
                    # 2. It may not be required to be transformer-compatible for certain backends.
                    logger.error(
                        f"Failed to evaluate model {model.name or model.readable_source}: {e}"
                    )

                if should_update_model:
                    await ModelService(session).update(model)

                await self._queue.put(instance)
            except Exception as e:
                try:
                    instance.state = ModelInstanceStateEnum.ERROR
                    instance.state_message = str(e)
                    await ModelInstanceService(session).update(instance)
                except Exception as ue:
                    logger.error(
                        f"Failed to update model instance: {ue}. Original error: {e}"
                    )

    async def check_model_distributability(
        self, session: AsyncSession, model: Model, instance: ModelInstance
    ):
        if (
            not model.distributable
            and model.gpu_selector
            and model.gpu_selector.gpu_ids
        ):
            worker_gpu_ids = group_gpu_ids_by_worker(model.gpu_selector.gpu_ids)
            if len(worker_gpu_ids) > 1:
                instance.state = ModelInstanceStateEnum.ERROR
                instance.state_message = (
                    "The model is not distributable to multiple workers."
                )
                await ModelInstanceService(session).update(instance)
                return True
        return False

    def _should_schedule(self, instance: ModelInstance) -> bool:
        """
        Check if the model instance should be scheduled.
        Args:
            instance: ModelInstance to check.
        """
        newly_created = (instance.updated_at - instance.created_at) < timedelta(
            seconds=1
        )
        update_delta = datetime.now(timezone.utc) - instance.updated_at.replace(
            tzinfo=timezone.utc
        )
        return (
            (
                # When enqueueing pending state model instances, handle two cases:
                # 1. Newly created model instances (updated_at - created_at < 1 second),
                #    which will be updated to ANALYZING in _evaluate.
                # 2. Existing PENDING model instances periodically enqueued by the scheduler job.
                #    In this case, update_delta is longer than 90s, as the scheduler runs every 180s.
                instance.worker_id is None
                and instance.state == ModelInstanceStateEnum.PENDING
                and (newly_created or update_delta > timedelta(seconds=90))
            )
            or (
                # Reschedule while it stays in anayzing state for too long,
                # maybe the server is restarted.
                instance.worker_id is None
                and instance.state == ModelInstanceStateEnum.ANALYZING
                and update_delta > timedelta(minutes=3)
            )
            or (
                # Reschedule while it stays in scheduled state for too long,
                # maybe the worker is down.
                instance.worker_id is not None
                and instance.state == ModelInstanceStateEnum.SCHEDULED
                and update_delta > timedelta(minutes=3)
            )
        )

    async def _schedule_cycle(self):
        while True:
            try:
                item = await self._queue.get()
                try:
                    await self._schedule_one(item)
                    self._queue.task_done()
                except Exception as e:
                    logger.error(f"Failed to schedule model instance: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Failed to get item from schedule queue: {e}")

    async def _try_schedule_group(
        self,
        session: AsyncSession,
        model: Model,
        model_instance: ModelInstance,
        workers: List[Worker],
        model_instances: List[ModelInstance],
    ) -> bool:
        """Place the whole group if this instance is the one that forms it.

        Returns True when the group path owned this item — either it placed
        every member, or it refused the group as a whole. False means "not a
        forming group", and the caller continues down the per-instance path
        unchanged.

        The queue hands over one instance at a time, so the first member of an
        unplaced group solves for all of them and writes every row. The
        siblings arrive later, find themselves already scheduled, and take the
        `False` branch — where `find_candidate` sees a placed instance and the
        existing logic leaves it alone.
        """
        group_instances = [
            i
            for i in model_instances
            if i.model_id == model.id
            and i.group_id == model_instance.group_id
            and i.group_id is not None
        ]
        if not is_group_forming(model, group_instances):
            return False

        by_instance, messages = await schedule_group(
            session,
            self._config,
            model,
            workers,
            model_instances,
            group_instances,
        )
        if by_instance is None:
            # All-or-nothing: not one member is placed, and the reason
            # is put on the instance that triggered the solve so it surfaces
            # somewhere rather than only in the log.
            model_instance.state = ModelInstanceStateEnum.PENDING
            model_instance.state_message = (
                "The group could not be placed.\nDetails:\n" + "".join(messages)
            )
            await ModelInstanceService(session).update(model_instance)
            logger.debug("Group %s not placeable: %s", model.name, "".join(messages))
            return True

        for instance_id, candidate in by_instance.items():
            row = next((i for i in group_instances if i.id == instance_id), None)
            if row is None:
                continue
            await apply_candidate_to_instance(session, model, row, candidate)
            # INFO, unlike the single-instance path's debug: a group forms
            # once per generation and its member-to-card mapping is the thing
            # anyone diagnosing a disaggregated deployment asks for first.
            # The per-instance path logs at debug because it runs constantly.
            logger.info(
                "Scheduled group member %s (role %s) to worker %s gpu %s",
                row.name,
                row.role,
                row.worker_name,
                candidate.gpu_indexes,
            )
        return True

    async def _schedule_one(self, instance: ModelInstance):  # noqa: C901
        """
        Schedule a model instance by picking one candidate.
        Args:
            item: Model instance to schedule.
        """
        logger.debug(f"Scheduling model instance {instance.name}")

        state_message = ""

        async with async_session() as session:
            workers = await Worker.all(session)
            if not workers:
                state_message = "No available workers"

            model = await Model.one_by_id(session, instance.model_id)
            if model is None:
                state_message = "Model not found"

            model_instance = await ModelInstance.one_by_id(session, instance.id)
            if model_instance is None:
                logger.debug(
                    f"Model instance(ID: {instance.id}) was deleted before scheduling due"
                )
                return

            model_instances = await ModelInstance.all(
                session, options=[selectinload(ModelInstance.model)]
            )

            # The group gate, and it is the whole safety argument. Only a
            # model with `roles` whose generation has no placed GPU member
            # gets here; a role-less deployment can never satisfy
            # `is_group_forming`, so the path below is byte-for-byte the one it
            # always took. See scheduler/group_schedule.py.
            if workers and model and model.roles:
                handled = await self._try_schedule_group(
                    session, model, model_instance, workers, model_instances
                )
                if handled:
                    return

            candidate = None
            messages = []
            if workers and model:
                try:
                    candidate, messages = await find_candidate(
                        session,
                        self._config,
                        model,
                        workers,
                        model_instances,
                        role=model_instance.role,
                        exclude_instance_id=model_instance.id,
                        group_id=getattr(model_instance, "group_id", None),
                    )
                except Exception as e:
                    state_message = f"Failed to find candidate: {e}"

            if candidate is None:
                # update model instance.
                if model_instance.state in (
                    ModelInstanceStateEnum.SCHEDULED,
                    ModelInstanceStateEnum.ANALYZING,
                ):
                    model_instance.state = ModelInstanceStateEnum.PENDING
                    model_instance.state_message = (
                        "No suitable workers.\nDetails:\n" + "".join(messages)
                    )
                if state_message != "":
                    model_instance.state_message = state_message

                await ModelInstanceService(session).update(model_instance)
                logger.debug(
                    f"No suitable workers for model instance {model_instance.name}, state: {model_instance.state}"
                )
            else:
                await apply_candidate_to_instance(
                    session, model, model_instance, candidate
                )

                logger.debug(
                    f"Scheduled model instance {model_instance.name} to worker "
                    f"{model_instance.worker_name} gpu {candidate.gpu_indexes}"
                )


async def apply_candidate_to_instance(
    session: AsyncSession,
    model: Model,
    model_instance: ModelInstance,
    candidate: ModelInstanceScheduleCandidate,
) -> None:
    """Write a chosen candidate onto its instance row.

    Extracted verbatim from `_schedule_one` so the group path writes members
    the same way the single path writes one. The alternative was a second
    write-back that starts identical and drifts: this block sets ten fields,
    two of which (`distributed_servers.mode`, the shared-cache re-resolve)
    are conditional on things a reader of the group path would not think to
    check.
    """
    model_instance.state = ModelInstanceStateEnum.SCHEDULED
    model_instance.state_message = ""
    model_instance.worker_id = candidate.worker.id
    model_instance.worker_name = candidate.worker.name
    model_instance.worker_ip = candidate.worker.ip
    model_instance.worker_advertise_address = candidate.worker.advertise_address
    model_instance.worker_ifname = candidate.worker.ifname
    model_instance.computed_resource_claim = candidate.computed_resource_claim
    model_instance.gpu_type = candidate.gpu_type
    model_instance.gpu_indexes = candidate.gpu_indexes
    model_instance.gpu_addresses = candidate.gpu_addresses
    model_instance.distributed_servers = DistributedServers(
        subordinate_workers=candidate.subordinate_workers,
    )
    if get_backend(model) in (
        BackendEnum.VLLM,
        BackendEnum.ASCEND_MINDIE,
        BackendEnum.SGLANG,
    ):
        model_instance.distributed_servers.mode = (
            DistributedServerCoordinateModeEnum.INITIALIZE_LATER
        )

    # Role-effective, not the Model's: attaching a cache to prefill alone is a
    # normal disaggregated configuration, and reading the deployment's value
    # would skip the re-resolve for exactly the member that asked for one.
    scheduled_cache = role_effective_model(model, model_instance.role).extended_kv_cache
    if scheduled_cache and scheduled_cache.is_shared():
        # The assigned worker is known now; re-resolve the shared-cache
        # snapshot so worker-dependent injection (e.g. the client's own
        # local_hostname) binds to this instance's node.
        model_instance.cache_config = await resolve_instance_cache_config_safe(
            session,
            model,
            worker=candidate.worker,
            spans_workers=model_instance.spans_workers,
            role=model_instance.role,
        )

    await ModelInstanceService(session).update(model_instance)


def _cards_per_member(model: Model) -> int:
    """How many whole cards one member of this deployment wants.

    Read off the engine's world size rather than `gpu_selector.gpus_per_replica`
    — on this path that field is `None`. `set_model_gpus_per_replica` returns
    early unless `gpu_selector.gpu_ids` is set, and manual card ids are
    mutually exclusive with `gpu_type_selector`, so it is never computed for an
    InstanceType claim. That is also why the admission check keyed on it never
    fires.
    """
    selector_map = {
        BackendEnum.VLLM.value: VLLMResourceFitSelector,
        BackendEnum.ASCEND_MINDIE.value: AscendMindIEResourceFitSelector,
        BackendEnum.SGLANG.value: SGLangResourceFitSelector,
    }
    selector = selector_map.get(model.backend)
    if selector is None:
        return 1
    try:
        result = selector.get_world_size_from_backend_parameters(model)
    except Exception as e:
        logger.warning(
            "Could not read the world size of %s from its backend parameters; "
            "assuming one card per member: %s",
            model.name,
            e,
        )
        return 1
    world_size, _ = result if result is not None else (None, None)
    return max(int(world_size or 1), 1)


def build_candidate_selector(
    config: Config,
    model: Model,
    model_instances: List[ModelInstance],
    cpu_only: bool = False,
    ram_claim: Optional[int] = None,
):
    """Which resource-fit selector answers "does one more member fit here".

    Extracted from `find_candidate` so the group scheduler's capacity count can
    ask the *same* question the placement path asks. `count_offer_slots` works
    by running a selector repeatedly against a growing instance list, and a
    second, separately-chosen selector would let the count and the placement
    disagree about the same worker — which is the one defect a capacity number
    must not have, because the disagreement surfaces as a group admitted into a
    domain that then cannot take it.

    `model` is expected to be role-projected already (`role_effective_model`):
    every branch here reads Model-level fields and none of them knows about
    roles.

    `ram_claim` is the accelerator-free role's declared memory, resolved by the
    caller before projection because `resources` is a role-OWN field. Only the
    `cpu_only` branch reads it — every other selector derives RAM from the
    weights it just sized.
    """
    if cpu_only:
        # Ahead of every backend branch, because the backend a router inherits
        # is the group's engine and every one of those selectors sizes the
        # model's weights. The router never loads them; asking for their VRAM
        # is what leaves it unschedulable on a host whose cards its own peers
        # have just filled.
        return CustomBackendResourceFitSelector(
            config, model, model_instances, cpu_only=True, ram_claim=ram_claim
        )
    if model.gpu_type_selector:
        # Whole cards and slices are two different questions on the same field.
        # A slice is a fraction of a card the node's device plugin picks, so
        # "one per worker" is a real limit there; whole cards have no such
        # difficulty and the operator hands out several at once. Branching here
        # rather than inside the selector keeps the slicing path byte-for-byte
        # unchanged — see policies/.../instance_type_whole_card_selector.py.
        if is_whole_card_claim(model.gpu_type_selector):
            return InstanceTypeWholeCardSelector(
                config,
                model,
                model_instances,
                cards_per_member=_cards_per_member(model),
            )
        return VGPUResourceFitSelector(config, model, model_instances)
    if is_gguf_model(model):
        return GGUFResourceFitSelector(model, model_instances, config.cache_dir)
    if model.backend == BackendEnum.ASCEND_MINDIE:
        return AscendMindIEResourceFitSelector(config, model, model_instances)
    if model.backend == BackendEnum.VLLM and not is_omni_model(model):
        # Note: Route omni categories to CustomSelector for vLLM-Omni.
        return VLLMResourceFitSelector(config, model, model_instances)
    if model.backend == BackendEnum.SGLANG:
        return SGLangResourceFitSelector(config, model, model_instances)
    return CustomBackendResourceFitSelector(config, model, model_instances)


def _pairing_affinity_max_score(
    configured: float, rest: Sequence[ScheduleCandidatesScorer]
) -> float:
    """What one opposite-role sibling has to be worth, on THIS chain.

    **Derived, because no constant stays true.** `CandidateScoreChain` sums,
    and `PairingAffinityScorer` deliberately pays a full `max_score` per
    sibling rather than normalising (see its own note), so "affinity first,
    capacity and topology only among equals" holds exactly while one sibling
    outweighs everything the rest of the chain can add. The configured 200 was
    chosen when the rest was `PlacementScorer` (100) plus the file-locality
    tiebreaker (5); `TopologyProximityScorer` then joined with a ceiling of
    150 x 3 = 450 on the built-in zone/rack/host chain, and 100 + 5 + 450 = 555
    quietly outranked it -- a scale-out went to the empty host in the group's
    own rack instead of to the host already holding two decodes. Writing a
    bigger number down would only postpone the next replay of that.

    Summed over `rest` -- the scorers actually appended, not the ones that
    could exist. `ModelFileLocalityScorer` is skipped when its weight is 0 and
    `TopologyProximityScorer` when the cluster declares no topology, and a
    chain that short should not have its affinity weight inflated on account of
    scorers that are not on it.

    The configured value is a floor, so raising
    `GPUSTACK_SCHEDULER_PAIRING_AFFINITY_MAX_SCORE` still raises it. 0 is not a
    floor but the documented off switch -- the scorer returns candidates
    untouched at `max_score <= 0` -- so it is handed back unchanged.
    """
    if configured <= 0:
        return configured
    return max(configured, sum(scorer.score_ceiling for scorer in rest) + 1)


def _group_scorer(
    group_id: Optional[str],
    role: Optional[str],
    cpu_only: bool,
    model_instances: List[ModelInstance],
    rest: Sequence[ScheduleCandidatesScorer] = (),
) -> Optional[ScheduleCandidatesScorer]:
    """The one extra scorer a group member gets, or None for everything else.

    **The whole PD placement policy on the per-instance path lives behind
    this single `group_id` check.** A model without `roles` has no `group_id`
    on any of its instances, so it returns None and the scoring chain is
    byte-for-byte what it was — which is the argument that none of this needs
    a compatibility case for existing deployments.

    Which scorer follows from which member this is, and the two are exclusive:

    - **The accelerator-free member** — the router today. It is deliberately
      outside the gang (`schedule_group` excludes it), and that left it
      outside everything: placed across the whole cluster, possibly a rack
      away from the members whose every token it forwards.
    - **A weight-holding member**, which on this path can only be a scale-out.
      Forming a group never reaches here — the solver places all of them at
      once — so a prefill or decode arriving alone means the group is already
      running and this is one more replica of one role. The solver cannot help
      with that, because it may not move what is already placed.

    `rest` is the chain this scorer is about to join, and only the pairing
    branch reads it -- the two branches being exclusive is also why the group
    locality weight never enters the sum that sizes pairing.
    """
    if not group_id:
        return None
    if cpu_only:
        # The router. `TopologyProximityScorer` already pulls it toward the
        # members it forwards for, and does it by the declared tree rather than
        # by host identity, so a second same-host bonus here would only restate
        # the tightest rung of an answer the chain already has.
        return None
    return PairingAffinityScorer(
        group_id,
        role,
        model_instances,
        max_score=_pairing_affinity_max_score(
            envs.SCHEDULER_PAIRING_AFFINITY_MAX_SCORE, rest
        ),
    )


async def _group_topology(
    session: AsyncSession,
    model: Model,
    group_id: Optional[str],
    workers: List[Worker],
) -> Optional[TopologyView]:
    """The cluster's tree, read only when a group member is being placed.

    Gated on `group_id` because that is the only thing the tree is used for
    here, and it is None for every deployment without roles -- so the read this
    costs is never paid by the models that were being scheduled before any of
    this existed. The worker list is the caller's, already in hand; only the
    cluster row is fetched.
    """
    if not group_id or not model.roles:
        return None
    try:
        cluster = await Cluster.one_by_id(session, model.cluster_id)
        return build_view(
            getattr(cluster, "topology", None),
            [w for w in workers if w.cluster_id == model.cluster_id],
        )
    except Exception as e:
        # A tree that cannot be read is not a floor of zero and not a distance
        # of infinity. Placing without it is the answer this path gave before
        # either policy existed, and the outcome is still reported by
        # `_gather_unmet`.
        logger.warning("Could not read the topology for model %s: %s", model.name, e)
        return None


def _gather_floor(
    model: Model,
    group_id: Optional[str],
    model_instances: List[ModelInstance],
    view: Optional[TopologyView],
    anchors: List[str],
) -> Optional[GatherFloorFilter]:
    """The floor filter for this member, or None when no floor was asked for.

    `PreferGather` gets none: it is a target, not a floor -- it is allowed to
    end up looser and says so on the model afterwards. Filtering for it would
    turn "aim for this" into "refuse below this", which is the other option and
    the one the operator did not pick.
    """
    if view is None:
        return None
    gather = getattr(model, "gather", None)
    if getattr(gather, "strategy", None) != GatherStrategyEnum.MUST_GATHER:
        return None
    return GatherFloorFilter(
        layer=getattr(gather, "layer", None),
        group_id=group_id,
        model_instances=model_instances,
        view=view,
        weight_bearing=anchors,
    )


async def find_candidate(
    session: AsyncSession,
    config: Config,
    model: Model,
    workers: List[Worker],
    model_instances: List[ModelInstance],
    role: Optional[str] = None,
    exclude_instance_id: Optional[int] = None,
    group_id: Optional[str] = None,
) -> Tuple[Optional[ModelInstanceScheduleCandidate], List[str]]:
    """
    Find a schedule candidate for the model instance.
    :param config: GPUStack configuration.
    :param model: Model to schedule.
    :param workers: List of workers to consider.
    :param role: Which role of a multi-role model is being placed. None for a
                 single-role deployment.
    :param exclude_instance_id: The instance being placed, if any. Its own
                 `computed_resource_claim` must not count against the GPUs it
                 is asking for -- see below.
    :param group_id: The generation this instance belongs to, when it is a
                 member of a role group. Used to place an accelerator-free
                 member near the siblings it forwards to, and to pull a
                 scaled-out prefill or decode toward the opposite role.
    :return: A tuple containing:
                - The schedule candidate.
                - A list of messages for the scheduling process.
    """

    # An instance may not be weighed against its own claim.
    #
    # `get_worker_allocatable_resource` derives a GPU's free VRAM as
    # `total - sum(claims of every instance on it) - system_reserved`, and the
    # caller hands it *every* row including the one being placed. A freshly
    # created instance has `computed_resource_claim = None`, so the sum skips
    # it and nothing goes wrong, so the ordinary first placement is unaffected.
    #
    # It stops holding as soon as an instance is placed, written a claim, and
    # then returns to PENDING (a retry, a re-deploy, a group re-solve). Now its
    # own claim is counted against the very GPUs it wants, and because the
    # claim is `gpu_memory_utilization x total`, allocatable collapses to the
    # remaining 10%. The GPU then fails the `allocatable/total >=
    # gpu_memory_utilization` test, every candidate is classed overcommit, and
    # a multi-replica model refuses overcommit outright. The retry re-reads the
    # same stale claim, so it never recovers on its own.
    #
    # The symptom is a replica stuck PENDING against cards that are empty
    # apart from its own claim.
    #
    # Filtered here and not in `ModelInstance.all()` at the call site: the
    # group path derives `group_instances` from that same list, and
    # `is_group_forming` has to be able to see the triggering member.
    if exclude_instance_id is not None:
        model_instances = [mi for mi in model_instances if mi.id != exclude_instance_id]

    # Read before projecting: the answer is a property of the ROLE — the router
    # is a proxy and loads no weights — and the projection flattens the role's
    # overrides onto the model, after which there is no role left to ask.
    cpu_only = role_takes_no_accelerator(model, role)
    # Same reason, same moment: `resources` is role-OWN too, and only the
    # accelerator-free branch consumes it.
    ram_claim = role_container_resources(model, role).memory if cpu_only else None
    # Read here too, and for the third time for the same reason: both of these
    # are properties of the MODEL and its ROLES, and the projection below
    # flattens the roles away.
    topology = await _group_topology(session, model, group_id, workers)
    anchors = [
        spec.name
        for spec in (model.roles or [])
        if not role_takes_no_accelerator(model, spec.name)
    ]
    floor = _gather_floor(model, group_id, model_instances, topology, anchors)
    proximity = (
        TopologyProximityScorer(
            group_id,
            model_instances,
            topology,
            anchors,
            max_score=envs.SCHEDULER_TOPOLOGY_PROXIMITY_MAX_SCORE,
        )
        if topology is not None
        else None
    )

    # Apply the role's overrides once, here. Every filter, selector and scorer
    # below is constructed from `model` and reads Model-level fields directly;
    # none of them knows about roles. A role-less model comes back unchanged.
    model = role_effective_model(model, role)

    # Filter workers.
    filters = [
        ClusterFilter(model),
        GPUMatchingFilter(model),
        LabelMatchingFilter(model),
        StatusFilter(model),
        BackendFrameworkFilter(model),
        LocalPathFilter(model),
        PDModeRuntimeFilter(model),
    ]
    if floor is not None:
        # Last, so its message names the workers the cheaper filters left --
        # "kept 0 of 1" after a label selector has already cut the fleet to one
        # host says something different from "kept 0 of 40".
        filters.append(floor)

    worker_filter_chain = WorkerFilterChain(filters)
    workers, filter_messages = await worker_filter_chain.filter(workers)
    messages = []
    if filter_messages:
        messages.append(str(ListMessageBuilder(filter_messages)) + "\n")
    if len(workers) == 0:
        return None, messages

    # Initialize candidate selector.
    try:
        candidates_selector = build_candidate_selector(
            config, model, model_instances, cpu_only=cpu_only, ram_claim=ram_claim
        )
    except Exception as e:
        return None, [f"Failed to initialize {model.backend} candidates selector: {e}"]

    # Select candidates.
    candidates = await candidates_selector.select_candidates(workers)

    if proximity is not None:
        # Before scoring, and by removal rather than by weight: "do not split
        # this member across machines more than it has to be" is the FIRST key
        # of the ordering, and a summing chain cannot hold two strict
        # priorities. `PairingAffinityScorer` already claims the other one and
        # is unbounded (`max_score` per opposite sibling), so two siblings
        # outweigh the split penalty and buy a member spread over two machines
        # -- trading an all-reduce that runs once per layer per token for a KV
        # transfer that runs once per request.
        #
        # A no-op unless the question actually arises: no topology, no group,
        # or -- the ordinary case -- every candidate on a single machine, where
        # they all tie at the tightest rung. It never empties the set, so a
        # role wider than any one machine still schedules.
        candidates = proximity.narrow_to_tightest_internal_spread(candidates)

    # Score candidates.
    candidate_scorers = [
        PlacementScorer(model, model_instances),
    ]
    locality_max_score = envs.SCHEDULER_SCALE_UP_LOCALITY_MAX_SCORE
    if locality_max_score > 0:
        candidate_scorers.append(
            ModelFileLocalityScorer(
                model,
                draft_model_source=await get_draft_model_source(session, model),
                max_score=locality_max_score,
            )
        )
    if proximity is not None:
        # Beside the group scorer rather than instead of it: that one answers
        # "which host has the most of the opposite role" -- the odds a
        # request's two ends land together -- and this one answers "how far is
        # this host from the group at all". Under 3P1D the first deliberately
        # prefers the host holding the single decode over the one holding two
        # prefills, which no notion of distance would produce.
        candidate_scorers.append(proximity)
    # Last, and that is not cosmetic: `_group_scorer` sizes pairing affinity
    # against the ceilings of the scorers already on the chain, so every other
    # scorer has to be appended by now. The chain resets `candidate.score`
    # between scorers and sums the results, so the position itself changes no
    # score.
    group_scorer = _group_scorer(
        group_id, role, cpu_only, model_instances, candidate_scorers
    )
    if group_scorer is not None:
        candidate_scorers.append(group_scorer)
    candidates = await CandidateScoreChain(candidate_scorers).score(candidates)

    # Pick the highest score candidate.
    candidate = pick_highest_score_candidate(candidates)

    # Collect messages.
    if candidate is None and len(workers) > 0:
        resource_fit_messages = candidates_selector.get_messages() or [
            "No workers meet the resource requirements."
        ]
        messages.extend(resource_fit_messages)
    elif candidate and candidate.overcommit:
        messages.extend(candidates_selector.get_messages())

    # Return the candidate and messages.
    return candidate, messages


def pick_highest_score_candidate(candidates: List[ModelInstanceScheduleCandidate]):
    """
    Pick the most offload layers from candidates.
    Args:
        candidates: List of ModelInstanceScheduleCandidate.
    """

    logger.debug(f"Pick highest score candidate from {len(candidates)} candidates")

    if len(candidates) == 0:
        return None

    candidate = candidates[0]
    for i in range(1, len(candidates)):
        if candidates[i].score > candidate.score:
            candidate = candidates[i]

    return candidate


async def evaluate_gguf_model(
    model: Model,
    workers: Optional[List[Worker]] = None,
) -> bool:

    task_output = await calculate_gguf_model_resource_claim(
        model, offload=GPUOffloadEnum.Full, workers=workers
    )
    if (
        task_output.resource_architecture
        and not task_output.resource_architecture.is_deployable()
    ):
        raise ValueError(
            "Unsupported model. To proceed with deployment, ensure the model is supported by backend, or deploy it using a custom backend version or custom backend."
        )

    should_update = False
    if task_output.resource_claim_estimate.reranking and not model.categories:
        should_update = True
        model.categories = [CategoryEnum.RERANKER]

    if task_output.resource_claim_estimate.embeddingOnly and not model.categories:
        should_update = True
        model.categories = [CategoryEnum.EMBEDDING]

    if task_output.resource_claim_estimate.imageOnly and not model.categories:
        should_update = True
        model.categories = [CategoryEnum.IMAGE]

    if not model.categories:
        should_update = True
        model.categories = [CategoryEnum.LLM]

    if task_output.resource_claim_estimate.distributable and not model.distributable:
        should_update = True
        model.distributable = True

    if model.gpu_selector and model.gpu_selector.gpu_ids:
        worker_gpu_ids = group_gpu_ids_by_worker(model.gpu_selector.gpu_ids)
        if (
            len(worker_gpu_ids) > 1
            and model.distributable
            and not model.distributed_inference_across_workers
        ):
            should_update = True
            model.distributed_inference_across_workers = True

        gpus_per_replica_modified = set_model_gpus_per_replica(model)
        should_update = should_update or gpus_per_replica_modified

    return should_update


async def evaluate_diffusion_model(
    model: Model,
    workers: Optional[List[Worker]] = None,
):
    """
    Evaluate diffusion model and update model categories.

    Args:
        model: Model to evaluate
        workers: Optional list of workers (for LOCAL_PATH remote read)

    Returns:
        True if the model is a diffusion model, False otherwise
    """
    # vLLM/SGLang support Diffusers (image) models.
    # If the source (HF/ModelScope/Local Path) contains model_index.json with "_diffusers_version",
    # classify as IMAGE directly.
    if model.categories and CategoryEnum.IMAGE not in model.categories:
        return False

    hf_token = get_global_config().huggingface_token

    # For Hub sources and local files, use hub.py function
    if model.source in (SourceEnum.HUGGING_FACE, SourceEnum.MODEL_SCOPE):
        is_diffusers = await asyncio.wait_for(
            asyncio.to_thread(has_diffusers_model_index, model, token=hf_token),
            timeout=10,
        )
    # For LOCAL_PATH, try local first, then workers
    elif model.source == SourceEnum.LOCAL_PATH:
        # Try local read first
        is_diffusers = await asyncio.wait_for(
            asyncio.to_thread(has_diffusers_model_index, model, token=hf_token),
            timeout=10,
        )
        # If not found locally and workers are provided, query workers
        if not is_diffusers and workers:
            is_diffusers = await asyncio.wait_for(
                check_diffusers_model_index_from_workers(model, workers),
                timeout=10,
            )
    else:
        return False

    if is_diffusers:
        model.categories = [CategoryEnum.IMAGE]
        return True
    return False


async def prioritize_workers_with_model_files(
    session: AsyncSession, model: Model, workers: List[Worker]
) -> List[Worker]:
    """
    Prioritize workers that have the model files. This helps optimization for getting model config from remote worker local paths.

    Args:
        session: Database session for querying worker files.
        model: Model to check for.
        workers: List of workers to prioritize.

    Returns:
        List of prioritized workers.
    """
    if not workers:
        return []

    source_index = model.model_source_index
    if not source_index:
        return workers

    model_files = await ModelFileService(session).get_by_source_index(source_index)
    if not model_files:
        return workers

    worker_ids_with_ready_files = {
        mf.worker_id for mf in model_files if mf.state == ModelFileStateEnum.READY
    }

    # Put workers with ready model files at the front
    sorted_workers = sorted(
        workers,
        key=lambda w: 0 if w.id in worker_ids_with_ready_files else 1,
    )
    return sorted_workers


async def evaluate_pretrained_config(
    model: Model,
    workers: Optional[List[Worker]] = None,
    raise_raw: bool = False,
) -> bool:
    """
    evaluate the model's pretrained config to determine its categories, meta and gpus_per_replica.
    Args:
        model: Model to evaluate.
        workers: Optional list of workers (for LOCAL_PATH).
        raise_raw: If True, raise the raw exception.
    Returns:
        True if the model's categories are updated, False otherwise.
    """
    # 1) try to evaluate as diffusion model
    try:
        is_image_category = await evaluate_diffusion_model(model, workers=workers)
        if is_image_category:
            return True
    except Exception:
        pass
    # 2) Check overrided architectures if specified in backend parameters.
    architectures = get_vllm_override_architectures(model)
    if not architectures:
        try:
            trust_remote_code = _extract_trust_remote_code(model)
            pretrained_config = await get_pretrained_config_with_workers(
                model,
                workers=workers,
                trust_remote_code=trust_remote_code,
            )
        except ValueError as e:
            # Skip value error exceptions and defaults to LLM catagory for certain cases.
            if should_skip_architecture_check(model):
                model.categories = model.categories or [CategoryEnum.LLM]
                return True

            if raise_raw:
                raise

            logger.debug(
                f"Failed to get config for model {model.name or model.readable_source}, ValueError: {e}"
            )
            raise simplify_auto_config_value_error(e)
        except (TimeoutError, asyncio.TimeoutError) as e:
            raise Exception(
                f"Timeout while getting config for model {model.name or model.readable_source}: {e}."
            )
        except Exception as e:
            raise Exception(
                f"Failed to get config for model {model.name or model.readable_source}: {e}"
            )

        architectures = getattr(pretrained_config, "architectures", []) or []
        if not architectures and not model.backend_version:
            raise ValueError(
                "Unrecognized architecture. To proceed with deployment, ensure the model is supported by backend, or deploy it using a custom backend version or custom backend."
            )

    model_type = detect_model_type(architectures)

    # TODO : Additional checks for unsupported architectures for other backends.
    if (
        model.backend == BackendEnum.VLLM
        and model_type == CategoryEnum.UNKNOWN
        and not model.backend_version
    ):
        raise ValueError(
            f"Unsupported architecture: {architectures}. To proceed with deployment, ensure the model is supported by backend, or deploy it using a custom backend version or custom backend."
        )

    meta_modified = False
    if not model.meta and (known_meta := get_model_meta(pretrained_config)):
        model.meta = known_meta
        meta_modified = True

    categories_modified = set_model_categories(model, model_type)
    gpus_per_replica_modified = set_model_gpus_per_replica(model)
    return categories_modified or gpus_per_replica_modified or meta_modified


def _extract_trust_remote_code(model: Model) -> bool:
    """Extract trust_remote_code from model backend parameters."""
    if model.backend_parameters and "--trust-remote-code" in model.backend_parameters:
        return True
    return False


def get_vllm_override_architectures(model: Model) -> List[str]:
    """
    Get the vLLM override architectures from the model's backend parameters.
    Args:
        model: Model to check.
    Returns:
        List of override architectures.
    """
    backend = get_backend(model)
    if backend != BackendEnum.VLLM:
        return []

    hf_overrides = find_parameter(model.backend_parameters, ["hf-overrides"])
    if hf_overrides:
        overrides_dict = json.loads(hf_overrides)
        return overrides_dict.get("architectures", [])
    return []


def should_skip_architecture_check(model: Model) -> bool:
    """
    Check if the model should skip architecture check.
    Args:
        model: Model to check.
    Returns:
        True if the model should skip architecture check, False otherwise.
    """

    if (
        model.backend == BackendEnum.CUSTOM
        or not is_built_in_backend(model.backend)
        or model.backend_version
    ):
        # New model architectures may be added with custom backend/version.
        return True

    if model.backend_parameters and find_parameter(
        model.backend_parameters, ["tokenizer-mode"]
    ):
        # Models like Pixtral may not provide compatible config but still work with custom parameters.
        return True

    return False


def simplify_auto_config_value_error(e: ValueError) -> ValueError:
    """
    Simplify the error message for ValueError exceptions.
    """
    message = str(e)
    if "trust_remote_code=True" in message:
        return ValueError(
            "The model contains custom code that must be executed to load correctly. If you trust the source, please pass the backend parameter `--trust-remote-code` to allow custom code to be run."
        )

    if "pip install --upgrade transformers" in message:
        return ValueError(
            "Unsupported model. To proceed with deployment, ensure the model is supported by backend, or deploy it using a custom backend version or custom backend."
        )

    return ValueError(f"Not a supported model.\n\n{message}")


def set_model_categories(model: Model, model_type: CategoryEnum) -> bool:
    if model.categories:
        return False

    if model_type == CategoryEnum.UNKNOWN:
        # Default to LLM for unknown architectures
        model.categories = [CategoryEnum.LLM]
    else:
        model.categories = [model_type]

    return True


def set_model_gpus_per_replica(model: Model) -> bool:
    """
    Set the model's gpu_selector.gpus_per_replica based on its gpu_selector.gpu_ids and backend parameters.
    Args:
        model: Model to set.
    Returns:
        True if the model's gpu_selector.gpus_per_replica is updated, False otherwise.
    """

    def calculate_gpus_per_replica(model: Model) -> int:
        if model.backend == BackendEnum.VOX_BOX.value:
            return 1

        replicas = model.replicas or 1
        gpus_per_replica = len(model.gpu_selector.gpu_ids) // replicas

        if should_skip_gpu_count_check(model):
            return gpus_per_replica

        # User-specified world size from backend parameters takes precedence.
        if model.backend_parameters is not None:
            selector_map = {
                BackendEnum.VLLM.value: VLLMResourceFitSelector,
                BackendEnum.ASCEND_MINDIE.value: AscendMindIEResourceFitSelector,
                BackendEnum.SGLANG.value: SGLangResourceFitSelector,
            }
            selector = selector_map.get(model.backend)
            world_size = None
            if selector:
                result = selector.get_world_size_from_backend_parameters(model)
                world_size, _ = result if result is not None else (None, None)
            if world_size and world_size > 0:
                return world_size

        # The largest power of 2 less than or equal to (total GPUs / replicas), used as the initial per-replica GPU count.
        return largest_power_of_2_leq(gpus_per_replica)

    if not model.gpu_selector or not model.gpu_selector.gpu_ids:
        return False

    if model.gpu_selector.gpus_per_replica and model.gpu_selector.gpus_per_replica > 0:
        return False

    gpus_per_replica = calculate_gpus_per_replica(model)
    model.gpu_selector.gpus_per_replica = gpus_per_replica
    try:
        flag_modified(model, "gpu_selector")
    except AttributeError:
        # Ignore if the given model is not a SQLModel instance.
        pass
    return True
