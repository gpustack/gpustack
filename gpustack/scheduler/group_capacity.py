"""The bridge between "how big is this domain" and the selectors that know.

`solve_group_placement` takes a `CapacityFn` and had no implementation of one:
the only thing that can answer "how many members of this role fit on that
worker" is `count_offer_slots`, and that needs a selector factory, which needs
the role projection and the worker filters — i.e. everything `find_candidate`
does before it starts scoring. This module is that assembly, and it exists as
its own file so both consumers (the deployment form's feasibility preview and,
later, the scheduler's own group placement) go through one implementation.

**Why not reuse `find_candidate` directly.** It answers "where does one more
member go", scores the result and returns a single winner. The group solver
needs the *count* per worker, which is a different question with a different
cost profile — and going through the scoring path once per hypothetical member
per worker per domain would multiply a full selector sweep by three dimensions.
`count_offer_slots` exists precisely to collapse that.

**A worker whose capacity could not be established is left out of the mapping,
never reported as zero.** The solver reads absence as "unknown" and counts it
separately, which is what lets a refusal say "we could not measure four
workers" instead of "the cluster is full" — and those two call for opposite
reactions from an operator.
"""

from __future__ import annotations

import logging
from typing import Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

from gpustack.config.config import Config
from gpustack.scheduler import port_budget
from gpustack.policies.base import MemberResourceClaim, WorkerFilterChain
from gpustack.policies.utils import (
    group_workers_by_gpu_memory_size,
    worker_largest_gpu_memory,
)
from gpustack.policies.worker_filters.backend_framework_filter import (
    BackendFrameworkFilter,
)
from gpustack.policies.worker_filters.cluster_filter import ClusterFilter
from gpustack.policies.worker_filters.gpu_matching_filter import GPUMatchingFilter
from gpustack.policies.worker_filters.label_matching_filter import LabelMatchingFilter
from gpustack.policies.worker_filters.local_path_filter import LocalPathFilter
from gpustack.policies.worker_filters.pd_mode_filter import PDModeRuntimeFilter
from gpustack.policies.worker_filters.status_filter import StatusFilter
from gpustack.schemas.models import (
    Model,
    ModelInstance,
    role_effective_model,
    role_container_resources,
    role_takes_no_accelerator,
)
from gpustack.schemas.workers import Worker
from gpustack.scheduler.offer_slot import _stand_in_for, count_offer_slots

logger = logging.getLogger(__name__)


def _reports_no_memory(worker: Worker) -> bool:
    """Whether this worker's system telemetry is absent rather than small.

    Zero total RAM is not a quantity, it is a gap: no host runs on none, and a
    worker that says so has failed to measure itself. Told apart from a real
    shortage because the two call for opposite reactions -- free some memory,
    versus go and look at why that agent is not reporting.
    """
    status = getattr(worker, "status", None)
    memory = getattr(status, "memory", None) if status else None
    return not getattr(memory, "total", None)


class _RoleProjection(NamedTuple):
    """What `_eligible_for` worked out for a role, read back by two callers.

    A NamedTuple rather than a bare tuple because the role-OWN fields read
    before projection are a growing set — the accelerator-free flag, now
    `ram_claim` — and a
    positional tuple couples every read site, plus every test that builds one,
    to that count. `ram_claim` defaults so a caller that only cares about
    placement need not spell it.
    """

    model: Model
    cpu_only: bool
    ram_claim: Optional[int] = None


class GroupCapacity:
    """A `CapacityFn` for one model, with the per-role setup done once.

    Stateful on purpose. The role projection, the filter sweep and the selector
    choice do not change between domains, and the solver asks for the same role
    against different worker subsets — redoing the setup per call was measured
    upstream at a full selector sweep per domain per layer.
    """

    def __init__(
        self,
        config: Config,
        model: Model,
        workers: Sequence[Worker],
        model_instances: Sequence[ModelInstance],
        cache_instances: Sequence[object] = (),
    ):
        self._config = config
        self._model = model
        self._workers = {w.id: w for w in workers}
        self._model_instances = list(model_instances)
        self._cache_instances = list(cache_instances)
        # worker_id -> ports already spoken for there. Computed on first use
        # and kept, because a solve asks about the same workers once per role.
        self._ports_taken: Dict[int, int] = {}
        # Rebuilt per counting pass from what the solve has committed;
        # empty until the first role is placed.
        self._committed_ports: Dict[int, int] = {}
        # role -> {worker_id: worker}, after that role's filters.
        self._eligible: Dict[Optional[str], Dict[int, Worker]] = {}
        self._projected: Dict[Optional[str], "_RoleProjection"] = {}
        # (role, worker_id) -> the candidates the count produced there, in the
        # order it produced them. See `_translate`.
        self._offers: Dict[tuple, List[object]] = {}
        # role -> the selectors' own account of why no more fit, deduplicated.
        # A group refusal counts members; these are what one member costs and
        # what stood in its way, and without them the two refusal paths
        # describe the same cluster in incomparable units.
        self._notes: Dict[str, List[str]] = {}
        # role -> what the FILTER CHAIN said while narrowing the fleet, kept in
        # a list of its own rather than pushed through `_remember_notes`.
        # The separation is the point: `_MAX_NOTES` caps the capacity notes
        # because on a fleet of equally-full workers those are one line per
        # worker, and a refusal nobody reads to the end explains nothing. The
        # filter lines are one per filter and they are the ones that name the
        # *mistake* — "Matched 0/3 workers by label selector: {...}" — so
        # sharing the budget would let a wide fleet's per-worker lines crowd
        # out either those or "The model requires approximately X GiB of VRAM",
        # which is the note the whole refusal is measured against. Two lists,
        # one cap, and `notes_for` puts the filter lines first because they
        # explain why the capacity lines are about so few workers. The
        # one-GPU-size sentence sits here for that same reason; see
        # `_note_one_gpu_size`.
        self._filter_notes: Dict[str, List[str]] = {}
        # Roles that already carry the one-GPU-size sentence, so the passes
        # after the first do not each add their own smaller count of the same
        # rule. See `_note_one_gpu_size`.
        self._size_noted: Set[str] = set()
        # role -> what the selector priced ONE member of it at. Kept from every
        # counting pass, so a refusal can show what each role wants even though
        # no placement exists to read a claim off.
        self._claims: Dict[str, MemberResourceClaim] = {}
        # (role, primary worker id) -> the machines one member of that role
        # occupies, primary first. Empty for every role that fits on a single
        # machine, which is every role placed today.
        self._spans: Dict[tuple, List[int]] = {}

    async def __call__(
        self,
        role: str,
        worker_ids: Sequence[int],
        already_placed: Sequence[object],
    ) -> Dict[int, int]:
        """How many more members of `role` each of `worker_ids` can take.

        `already_placed` is what the solve has committed so far, in the shape
        the allocation accounting reads. It is appended to the instance list
        rather than merged into it, which is what keeps the second role honest
        about what the first one took.
        """
        return await self._count(role, worker_ids, already_placed, one_gpu_size=True)

    async def sizing(self, role: str, worker_ids: Sequence[int]) -> Dict[int, int]:
        """The same count, with the one-GPU-size rule left off.

        Used to order the domains of the search and to skip the ones proven
        empty -- both questions about a domain's *size*, asked once over the
        whole tree.

        🔑 **The size rule must not run here.** It picks the band holding the
        most members, and picking one across the whole cluster zeroes every
        worker in the losing bands -- including every worker of a domain made
        entirely of them. That domain then sizes as empty and is dropped from
        the search, although its members would have sat on one size of card
        perfectly well. The rule belongs where it is decided: inside a domain,
        where all of one role's members actually land.

        Ordering on the unbanded count overstates what a banded placement can
        use, and that is the safe direction: a domain worth examining is
        examined, and `_fit_in_domain` applies the rule for real.

        Args:
            role: The role to size domains for -- the hungriest one.
            worker_ids: Every worker under the tree.

        Returns:
            worker_id -> slots, with absence still meaning unmeasured.
        """
        return await self._count(role, worker_ids, (), one_gpu_size=False)

    async def _count(  # noqa: C901
        self,
        role: str,
        worker_ids: Sequence[int],
        already_placed: Sequence[object],
        one_gpu_size: bool,
    ) -> Dict[int, int]:
        eligible = await self._eligible_for(role)
        # No early return when nothing is eligible. An empty `eligible` is
        # not a gap in what we know — it is the filter chain having excluded
        # every worker on purpose, which is a measured zero and exactly the
        # verdict the per-worker branch below reaches one worker at a time
        # (`eligible.get(worker_id) is None` -> `out[worker_id] = 0`). Handing
        # back `{}` made the solver read every worker as unmeasured, and an
        # e2e run whose `worker_selector` named a label no worker carries was
        # refused with "the cluster has room for 0, but capacity could not be
        # measured on 17 worker(s) — the shortfall may be smaller than it looks,
        # or there may be none": a hedge about broken telemetry put in front of
        # an operator whose selector was simply a typo, while the
        # single-instance path answered the same question with "Matched 0/3
        # workers by label selector".
        projected = self._projected[role]
        instances = self._model_instances + self._translate(already_placed)
        limit = self._limit_for(role)
        # VRAM crosses roles through `_translate`; ports did not. `_share_out`
        # deals two roles onto the same roomiest worker by design, and the
        # second role was priced against the free-port count the snapshot had
        # before the first role committed anything -- so a concentrated group
        # is admitted for more ports than the range holds and its trailing
        # members wedge in `starting`, which is the wedge this budget exists
        # to prevent.
        self._committed_ports = self._committed_port_demand(already_placed)

        out: Dict[int, int] = {}
        for worker_id in worker_ids:
            worker = eligible.get(worker_id)
            if worker is None:
                # Filtered out for this role — a measured, definite zero, not
                # an unknown. Present-and-zero is what tells the solver the
                # domain is genuinely too small rather than unmeasurable.
                out[worker_id] = 0
                continue
            if _reports_no_memory(worker):
                # Left out, not zeroed. A host that reports `memory.total`
                # of 0 has not told us it is full, it has told us nothing --
                # and every role wants some RAM, so a definite zero here reads
                # to the solver as a host with no room and the refusal comes
                # out as "not enough room" on a fleet with idle cards. Observed
                # on a worker whose GPU telemetry was fine and whose system
                # telemetry was empty: the group was refused while single
                # instances kept landing there, because that path accepts the
                # overcommitted candidate this one is right to refuse.
                logger.debug(
                    "Worker %s reports no system memory; its capacity for role "
                    "%r is unknown rather than zero",
                    worker_id,
                    role,
                )
                continue
            offer = await count_offer_slots(
                make_selector=lambda instances_now, p=projected: (
                    self._selector(p.model, instances_now, p.cpu_only, p.ram_claim)
                ),
                workers=[worker],
                model_instances=instances,
                limit=limit,
            )
            if offer.placements:
                # Learned here so `_translate` can turn the solver's commits
                # into something the allocation accounting can read. The whole
                # list, in order: the second member of a role on this worker
                # gets the second set of cards, and remembering only the first
                # is what made every later member invisible to the next role.
                self._offers[(role, worker_id)] = list(offer.placements)
            self._remember_notes(role, offer.notes)
            self._remember_claim(role, offer.claim)
            if offer.unavailable and offer.slots == 0:
                # Nothing was proven about this worker. Leaving it out is the
                # difference between "no room" and "we could not look".
                continue
            # One offer per worker here, so its first placement is this
            # worker's own.
            first = (getattr(offer, "placements", None) or [None])[0]
            out[worker_id] = self._within_port_budget(
                role,
                worker_id,
                offer,
                len(getattr(first, "gpu_indexes", None) or []) if first else 0,
            )

        # Narrowed before the spanning fallback below, and it cannot divert a
        # domain into it: the band that wins is the one holding the most
        # members, so it is non-empty whenever anything fit at all.
        if one_gpu_size:
            out = self._one_gpu_size_only(role, eligible, out)
        if any(out.values()):
            return out
        return await self._spanning(
            role, worker_ids, eligible, instances, limit, out, one_gpu_size
        )

    def _one_gpu_size_only(
        self,
        role: str,
        eligible: Dict[int, Worker],
        measured: Dict[int, int],
    ) -> Dict[int, int]:
        """`measured`, with every GPU size but the roomiest one zeroed.

        A member reserves a fraction of its card's total memory (vLLM's
        `--gpu-memory-utilization`, SGLang's `--mem-fraction-static`), so its KV
        cache grows with that total. Two members of one role on a 48 GiB and a
        32 GiB card therefore hold caches of different sizes while the router
        shares requests between them as equals, and the small one queues. The
        engines do not take uneven VRAM inside one replica either, and nothing
        in a deployment pins a role to one kind of card -- `gpu_type` names a
        framework and `gpu_type_selector` names an operator's vGPU pool -- so
        the scheduler is where the sizes have to be kept together.

        The band holding the most members wins. A tie goes to the bigger card,
        which leaves the most headroom behind, and then to the lowest worker id
        so an unchanged cluster keeps producing the same plan.

        **The losing bands are zeroed, never dropped.** The solver reads an
        absent worker as unmeasured and hedges its refusal accordingly
        ("capacity could not be measured on 4 worker(s)"), which sends an
        operator to look at agent telemetry. These workers were measured and
        then ruled out by a rule, and only a definite zero says so.

        Only within one role: prefill on 48 GiB cards and decode on 32 GiB ones
        is a deployment people run on purpose. A role that occupies no
        accelerator -- the router -- has no card to match and is handed back
        untouched.

        Args:
            role: The role these slots were counted for.
            eligible: That role's workers by id, as `_eligible_for` left them.
            measured: worker_id -> slots, as the counting pass established it.

        Returns:
            `measured` with the losing bands' workers set to zero.
        """
        if role_takes_no_accelerator(self._model, role):
            return measured
        bands = group_workers_by_gpu_memory_size(
            [eligible[worker_id] for worker_id in measured if worker_id in eligible]
        )
        if len(bands) < 2:
            # One size, or no worker that reports one. Nothing to choose
            # between, and the answer is what was measured.
            return measured

        ranked = sorted(bands, key=lambda band: self._band_rank(band, measured))
        winner = ranked[0]
        out = dict(measured)
        for band in ranked[1:]:
            for worker in band:
                out[worker.id] = 0
        held = sum(measured.get(worker.id, 0) for worker in winner)
        logger.debug(
            "Role %r is held to one GPU size: %d of %d band(s) zeroed, the "
            "one kept holds %d member(s)",
            role,
            len(bands) - 1,
            len(bands),
            held,
        )
        self._note_one_gpu_size(role, held)
        return out

    @staticmethod
    def _band_rank(
        band: Sequence[Worker], measured: Dict[int, int]
    ) -> Tuple[int, int, int]:
        """How good a band is, as a sort key: most members first.

        The card size comes second and negated, so the bigger card wins a tie
        on member count; the lowest worker id comes last, which is a tie-break
        that exists only to be reproducible.
        """
        return (
            -sum(measured.get(worker.id, 0) for worker in band),
            -max((worker_largest_gpu_memory(worker) or 0) for worker in band),
            min(worker.id for worker in band),
        )

    def _widest_band(self, workers: List[Worker]) -> List[Worker]:
        """`workers`, narrowed to the roomiest same-size band among them.

        Ranked by `_band_rank` with nothing measured, which is what this case
        is: no machine held a member on its own, so the key falls through to
        "the bigger card, then the lowest worker id" — the most headroom, and
        reproducible.
        """
        bands = group_workers_by_gpu_memory_size(workers)
        if len(bands) < 2:
            return workers
        return sorted(bands, key=lambda band: self._band_rank(band, {}))[0]

    def _note_one_gpu_size(self, role: str, held: int) -> None:
        """Say that mixed card sizes, not a full cluster, is what ran short.

        Kept beside the FILTER lines rather than under the `_MAX_NOTES` budget,
        for the reason that budget exists: this is one sentence naming the
        *rule* a fleet fell foul of, and the capacity lines it would share a cap
        with are one per worker on a wide fleet. It also reads first, which is
        the order `notes_for` wants -- it explains why the capacity lines beside
        it are about so few workers.

        Written once per role. A solve measures a role several times -- the
        whole tree for sizing, then domain by domain -- and the later passes see
        smaller slices of the same fleet, so a second sentence would report a
        worse shortfall about a narrower question.

        Recorded only on a shortfall, because the refusal it belongs to only
        happens then. A line saying the matching set holds every member the role
        needs would read as a contradiction of the refusal carrying it.

        Args:
            role: The role whose bands were narrowed.
            held: How many members the winning band was measured to hold.
        """
        needed = self._limit_for(role)
        if held >= needed or role in self._size_noted:
            return
        self._size_noted.add(role)
        self._filter_notes.setdefault(role, []).append(
            f"Role {role!r} must sit on GPUs of the same size; the largest "
            f"matching set holds {held} of the {needed} members it needs."
        )

    async def _spanning(
        self,
        role: str,
        worker_ids: Sequence[int],
        eligible: Dict[int, Worker],
        instances: List[object],
        limit: int,
        measured: Dict[int, int],
        one_gpu_size: bool = False,
    ) -> Dict[int, int]:
        """Capacity for a member that needs more machines than any one has.

        **Reached only when no single machine holds even one.** That is the
        whole compatibility argument: a role that fits on a host never comes
        here, so every deployment placed today is placed by exactly the code
        that placed it yesterday, and pays nothing for this — not even the
        extra selector pass.

        **The width is discovered, not computed.** Deriving it would mean
        reimplementing the selectors' own rules about which machines may be
        combined — equal GPU counts, one accelerator type, attention heads
        divisible by the tensor-parallel size — and a second implementation of
        those is a second set of answers. So the domain is handed over whole
        and the selector says what it made of it; how many machines it took is
        read back off the candidate.

        The results are keyed by each combination's *primary* machine, which
        keeps the solver's arithmetic untouched: it goes on dealing members to
        worker ids, one per slot, and never learns that some of those slots are
        two machines wide. `commit` reassembles the rest from `_spans`.
        """
        if role_takes_no_accelerator(self._model, role):
            # A proxy is one process. There is nothing to spread and no
            # collective to pay for spreading it.
            return measured
        # Read off the PROJECTION, not the Model. `Model` is a SQLModel table
        # class, so its validators do not run on construction and the field
        # arrives as None -- the default that makes this true for vLLM, SGLang
        # and MindIE is applied by `ModelSpecBase.set_defaults`, which the role
        # projection goes through. Reading the raw row would switch the whole
        # path off for every deployment whose column was never written.
        if not getattr(
            self._projected[role].model, "distributed_inference_across_workers", False
        ):
            # The deployment said its members may not be split. That is an
            # answer, not a gap: the refusal that follows should send the
            # reader to this switch rather than to a bigger machine.
            return measured
        usable = [
            eligible[worker_id]
            for worker_id in worker_ids
            if worker_id in eligible and not _reports_no_memory(eligible[worker_id])
        ]
        if one_gpu_size:
            # The same rule the single-machine pass applied, and it has to be
            # applied again here rather than inherited: that pass zeroed the
            # losing bands in `measured`, but this one rebuilds its own
            # candidate set from `eligible` and would otherwise combine the
            # machines it just ruled out -- or mix sizes across the combination
            # itself, which the selectors' cross-node rules do not prevent
            # (they govern GPU type and count, not per-card memory). A replica
            # spread over a 48 GiB card and a 32 GiB one is uneven VRAM inside
            # one replica, which the engines do not take.
            usable = self._widest_band(usable)
        if len(usable) < 2:
            # Fewer than two machines to combine, so there is no combination to
            # try and `measured` is already the whole answer. This is also the
            # door the all-filtered-out case leaves by: with nothing eligible,
            # `usable` is empty and every entry in `measured` is the definite
            # zero the caller just wrote. Nothing above it reads `eligible`
            # except through `worker_ids`, and `self._projected[role]` is
            # written by `_eligible_for` before it filters anything, so both
            # reads on the way here hold for a role no worker can host.
            return measured

        offer = await count_offer_slots(
            make_selector=lambda instances_now, p=self._projected[role]: (
                self._selector(p.model, instances_now, p.cpu_only, p.ram_claim)
            ),
            workers=usable,
            model_instances=instances,
            limit=limit,
        )
        self._remember_notes(role, offer.notes)
        self._remember_claim(role, offer.claim)
        if not offer.placements:
            return measured

        out = dict(measured)
        for candidate in offer.placements:
            primary = candidate.worker.id
            span = [primary] + [
                subordinate.worker_id
                for subordinate in (
                    getattr(candidate, "subordinate_workers", None) or []
                )
            ]
            self._offers[(role, primary)] = [candidate]
            self._spans[(role, primary)] = span
            # The cap is the tightest of the machines the member lands on,
            # not the primary's. A member needs its side-channel ports on every
            # host that carries one of its ranks, so a host short of them stops
            # the whole combination rather than a fraction of it.
            cards = len(getattr(candidate, "gpu_indexes", None) or [])
            allowed = min(
                self._within_port_budget(role, worker_id, offer, cards)
                for worker_id in span
            )
            out[primary] = min(out.get(primary, 0) + 1, max(allowed, 0))
        logger.debug(
            "Role %r does not fit on any single machine; %d combination(s) of "
            "%d machines each",
            role,
            len(offer.placements),
            len(self._spans.get((role, offer.placements[0].worker.id), [])),
        )
        return out

    def spans_for(self, role: str, worker_id: int) -> List[int]:
        """The machines one member of `role` occupies when placed at `worker_id`.

        A single machine for every role placed today, and that is what the
        caller is usually checking: the solver ranks the placements of one
        layer and has to drop any that split a member across machines before
        it scores the rest, because an all-reduce per layer per token is not a
        cost a bounded score can be trusted to outweigh.

        It cannot read that off the placement -- a member spanning machines
        appears in `assignments` once, keyed by its primary -- so it asks
        here, where the width was discovered.

        Args:
            role: The role the member belongs to.
            worker_id: The member's primary machine, as the solve assigned it.

        Returns:
            The machines, primary first. A single-element list when the member
            fits on one, which is the answer for every role placed today.
        """
        return list(self._spans.get((role, worker_id), [worker_id]))

    # How many lines of explanation a refusal may carry. The claim is the same
    # sentence on every worker, so deduplication does most of the work; the cap
    # is for the per-worker lines on a fleet where dozens are equally full, and
    # a refusal nobody reads to the end explains nothing.
    _MAX_NOTES = 6

    def _remember_notes(self, role: str, notes: Sequence[str]) -> None:
        """Keep each distinct explanation once, in the order first seen.

        Across workers rather than per worker: the line that matters most --
        what one member of this role costs -- is identical everywhere, and the
        ones that differ name the worker they came from.
        """
        if not notes:
            return
        kept = self._notes.setdefault(role, [])
        for note in notes:
            text = note.strip()
            if text and text not in kept and len(kept) < self._MAX_NOTES:
                kept.append(text)

    def _remember_claim(self, role: str, claim: Optional[MemberResourceClaim]) -> None:
        """Keep what a member of this role costs, once it has been priced.

        First answer wins. Every pass for one role prices the same member the
        same way, and a later `None` -- a stub selector, a pass that raised
        before pricing -- must not erase a figure already established.
        """
        if claim is not None and role not in self._claims:
            self._claims[role] = claim

    async def demand_for(
        self, role: str, worker_ids: Sequence[int]
    ) -> Tuple[Optional[MemberResourceClaim], int]:
        """What one member of `role` costs, and how many of them fit.

        For a refusal's breakdown, where there is no placement to read a claim
        off. Measured with **nothing of this group standing in**, so every role
        in one breakdown is answered in the same unit: "this role, on its own,
        in this cluster". Mixing that with a figure taken mid-solve -- where
        the roles placed before it are already occupying cards -- would put two
        different questions in one list and make the smaller number look like
        the shortfall.

        Roles that each fit alone and do not fit together therefore all report
        their full count, and that is the intended reading: the group-level
        line above the breakdown is what carries the shortfall, and a
        breakdown where every role is satisfied is itself the finding.

        **Refusal path only.** It re-measures against an empty placement
        set, which overwrites what the solve learned about (role, worker); a
        `commit` after this would hand out cards priced for a solve that is no
        longer the one being committed.
        """
        slots = await self(role, worker_ids, [])
        placeable = min(sum(slots.values()), self._limit_for(role))
        return self._claims.get(role), placeable

    def notes_for(self, role: Optional[str]) -> List[str]:
        """Why this role found no room, for a caller building a refusal.

        Filter notes first, capacity notes after, and the order is the reading
        order: which workers were even considered, and only then what one
        member costs on them. A refusal that opens with "the largest worker
        offered 576 GiB" after a selector matched nothing is answering a
        question about a fleet the deployment was never allowed to use.

        Deduplicated across the two lists as well as within them: a filter and
        a selector can reach the same sentence about the same worker, and the
        same line twice in a refusal reads as two separate findings.
        """
        if not role:
            return []
        out: List[str] = []
        for note in list(self._filter_notes.get(role, [])) + list(
            self._notes.get(role, [])
        ):
            if note not in out:
                out.append(note)
        return out

    def _committed_port_demand(
        self, already_placed: Sequence[object]
    ) -> Dict[int, int]:
        """Ports the members this solve has committed so far will reserve.

        Priced per member with its OWN role's demand, not the asking role's: a
        prefill on eight cards and a router on none take different numbers of
        ports from the same range, and charging one at the other's rate is how
        a budget stops being one.

        **The whole demand on every machine the member spans**, not a share of
        it split between them, because that is what the worker does: one
        `ModelInstance` row holds one set of ports, `_assign_named_ports` sizes
        the bands once on the primary, and every host the member lands on then
        fences that same set through `_register_assigned_ports`. Charging only
        the primary leaves a later role priced against free ports its own
        allocator will find taken -- the scheduler promising room the worker
        cannot produce, which is the failure this budget exists to prevent.
        `_within_port_budget` reads the same way for the same reason.

        Empty for the first role of a solve, which is the case every
        single-role deployment takes.
        """
        out: Dict[int, int] = {}
        for entry in already_placed:
            worker_id = getattr(entry, "worker_id", None)
            role = getattr(entry, "role", None)
            if worker_id is None or not role or role not in self._projected:
                continue
            offers = self._offers.get((role, worker_id)) or []
            cards = len(getattr(offers[0], "gpu_indexes", None) or []) if offers else 0
            demand = port_budget.member_port_demand(
                self._projected[role].model, role, cards
            )
            for machine in self._spans.get((role, worker_id), [worker_id]):
                out[machine] = out.get(machine, 0) + demand
        return out

    def _within_port_budget(self, role: str, worker_id: int, offer, cards: int) -> int:
        """`offer.slots`, capped by what the host has ports for.

        Applied here rather than as a filter of its own so a port shortage
        reads to the solver exactly like a card shortage: fewer slots on this
        worker, and the same walk to the next domain. The alternative — a
        worker that passes capacity and fails at start-up — puts the failure
        after the placement decision, where nothing reconsiders it.

        `cards` is passed in rather than read off the offer, because an offer
        is not always about one placement. A single-machine pass makes one per
        worker and its placements are that worker's, but the spanning pass
        makes ONE offer holding every combination it found — so reading the
        first placement there prices every combination at the width of
        whichever happened to come back first. It is the primary's card count
        either way: `{{accelerator_count}}` resolves from `gpu_indexes` on the
        instance row, and that row carries the primary's cards.
        """
        if offer.slots <= 0:
            return offer.slots

        # The role's projection, which is what the worker's own resolver
        # receives: `backend_parameters` there are the role's effective ones.
        projected = self._projected[role].model
        demand = port_budget.member_port_demand(projected, role, cards)

        if worker_id not in self._ports_taken:
            self._ports_taken[worker_id] = port_budget.ports_taken_on(
                worker_id, self._model_instances, self._cache_instances
            )
        # Added on the read rather than written back: the cache holds what the
        # snapshot saw, and what this solve has committed changes per role.
        taken = self._ports_taken[worker_id] + self._committed_ports.get(worker_id, 0)
        allowed = port_budget.port_capacity(
            getattr(self._config, "service_port_range", None),
            demand,
            taken,
        )
        if allowed is None or allowed >= offer.slots:
            return offer.slots

        logger.debug(
            "Port budget caps role %r on %s",
            role,
            port_budget.describe(worker_id, allowed, demand, taken),
        )
        return allowed

    def _translate(self, already_placed: Sequence[object]) -> List[object]:
        """The solver's commits, in the shape the allocation accounting reads.

        Found by running this on a live server, not by a test: the solver
        commits `_Committed(worker_id, role)` — deliberately just those two,
        with its docstring saying "the capacity function re-derives the real
        resource claim itself, and inventing one here would be a second,
        quieter accounting of the same placement". That re-derivation is this
        method, and its absence surfaced as
        `'_Committed' object has no attribute 'gpu_type'` from deep inside
        `compute_worker_allocated`, which reads four fields off every entry.

        The claim comes from what the selector already produced for that
        (role, worker) while counting — the same number the placement decision
        was made against. Anything computed a second way here would be the
        second accounting the solver's docstring warns about.

        Entries that already look like instances (or stand-ins) pass through:
        the commit pass builds those itself.

        **Which cards, not just how much.** A claim handed back with
        `gpu_indexes=None` is one the allocation accounting cannot subtract
        from any GPU, so the next role sees every card on the worker as free.
        On a two-card host a 2P1D would then count two prefill slots and,
        separately, one decode slot, and the solver would deal three members
        onto two cards -- nothing catching it until
        `commit` tried to turn that assignment into real cards and came up
        short -- reported as "the cluster changed during scheduling", which was
        never true. The group was refused outright while the same fleet ran the
        same spec perfectly if it was grown one member at a time, because
        scale-out never goes through here.

        The n-th commit for a (role, worker) therefore takes the n-th candidate
        the count produced there. They are distinct by construction:
        `count_offer_slots` stands each one in before looking for the next, so
        the cards it hands out in a single pass are already disjoint.
        """
        out: List[object] = []
        taken: Dict[tuple, int] = {}
        for entry in already_placed:
            if getattr(entry, "computed_resource_claim", None) is not None or hasattr(
                entry, "distributed_servers"
            ):
                out.append(entry)
                continue
            worker_id = getattr(entry, "worker_id", None)
            role = getattr(entry, "role", "") or ""
            key = (role, worker_id)
            index = taken.get(key, 0)
            taken[key] = index + 1
            offers = self._offers.get(key) or []
            if index >= len(offers):
                # Unreachable while the solver hands out no more than the count
                # reported: that count is the length of this list. Skipping
                # rather than inventing a claim, and saying so, because a
                # made-up one would make the domain look either roomier or
                # tighter than the placement it is meant to reflect.
                logger.warning(
                    "No learned placement #%d for role %r on worker %s; the "
                    "commit is not counted against remaining capacity.",
                    index + 1,
                    role,
                    worker_id,
                )
                continue
            # The candidate carries its own primary worker and, when it spans
            # machines, its subordinates -- which `compute_worker_allocated`
            # bills to the right one. Nothing here has to know which.
            out.append(_stand_in_for(offers[index]))
        return out

    def _selector(self, model, instances, cpu_only, ram_claim=None):
        # Imported here rather than at module scope: `scheduler` imports a wide
        # slice of the policy stack, and this module is imported by a route.
        from gpustack.scheduler.scheduler import build_candidate_selector

        return build_candidate_selector(
            self._config,
            model,
            instances,
            cpu_only=cpu_only,
            ram_claim=ram_claim,
        )

    def _limit_for(self, role: str) -> int:
        """Never count past what the group could place.

        Every extra slot costs a full selector pass, and capacity beyond the
        group's own member count answers a question nobody asked.
        """
        for spec in self._model.roles or []:
            if spec.name == role:
                return max(int(spec.replicas or 0), 0)
        return max(int(self._model.replicas or 1), 0)

    async def _eligible_for(self, role: str) -> Dict[int, Worker]:
        if role in self._eligible:
            return self._eligible[role]

        # Read before projecting: the answer is a property of the ROLE, and the
        # projection flattens the role's overrides onto the model.
        cpu_only = role_takes_no_accelerator(self._model, role)
        # `resources` is role-OWN as well, and only the accelerator-free
        # branch consumes it — same read-before-projection reason.
        ram_claim = (
            role_container_resources(self._model, role).memory if cpu_only else None
        )
        model = role_effective_model(self._model, role)
        self._projected[role] = _RoleProjection(model, cpu_only, ram_claim)

        chain = WorkerFilterChain(
            [
                ClusterFilter(model),
                GPUMatchingFilter(model),
                LabelMatchingFilter(model),
                StatusFilter(model),
                BackendFrameworkFilter(model),
                LocalPathFilter(model),
                PDModeRuntimeFilter(model),
            ]
        )
        try:
            # The second half of that tuple is why a worker is not here, in
            # the filters' own words, and it is carried rather than dropped. It
            # is the same text `find_candidate` prints on the single-instance path
            # ("Matched 3 workers by cluster selector.", "Matched 0/3 workers by
            # label selector: {'worker-name': ...}."), and without it a group
            # refusal could count members but never say that the count is zero
            # because a selector matched nothing — the two paths described the
            # same cluster and only one of them named the reason.
            kept, messages = await chain.filter(list(self._workers.values()))
        except Exception as e:
            # A broken filter is not an empty cluster. Returning nothing here
            # would make every domain look too small and the refusal would
            # blame capacity.
            logger.warning(
                "Could not filter workers for role %r; treating every worker "
                "as eligible and letting the selectors decide: %s",
                role,
                e,
            )
            kept = list(self._workers.values())
            # No messages either: the chain stopped part-way, so whatever it
            # had said so far describes a narrowing that was then thrown away.
            # Repeating it beside a fleet we are deliberately treating as
            # wholly eligible would contradict the very list we return.
            messages = []

        # Deduplicated and stripped here rather than in `notes_for`, because
        # this runs once per role (the result is cached on `_eligible`) while
        # `notes_for` runs once per refusal read.
        filter_notes = self._filter_notes.setdefault(role, [])
        for message in messages:
            text = (message or "").strip()
            if text and text not in filter_notes:
                filter_notes.append(text)

        self._eligible[role] = {w.id: w for w in kept}
        return self._eligible[role]

    async def commit(
        self,
        role: str,
        worker_ids: Sequence[int],
        already_placed: Sequence[object],
    ) -> List[object]:
        """The concrete candidates for `worker_ids`, in the order given.

        This is the half of step 6 the solver cannot do. `GroupPlacement`
        answers *which worker* each member goes to; a member also needs which
        cards, and those only exist inside the candidates `count_offer_slots`
        already produced and threw away.

        Re-derived here rather than cached during the count, and that is
        deliberate: the count runs once per domain per layer while searching,
        so a cache would hold whichever domain was examined last — not the one
        that won. Running it again against the winning assignment is one extra
        pass and is consistent by construction.

        `worker_ids` may repeat: two members of one role on one worker means
        that worker appears twice, and the second entry must be the *second*
        candidate the selector offers, computed with the first already
        standing in. Anything else hands out the same cards twice.
        """
        eligible = await self._eligible_for(role)
        projected = self._projected[role]
        instances = self._model_instances + self._translate(already_placed)

        wanted: Dict[int, int] = {}
        for worker_id in worker_ids:
            wanted[worker_id] = wanted.get(worker_id, 0) + 1

        by_worker: Dict[int, List[object]] = {}
        for worker_id, count in wanted.items():
            worker = eligible.get(worker_id)
            if worker is None:
                return []
            # The machines this member lands on. For a role that fits on one,
            # that is the worker the solve named -- which is every role placed
            # today. For a wider one it is the combination the count recorded
            # under that primary, re-derived here rather than replayed, for the
            # same reason the single-machine case is: `already_placed` has
            # moved on since the count, and a cached candidate would be an
            # answer to the older question.
            span = [
                eligible[w]
                for w in self._spans.get((role, worker_id), [worker_id])
                if w in eligible
            ]
            if not span:
                return []
            offer = await count_offer_slots(
                make_selector=lambda instances_now, p=projected: (
                    self._selector(p.model, instances_now, p.cpu_only, p.ram_claim)
                ),
                workers=span,
                model_instances=instances,
                limit=count,
            )
            if offer.slots < count:
                # The winning solve said this fits, and the two passes read
                # one snapshot of the instance list -- taken when this object
                # was built -- so a concurrent change cannot be what made them
                # disagree. Reaching here means the search and the commit
                # computed different things from the same inputs, which is a
                # defect. Reported whole rather than partially placed: a group
                # with some members on workers is the one state the group
                # solve exists to prevent.
                logger.warning(
                    "Commit pass came up short for role %r on worker %s: "
                    "wanted %d, got %d. The search and the commit disagree on "
                    "one snapshot, which is a bug; treating the group as "
                    "unplaceable.",
                    role,
                    worker_id,
                    count,
                    offer.slots,
                )
                return []
            by_worker[worker_id] = list(offer.placements[:count])

        # Handed back in the order the caller asked, so a member's index in
        # the assignment list matches its candidate.
        cursor: Dict[int, int] = {}
        out: List[object] = []
        for worker_id in worker_ids:
            index = cursor.get(worker_id, 0)
            cursor[worker_id] = index + 1
            out.append(by_worker[worker_id][index])
        return out


def role_demands(model: Model) -> List[dict]:
    """The group's shape, as the solver's `RoleDemand` fields.

    Returned as plain dicts so this module does not have to import the solver's
    dataclass — the route that owns both does the construction.

    `weight` orders the roles when they compete for the same cards: the
    hungriest first. Cards-per-member is the honest proxy available here, and
    getting the order wrong is the difference between "4P4D fits" and "4P4D
    fits only if you happen to place P first".
    """
    demands: List[dict] = []
    for spec in model.roles or []:
        if role_takes_no_accelerator(model, spec.name):
            # A router occupies no accelerator, so it neither competes for
            # cards nor constrains which domain the group lands in. Counting it
            # would make a 4P4D look like nine members needing one domain.
            # `attendant_demands` picks it up instead: its container memory
            # still has to exist somewhere, it just must not inflate the
            # gang's size.
            continue
        projected = role_effective_model(model, spec.name)
        per_member = _cards_per_member(projected)
        demands.append(
            {
                "role": spec.name,
                "replicas": max(int(spec.replicas or 0), 0),
                "weight": float(per_member),
            }
        )
    return demands


def attendant_demands(model: Model) -> List[dict]:
    """The group's members that occupy no accelerator -- the router.

    The complement of `role_demands`, and the two must stay complementary: a
    role counted in both would be placed twice, and a role in neither is the
    state this function exists to end.

    They are demands for feasibility and not for sizing. A router answers every
    request, so a group whose router cannot be scheduled serves nothing -- but
    it competes for no card, and adding it to the gang would make a 4P4D need
    nine placements in one domain and refuse racks that would have served.

    `weight` is zero because ordering is a contest for cards and they are not
    in it; they are checked after the gang is placed, against what it left.
    """
    return [
        {"role": spec.name, "replicas": max(int(spec.replicas or 0), 0), "weight": 0.0}
        for spec in model.roles or []
        if role_takes_no_accelerator(model, spec.name)
        and max(int(spec.replicas or 0), 0) > 0
    ]


def _cards_per_member(model) -> int:
    """How many accelerators one member of this role wants.

    Only used to order the roles, so a wrong answer costs placement quality
    rather than correctness — which is why it reads the declared selector
    rather than re-deriving parallelism from backend parameters.
    """
    selector = getattr(model, "gpu_selector", None)
    per_replica = getattr(selector, "gpus_per_replica", None) if selector else None
    try:
        return max(int(per_replica or 1), 1)
    except (TypeError, ValueError):
        return 1
