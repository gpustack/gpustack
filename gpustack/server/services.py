import logging
from datetime import datetime, timezone
from uuid import uuid4
from typing import List, NamedTuple, Optional, Tuple, Union, Set
from sqlalchemy import delete, func, or_, inspect
from sqlalchemy.exc import IntegrityError
from sqlmodel import SQLModel, select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.orm import selectinload
from sqlalchemy.orm.attributes import set_committed_value

from gpustack.api.exceptions import (
    ConflictException,
    HTTPException,
    InternalServerErrorException,
)

from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.cache_services import (
    CacheService,
    CacheServiceInstance,
    CacheServiceInstanceCreate,
)
from gpustack.schemas.links import (
    ModelRoutePrincipalLink,
    ModelInstanceModelFileLink,
    ModelInstanceDraftModelFileLink,
)
from gpustack.schemas.model_files import ModelFile
from gpustack.schemas.models import (
    BackendEnum,
    get_backend,
    ROUTER_DEFAULT_MEMORY,
    Model,
    ModelInstance,
    ModelSource,
    ModelInstanceStateEnum,
    ModelInstanceWorkload,
    ModelInstanceLegacyStop,
    DistributedServers,
    ModelInstanceWorkloadStateEnum,
    member_worker_ids,
)
from gpustack.schemas.model_routes import (
    ModelRoute,
    MyModel,
    ModelRouteTarget,
    TargetStateEnum,
    AccessPolicyEnum,
    effective_route_name,
)
from gpustack.schemas.principals import (
    OrgRole,
    Principal,
    PrincipalMembership,
    PrincipalType,
    get_platform_principal_id,
    platform_principal_id,
)
from gpustack.schemas.users import AuthProviderEnum, User
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.workers import Worker, WorkerStateEnum
from gpustack import envs
from gpustack.utils.network import is_offline
from gpustack.server.cache import (
    delete_cache_by_key,
    locked_cached,
)
from gpustack.server.cache_services import resolve_instance_cache_config_safe
from gpustack.server.cache_service_resources import cache_ram_by_worker
from gpustack.server.worker_allocated_cache import (
    invalidate_cache_workers_ram,
    invalidate_workers_allocated,
)
from gpustack.utils.usage_snapshots import propagate_user_rename
from gpustack.utils.model_instance_workers import (
    get_model_instance_workloads,
    has_model_instance_stop_barrier,
    requires_model_instance_stop_barrier,
    assigned_managed_workload_worker_ids,
    get_model_instance_worker_match,
    model_instance_workloads_stopped,
    model_instance_legacy_stop,
    requires_group_recovery,
    calculate_group_recovery_policy,
)

logger = logging.getLogger(__name__)


class RouteTargetResolution(NamedTuple):
    """Routing decision for a single ModelRouteTarget."""

    model_id: int
    overridden_model_name: Optional[str]
    weight: int


class UserService:

    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached()
    async def get_by_id(self, user_id: int) -> Optional[User]:
        result = await User.one_by_id(
            self.session,
            user_id,
            options=[selectinload(User.worker), selectinload(User.cluster)],
        )
        if result is None:
            return None
        if result.worker is not None:
            # detach worker to avoid lazy loading
            self.session.expunge(result.worker)
        if result.cluster is not None:
            # detach cluster too — the cached User outlives this session
            # and tenant-context resolution reads cluster.id /
            # cluster.owner_principal_id on later requests; an instance
            # still tied to the closed session risks DetachedInstanceError
            # if anything expires it.
            self.session.expunge(result.cluster)
        self.session.expunge(result)
        return result

    @locked_cached()
    async def get_by_username(self, username: str) -> Optional[User]:
        # ``username`` is the wire-level name for the storage column
        # ``name`` (k8s-style identifier). The cache key passes
        # through the legacy parameter name so OAuth2 password
        # callers don't need to know about the rename.
        #
        # Scoped to ``kind == USER``: login is USER-only, and since USER
        # now has its own name partition a same-named ORG must not be
        # returned here (it would otherwise shadow a real user login).
        # Persisted SYSTEM principals never reach this path — they
        # authenticate via API token (``get_by_id``) or in-memory.
        #
        # ``cluster`` / ``worker`` are eager-loaded for parity with
        # ``get_by_id``; for a USER row they resolve to None.
        result = await User.one_by_fields(
            self.session,
            {"name": username, "kind": PrincipalType.USER},
            options=[selectinload(User.cluster), selectinload(User.worker)],
        )
        if result is None:
            return None
        self.session.expunge(result)
        return result

    async def update(
        self,
        user: User,
        source: Union[dict, SQLModel, None] = None,
        *,
        auto_commit: bool = True,
    ):
        old_name = user.name
        result = await user.update(self.session, source, auto_commit=False)
        # Refresh denormalized ``user_name`` snapshots on usage rows so
        # dashboards reflect the new name. Bundled in the same
        # transaction as the user-row write so a rollback rolls back
        # both. See :func:`propagate_user_rename` for scope notes.
        if user.name != old_name:
            await propagate_user_rename(self.session, user.id, user.name)
        # ``auto_commit=False`` defers the commit to the caller so
        # this write can share a transaction with other writes; the
        # caller is then also responsible for invoking
        # :meth:`invalidate_cache` *after* it commits. Invalidating
        # here would evict the entry while the row write is still
        # uncommitted, letting a concurrent read refill the cache
        # from the pre-commit row and leave stale data past the commit.
        if auto_commit:
            await self.session.commit()
            await self.invalidate_cache(user, old_name=old_name)
        return result

    async def invalidate_cache(
        self, user: User, *, old_name: Optional[str] = None
    ) -> None:
        """Drop this user's read-through cache entries.

        Meant to run after a commit; :meth:`update` calls it inline
        when ``auto_commit=True``, otherwise the caller invokes it
        once its enclosing transaction has committed. Accepts the
        pre-update ``old_name`` so a rename can drop the entry keyed
        by the previous username too.
        """
        await delete_cache_by_key(self.get_by_id, user.id)
        await delete_cache_by_key(self.get_user_accessible_model_names, user.id)
        await delete_cache_by_key(self.get_by_username, user.name)
        if old_name and old_name != user.name:
            await delete_cache_by_key(self.get_by_username, old_name)

    async def delete(self, user: User):
        # The keys have to be read before the delete and invalidated by hand:
        # ``Principal.api_keys`` is ``lazy="noload"``, so ``_handle_cascade_delete``
        # sees an empty list and ``ApiKey.delete`` — which would otherwise drop
        # each entry — never runs. The rows themselves do go: both principal FKs
        # on ``api_keys`` carry ``ON DELETE CASCADE`` in the schema the
        # migrations build (``c45e397531d1``, ``7c5e3f9a2d18``), whatever the
        # model's own ``user_id`` declaration says — nothing ever builds this
        # table from the model. That is exactly why the cache is what's left:
        # losing this loop leaves every key of a deleted user authenticating
        # from its cached entry for the rest of the TTL.
        apikeys = await APIKeyService(self.session).get_by_user_id(user.id)
        result = await user.delete(self.session)
        await delete_cache_by_key(self.get_by_id, user.id)
        await delete_cache_by_key(self.get_user_accessible_model_names, user.id)
        await delete_cache_by_key(self.get_by_username, user.name)
        for apikey in apikeys:
            await delete_cache_by_key(
                APIKeyService.get_by_access_key, apikey.access_key
            )
        return result

    async def model_allowed_for_user(
        self, model_name: str, user_id: int, api_key: Optional[ApiKey]
    ) -> bool:
        limited_model_names: Optional[Set[str]] = (
            set(api_key.allowed_model_names)
            if api_key is not None
            and api_key.allowed_model_names is not None
            and len(api_key.allowed_model_names) > 0
            else None
        )
        accessible_model_names: Set[str] = await self.get_user_accessible_model_names(
            user_id
        )
        allowed = model_name in intersection_nullable_set(
            accessible_model_names, limited_model_names
        )
        if not allowed:
            logger.info(
                "Access denied: model_name=%r user_id=%d " "accessible=%s limited=%s",
                model_name,
                user_id,
                sorted(accessible_model_names),
                sorted(limited_model_names) if limited_model_names else None,
            )
        return allowed

    @locked_cached()
    async def get_user_accessible_model_names(self, user_id: int) -> Set[str]:
        # Get all accessible model names for the user. The set holds two
        # forms per route:
        #   1. Org-effective name (`<owner-name>/<route>` for
        #      non-platform Orgs, raw for platform) — matches
        #      `/v1/models` output and the gateway's ingress header
        #      matcher.
        #   2. Raw `route.name` — matches the post-`modelMapping` value
        #      that Higress's AI proxy hands back via
        #      `x-higress-llm-model` on the auth callback. Without this
        #      the callback would deny chat traffic for non-platform
        #      Orgs even though the gateway already routed it to the
        #      correct ingress.
        # Cross-Org collisions on raw names are fine: each user's set is
        # isolated, and Higress's per-Org ingress already disambiguates
        # which underlying instance receives the request.
        user: User = await self.get_by_id(user_id)
        if user is None:
            return set()
        if user.is_admin or user.kind == PrincipalType.SYSTEM:
            routes = await ModelRoute.all_by_field(self.session, "deleted_at", None)
        else:
            routes = await MyModel.all_by_fields(
                self.session, {"user_id": user.id, "deleted_at": None}
            )
        principal_ids = {
            r.owner_principal_id for r in routes if r.owner_principal_id is not None
        }
        principal_by_id = {}
        if principal_ids:
            rows = (
                await self.session.exec(
                    select(Principal).where(Principal.id.in_(principal_ids))
                )
            ).all()
            principal_by_id = {p.id: p for p in rows}
        names: Set[str] = set()
        for r in routes:
            owner = (
                principal_by_id.get(r.owner_principal_id)
                if r.owner_principal_id
                else None
            )
            names.add(
                effective_route_name(
                    r.name,
                    getattr(owner, "name", None),
                    getattr(owner, "id", None) == platform_principal_id(),
                )
            )
            names.add(r.name)
        return names


async def provision_bootstrap_admin_orgs(session: AsyncSession, user: User) -> None:
    """Add the bootstrap admin as OWNER of the platform Org.

    Caller commits.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    platform_id = await get_platform_principal_id(session)
    session.add(
        PrincipalMembership(
            parent_principal_id=platform_id,
            member_principal_id=user.id,
            role=OrgRole.OWNER,
            created_at=now,
            updated_at=now,
        )
    )


async def _insert_group_or_refetch(
    session: AsyncSession,
    name: str,
    provider: AuthProviderEnum,
    now: datetime,
) -> Optional[Principal]:
    """Insert a new GROUP-principal, or re-fetch if a concurrent
    transaction beat us to it.

    SAVEPOINT-wraps the insert so an IntegrityError from
    ``uix_principals_group_name`` (PG partial unique; MySQL falls
    back to a plain index) doesn't roll back the caller's
    transaction — we re-read the winning row instead.

    Returns ``None`` when the winning row's source disagrees with
    ``provider`` (cross-source adoption is refused). Raises only if
    IntegrityError fired but no row is visible on re-read.
    """
    grp = Principal(
        kind=PrincipalType.GROUP,
        name=name,
        source=provider,
        created_at=now,
        updated_at=now,
    )
    try:
        async with session.begin_nested():
            session.add(grp)
            await session.flush([grp])
        return grp
    except IntegrityError:
        existing = (
            await session.exec(
                select(Principal).where(
                    Principal.kind == PrincipalType.GROUP,
                    Principal.deleted_at.is_(None),
                    Principal.name == name,
                )
            )
        ).first()
        if existing is None:
            raise
        if existing.source != provider:
            logger.warning(
                "Group '%s' exists with source '%s'; refusing to adopt "
                "for provider '%s' sync.",
                name,
                existing.source,
                provider,
            )
            return None
        return existing


async def sync_user_group_memberships(
    session: AsyncSession,
    user_principal_id: int,
    provider: AuthProviderEnum,
    group_names: List[str],
) -> None:
    """Authoritatively reconcile a user's Group memberships from an IdP.

    Called from OIDC / SAML callbacks with the user's group set as the
    IdP currently sees it. Resolves each name to a Group-principal
    (creating one with ``kind=GROUP`` for names we've never seen),
    then diffs against the user's current Group memberships **scoped
    to** ``source == provider``:

    - Names present in the IdP but not on the user → insert (or revive
      a soft-deleted row) with ``source=provider``.
    - Memberships present locally with matching source but absent from
      the IdP → soft-delete.
    - Admin-curated memberships (``source=Local``) and memberships
      sourced by a *different* IdP are never touched. The same group
      can have one Local membership and one OIDC-sourced membership
      side by side for the same user; that's how the bookkeeping
      survives mixed setups.

    Group entities themselves are never deleted by this function — a
    Group may have outstanding (Org → Group) bindings the admin
    relies on; lifecycle for Group rows lives on the ``/groups`` admin
    surface. Provider must be ``OIDC`` or ``SAML``; ``Local`` would
    be a logic error and is rejected.

    Caller commits.
    """
    if provider == AuthProviderEnum.Local:
        raise InternalServerErrorException(
            message="sync_user_group_memberships is not for Local provider"
        )

    # De-dupe + drop blanks. Caller (auth ``_coerce_group_claim``)
    # already trims and drops empty entries; we defensively re-check
    # so direct callers of this service (tests, future SCIM endpoint,
    # ...) can't slip an empty / duplicate name through.
    wanted_names: List[str] = []
    seen = set()
    for name in group_names:
        if not name or name in seen:
            continue
        seen.add(name)
        wanted_names.append(name)

    # Cross-source name matches are refused, not adopted — the IdP
    # claim is skipped so the user's other groups still sync.
    desired_group_ids: Set[int] = set()
    if wanted_names:
        existing = (
            await session.exec(
                select(Principal).where(
                    Principal.kind == PrincipalType.GROUP,
                    Principal.deleted_at.is_(None),
                    Principal.name.in_(wanted_names),
                )
            )
        ).all()
        existing_by_name = {p.name: p for p in existing}

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        for name in wanted_names:
            grp = existing_by_name.get(name)
            if grp is not None and grp.source != provider:
                logger.warning(
                    "Skipping group sync for '%s': exists with source "
                    "'%s' but provider is '%s'.",
                    name,
                    grp.source,
                    provider,
                )
                continue
            if grp is None:
                grp = await _insert_group_or_refetch(session, name, provider, now)
                if grp is None:
                    continue
            desired_group_ids.add(grp.id)

    # Memberships this sync owns: this user's rows with matching
    # provider on both the membership and the parent group. The
    # group-source filter keeps legacy mis-attached rows out of the
    # diff so first sync after upgrade doesn't silently delete them.
    user_pm_stmt = (
        select(PrincipalMembership)
        .join(Principal, Principal.id == PrincipalMembership.parent_principal_id)
        .where(
            PrincipalMembership.member_principal_id == user_principal_id,
            PrincipalMembership.source == provider,
            Principal.kind == PrincipalType.GROUP,
            Principal.source == provider,
        )
    )
    owned_rows: List[PrincipalMembership] = list(
        (await session.exec(user_pm_stmt)).all()
    )
    owned_by_group: dict[int, PrincipalMembership] = {
        m.parent_principal_id: m for m in owned_rows
    }

    now = datetime.now(timezone.utc).replace(tzinfo=None)

    # Adds + revives.
    for gid in desired_group_ids:
        existing = owned_by_group.get(gid)
        if existing is None:
            session.add(
                PrincipalMembership(
                    parent_principal_id=gid,
                    member_principal_id=user_principal_id,
                    role=None,
                    source=provider,
                    created_at=now,
                    updated_at=now,
                )
            )
        elif existing.deleted_at is not None:
            existing.deleted_at = None
            existing.updated_at = now
            session.add(existing)

    # Removals: anything we own but the IdP no longer claims.
    for gid, row in owned_by_group.items():
        if gid in desired_group_ids:
            continue
        if row.deleted_at is not None:
            continue
        row.deleted_at = now
        row.updated_at = now
        session.add(row)


class APIKeyService:
    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached()
    async def get_by_access_key(self, access_key: str) -> Optional[ApiKey]:
        """Look up a live key by its access key, for authentication.

        ``deleted_at IS NULL`` is part of the predicate, not a caller's
        responsibility. This is the query behind ``/token-auth``'s credential
        path, which is what every request falls back to when the gateway's own
        local key table cannot answer -- and that table is built with the same
        filter (``build_local_auth_tables``). Without it here, a soft-deleted
        row keeps authenticating on the fallback path forever while the gateway
        has already stopped honouring it, so revocation never converges.

        No caller soft-deletes a key today: ``ApiKey.delete()`` hard-deletes
        unless asked otherwise, and an owner's removal never reaches the row
        through the ORM at all -- ``Principal.api_keys`` is ``lazy="noload"``,
        so the cascade reads an empty list. The filter is here because a
        revocation that depends on which delete path ran is not a revocation --
        the first ``soft=True`` caller must not silently keep the credential
        alive on the one path that still answers when the gateway cannot.
        """
        result = await ApiKey.one_by_fields(
            self.session, {"access_key": access_key, "deleted_at": None}
        )
        if result is None:
            return None
        self.session.expunge(result)
        return result

    async def get_by_user_id(self, user_id: int) -> List[ApiKey]:
        results = await ApiKey.all_by_field(self.session, "user_id", user_id)
        if results is None:
            return []
        for result in results:
            self.session.expunge(result)
        return results

    async def update(self, api_key: ApiKey, source: Union[dict, SQLModel, None] = None):
        result = await api_key.update(self.session, source)
        await delete_cache_by_key(self.get_by_access_key, api_key.access_key)
        return result

    async def delete(self, api_key: ApiKey):
        result = await api_key.delete(self.session)
        await delete_cache_by_key(self.get_by_access_key, api_key.access_key)
        return result


class ClusterService:
    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached()
    async def get_by_id(self, cluster_id: int) -> Optional[Cluster]:
        result = await Cluster.one_by_id(self.session, cluster_id)
        if result is None:
            return None
        self.session.expunge(result)
        return result


class PrincipalService:
    """Cached principal (user / org / group) lookups by id.

    Mirrors ``ClusterService`` — an in-memory, TTL'd, coordinator-invalidated
    cache shared across sessions (the result is expunged so it's detached and
    safe to hold). Intended for hot paths that only need a principal's ``name``,
    e.g. the resource-event logger resolving owner / consumer / creator names on
    every event. Rename staleness is bounded by the cache TTL, which is fine for
    display / audit names.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached()
    async def get_by_id(self, principal_id: int) -> Optional[Principal]:
        result = await Principal.one_by_id(self.session, principal_id)
        if result is None:
            return None
        self.session.expunge(result)
        return result


class WorkerService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_workers_for_update(
        self,
        worker_ids: List[int],
        extra_conditions: Optional[List] = None,
    ) -> List[Worker]:
        """Read live workers under row locks for a label read-modify-write.

        Args:
            worker_ids: Workers to lock.
            extra_conditions: Caller-provided tenant or cluster scope.

        Returns:
            Matching workers in ID order, refreshed from the database.
        """
        statement = (
            select(Worker)
            .where(
                Worker.id.in_(worker_ids),
                Worker.deleted_at.is_(None),
                *(extra_conditions or []),
            )
            .order_by(Worker.id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        result = await self.session.exec(statement)
        return list(result.all())

    @locked_cached()
    async def get_by_id(self, worker_id: int) -> Optional[Worker]:
        result = await Worker.one_by_id(self.session, worker_id)
        if result is None:
            return None
        self.session.expunge(result)
        return result

    @locked_cached()
    async def get_by_cluster_id_name(
        self, cluster_id: int, name: str
    ) -> Optional[Worker]:
        result = await Worker.one_by_fields(
            self.session, fields={"cluster_id": cluster_id, "name": name}
        )
        if result is None:
            return None
        self.session.expunge(result)
        return result

    @locked_cached()
    async def get_by_name(self, name: str) -> Optional[Worker]:
        result = await Worker.one_by_field(self.session, "name", name)
        if result is None:
            return None
        self.session.expunge(result)
        return result

    async def update(
        self, worker: Worker, source: Union[dict, SQLModel, None] = None, **kwargs
    ):
        history = inspect(worker).attrs
        old_key = (
            (
                history.cluster_id.history.deleted[0]
                if history.cluster_id.history.deleted
                else worker.cluster_id
            ),
            (
                history.name.history.deleted[0]
                if history.name.history.deleted
                else worker.name
            ),
        )
        result = await worker.update(self.session, source, **kwargs)
        # Worker cache is high-frequency, non-security-critical, skip coordinator sync
        await delete_cache_by_key(self.get_by_id, worker.id, sync_coordinator=False)
        await delete_cache_by_key(self.get_by_name, worker.name, sync_coordinator=False)
        for cluster_id, name in {old_key, (worker.cluster_id, worker.name)}:
            await delete_cache_by_key(
                self.get_by_cluster_id_name, cluster_id, name, sync_coordinator=False
            )
        return result

    async def batch_update(
        self,
        workers: List[Worker],
        source: Union[dict, SQLModel, None] = None,
        **kwargs,
    ) -> int:
        if not workers:
            return 0

        old_keys = []
        for worker in workers:
            history = inspect(worker).attrs
            old_keys.append(
                (
                    (
                        history.cluster_id.history.deleted[0]
                        if history.cluster_id.history.deleted
                        else worker.cluster_id
                    ),
                    (
                        history.name.history.deleted[0]
                        if history.name.history.deleted
                        else worker.name
                    ),
                )
            )
        updated = await Worker.batch_update(self.session, workers)

        for w, old_key in zip(workers, old_keys):
            # Worker cache is high-frequency, non-security-critical, skip coordinator sync
            await delete_cache_by_key(self.get_by_id, w.id, sync_coordinator=False)
            await delete_cache_by_key(self.get_by_name, w.name, sync_coordinator=False)

            for cluster_id, name in {old_key, (w.cluster_id, w.name)}:
                await delete_cache_by_key(
                    self.get_by_cluster_id_name,
                    cluster_id,
                    name,
                    sync_coordinator=False,
                )

        return updated

    @staticmethod
    def assert_model_cohort_stopped(
        worker: Worker, instances: List[ModelInstance]
    ) -> None:
        """Preserve a cohort member's credentials until its own stop is proven."""
        groups = {}
        for instance in instances:
            groups.setdefault(instance.model_id, []).append(instance)
        for members in groups.values():
            if worker.id not in assigned_managed_workload_worker_ids(members):
                continue
            if any(
                get_model_instance_worker_match(instance, worker_id=worker.id).matched
                and (
                    not has_model_instance_stop_barrier(instance)
                    or not model_instance_workloads_stopped(instance)
                )
                for instance in members
            ):
                raise ConflictException(
                    message="Stop all assigned managed model tasks before deleting or rebinding this worker."
                )

    async def delete(self, worker: Worker, **kwargs):
        # Admission and rank reports lock Worker before Instance. Preserve that
        # order while keeping its credentials until physical stopping is proven.
        worker = await Worker.one_by_id(
            self.session,
            worker.id,
            for_update=True,
            options=[selectinload(Worker.system_principal)],
            refresh=True,
        )
        if worker is None:
            return
        instances = await ModelInstance.all_by_field(
            self.session, "cluster_id", worker.cluster_id, for_update=True, refresh=True
        )
        self.assert_model_cohort_stopped(worker, instances)
        worker_id = worker.id
        worker_name = worker.name
        worker_cluster_id = worker.cluster_id
        if kwargs.get("soft"):
            worker.state = WorkerStateEnum.DELETING
        result = await worker.delete(self.session, **kwargs)
        # Worker cache is high-frequency, non-security-critical, skip coordinator sync
        await delete_cache_by_key(self.get_by_id, worker_id, sync_coordinator=False)
        await delete_cache_by_key(self.get_by_name, worker_name, sync_coordinator=False)
        await delete_cache_by_key(
            self.get_by_cluster_id_name,
            worker_cluster_id,
            worker_name,
            sync_coordinator=False,
        )
        return result


async def collect_route_cache_names(
    session: AsyncSession, route_id: int, route_name: str
) -> Set[str]:
    """Names to invalidate when a route's resolution may change.

    Callers in routes/openai.py and gateway/auth callbacks key the
    @locked_cached entries by whatever string they receive — that is
    the raw route_name for the platform Org and the
    ``<owner-name>/<route-name>`` effective name for any other Org.
    Both must be cleared.
    """
    names = {route_name}
    route = await ModelRoute.one_by_id(session, route_id)
    if route:
        owner = await Principal.one_by_id(session, route.owner_principal_id)
        if owner:
            names.add(
                effective_route_name(
                    route_name, owner.name, owner.id == platform_principal_id()
                )
            )
    return names


class ModelRouteService:
    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached()
    async def get_model_auth_info_by_name(
        self, name: str
    ) -> Optional[Tuple[AccessPolicyEnum, str]]:
        # Higress's auth callback may hand us either the Org-effective
        # name (`<owner-name>/<route>`) or the raw `route.name`
        # depending on whether `modelMapping` has fired yet. Resolve
        # both forms.
        route: Optional[ModelRoute] = None
        if "/" in name:
            owner_name, _, rest = name.partition("/")
            if rest:
                # Route owners are always ORG principals — USER /
                # SYSTEM / GROUP never own routes. Scope the lookup
                # accordingly so a same-named non-ORG principal (e.g.
                # a GROUP under the partitioned name model) can't be
                # mis-resolved as the URL prefix owner.
                owner = (
                    await self.session.exec(
                        select(Principal).where(
                            Principal.name == owner_name,
                            Principal.kind == PrincipalType.ORG,
                            Principal.deleted_at.is_(None),
                        )
                    )
                ).first()
                if owner is not None:
                    route = await ModelRoute.one_by_fields(
                        self.session,
                        {"name": rest, "owner_principal_id": owner.id},
                    )
        if route is None:
            route = await ModelRoute.one_by_field(self.session, "name", name)
        if route is None:
            return None
        route_targets = await ModelRouteTarget.all_by_fields(
            self.session,
            fields={"route_id": route.id},
        )
        if len(route_targets) == 0:
            return None
        models = await Model.all_by_fields(
            session=self.session,
            fields={},
            extra_conditions=[
                Model.id.in_(
                    [e.model_id for e in route_targets if e.model_id is not None]
                )
            ],
        )
        # set a default static token to avoid empty token response for public maas model route
        registration_token = "static_token_not_found"
        for model in models:
            cluster = await Cluster.one_by_id(self.session, model.cluster_id)
            if cluster.registration_token is not None:
                registration_token = cluster.registration_token
                break

        return route.access_policy, registration_token

    @locked_cached()
    async def resolve_route_targets(self, name: str) -> List[RouteTargetResolution]:
        """Resolve a request model name to routing decisions, honoring an
        optional `<owner-name>/<route>` principal prefix.
        """
        owner_principal_id: Optional[int] = None
        raw_name = name
        if "/" in name:
            owner_name, _, rest = name.partition("/")
            if rest:
                # Route owners are always ORG principals — USER /
                # SYSTEM / GROUP never own routes. Scope the lookup
                # accordingly so a same-named non-ORG principal (e.g.
                # a GROUP under the partitioned name model) can't be
                # mis-resolved as the URL prefix owner.
                owner = (
                    await self.session.exec(
                        select(Principal).where(
                            Principal.name == owner_name,
                            Principal.kind == PrincipalType.ORG,
                            Principal.deleted_at.is_(None),
                        )
                    )
                ).first()
                if owner is not None:
                    owner_principal_id = owner.id
                    raw_name = rest
                # If the principal name didn't match, fall through and
                # try the literal name (handles edge cases like a route
                # called "literal/with/slashes" before the prefix
                # convention existed).
        # A raw (unprefixed) name is the platform Org's namespace by the
        # effective-route-name scheme; non-platform Orgs are addressed only
        # via their `<owner-name>/<route>` prefix. Scope an unprefixed lookup
        # to platform-owned routes so a raw name resolves within the platform
        # Org rather than a same-named route in another Org.
        if owner_principal_id is None:
            owner_principal_id = platform_principal_id()
        target_fields = {
            "route_name": raw_name,
            "state": TargetStateEnum.ACTIVE,
            "deleted_at": None,
        }
        targets = await ModelRouteTarget.all_by_fields(
            self.session,
            fields=target_fields,
        )
        # Narrow to the resolved owner's route (the parsed prefix owner, or
        # the platform Org for a raw name) by joining through the parent
        # ModelRoute's ``owner_principal_id``.
        if owner_principal_id is not None and len(targets) > 0:
            route_ids = {t.route_id for t in targets if t.route_id is not None}
            owner_routes = await ModelRoute.all_by_fields(
                self.session,
                fields={
                    "owner_principal_id": owner_principal_id,
                    "deleted_at": None,
                },
            )
            allowed_route_ids = {r.id for r in owner_routes if r.id in route_ids}
            targets = [t for t in targets if t.route_id in allowed_route_ids]
        return [
            RouteTargetResolution(
                model_id=target.model_id,
                overridden_model_name=target.overridden_model_name,
                weight=target.weight,
            )
            for target in targets
            if target.model_id is not None
        ]

    @locked_cached()
    async def get_by_name(self, name: str) -> Optional[ModelRoute]:
        """Resolve a request model name to its ``ModelRoute`` row.

        Mirrors the ``<owner-name>/<route>`` prefix handling from
        :meth:`get_model_auth_info_by_name` so the OpenAI proxy can
        attribute requests to the route they entered through regardless
        of whether Higress's ``modelMapping`` has rewritten the name yet.
        """
        if "/" in name:
            owner_name, _, rest = name.partition("/")
            if rest:
                owner = (
                    await self.session.exec(
                        select(Principal).where(
                            Principal.name == owner_name,
                            Principal.kind == PrincipalType.ORG,
                            Principal.deleted_at.is_(None),
                        )
                    )
                ).first()
                if owner is not None:
                    route = await ModelRoute.one_by_fields(
                        self.session,
                        {"name": rest, "owner_principal_id": owner.id},
                    )
                    if route is not None:
                        return route
        # An unprefixed name is the platform Org's namespace, matching
        # resolve_route_targets — so attribution resolves to the same route
        # the request was actually routed to.
        return await ModelRoute.one_by_fields(
            self.session,
            {"name": name, "owner_principal_id": platform_principal_id()},
        )

    async def update(
        self,
        model_route: ModelRoute,
        source: Union[dict, SQLModel, None] = None,
        auto_commit: bool = True,
    ):
        result = await model_route.update(self.session, source, auto_commit=auto_commit)
        names = await collect_route_cache_names(
            self.session, model_route.id, model_route.name
        )
        for name in names:
            await delete_cache_by_key(self.get_model_auth_info_by_name, name)
            await delete_cache_by_key(self.resolve_route_targets, name)
            await delete_cache_by_key(self.get_by_name, name)
        return result

    async def delete(self, model_route: ModelRoute, auto_commit: bool = True):
        # Owner principal must be resolved before the cascade removes the route.
        names = await collect_route_cache_names(
            self.session, model_route.id, model_route.name
        )
        result = await model_route.delete(self.session, auto_commit=auto_commit)
        for name in names:
            await delete_cache_by_key(self.get_model_auth_info_by_name, name)
            await delete_cache_by_key(self.resolve_route_targets, name)
            await delete_cache_by_key(self.get_by_name, name)
        return result


class ModelService:
    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached()
    async def get_by_id(self, model_id: int) -> Optional[Model]:
        result = await Model.one_by_id(self.session, model_id)
        if result is None:
            return None
        self.session.expunge(result)
        return result

    @locked_cached()
    async def get_by_name(self, name: str) -> Optional[Model]:
        result = await Model.one_by_field(self.session, "name", name)
        if result is None:
            return None
        self.session.expunge(result)
        return result

    async def update(
        self,
        model: Model,
        source: Union[dict, SQLModel, None] = None,
        *,
        auto_commit: bool = True,
    ):
        if isinstance(source, SQLModel):
            source = {key: getattr(source, key) for key in source.model_fields_set}
        if model.replicas == 0 and source and (source.get('replicas') or 0) > 0:
            instances = await ModelInstance.all_by_field(
                self.session, 'model_id', model.id
            )
            if any(
                has_model_instance_stop_barrier(instance)
                and instance.draining_since is not None
                for instance in instances
            ):
                # Physical teardown cannot be cancelled. Keep the whole old
                # group behind the restart barrier before forming replacements.
                source = {
                    **source,
                    'restarting_since': model.restarting_since
                    or datetime.now(timezone.utc),
                }
        result = await model.update(self.session, source, auto_commit=auto_commit)
        await delete_cache_by_key(self.get_by_id, model.id)
        await delete_cache_by_key(self.get_by_name, model.name)
        return result

    async def stop_instances(
        self,
        model: Model,
        instances: List[ModelInstance],
        source: Optional[dict] = None,
    ) -> None:
        """Commit model operation intent and all child stop requests together.

        Args:
            model: Parent model locked by the caller when changing its intent.
            instances: Old instances whose tasks must stop before deletion.
            source: Optional parent fields to commit with the child requests.
        """
        service = ModelInstanceService(self.session)
        try:
            locked = []
            for instance in sorted(instances, key=lambda item: item.id):
                current = await ModelInstance.one_by_id(
                    self.session, instance.id, for_update=True, refresh=True
                )
                if current is not None and current.model_id == model.id:
                    locked.append(current)
            await service.request_stop(locked, auto_commit=False)
            if source:
                await model.update(self.session, source, auto_commit=False)
            await self.session.commit()
        except Exception:
            await self.session.rollback()
            raise
        await service._invalidate_instances(locked)
        if source:
            await Model._invalidate_cached_all()
            await delete_cache_by_key(self.get_by_id, model.id)
            await delete_cache_by_key(self.get_by_name, model.name)

    async def delete(self, model: Model) -> bool:
        """Return False while workload stop confirmation prevents record deletion."""
        model = await Model.one_by_id(
            self.session, model.id, for_update=True, refresh=True
        )
        if model is None:
            return True
        instances = await ModelInstance.all_by_field(
            self.session, "model_id", model.id, for_update=True, refresh=True
        )
        managed = (
            [
                instance
                for instance in instances
                if has_model_instance_stop_barrier(instance)
            ]
            if any(
                requires_model_instance_stop_barrier(instance) for instance in instances
            )
            else []
        )
        pending = await ModelInstanceService(self.session).pending_stop_instances(
            managed, retire_unreachable=True
        )
        if managed and (
            model.deleted_at is None or pending or len(managed) != len(instances)
        ):
            if (
                model.deleted_at is None
                or model.replicas != 0
                or any(instance.draining_since is None for instance in instances)
                or any(
                    get_model_instance_workloads(instance) is not None
                    and not instance.distributed_servers.stop_requested
                    for instance in managed
                )
                or len(managed) != len(instances)
            ):
                await self.stop_instances(
                    model,
                    instances,
                    {
                        "deleted_at": model.deleted_at or datetime.now(timezone.utc),
                        "replicas": 0,
                    },
                )
            return False
        # ORM cascade bypasses child service caches; collect ids and route
        # names (raw + owner-name-prefixed effective) BEFORE deleting so
        # we can invalidate them explicitly below.
        instance_ids = list(
            (
                await self.session.exec(
                    select(ModelInstance.id).where(ModelInstance.model_id == model.id)
                )
            ).all()
        )
        route_names: Set[str] = set()
        stmt = (
            select(ModelRouteTarget.route_name, Principal.name, Principal.id)
            .join(ModelRoute, ModelRoute.id == ModelRouteTarget.route_id)
            .join(Principal, Principal.id == ModelRoute.owner_principal_id)
            .where(ModelRouteTarget.model_id == model.id)
        )
        for r_name, owner_name, p_id in (await self.session.exec(stmt)).all():
            if not r_name:
                continue
            route_names.add(r_name)
            route_names.add(
                effective_route_name(
                    r_name, owner_name, p_id == platform_principal_id()
                )
            )

        route_service = ModelRouteService(self.session)
        # A route this model created (enable_model_route, plus the LoRA child
        # routes) must not outlive the model: while one exists, the
        # unique-name lookup in create_model resolves it and rejects a new
        # model of that name. Delete it in the same transaction as the model,
        # unless targets other than this model's still point at it — a shared
        # route stays, and ModelRouteTargetController owns it from there.
        created_routes = await ModelRoute.all_by_fields(
            self.session,
            fields={"created_model_id": model.id, "deleted_at": None},
        )
        for route in created_routes or []:
            other_targets = (
                await self.session.exec(
                    select(func.count())
                    .select_from(ModelRouteTarget)
                    .where(
                        ModelRouteTarget.route_id == route.id,
                        ModelRouteTarget.deleted_at.is_(None),
                        or_(
                            ModelRouteTarget.model_id.is_(None),
                            ModelRouteTarget.model_id != model.id,
                        ),
                    )
                )
            ).one()
            if other_targets == 0:
                await route_service.delete(route, auto_commit=False)

        for instance in instances:
            await instance.delete(self.session, auto_commit=False)
        await model.delete(self.session)
        await delete_cache_by_key(self.get_by_id, model.id)
        await delete_cache_by_key(self.get_by_name, model.name)

        instance_service = ModelInstanceService(self.session)
        await delete_cache_by_key(instance_service.get_running_instances, model.id)
        for instance_id in instance_ids:
            await delete_cache_by_key(instance_service.get_by_id, instance_id)
        await invalidate_workers_allocated(instances)

        for route_name in route_names:
            await delete_cache_by_key(route_service.resolve_route_targets, route_name)
            await delete_cache_by_key(
                route_service.get_model_auth_info_by_name, route_name
            )

        return True


class CacheServiceService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def delete(self, cache_service: CacheService) -> None:
        # The parent commits all instance deletions in one transaction.
        await cache_service.delete(self.session)
        await invalidate_cache_workers_ram()


class CacheServiceInstanceService:
    """Mutate cache instances and invalidate reservations after commit."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self, instance: Union[CacheServiceInstance, CacheServiceInstanceCreate]
    ) -> CacheServiceInstance:
        result = await CacheServiceInstance.create(self.session, instance)
        await invalidate_cache_workers_ram()
        return result

    async def update(
        self,
        instance: CacheServiceInstance,
        source: Union[dict, SQLModel, None] = None,
    ) -> None:
        reservations = cache_ram_by_worker([instance])
        await instance.update(self.session, source)
        # With no patch, the caller may have mutated the instance before entry.
        if source is None or reservations != cache_ram_by_worker([instance]):
            await invalidate_cache_workers_ram()

    async def delete(self, instance: CacheServiceInstance) -> None:
        await instance.delete(self.session)
        await invalidate_cache_workers_ram()


class ModelInstanceService:
    def __init__(self, session: AsyncSession):
        self.session = session

    @locked_cached()
    async def get_by_id(self, id: int) -> Optional[ModelInstance]:
        result = await ModelInstance.one_by_id(self.session, id)
        if result is None:
            return None
        self.session.expunge(result)
        return result

    @locked_cached()
    async def get_running_instances(self, model_id: int) -> List[ModelInstance]:
        results = await ModelInstance.all_by_fields(
            self.session,
            fields={
                "model_id": model_id,
                "state": ModelInstanceStateEnum.RUNNING,
                "draining_since": None,
            },
        )
        if results is None:
            return []

        for result in results:
            self.session.expunge(result)
        return results

    async def create(self, model_instance):
        await self._prepare_recovery_policy(model_instance, None)
        # Denormalize the shared-cache connection info onto the instance so
        # the worker can inject engine config without a server round-trip.
        model = await Model.one_by_id(self.session, model_instance.model_id)
        if model is not None:
            # The degrading variant: replica creation must not fail on a
            # resolution error — the instance starts without the shared
            # cache instead (same contract as the scheduler's re-resolve).
            model_instance.cache_config = await resolve_instance_cache_config_safe(
                self.session, model, role=model_instance.role
            )
        result = await ModelInstance.create(self.session, model_instance)
        await delete_cache_by_key(self.get_running_instances, model_instance.model_id)
        await invalidate_workers_allocated([model_instance])
        return result

    async def batch_create(self, model_instances: List[ModelInstance]):
        """Create every instance in ONE transaction.

        The mirror of ``batch_delete``, and it exists for the same reason that
        one does: a PD group's members have to appear together. Kueue's
        pod-group admission counts the members it can see against the declared
        total, so a group whose rows land one commit at a time is repeatedly
        seen as incomplete. The group's feasibility verdict also has to be
        settled before any member row exists, which a per-row loop cannot
        express.

        Cache resolution is per instance and per the same degrading contract
        as ``create``: a group must not fail to form because a shared cache
        could not be resolved.
        """
        if not model_instances:
            return []

        model_ids = {mi.model_id for mi in model_instances}
        models = {}
        for model_id in model_ids:
            models[model_id] = await Model.one_by_id(self.session, model_id)

        try:
            created = []
            for model_instance in model_instances:
                await self._prepare_recovery_policy(model_instance, None)
                model = models.get(model_instance.model_id)
                if model is not None:
                    model_instance.cache_config = (
                        await resolve_instance_cache_config_safe(
                            self.session, model, role=model_instance.role
                        )
                    )
                result = await ModelInstance.create(
                    self.session, model_instance, auto_commit=False
                )
                created.append(result)
            await self.session.commit()
        except Exception as e:
            await self.session.rollback()
            names = [mi.name for mi in model_instances]
            raise InternalServerErrorException(
                message=f"Failed to create model instances {names}: {e}"
            )

        # Past the commit, so past the point where anything can be undone. The
        # cache work below used to sit inside the block above, where a failure
        # in it rolled back a transaction that had already landed -- a no-op --
        # and then told the caller the members had not been created. For a
        # group that is the worst answer available: the rows exist, and a
        # caller acting on "not created" creates them again.
        #
        # create(auto_commit=False) returns before invalidating cached_all, so
        # the batch has to do it or subscribe()'s replay snapshot will not
        # carry the new rows. A failure there is reported rather than raised:
        # a stale cache is a pass of lag that the next invalidation clears,
        # and losing a created group to it is not a trade worth making.
        try:
            await ModelInstance._invalidate_cached_all()
            for model_id in model_ids:
                await delete_cache_by_key(self.get_running_instances, model_id)
            await invalidate_workers_allocated(model_instances)
        except Exception as e:
            logger.warning(
                "Model instances %s were created, but a cache could not be "
                "invalidated: %s. Readers may lag by a pass.",
                [mi.name for mi in model_instances],
                e,
            )

        return created

    async def update(
        self, model_instance: ModelInstance, source: Union[dict, SQLModel, None] = None
    ):
        source = await self._prepare_recovery_policy(model_instance, source)
        result = await model_instance.update(self.session, source)
        await delete_cache_by_key(self.get_running_instances, model_instance.model_id)
        await delete_cache_by_key(self.get_by_id, model_instance.id)
        await invalidate_workers_allocated([model_instance])
        return result

    async def ensure_recovery_policy(self, instance: ModelInstance) -> None:
        """Backfill a bound legacy round before exposing it to a Worker."""
        if (
            calculate_group_recovery_policy(instance, None) is not None
            and instance.distributed_servers.group_recovery is None
        ):
            await self.update(instance, {})

    async def _prepare_recovery_policy(self, instance, source):
        fields = (
            {key: getattr(source, key) for key in source.model_fields_set}
            if isinstance(source, SQLModel)
            else dict(source or {})
        )
        # A shallow ORM copy shares attribute history with the attached row.
        # Policy calculation must not mark incoming values as already persisted.
        values = {key: getattr(instance, key) for key in type(instance).model_fields}
        candidate = ModelInstance(**{**values, **fields})
        if calculate_group_recovery_policy(candidate, None) is None:
            return source
        servers = candidate.distributed_servers.model_copy(deep=True)
        new_round = (candidate.restart_count or 0) != (instance.restart_count or 0)
        if not new_round and instance.distributed_servers is not None:
            servers.group_recovery = instance.distributed_servers.group_recovery
        if new_round:
            # An old launch snapshot must not select the next launch's policy.
            servers.group_recovery = None
            servers.legacy_stop = None
            fields['model_config_snapshot'] = None
            candidate.model_config_snapshot = None
        candidate.distributed_servers = servers
        if servers.group_recovery is None:
            model = await Model.one_by_id(self.session, instance.model_id, refresh=True)
            if model is None:
                raise ConflictException(
                    message='Model is unavailable for launch policy'
                )
            servers.group_recovery = requires_group_recovery(candidate, model)
        if (
            new_round
            and instance.distributed_servers is not None
            and instance.distributed_servers.group_recovery is False
            and servers.group_recovery
            and not model_instance_workloads_stopped(instance)
        ):
            return {
                'state': ModelInstanceStateEnum.ERROR,
                'state_message': 'Stopping all members before changing recovery policy',
                'distributed_servers': await self._legacy_stop_servers(instance),
            }
        if not new_round and servers == instance.distributed_servers:
            return source
        fields['distributed_servers'] = servers
        if source is None:
            for key, value in fields.items():
                setattr(instance, key, value)
            return None
        return fields

    async def reconcile_managed_instance(
        self, instance_id: int, restart_count: int, *, allow_restart: bool = True
    ) -> Optional[float]:
        """Recover an SMG alone, or stop and advance a managed or legacy model group.

        Returns the remaining restart backoff, if a timer is needed. This owns
        its transaction; callers must not have pending writes in the session.

        Args:
            instance_id: Instance to reconcile from authoritative state.
            restart_count: Generation observed by the controller.
            allow_restart: Whether the current deployment specification permits recovery.
        """
        instance = await ModelInstance.one_by_id(
            self.session, instance_id, for_update=True, refresh=True
        )
        source, due = {}, None
        if (
            instance is not None
            and instance.deleted_at is None
            and (instance.restart_count or 0) == restart_count
        ):
            workloads = get_model_instance_workloads(instance)
            if workloads is None:
                source, due = await self._legacy_recovery(instance, allow_restart)
            else:
                # A non-SMG failure withdraws the generation before surviving
                # tasks stop; the restart fence advances only after teardown.
                failure = next(
                    (
                        item
                        for item in workloads
                        if (
                            item.component is None
                            or item.component.kind != "sglang_smg"
                        )
                        and (
                            item.state == ModelInstanceWorkloadStateEnum.FAILED
                            or (
                                item.state == ModelInstanceWorkloadStateEnum.STOPPED
                                and instance.draining_since is None
                            )
                        )
                    ),
                    None,
                )
                if (
                    failure is not None
                    and instance.state != ModelInstanceStateEnum.ERROR
                ):
                    ranks = ", ".join(
                        f"dp_rank={p.dp_rank}/node_rank={p.node_rank}"
                        for p in failure.dp_rank_placements
                    )
                    source = {
                        "state": ModelInstanceStateEnum.ERROR,
                        "state_message": (
                            f"Workload {failure.name} on Worker {failure.worker_id} "
                            f"({ranks}) is {failure.state.value}."
                        ),
                    }
                elif (
                    instance.state == ModelInstanceStateEnum.ERROR
                    and instance.draining_since is None
                    and allow_restart
                ):
                    source, due = await self._managed_restart(instance, workloads)
                elif any(
                    item.component is not None and item.component.kind == "sglang_smg"
                    for item in workloads
                ):
                    source, due = await self._managed_smg_reconcile(
                        instance, workloads, allow_restart=allow_restart
                    )
                elif (
                    instance.state
                    in (
                        ModelInstanceStateEnum.STARTING,
                        ModelInstanceStateEnum.UNREACHABLE,
                    )
                    and instance.draining_since is None
                    and await self._managed_instance_ready(instance, workloads)
                ):
                    source = {
                        "state": ModelInstanceStateEnum.RUNNING,
                        "state_message": "",
                    }
        if source and any(
            getattr(instance, key) != value for key, value in source.items()
        ):
            await self.update(instance, source)
        else:
            # Release the row lock before routing or other external work runs.
            await self.session.commit()
        return due

    async def _legacy_recovery(self, instance, allow_restart):
        model = await Model.one_by_id(self.session, instance.model_id, refresh=True)
        if model is None or not requires_group_recovery(instance, model):
            return {}, None
        if instance.draining_since is not None:
            return {}, None
        failed = next(
            (
                worker
                for worker in instance.distributed_servers.subordinate_workers
                if worker.state == ModelInstanceStateEnum.ERROR
            ),
            None,
        )
        if instance.state != ModelInstanceStateEnum.ERROR and failed is None:
            return {}, None
        if model_instance_legacy_stop(instance) is None:
            return {
                'state': ModelInstanceStateEnum.ERROR,
                'state_message': (
                    instance.state_message
                    if instance.state == ModelInstanceStateEnum.ERROR
                    else (
                        f'Worker {failed.worker_id}: '
                        f'{failed.state_message or "Inference server failed."}'
                    )
                ),
                'distributed_servers': await self._legacy_stop_servers(instance),
            }, None
        if allow_restart:
            return await self._managed_restart(instance, None)
        return {}, None

    async def _legacy_stop_servers(self, instance, bindings=None):
        workers = sorted(member_worker_ids(instance))
        if bindings is None:
            bindings = {}
            for worker_id in workers:
                worker = await Worker.one_by_id(self.session, worker_id, refresh=True)
                bindings[worker_id] = getattr(worker, 'system_principal_id', None)
        servers = (
            instance.distributed_servers.model_copy(deep=True)
            if instance.distributed_servers
            else DistributedServers()
        )
        servers.legacy_stop = ModelInstanceLegacyStop(
            token=uuid4(),
            restart_count=instance.restart_count or 0,
            worker_ids=workers,
            worker_principal_ids={
                worker_id: bindings[worker_id] for worker_id in workers
            },
        )
        return servers

    async def _managed_instance_ready(
        self, instance: ModelInstance, workloads: List[ModelInstanceWorkload]
    ) -> bool:
        members = set(member_worker_ids(instance))
        if not workloads or {item.worker_id for item in workloads} != members:
            return False
        if any(
            item.component is not None and item.component.kind == "vllm_hybrid"
            for item in workloads
        ):
            apis = [w for w in workloads if w.component and w.component.api_port]
            ready = (
                len(workloads) == len(members)
                and all(
                    w.component is not None and w.component.kind == "vllm_hybrid"
                    for w in workloads
                )
                and bool(apis)
                and all(w.component.ready for w in apis)
            )
        elif any(item.component is not None for item in workloads):
            engines = [
                w
                for w in workloads
                if w.component is not None and w.component.kind == "sglang_engine"
            ]
            smgs = [
                w
                for w in workloads
                if w.component is not None and w.component.kind == "sglang_smg"
            ]
            mains = [w for w in engines if w.component.api_port is not None]
            ready = (
                len(engines) == len(members)
                and {w.worker_id for w in engines} == members
                and len(mains) == 1
                and mains[0].worker_id == instance.worker_id
                and mains[0].component.ready
                and len(smgs) == 1
                and smgs[0].worker_id == instance.worker_id
                and smgs[0].component.ready
                and not smgs[0].component.stop_requested
                and all(w.component is not None for w in workloads)
            )
        else:
            mains = [
                placement
                for item in workloads
                for placement in item.dp_rank_placements
                if placement.node_rank == 0
            ]
            ready = bool(mains) and all(p.ready for p in mains)
        if not ready or any(
            item.state != ModelInstanceWorkloadStateEnum.RUNNING for item in workloads
        ):
            return False
        return await self._managed_workers_ready(members)

    async def _managed_workers_ready(self, members: Set[int]) -> bool:
        workers = await Worker.all_by_ids(self.session, members, refresh=True)
        if {worker.id for worker in workers} != members:
            return False
        for worker in workers:
            if (
                worker.deleted_at is not None
                or worker.unreachable
                or worker.state
                not in (WorkerStateEnum.READY, WorkerStateEnum.MAINTENANCE)
                or is_offline(
                    worker.heartbeat_time, envs.WORKER_HEARTBEAT_GRACE_PERIOD
                )[0]
            ):
                return False
        return True

    async def _managed_smg_reconcile(
        self,
        instance: ModelInstance,
        workloads: List[ModelInstanceWorkload],
        *,
        allow_restart: bool,
    ) -> Tuple[dict, Optional[float]]:
        if instance.draining_since is not None or instance.state not in (
            ModelInstanceStateEnum.RUNNING,
            ModelInstanceStateEnum.STARTING,
            ModelInstanceStateEnum.UNREACHABLE,
        ):
            return {}, None
        smgs = [
            w for w in workloads if w.component and w.component.kind == "sglang_smg"
        ]
        if len(smgs) != 1:
            return {}, None
        smg = smgs[0]
        # Lost contact is not evidence that the CPU process failed. Wait for
        # the owning Worker to observe it again unless a stop was already issued.
        if (
            smg.state == ModelInstanceWorkloadStateEnum.UNKNOWN
            and not smg.component.stop_requested
        ):
            return {}, None
        if not smg.component.stop_requested and smg.state not in (
            ModelInstanceWorkloadStateEnum.FAILED,
            ModelInstanceWorkloadStateEnum.STOPPED,
            ModelInstanceWorkloadStateEnum.UNKNOWN,
        ):
            if instance.state in (
                ModelInstanceStateEnum.STARTING,
                ModelInstanceStateEnum.UNREACHABLE,
            ) and await self._managed_instance_ready(instance, workloads):
                return {
                    "state": ModelInstanceStateEnum.RUNNING,
                    "state_message": "",
                }, None
            return {}, None
        model = await Model.one_by_id(self.session, instance.model_id, refresh=True)
        if (
            model is None
            or model.deleted_at is not None
            or model.replicas == 0
            or model.restarting_since is not None
        ):
            return {}, None
        servers = instance.distributed_servers.model_copy(deep=True)
        replacement = next(w for w in servers.workloads if w.name == smg.name)
        replacement.component.ready = False
        replacement.component.stop_requested = True
        source = {
            "state": ModelInstanceStateEnum.STARTING,
            "state_message": "SMG is unavailable.",
            "distributed_servers": servers,
        }
        if not model.restart_on_error:
            source["state_message"] = "SMG unavailable; automatic recovery is disabled."
        if not allow_restart or not await self._managed_smg_restart_allowed(
            instance, workloads, smg, model
        ):
            return source, None
        component = smg.component
        count = component.attempt
        delay = min(10 * 2 ** min(count - 1, 5), 300) if count else 0
        now = datetime.now(timezone.utc)
        remaining = (
            delay - (now - component.last_restart_time).total_seconds()
            if component.last_restart_time is not None
            else 0
        )
        if remaining > 0:
            return source, remaining
        replacement.name = (
            f"{instance.name}-r{instance.restart_count or 0}-smg-a{count + 1}"
        )
        replacement.state = ModelInstanceWorkloadStateEnum.PENDING
        replacement.component.attempt = count + 1
        replacement.component.stop_requested = False
        replacement.component.last_restart_time = now
        source["state_message"] = ""
        return source, None

    async def _managed_smg_restart_allowed(
        self,
        instance: ModelInstance,
        workloads: List[ModelInstanceWorkload],
        smg: ModelInstanceWorkload,
        model: Optional[Model],
    ) -> bool:
        if (
            smg.state != ModelInstanceWorkloadStateEnum.STOPPED
            or model is None
            or model.deleted_at is not None
            or not model.restart_on_error
            or model.replicas == 0
            or model.restarting_since is not None
            or (model.stale and (instance.group_id or not instance.spec_digest))
        ):
            return False
        engines = [
            w for w in workloads if w.component and w.component.kind == "sglang_engine"
        ]
        mains = [w for w in engines if w.component.api_port is not None]
        members = set(member_worker_ids(instance))
        if (
            len(engines) != len(members)
            or {w.worker_id for w in engines} != members
            or len(mains) != 1
            or mains[0].worker_id != instance.worker_id
            or not mains[0].component.ready
            or smg.worker_id != instance.worker_id
            or any(w.state != ModelInstanceWorkloadStateEnum.RUNNING for w in engines)
            or not await self._managed_workers_ready(members)
        ):
            return False
        if instance.spec_digest:
            from gpustack.server.controllers import model_members_stale

            if await model_members_stale(self.session, model, [instance]):
                return False
        return True

    async def _managed_restart(
        self, instance: ModelInstance, workloads: Optional[List[ModelInstanceWorkload]]
    ) -> Tuple[dict, Optional[float]]:
        if workloads is None and not model_instance_workloads_stopped(instance):
            return {}, None
        if workloads is not None and any(
            item.state != ModelInstanceWorkloadStateEnum.STOPPED for item in workloads
        ):
            return {}, None
        model = await Model.one_by_id(self.session, instance.model_id, refresh=True)
        if (
            model is None
            or model.deleted_at is not None
            or not model.restart_on_error
            or model.replicas == 0
            or model.restarting_since is not None
            or (instance.group_id and model.stale)
        ):
            return {}, None
        count = instance.restart_count or 0
        delay = min(10 * 2 ** min(count - 1, 5), 300) if count else 0
        now = datetime.now(timezone.utc)
        remaining = (
            delay - (now - instance.last_restart_time).total_seconds()
            if instance.last_restart_time is not None
            else 0
        )
        if remaining > 0:
            return {}, remaining
        # Registration precedes every create. An unregistered task cannot be
        # running; late reports for this generation fail after the locked bump.
        servers = instance.distributed_servers.model_copy(deep=True)
        servers.workloads = []
        source = {
            "restart_count": count + 1,
            "last_restart_time": now,
            "state": ModelInstanceStateEnum.SCHEDULED,
            "state_message": "",
            "distributed_servers": servers,
        }
        if workloads is None:
            servers.workloads = None
            servers.legacy_stop = None
            for worker in servers.subordinate_workers:
                worker.state = ModelInstanceStateEnum.PENDING
                worker.state_message = None
                worker.pid = None
            source['pid'] = None
            return source, None
        configuration_changed = False
        if not instance.group_id and instance.spec_digest:
            from gpustack.server.controllers import model_spec_digest

            target = await model_spec_digest(self.session, model)
            source['spec_digest'] = target
            configuration_changed = instance.spec_digest != target
            if configuration_changed:
                # Every old task has stopped; preparation can reserve a new layout.
                source.update(
                    await self._managed_restart_configuration(instance, model)
                )
                source.update(
                    port=None,
                    ports=None,
                    named_ports=None,
                    backend=get_backend(model),
                    backend_version=model.backend_version,
                    api_detected_backend_version=None,
                )
        from gpustack.utils.model_instance_workers import sglang_smg_mode
        from gpustack.utils.vllm_topology import vllm_lb_mode

        sglang_engines = any(
            item.component is not None and item.component.kind == "sglang_engine"
            for item in workloads
        )
        single_entry = (
            vllm_lb_mode(model.backend_parameters) == 'internal'
            if get_backend(model) == BackendEnum.VLLM
            else not sglang_smg_mode(model)
        )
        if (
            instance.role is None
            and not instance.group_id
            and model.disaggregation is None
            and single_entry
            and (
                configuration_changed
                or (sglang_engines and get_backend(model) == BackendEnum.SGLANG)
            )
        ):
            # All old tasks have stopped before the next launch changes its layout.
            servers.workloads = None
            servers.orchestration_mode = "legacy"
            source.update(port=None, ports=None, named_ports=None)
            claim = instance.computed_resource_claim
            if claim is not None and sglang_engines:
                source["computed_resource_claim"] = claim.model_copy(
                    update={"ram": max(0, (claim.ram or 0) - ROUTER_DEFAULT_MEMORY)}
                )
        return source, None

    async def _managed_restart_configuration(
        self, instance: ModelInstance, model: Model
    ) -> dict:
        """Rebind launch inputs and file associations after every old task stops."""
        from gpustack.utils.model_source import get_draft_model_source

        draft_source = await get_draft_model_source(self.session, model)
        worker = await Worker.one_by_id(self.session, instance.worker_id)
        cache_config = await resolve_instance_cache_config_safe(
            self.session, model, worker=worker, spans_workers=instance.spans_workers
        )
        # Cached files remain reusable; only this instance's old associations go.
        for link in (ModelInstanceModelFileLink, ModelInstanceDraftModelFileLink):
            await self.session.execute(
                delete(link).where(link.model_instance_id == instance.id)
            )
        set_committed_value(instance, 'model_files', [])
        set_committed_value(instance, 'draft_model_files', [])
        return {
            **model.model_dump(include=set(ModelSource.model_fields)),
            'draft_model_source': draft_source,
            'cache_config': cache_config,
            'resolved_path': None,
            'draft_model_resolved_path': None,
            'download_progress': None,
            'draft_model_download_progress': None,
            'mounted_loras': [],
            'model_config_snapshot': None,
        }

    async def delete(self, model_instance: ModelInstance) -> bool:
        """Return whether the row is absent, or False while stopping is pending."""
        current = await ModelInstance.one_by_id(
            self.session, model_instance.id, for_update=True, refresh=True
        )
        if current is None:
            return True
        model_instance = current
        if requires_model_instance_stop_barrier(model_instance):
            deleted = await self.batch_delete([model_instance])
            return model_instance.name in deleted
        await model_instance.delete(self.session)
        await delete_cache_by_key(self.get_running_instances, model_instance.model_id)
        await delete_cache_by_key(self.get_by_id, model_instance.id)
        await invalidate_workers_allocated([model_instance])
        return True

    async def batch_delete(
        self,
        model_instances: List[ModelInstance],
        *,
        require_all_stopped: bool = False,
        retire_unreachable: bool = False,
    ):
        if not model_instances:
            return []

        names = []
        ids = set()
        try:
            locked = []
            guarded = any(
                requires_model_instance_stop_barrier(m) for m in model_instances
            )
            for m in sorted(model_instances, key=lambda instance: instance.id):
                expected = m.restart_count or 0
                m = await ModelInstance.one_by_id(
                    self.session, m.id, for_update=True, refresh=True
                )
                if m is None or (m.restart_count or 0) != expected:
                    continue
                locked.append(m)
            guarded = guarded or any(
                requires_model_instance_stop_barrier(m) for m in locked
            )
            if guarded:
                await self.request_stop(locked, auto_commit=False)
            pending = (
                await self.pending_stop_instances(
                    locked, retire_unreachable=retire_unreachable
                )
                if guarded
                else []
            )
            await self.request_stop(pending, auto_commit=False)
            deletable = (
                []
                if require_all_stopped
                and (pending or len(locked) != len(model_instances))
                else [m for m in locked if m not in pending]
            )
            for m in deletable:
                await m.delete(self.session, auto_commit=False)
                names.append(m.name)
                ids.add(m.model_id)
            ids.update(m.model_id for m in pending)
            await self.session.commit()

            # delete(auto_commit=False) returns before invalidating cached_all,
            # so the batch commit must do it. Otherwise subscribe()'s replay
            # snapshot keeps serving the deleted rows and watch reconnects
            # resurrect them as ghost CREATED events.
            await ModelInstance._invalidate_cached_all()

            for id in ids:
                await delete_cache_by_key(self.get_running_instances, id)
            for instance in model_instances:
                await delete_cache_by_key(self.get_by_id, instance.id)
            await invalidate_workers_allocated(model_instances)

            return names
        except Exception as e:
            await self.session.rollback()
            if isinstance(e, HTTPException):
                raise
            raise InternalServerErrorException(
                message=f"Failed to delete model instances {names}: {e}"
            )

    async def pending_stop_instances(
        self,
        instances: List[ModelInstance],
        *,
        retire_unreachable: bool = False,
    ) -> List[ModelInstance]:
        """Wait for live Workers; explicit deployment teardown may retire lost ones.

        Retired ownership is only used to delete the old records. It does not
        fabricate stop reports or permit automatic recovery of disconnected runs.
        Reconnecting Workers clean up tasks no longer owned by an instance.
        """
        pending = [
            instance
            for instance in instances
            if has_model_instance_stop_barrier(instance)
            and not model_instance_workloads_stopped(instance)
        ]
        if not retire_unreachable or not pending:
            return pending
        members = {
            worker_id
            for instance in pending
            for worker_id in member_worker_ids(instance)
        }
        workers = await Worker.all_by_ids(self.session, members, refresh=True)
        available = {
            worker.id
            for worker in workers
            if worker.deleted_at is None
            and not worker.unreachable
            and worker.state != WorkerStateEnum.UNREACHABLE
            and not is_offline(
                worker.heartbeat_time, envs.WORKER_HEARTBEAT_GRACE_PERIOD
            )[0]
        }
        return [
            instance
            for instance in pending
            if not model_instance_workloads_stopped(
                instance, retired_worker_ids=members - available
            )
        ]

    async def request_stop(
        self, instances: List[ModelInstance], *, auto_commit: bool = True
    ) -> None:
        """Release physical teardown without deleting manifests or claims."""
        now = datetime.now(timezone.utc)
        guarded_models = {
            instance.model_id
            for instance in instances
            if requires_model_instance_stop_barrier(instance)
        }
        to_guard = [
            instance
            for instance in instances
            if instance.model_id in guarded_models
            and not has_model_instance_stop_barrier(instance)
        ]
        bindings = {}
        for worker_id in sorted(
            {
                worker_id
                for instance in to_guard
                for worker_id in member_worker_ids(instance)
            }
        ):
            worker = await Worker.one_by_id(self.session, worker_id, refresh=True)
            bindings[worker_id] = getattr(worker, "system_principal_id", None)
        for instance in instances:
            source = {}
            if instance in to_guard:
                workers = sorted(member_worker_ids(instance))
                if not workers and (
                    instance.pid is not None
                    or instance.port
                    or instance.ports
                    or instance.named_ports
                    or instance.state
                    not in (
                        ModelInstanceStateEnum.PENDING,
                        ModelInstanceStateEnum.SCHEDULED,
                    )
                ):
                    raise ConflictException(
                        message="Cannot prove an unassigned legacy instance has no tasks"
                    )
                source["distributed_servers"] = await self._legacy_stop_servers(
                    instance, bindings
                )
            if (
                get_model_instance_workloads(instance) is not None
                and not instance.distributed_servers.stop_requested
            ):
                servers = instance.distributed_servers.model_copy(deep=True)
                servers.stop_requested = True
                source["distributed_servers"] = servers
            if instance.draining_since is None:
                source["draining_since"] = now
            if source:
                await instance.update(self.session, source, auto_commit=False)
        if auto_commit:
            await self.session.commit()
            await self._invalidate_instances(instances)

    async def _invalidate_instances(self, instances: List[ModelInstance]) -> None:
        await ModelInstance._invalidate_cached_all()
        for model_id in {instance.model_id for instance in instances}:
            await delete_cache_by_key(self.get_running_instances, model_id)
        for instance in instances:
            await delete_cache_by_key(self.get_by_id, instance.id)
        await invalidate_workers_allocated(instances)

    async def batch_update(
        self,
        model_instances: List[ModelInstance],
        source: Union[dict, SQLModel, None] = None,
    ):
        if not model_instances:
            return []

        names = [mi.name for mi in model_instances]
        ids = set()
        try:
            for m in model_instances:
                prepared = await self._prepare_recovery_policy(m, source)
                await m.update(self.session, prepared, auto_commit=False)
                ids.add(m.model_id)
            await self.session.commit()

            # Per-instance update() invalidates cached_all at flush time, i.e.
            # before this manual commit — a concurrent read in that window could
            # repopulate it with pre-commit state. Invalidate again after commit
            # so the replay snapshot reflects the committed rows.
            await ModelInstance._invalidate_cached_all()

            for id in ids:
                await delete_cache_by_key(self.get_running_instances, id)
            await invalidate_workers_allocated(model_instances)

            return names
        except Exception as e:
            await self.session.rollback()
            raise InternalServerErrorException(
                message=f"Failed to update model instances {names}: {e}"
            )


class ModelFileService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_resolved_path(self, path: str) -> List[ModelFile]:
        results = await ModelFile.all_by_fields(
            self.session,
        )
        filtered_results = []
        for result in results:
            self.session.expunge(result)
            if path in result.resolved_paths:
                filtered_results.append(result)

        return filtered_results

    async def get_by_source_index(self, source_index: str) -> List[ModelFile]:
        results = await ModelFile.all_by_field(
            self.session, "source_index", source_index
        )
        if results is None:
            return None

        for result in results:
            self.session.expunge(result)
        return results

    async def create(self, model_file: ModelFile):
        return await ModelFile.create(self.session, model_file)


def intersection_nullable_set(set1: Set[str], set2: Optional[Set[str]]) -> Set[str]:
    if set2 is None:
        return set1
    return set1.intersection(set2)


async def delete_accessible_model_cache(
    *user_ids: int,
):
    for user_id in user_ids:
        await delete_cache_by_key(UserService.get_user_accessible_model_names, user_id)


async def revoke_model_access_cache(
    session: AsyncSession,
    model: Optional[ModelRoute] = None,
    extra_user_ids: Optional[set[int]] = None,
):
    user_ids = set()
    if model is None:
        # Cache-bust everyone — restrict to actual users so we don't
        # waste time invalidating ORG / GROUP keys that never existed.
        result = await session.exec(
            select(User.id).where(User.kind == PrincipalType.USER)
        )
        user_ids = set(result.all())
    else:
        # Users with a direct grant on this route's ACL — i.e. their
        # USER-principal appears in ``model_route_principals`` for this
        # route. Group / Org grants are intentionally not expanded
        # here: this helper invalidates per-user caches and the broader
        # invalidation path uses ``model=None`` (cache-bust everyone).
        stmt = (
            select(User.id)
            .join(
                ModelRoutePrincipalLink,
                ModelRoutePrincipalLink.principal_id == User.id,
            )
            .where(ModelRoutePrincipalLink.route_id == model.id)
        )
        user_ids = set((await session.exec(stmt)).all())
    if extra_user_ids:
        user_ids.update(extra_user_ids)
    await delete_accessible_model_cache(*user_ids)
