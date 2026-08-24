"""Keeps the gateway's copy of "which keys exist" in step with the database.

The ext-auth plugin authenticates locally, which means it needs the set of keys
the server would accept. That set is mutable state the server owns, so the
point of this reconciler is not to make the gateway independent of the server
-- it is to move the server off the *request* path and onto the *change
propagation* path. A server outage should then cost freshness (a key created
seconds ago is not recognized yet, a revocation lands late) and not
availability (keys that already work stop working).

Two things follow from that, and they are the whole design of this module:

**Events trigger; the recomputation is what produces state.** Every pass
rebuilds the tables from the database in full, and the watches below never
compute anything -- they only say "look now". The tables are a projection over
a join (api_keys x principals) plus ``ModelRoute.access_policy``, so an
incremental updater would have to subscribe to every input and translate each
event into the right mutation. Miss one input and the state diverges silently
and permanently, whereas a level-triggered pass heals within one interval. The
failure is also asymmetric: a missed addition costs a key some round-trips to
the server, a missed removal keeps a revoked credential authenticating locally
-- and on a PUBLIC route nothing behind the gateway catches it.

Deletion via the ORM does raise events (``Principal.api_keys`` carries
``cascade: delete``, so each key is deleted individually and publishes), but a
``DELETE`` issued straight against the database does not, and neither does a
migration. Those are exactly the changes nobody remembers to think about, so
the periodic interval is a security parameter rather than a refresh rate.

**Tightening flushes now, widening waits.** The direction of a change decides
its urgency. A revocation, a deactivated principal, a route losing PUBLIC
status: those are all "someone should stop being let in", and they go out
immediately. A new digest or a new PUBLIC rule is monotonic and idempotent --
the key already works through the fallback path, the route already authorizes
per request -- so those ride the next tick, which is what keeps a mass digest
backfill from turning into thousands of CR writes.

Expiry needs neither: ``exp`` travels in the entry itself and the plugin
compares it against its own clock, so a key expiring changes nothing here. It
is only dropped from the tables on the next pass, to stop dead rows from
spending the entry budget.
"""

import asyncio
import logging
from datetime import datetime, timezone
from functools import partial
from typing import Any, Dict, List, Optional, Set, Tuple

from kubernetes_asyncio import client as k8s_client
from sqlalchemy import or_
from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession

from gpustack import envs
from gpustack.config.config import Config
from gpustack.gateway import get_async_k8s_config, get_gpustack_higress_registry
from gpustack.gateway.client.extensions_higress_io_v1_api import (
    ExtensionsHigressIoV1Api,
)
from gpustack.gateway.client.networking_higress_io_v1_api import McpBridgeRegistry
from gpustack.gateway.ext_auth import (
    ext_auth_reconcile_spec_diff,
    ext_auth_resource_name,
)
from gpustack.gateway.utils import ensure_wasm_plugin, route_ingress_names_for_plugins
from gpustack.schemas.api_keys import ApiKey
from gpustack.schemas.config import GatewayModeEnum
from gpustack.schemas.model_routes import AccessPolicyEnum, ModelRoute
from gpustack.schemas.principals import Principal, PrincipalType
from gpustack.security import gateway_digest
from gpustack.server.bus import EventType, event_field
from gpustack.server.db import async_session

logger = logging.getLogger(__name__)


# Rendered sizes, measured against a real CR rather than estimated: a key entry
# with a truncated digest (``"<16 hex ak>":{"digest":"s128$...","user_id":N}``)
# came out at 112 bytes and one PUBLIC match rule (two ingress names plus the
# access policy) at 169. Both are rounded up here, and a refs entry is far
# smaller than a key entry, so sizing every entry as a key entry errs toward
# leaving room -- which is the direction that matters, since the cost of
# underestimating is a refused write rather than a smaller table.
KEY_ENTRY_BYTES = 115
MATCH_RULE_BYTES = 170

# Bounds on how fast a dropped event watch reconnects.
_WATCH_RETRY_MIN_SECONDS = 1
_WATCH_RETRY_MAX_SECONDS = 30


def split_cr_budget(budget: int, public_route_count: int) -> Tuple[int, int]:
    """``(public routes to keep, key entries that then fit)``.

    One budget rather than two independent caps. The key tables and the PUBLIC
    match rules share a single CR and therefore a single etcd object limit;
    capping them separately means the sum can still overrun it, and overrunning
    it is not a partial failure -- the write is refused, the tables freeze, and
    revocations stop propagating.

    Routes are served first because there are orders of magnitude fewer of them
    (one per public model, against one per API key), so in any realistic mix the
    keys absorb the variation. Both overflows are benign and identical in kind:
    a key past the budget authenticates at the server on every request, a route
    past it authorizes there per request.
    """
    routes = min(public_route_count, budget // MATCH_RULE_BYTES)
    entries = max(0, (budget - routes * MATCH_RULE_BYTES) // KEY_ENTRY_BYTES)
    return routes, entries


def gateway_digest_publishable(is_custom: bool, digest: Optional[str]) -> bool:
    """Whether this key's digest may be written into the gateway's ``keys``.

    Not the same question as :func:`security.secret_key_digest_eligible`, which
    decides whether a digest may be *computed* and answers once, when the key is
    created. This one is asked on every pass, and has to be, because the two can
    disagree: a custom key given a digest while
    ``GATEWAY_AUTH_ALLOW_CUSTOM_KEYS`` was on keeps that digest in the database
    after the switch is turned off. Deciding only at creation time would leave
    those keys published, so an operator who turns the switch off in response to
    a review would get no effect at all on the keys they turned it off for --
    every custom key that had ever been used.

    Leaving the stored digest alone is deliberate. It never leaves the database,
    where it sits beside the argon2 hash it is derived from the same secret as,
    and it is what lets the switch be turned back on without every custom key
    having to be re-authenticated to earn its digest again. What the switch
    governs is publication, so publication is where it is enforced.
    """
    if not digest:
        return False
    if is_custom:
        return envs.GATEWAY_AUTH_ALLOW_CUSTOM_KEYS
    return True


def gateway_ref_eligible(is_custom: bool) -> bool:
    """Whether ``refs`` is where this kind of key belongs.

    Only a custom key in a deployment that has turned
    ``GATEWAY_AUTH_ALLOW_CUSTOM_KEYS`` off. That is the one case where a key can
    never reach ``keys``: the switch is what refuses it a publishable digest,
    and nothing about the key itself will change to earn one.

    Everything else that lacks a publishable digest is *on its way* to one, and
    is deliberately in neither table until it arrives -- a generated key
    predating the digest column, and equally a custom key predating this feature
    or created while the switch was off. All three converge on their first
    successful authentication, which backfills the digest, and the CR write that
    adds them to ``keys`` is the same write that would have had to remove them
    from ``refs``. Listing them saves no write and leaves behind a shared-data
    entry the plugin can never read again or reclaim.

    Deliberately a function of the key *kind* and the deployment, not of the
    row's current digest: a stale digest left over from when the switch was on
    must not keep a key out of ``refs``, or turning the switch off would strand
    exactly the keys it was turned off for.
    """
    return is_custom and not envs.GATEWAY_AUTH_ALLOW_CUSTOM_KEYS


def gateway_ref_indexable(api_key: ApiKey, principal: Principal) -> bool:
    """Whether the gateway may hold a ``refs`` entry for this key.

    Also decides whether ``/token-auth`` hands the gateway an
    ``X-GPUStack-Key-Ref`` for it, and the two must agree. A ref the gateway
    cannot look up is worse than no ref at all: the plugin takes it, mints a
    marker naming ``ref:<id>``, overwrites the server's marker with it, and
    then refuses its own marker on the fallback pass because the id validates
    against nothing. The request falls back to forwarding a credential that
    ``ai-proxy`` has by then replaced with the cluster registration token, and
    the server resolves the caller as the SYSTEM principal -- wrong consumer,
    and policy evaluated for the wrong subject.

    Which keys qualify is :func:`gateway_ref_eligible`, shared with
    :func:`build_local_auth_tables` rather than restated. The two must agree for
    the reason above, and turning ``GATEWAY_AUTH_ALLOW_CUSTOM_KEYS`` off moves
    keys across that boundary at runtime -- an agreement held by a comment would
    not have survived it. What this adds is the state the endpoint has to check
    and the table build gets from its WHERE clause.
    """
    return (
        gateway_ref_eligible(api_key.is_custom)
        and api_key.deleted_at is None
        and principal.kind != PrincipalType.SYSTEM
        and principal.is_active
    )


async def build_local_auth_tables(
    session: AsyncSession, max_entries: Optional[int] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(keys, refs)`` -- every key the server would authenticate, split by
    whether the gateway can verify it on its own.

    Two predicates, and most keys satisfy neither for long.
    :func:`gateway_digest_publishable` puts a key in ``keys``, where the plugin
    checks the secret itself. :func:`gateway_ref_eligible` puts it in ``refs``,
    a validity index that cannot verify anything -- an id and an expiry, nothing
    derived from the secret -- and holds only for a custom key in a deployment
    that has turned ``GATEWAY_AUTH_ALLOW_CUSTOM_KEYS`` off. In the default
    configuration ``refs`` is therefore empty, and stays that way.

    A key that satisfies neither is in no table at all: it is awaiting the
    digest that will put it in ``keys``, whether it predates the digest column,
    predates custom-key support, or was created while the switch was off. That
    is deliberate and identical in all three cases -- the first successful
    authentication backfills the digest, and the CR write that adds the key to
    ``keys`` is the same write that would have had to remove it from ``refs``.
    See :func:`gateway_ref_indexable` for why the omission has to be mirrored by
    ``/token-auth``.

    Both predicates are evaluated here on every pass rather than trusted from
    key creation, because a stored digest outlives the switch that allowed it.

    Excluded, and each for its own reason:

    * **SYSTEM principals.** Not merely a CR-exposure concern. A cluster's
      registration token is built by the same generator as any other key, so
      it is digest-eligible and structurally indistinguishable -- the digest
      predicate cannot catch it. It is also exactly the credential ai-proxy
      puts into ``Authorization`` on a fallback trip, so leaving it in would
      have the plugin authenticate the provider credential as a SYSTEM
      identity and assert *that* to the server.
    * **deactivated or deleted principals.** ``authenticate_request`` rejects
      an inactive principal right after identifying the key, so those keys do
      not authenticate on the server either. Filtering them here makes
      deactivation show up as absence from the tables, which sends the request
      back to the server and reproduces today's behavior exactly -- including
      the ``'none'`` consumer on a PUBLIC route.
    * **soft-deleted keys.** ``ApiKey.delete()`` hard-deletes by default, but
      the column exists and a cascade takes the soft path.
    * **keys that have already expired.** Not for correctness -- ``exp`` rides
      along and both the plugin and the server reject them anyway -- but for
      the budget below. The cap drops rows in id order, i.e. it keeps the
      oldest, so a long-dead key would sit in the table while a newly created
      live one is the thing dropped.

    ``max_entries`` caps the combined size, in SQL as well as here, so the work
    a pass does is bounded by the cap rather than by the size of the table. A
    key past the cap still works, it just asks the server on every request.
    ``None`` means no cap; zero is a real cap and must stay distinguishable
    from it, because :func:`split_cr_budget` returns zero for a budget that the
    public routes have already spent -- read as "no cap" that would publish
    every key in the deployment, which is the exact opposite.
    """
    now = datetime.now(timezone.utc)
    # Columns rather than entities. This runs on a timer over every key in the
    # deployment and the hydration is synchronous, so building 10k ApiKey +
    # Principal instances would block the event loop for ~110 ms a pass against
    # ~2 ms of actual query -- serving tuples instead costs ~22 ms. It also
    # avoids fetching the argon2 hash, the description and the allowed-model
    # JSON, none of which is read here.
    statement = (
        select(
            ApiKey.id,
            ApiKey.access_key,
            ApiKey.secret_key_digest,
            ApiKey.expires_at,
            ApiKey.user_id,
            ApiKey.is_custom,
        )
        .join(Principal, Principal.id == ApiKey.user_id)
        .where(
            ApiKey.deleted_at.is_(None),
            or_(ApiKey.expires_at.is_(None), ApiKey.expires_at > now),
            # Belongs in one of the two tables. Stated here as well as in the
            # loop so ``limit`` counts only rows that will be used -- otherwise
            # keys awaiting a digest would spend the budget without appearing.
            #
            # The two arms are the two predicates below. With the switch on,
            # ``refs`` has no members and a stored digest is the whole test;
            # with it off, every custom key qualifies for ``refs`` whatever its
            # digest column says, which is what subsumes the first arm for them.
            (
                ApiKey.secret_key_digest.is_not(None)
                if envs.GATEWAY_AUTH_ALLOW_CUSTOM_KEYS
                else or_(
                    ApiKey.secret_key_digest.is_not(None),
                    ApiKey.is_custom.is_(True),
                )
            ),
            Principal.deleted_at.is_(None),
            Principal.kind != PrincipalType.SYSTEM,
            Principal.is_active.is_(True),
        )
        # Stable order, so the cap drops the same rows on every pass instead of
        # producing a different table each time and rewriting the CR forever.
        .order_by(ApiKey.id)
    )
    if max_entries is not None:
        # One past the cap: enough to know something was left out without
        # reading a table that may be many times the cap.
        statement = statement.limit(max_entries + 1)
    rows = (await session.exec(statement)).all()

    keys: Dict[str, Any] = {}
    refs: Dict[str, Any] = {}
    dropped = False
    for key_id, access_key, digest, expires_at, user_id, is_custom in rows:
        publishable = gateway_digest_publishable(is_custom, digest)
        if not publishable and not gateway_ref_eligible(is_custom):
            # A key on its way to a digest: neither table serves it until the
            # first authentication backfills one. The query says the same thing;
            # this is the belt to that brace, and what keeps the two halves of
            # the rule readable side by side. The other conditions of
            # ``gateway_ref_indexable`` are enforced by the WHERE above for
            # every row here.
            continue
        if max_entries is not None and len(keys) + len(refs) >= max_entries:
            dropped = True
            continue
        # Rows skipped below this point -- an unparseable digest, an empty
        # access key -- have already spent a slot of ``limit(max_entries + 1)``,
        # so the tables can come out under the cap with ``dropped`` still false.
        # Harmless: the budget is a ceiling, and both cases are rare enough that
        # tightening the accounting would cost more than it returns.
        # Still carried even though the query already excludes what has
        # expired: a key can expire between two passes, and this is what makes
        # that immediate instead of waiting one out. Unix seconds rather than
        # RFC 3339, so the plugin compares against proxy-wasm's integer clock
        # instead of parsing timestamps.
        entry: Dict[str, Any] = {}
        if expires_at is not None:
            entry["exp"] = int(expires_at.timestamp())
        if publishable:
            gateway_digest_value = gateway_digest(digest)
            if gateway_digest_value is None:
                # A stored value this build cannot parse says nothing about the
                # secret, so the key falls through to the server rather than
                # being published under a digest the plugin would reject.
                continue
            if not access_key:
                # The legacy cluster token's row has no access key, so there is
                # nothing to index it by. It belongs to a SYSTEM principal and
                # should already be gone above; this is the belt to that brace.
                # It also has to be a skip rather than an empty-string entry:
                # the plugin rejects an empty access_key outright, and that
                # error fails the *whole* config, taking every other key with
                # it.
                continue
            entry["digest"] = gateway_digest_value
            # Just the id: the plugin rebuilds "<access_key>.gpustack-<id>"
            # locally for PUBLIC routes, and a few bytes of integer replace a
            # ~40-byte string on the larger of the two tables.
            entry["user_id"] = user_id
            keys[access_key] = entry
        else:
            # A custom key in a deployment that publishes none of them, whatever
            # its digest column holds. Keyed by id and carrying only an expiry --
            # not by access key, which for a custom key is itself an unsalted
            # hash of the secret, and so is the thing the switch withholds.
            refs[str(key_id)] = entry

    if dropped:
        logger.warning(
            f"Gateway auth: the ext-auth config is at its {max_entries} key "
            "entry budget and further API keys are left out. They still "
            "authenticate, via the server on every request."
        )
    return keys, refs


async def build_public_route_ids(session: AsyncSession) -> List[int]:
    """Ids of every PUBLIC route, which is the whole input to the rules.

    Ids rather than names because the names are derived from them
    (``ai-route-route-<id>.internal``), which is why renaming a route, changing
    its weight or adding a target cannot move this value -- only joining or
    leaving the PUBLIC set can. That is what keeps the rules stable across the
    high-frequency churn ``ModelRoute`` rows see.

    Each id here is a standing authorization to allow requests through without
    asking the server, which is why *losing* PUBLIC status has to propagate
    immediately while gaining it can wait.
    """
    statement = (
        # Only the id is read; the names are derived from it.
        select(ModelRoute.id)
        .where(
            ModelRoute.deleted_at.is_(None),
            ModelRoute.access_policy == AccessPolicyEnum.PUBLIC,
        )
        .order_by(ModelRoute.id)
    )
    return [row for row in (await session.exec(statement)).all() if row is not None]


def public_route_ingresses(route_ids: List[int], cfg: Config) -> List[List[str]]:
    """Ingress names for those routes, main and fallback together.

    Both names, always: the fallback trip re-runs the filter chain under the
    fallback route's name, and listing only the main one would drop it to the
    catch-all rule at the exact moment its credential is gone.
    """
    return [
        list(
            route_ingress_names_for_plugins(
                model_route_id=route_id,
                resource_namespace=cfg.get_namespace(),
                gateway_namespace=cfg.gateway_namespace,
            )
        )
        for route_id in route_ids
    ]


def _is_public_policy(policy: Any) -> bool:
    """Whether an access policy read off an event payload means PUBLIC.

    The value arrives in three shapes depending on how the event travelled:
    the enum member on a hydrated model, its value (``"public"``) once it has
    been through JSON, and its name (``"PUBLIC"``) as the column stores it.
    """
    if isinstance(policy, AccessPolicyEnum):
        return policy is AccessPolicyEnum.PUBLIC
    return policy in (AccessPolicyEnum.PUBLIC.value, AccessPolicyEnum.PUBLIC.name)


class GatewayAuthReconciler:
    """Writes the key tables and PUBLIC rules into the ext-auth CR."""

    def __init__(self, cfg: Config):
        self._config = cfg
        self._disabled = cfg.gateway_mode == GatewayModeEnum.disabled
        self._interval = envs.GATEWAY_AUTH_RECONCILE_INTERVAL_SECONDS
        self._budget = envs.GATEWAY_AUTH_MAX_CR_BYTES
        self._flush_now = asyncio.Event()
        self._extensions_api: Optional[ExtensionsHigressIoV1Api] = None
        self._registry: Optional[McpBridgeRegistry] = None
        # PUBLIC route ids as of the last successfully applied CR -- not "as of
        # the last database read". The invariant that buys is
        #
        #     the CR carries a rule for a route  <=>  its id is in this set
        #
        # which is what lets ``_model_route_may_change_rules`` rule an event
        # out. None means nothing has been applied yet, so nothing can be ruled
        # out.
        self._applied_public_route_ids: Optional[Set[int]] = None
        # The same thing the tuple above records, kept whole so a pass can tell
        # whether it changed anything. Compared rather than hashed: the tables
        # are already in hand, equality short-circuits on the first difference,
        # and serializing them again to fingerprint them would cost more than
        # the comparison saves.
        self._applied_state: Optional[
            Tuple[Dict[str, Any], Dict[str, Any], List[int]]
        ] = None

    async def start(self):
        if self._disabled:
            return
        self._extensions_api = ExtensionsHigressIoV1Api(
            k8s_client.ApiClient(configuration=get_async_k8s_config(cfg=self._config))
        )
        # Derived from static config, but resolved here rather than in
        # __init__: with the gateway disabled it has no address to build from
        # and raises, while the reconciler is constructed either way.
        self._registry = get_gpustack_higress_registry(cfg=self._config)
        # Reconcile once up front instead of waiting out an interval: a server
        # that just restarted may be the one that missed the deletion.
        self._flush_now.set()
        await asyncio.gather(
            self._watch(ApiKey, "api_key"),
            # Principals are an input too, not just an owner: deactivating one,
            # or deleting it outside the ORM's cascade, takes its keys out of
            # the tables. Without this watch that tightening would wait out the
            # periodic pass, which is the one direction that must not wait.
            self._watch(Principal, "principal"),
            self._watch(
                ModelRoute, "model_route", relevant=self._model_route_may_change_rules
            ),
            self._flush_loop(),
        )

    def _model_route_may_change_rules(self, event) -> bool:
        """Whether this route event could change the rules array.

        ``ModelRoute`` rows are written constantly -- ``targets`` and
        ``ready_targets`` are restamped on every target state transition, so a
        cluster with instances coming and going produces a steady stream of
        updates. Almost none of them can affect this CR: the rules are derived
        from ``(id, access_policy)`` alone, so a rename, a weight change or a
        new target renders identically and the write is diffed away. Acting on
        those events costs a pointless pass -- two queries and a call to the
        API server -- which is the thing worth avoiding.

        Two cases can matter, and between them they are exhaustive:

        * the route **is** PUBLIC now, so a rule may need adding;
        * its id is in the applied set, so the CR has a rule for it and this
          event may be taking it away. This is the one that catches PUBLIC ->
          AUTHED, where the payload already reads as AUTHED.

        Anything else is a route that has no rule and would not get one, and
        nothing about it can move the array.

        The rest is deliberately conservative: nothing applied yet, no id, or a
        policy the payload does not carry (distributed mode delivers id-only
        events) all mean "do the work". Being wrong in that direction costs one
        redundant pass; being wrong the other way would leave a route allowed
        locally after it stopped being public, until the periodic pass caught
        it.
        """
        applied = self._applied_public_route_ids
        if applied is None:
            return True
        route_id = event_field(event.data, "id")
        if route_id is None:
            return True
        if route_id in applied:
            return True
        policy = event_field(event.data, "access_policy")
        if policy is None:
            return True
        return _is_public_policy(policy)

    async def _watch(self, resource, label: str, relevant=None):
        """Turn events into flush urgency -- never into incremental state.

        Only the direction matters here. A deletion or an update may take
        something away (revoked key, a deactivated principal, a route leaving
        PUBLIC), so it flushes immediately. A creation can only add, and adding
        late is harmless: until the push lands the key authenticates via the
        server and the route authorizes per request, exactly as before.

        ``relevant`` narrows that further for a resource whose rows change far
        more often than the config derived from them; see
        :meth:`_model_route_may_change_rules`. It is only ever allowed to
        suppress work the periodic pass would redo anyway, so a bug in it costs
        latency rather than correctness -- and a predicate that raises is
        treated as "relevant" for the same reason.

        The digest backfill deliberately does not show up as an event -- it is
        a bulk UPDATE that bypasses the ORM -- so a mass backfill converges on
        the periodic pass as one CR write rather than thousands.
        """
        urgent = {EventType.UPDATED, EventType.DELETED, EventType.UNKNOWN}
        backoff = _WATCH_RETRY_MIN_SECONDS
        while True:
            try:
                # No initial snapshot: it would push every existing row through
                # the bus for nothing, since the first pass reads the database
                # in full anyway.
                async for event in resource.subscribe(
                    source=f"gateway_auth_reconciler.{label}", replay_existing=False
                ):
                    backoff = _WATCH_RETRY_MIN_SECONDS
                    if event.type not in urgent:
                        continue
                    try:
                        if relevant is not None and not relevant(event):
                            continue
                    except Exception:
                        logger.exception(
                            f"Gateway auth {label} filter failed; reconciling anyway"
                        )
                    self._flush_now.set()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception(
                    f"Gateway auth {label} watch failed, retrying in {backoff}s: {e}"
                )
            # Resubscribed rather than left dead. Events are what makes a
            # tightening -- a revocation, a deactivated principal, a route
            # leaving PUBLIC -- take effect on the next request instead of
            # waiting out the periodic pass; a watch that exits silently
            # downgrades every one of those to that interval, and stays
            # downgraded for the life of the process.
            #
            # Backed off because the plausible causes are not transient: a bus
            # that is gone stays gone, and a tight loop would bury the log line
            # that says so. The first reconnect still happens promptly.
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _WATCH_RETRY_MAX_SECONDS)
            # Anything that changed while the watch was down is invisible to it,
            # so make the pass that follows an unconditional one.
            self._flush_now.set()

    async def _flush_loop(self):
        while True:
            try:
                await asyncio.wait_for(self._flush_now.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                pass
            # Cleared before the work, not after: anything that arrives while
            # this pass runs sets it again and gets its own pass, instead of
            # being swallowed by a pass that had already read the database.
            self._flush_now.clear()
            try:
                await self.reconcile()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.exception(f"Failed to reconcile gateway auth config: {e}")

    async def reconcile(self):
        async with async_session() as session:
            # Routes first: they decide how much of the shared budget is left
            # for keys.
            public_route_ids = await build_public_route_ids(session)
            route_count, max_entries = split_cr_budget(
                self._budget, len(public_route_ids)
            )
            if route_count < len(public_route_ids):
                logger.warning(
                    f"Gateway auth: {len(public_route_ids) - route_count} public "
                    f"routes left out of the ext-auth config ({self._budget} byte "
                    "budget). They keep authorizing via the server per request."
                )
                public_route_ids = public_route_ids[:route_count]
            keys, refs = await build_local_auth_tables(session, max_entries=max_entries)
        ingresses = public_route_ingresses(public_route_ids, self._config)

        # ``ensure_wasm_plugin`` compares the rendered spec and skips the write
        # when nothing moved, so an unchanged recomputation costs no CR write,
        # no resourceVersion bump and no xDS push. That is what makes a
        # seconds-level full recompute affordable.
        await ensure_wasm_plugin(
            api=self._extensions_api,
            name=ext_auth_resource_name,
            namespace=self._config.gateway_namespace,
            spec_diff=partial(
                ext_auth_reconcile_spec_diff,
                keys=keys,
                refs=refs,
                public_route_ingresses=ingresses,
                cfg=self._config,
                # Only used to rebuild a CR that has gone missing, but it has
                # to be passed in: this module can import the gateway package,
                # ext_auth cannot import back into it.
                registry=self._registry,
            ),
        )
        # After the apply, never before: the event filter reads this as "what
        # the CR holds". A failed write leaves the old rules in place, and
        # recording the new set anyway would make the filter discard the very
        # events that would have retried it.
        #
        # Logged only when it moved. The pass runs every interval whether or not
        # anything changed, so an unconditional line is a heartbeat that says
        # nothing about the config -- and it cannot even be read as one, since
        # it looks identical whether the CR was rewritten or diffed away. This
        # way its presence *is* the signal. Liveness, when it is the question,
        # comes from the exception logs in the flush loop and the watches.
        if (keys, refs, public_route_ids) != self._applied_state:
            logger.debug(
                f"Gateway auth: {len(keys)} locally verifiable keys, "
                f"{len(refs)} refs, {len(public_route_ids)} public routes."
            )
        self._applied_state = (keys, refs, public_route_ids)
        self._applied_public_route_ids = set(public_route_ids)
