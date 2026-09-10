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
its urgency. A revocation, a deactivated principal, a key that stops being
``unrestricted``, a route leaving the set of policies the gateway may act on:
those are all "someone should stop being let in", and they go out immediately.
A new digest or a new rule is monotonic and idempotent -- the key already works
through the fallback path, the route already authorizes per request -- so those
ride the next tick, which is what keeps a mass digest backfill from turning
into thousands of CR writes.

Expiry needs neither: ``exp`` travels in the entry itself and the plugin
compares it against its own clock, so a key expiring changes nothing here. It
is only dropped from the tables on the next pass, to stop dead rows from
spending the entry budget.
"""

import asyncio
import logging
from datetime import datetime, timezone
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

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
from gpustack.schemas.api_keys import ApiKey, PermissionScope
from gpustack.schemas.config import GatewayModeEnum
from gpustack.schemas.model_routes import AccessPolicyEnum, ModelRoute
from gpustack.schemas.principals import Principal, PrincipalType
from gpustack.security import gateway_digest
from gpustack.server.bus import EventType, event_field
from gpustack.server.db import async_session

logger = logging.getLogger(__name__)


# Rendered sizes, measured against a real CR rather than estimated: a key entry
# with a truncated digest (``"<16 hex ak>":{"digest":"s128$...","user_id":N}``)
# came out at 112 bytes, ``"unrestricted":true`` adds about 20 more, and one
# match rule (two ingress names plus the access policy) at 169. All are rounded
# up here, and a refs entry is far smaller than a key entry, so sizing every
# entry as a key entry that carries the flag errs toward leaving room -- which
# is the direction that matters, since the cost of underestimating is a refused
# write rather than a smaller table. The two policy values render to the same
# width, so one rule size covers both.
KEY_ENTRY_BYTES = 135
MATCH_RULE_BYTES = 170

# The route policies whose authorization verdict the gateway can reproduce from
# what it already holds, and so the only values that may reach a match rule.
# PUBLIC first: the order is the order the budget serves them in, and the two
# are not worth the same -- see :func:`split_cr_budget`.
#
# ``ALLOWED_PRINCIPALS`` is deliberately absent. Its verdict turns on
# per-principal grants that are published nowhere near the edge, so it is
# authorization proper and stays on the server.
SKIPPABLE_ROUTE_POLICIES = (AccessPolicyEnum.PUBLIC, AccessPolicyEnum.AUTHED)

# The same order, as the wire values a rule carries, for sorting by.
_POLICY_RANK = {
    policy.value: rank for rank, policy in enumerate(SKIPPABLE_ROUTE_POLICIES)
}

# Bounds on how fast a dropped event watch reconnects.
_WATCH_RETRY_MIN_SECONDS = 1
_WATCH_RETRY_MAX_SECONDS = 30


def split_cr_budget(budget: int, public_route_count: int) -> Tuple[int, int]:
    """``(PUBLIC rules to keep, key entries that then fit)`` -- the first two
    of the three claims on one budget.

    One budget rather than independent caps. The key tables and the match rules
    share a single CR and therefore a single etcd object limit; capping them
    separately means the sum can still overrun it, and overrunning it is not a
    partial failure -- the write is refused, the tables freeze, and revocations
    stop propagating.

    The order is **PUBLIC rules, then key entries, then AUTHED rules** (the
    third served by :func:`authed_rules_budget` from whatever the first two
    leave). Serving *all* rules first, as this did when PUBLIC was the only
    policy that got one, rested on there being orders of magnitude fewer routes
    than keys. AUTHED is the default policy, so that premise is gone: at ~170
    bytes a rule against the default 1.1 MB budget, 2000 routes cut the key
    table by about 41% and ~6500 reduce it to nothing.

    What settles the order is that the two overflows are not equivalent:

    * a **route** past the budget loses only its authorization skip -- its
      callers still authenticate locally and the server evaluates policy per
      request, exactly as it does today;
    * a **key** past the budget loses local authentication too, so every one of
      its requests carries a credential to the server.

    PUBLIC is the exception that goes first, because a public route with no
    rule needs a live server even for anonymous traffic -- nothing else in the
    chain can name that caller.
    """
    public_rules = min(public_route_count, budget // MATCH_RULE_BYTES)
    entries = max(0, (budget - public_rules * MATCH_RULE_BYTES) // KEY_ENTRY_BYTES)
    return public_rules, entries


def authed_rules_budget(budget: int, public_rules: int, key_entries: int) -> int:
    """How many AUTHED rules fit in what the first two claims left.

    Takes the key entries actually published rather than the cap
    :func:`split_cr_budget` handed out, so a deployment with fewer keys than
    the cap spends the difference on rules instead of reserving it for keys
    that do not exist. That is the whole reason this is a second call and not
    a third return value: the real count is only known once the tables are
    built.
    """
    spent = public_rules * MATCH_RULE_BYTES + key_entries * KEY_ENTRY_BYTES
    return max(0, (budget - spent) // MATCH_RULE_BYTES)


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


def gateway_key_unrestricted(
    scope: Optional[List[Any]],
    allowed_model_names: Optional[List[str]],
    principal_kind: Any,
) -> bool:
    """Whether this key adds nothing to what its user may already reach.

    What the plugin does with it: on an ``authed`` route it skips the
    authorization call for a caller whose entry carries this flag. So the flag
    has to stand in for everything ``/token-auth`` would have evaluated after
    authentication, which on a non-PUBLIC route is exactly two things
    (``routes/token.py``)::

        inference_scope(request, user)
        model_allowed_for_user(model_name, user.id, api_key)

    The first is the scope test below, transcribed from ``api.auth``. The
    second is ``model_name in accessible_model_names(user) ∩
    allowed_model_names(key)``, whose second term drops out entirely when the
    key names no models -- which is the other half of this predicate.

    Its *first* term is not checked here because on an AUTHED route it is
    constant-true. ``non_admin_user_models`` (``schemas/stmt.py``) cross-joins
    every live non-admin USER-principal with every route whose policy is
    ``PUBLIC`` or ``AUTHED``, and ``get_user_accessible_model_names`` hands an
    admin every route unconditionally. A rule is only emitted for a route with
    one of those two policies, so for any caller the tables can name, the
    accessible set contains it.

    ``principal_kind`` is what makes that last sentence true rather than nearly
    true. Both branches above are about a USER: the view filters
    ``u.kind = 'USER'``, and a non-admin principal of any other kind falls
    through it to an *empty* accessible set -- the server would refuse, while a
    gateway acting on this flag would allow. Nothing in the product gives an
    ORG- or GROUP-principal an API key today, and the table build already
    excludes SYSTEM; naming the condition is what keeps that from being load
    bearing.

    The encoding is positive on purpose, and that direction is not negotiable:
    absent means "ask the server", so a gateway config written before this
    field existed, or one this server has withdrawn the flag from, costs a
    round trip rather than a wrong verdict. Withdrawing it is a tightening in
    the same class as a revocation -- on a skipped route nothing else asks the
    server whether the key still qualifies -- so it has to ride the same
    immediate flush, which is what the ``ApiKey`` watch already gives it.
    """
    if principal_kind != PrincipalType.USER:
        return False
    scopes = scope or []
    if PermissionScope.ALL not in scopes and PermissionScope.INFERENCE not in scopes:
        return False
    return not allowed_model_names


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

    An entry in either table may also carry ``unrestricted``, which is what
    lets the plugin skip the authorization call on an AUTHED route. It rides
    the entry rather than a table of its own precisely so that withdrawing it
    is the same write, on the same immediate flush, as withdrawing the entry --
    see :func:`gateway_key_unrestricted`.

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
    match rules have already spent -- read as "no cap" that would publish
    every key in the deployment, which is the exact opposite.
    """
    now = datetime.now(timezone.utc)
    # Columns rather than entities. This runs on a timer over every key in the
    # deployment and the hydration is synchronous, so building 10k ApiKey +
    # Principal instances would block the event loop for ~110 ms a pass against
    # ~2 ms of actual query -- serving tuples instead costs ~22 ms. It also
    # avoids fetching the argon2 hash and the description, neither of which is
    # read here.
    #
    # ``scope`` and ``allowed_model_names`` are the two JSON columns
    # :func:`gateway_key_unrestricted` folds into one bit; only their emptiness
    # is read, never their contents. ``Principal.kind`` is selected as well as
    # filtered on because that predicate needs the value, not just the
    # exclusion the WHERE below already applies.
    statement = (
        select(
            ApiKey.id,
            ApiKey.access_key,
            ApiKey.secret_key_digest,
            ApiKey.expires_at,
            ApiKey.user_id,
            ApiKey.is_custom,
            ApiKey.scope,
            ApiKey.allowed_model_names,
            Principal.kind,
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
    for (
        key_id,
        access_key,
        digest,
        expires_at,
        user_id,
        is_custom,
        scope,
        allowed_model_names,
        principal_kind,
    ) in rows:
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
        # Emitted only when true, never as ``false``. Absent already means
        # false to the plugin, so writing the negative case would spend ~20
        # bytes of the shared budget per entry to say what silence says.
        if gateway_key_unrestricted(scope, allowed_model_names, principal_kind):
            entry["unrestricted"] = True
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
            # locally on any route it skips, and a few bytes of integer replace
            # a ~40-byte string on the larger of the two tables.
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


async def build_skippable_routes(session: AsyncSession) -> List[Tuple[int, str]]:
    """``(id, policy)`` for every route the gateway may act on by itself.

    Ids rather than names because the names are derived from them
    (``ai-route-route-<id>.internal``), which is why renaming a route, changing
    its weight or adding a target cannot move this value -- only its policy
    crossing into or out of :data:`SKIPPABLE_ROUTE_POLICIES` can. That is what
    keeps the rules stable across the high-frequency churn ``ModelRoute`` rows
    see.

    The policy travels with the id because the rule carries it verbatim: the
    two are different standing authorizations, and the plugin decides what it
    may skip from the value, not from the rule's presence. Each entry is one
    such authorization, which is why *losing* a skippable policy has to
    propagate immediately while gaining one can wait.

    PUBLIC routes are ordered ahead of AUTHED ones, then by id within each,
    because the budget serves the two groups on either side of the key table --
    see :func:`split_cr_budget`. Keeping them in one ordered list means the
    caller splits rather than re-sorts.
    """
    statement = (
        # Only the id and the policy are read; the names are derived from the
        # id.
        select(ModelRoute.id, ModelRoute.access_policy)
        .where(
            ModelRoute.deleted_at.is_(None),
            ModelRoute.access_policy.in_(SKIPPABLE_ROUTE_POLICIES),
        )
        .order_by(ModelRoute.id)
    )
    rows = (await session.exec(statement)).all()
    # Through the same normaliser the event filter uses, rather than
    # ``AccessPolicyEnum(policy)``. A typed column select hands back the enum
    # member on every dialect we support, but stating the mapping twice is what
    # lets the two drift, and this form also survives a driver that returns the
    # stored name. A value it cannot name is dropped rather than raised on: a
    # route the gateway cannot place is one it should authorize at the server,
    # which is what no rule already means.
    routes = [
        (route_id, rule_value)
        for route_id, rule_value in (
            (route_id, _policy_rule_value(policy)) for route_id, policy in rows
        )
        if route_id is not None and rule_value is not None
    ]
    # Stable, so ids stay ascending inside each policy group and the rules
    # array is byte-identical between two passes that read the same rows.
    routes.sort(key=lambda route: _POLICY_RANK[route[1]])
    return routes


def route_rule_ingresses(
    routes: List[Tuple[int, str]], cfg: Config
) -> List[Tuple[List[str], str]]:
    """``(ingress names, policy)`` per route, main and fallback together.

    Both names, always: the fallback trip re-runs the filter chain under the
    fallback route's name, and listing only the main one would drop it to the
    catch-all rule at the exact moment its credential is gone.
    """
    return [
        (
            list(
                route_ingress_names_for_plugins(
                    model_route_id=route_id,
                    resource_namespace=cfg.get_namespace(),
                    gateway_namespace=cfg.gateway_namespace,
                )
            ),
            policy,
        )
        for route_id, policy in routes
    ]


def _policy_rule_value(policy: Any) -> Optional[str]:
    """The rule value an access policy off an event payload would render as.

    ``None`` for a policy that gets no rule, which includes one this build does
    not recognise -- the same rollback-safe direction the plugin takes for a
    value it cannot name.

    The value arrives in three shapes depending on how the event travelled:
    the enum member on a hydrated model, its value (``"public"``) once it has
    been through JSON, and its name (``"PUBLIC"``) as the column stores it.
    """
    for skippable in SKIPPABLE_ROUTE_POLICIES:
        if policy is skippable or policy in (skippable.value, skippable.name):
            return skippable.value
    return None


class GatewayAuthReconciler:
    """Writes the key tables and the per-route rules into the ext-auth CR."""

    def __init__(self, cfg: Config):
        self._config = cfg
        self._disabled = cfg.gateway_mode == GatewayModeEnum.disabled
        self._interval = envs.GATEWAY_AUTH_RECONCILE_INTERVAL_SECONDS
        self._budget = envs.GATEWAY_AUTH_MAX_CR_BYTES
        self._flush_now = asyncio.Event()
        self._extensions_api: Optional[ExtensionsHigressIoV1Api] = None
        self._registry: Optional[McpBridgeRegistry] = None
        # The policy each skippable route had as of the last successfully
        # applied CR -- not "as of the last database read". The invariant that
        # buys is
        #
        #     the last pass saw route R with policy P  <=>  this maps R to P
        #
        # which is what lets ``_model_route_may_change_rules`` rule an event
        # out. None means nothing has been applied yet, so nothing can be ruled
        # out.
        #
        # The policy is carried, not just the id, and with AUTHED in the
        # picture it has to be. AUTHED is the default policy, so nearly every
        # route now has a rule -- an id-keyed set would match every route event
        # and the filter would suppress nothing, turning the steady restamping
        # of ``targets`` / ``ready_targets`` into a reconcile pass apiece.
        # Comparing the policy is what keeps "this route already renders like
        # this" separable from "this route's rule is about to change".
        #
        # Every skippable route is recorded, including the ones the byte budget
        # left out. What the filter asks is whether an event changes the
        # *input*, and a route dropped for want of budget has the same input as
        # one that fits: re-running the pass would drop it again. Recording only
        # what was written would leave those routes matching nothing, so each of
        # their churn events would reconcile -- the suppression would fail
        # exactly where the deployment is already large enough to need it.
        self._applied_route_policies: Optional[Dict[int, str]] = None
        # The same thing the tuple above records, kept whole so a pass can tell
        # whether it changed anything. Compared rather than hashed: the tables
        # are already in hand, equality short-circuits on the first difference,
        # and serializing them again to fingerprint them would cost more than
        # the comparison saves.
        self._applied_state: Optional[
            Tuple[Dict[str, Any], Dict[str, Any], List[Tuple[int, str]]]
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

        The whole test is therefore "would this route's policy still render the
        way the last pass saw it", and it is one comparison because both sides
        reduce to the same thing: the rule value the policy produces, with
        ``None`` for a policy that gets no rule. Equal means nothing can move --
        the churn case, and by far the common one. Unequal covers all four ways
        it can: a route gaining a rule, losing one, or crossing between PUBLIC
        and AUTHED in either direction. That last pair is the reason the applied
        state carries the policy and not just the id.

        A deletion never reaches that comparison. The event carries the row as
        it was, so its policy still matches what the last pass recorded and the
        comparison would read the removal as "nothing moved" -- which is the one
        direction this filter must never be wrong in, since the rule outlives
        the route it names until the next periodic pass.

        The rest is deliberately conservative: nothing applied yet, no id, or a
        policy the payload does not carry (distributed mode delivers id-only
        events) all mean "do the work". Being wrong in that direction costs one
        redundant pass; being wrong the other way would leave a route skipped
        locally after it stopped qualifying, until the periodic pass caught it.
        """
        if event.type is EventType.DELETED:
            return True
        applied = self._applied_route_policies
        if applied is None:
            return True
        route_id = event_field(event.data, "id")
        if route_id is None:
            return True
        policy = event_field(event.data, "access_policy")
        if policy is None:
            return True
        return _policy_rule_value(policy) != applied.get(route_id)

    async def _watch(self, resource, label: str, relevant=None):
        """Turn events into flush urgency -- never into incremental state.

        Only the direction matters here. A deletion or an update may take
        something away (revoked key, a deactivated principal, a key narrowed
        to a model list, a route leaving the skippable policies), so it flushes
        immediately. A creation can only add, and adding late is harmless:
        until the push lands the key authenticates via the server and the route
        authorizes per request, exactly as before.

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
            # leaving the skippable policies -- take effect on the next
            # request instead of
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
            # Three claims on one budget, served in the order
            # :func:`split_cr_budget` explains: PUBLIC rules, key entries, then
            # AUTHED rules from whatever is left. The query returns them
            # PUBLIC-first, so the split is a partition, not a re-sort.
            skippable = await build_skippable_routes(session)
            public_routes = [
                r for r in skippable if r[1] == AccessPolicyEnum.PUBLIC.value
            ]
            authed_routes = skippable[len(public_routes) :]
            public_kept, max_entries = split_cr_budget(self._budget, len(public_routes))
            keys, refs = await build_local_auth_tables(session, max_entries=max_entries)
        # Against the entries actually published, not the cap: a deployment
        # with fewer keys than the cap spends the remainder on AUTHED rules
        # rather than reserving it for keys that do not exist.
        authed_kept = authed_rules_budget(
            self._budget, public_kept, len(keys) + len(refs)
        )
        dropped = (len(public_routes) - public_kept) + (
            len(authed_routes) - min(authed_kept, len(authed_routes))
        )
        if dropped:
            logger.warning(
                f"Gateway auth: {dropped} routes left out of the ext-auth config "
                f"({self._budget} byte budget). They keep authorizing via the "
                "server per request."
            )
        routes = public_routes[:public_kept] + authed_routes[:authed_kept]
        rule_ingresses = route_rule_ingresses(routes, self._config)

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
                route_rules=rule_ingresses,
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
        if (keys, refs, routes) != self._applied_state:
            public = sum(
                1 for _, policy in routes if policy == AccessPolicyEnum.PUBLIC.value
            )
            logger.debug(
                f"Gateway auth: {len(keys)} locally verifiable keys, "
                f"{len(refs)} refs, {len(routes)} skippable routes "
                f"({public} public)."
            )
        self._applied_state = (keys, refs, routes)
        # Every skippable route, not just the ones that fit -- see the field's
        # own comment for why the budget must not narrow this.
        self._applied_route_policies = dict(skippable)
