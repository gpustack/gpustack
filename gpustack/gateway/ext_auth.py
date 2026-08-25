"""The ext-auth WasmPlugin CR, in the shape that lets the gateway authenticate.

Authentication used to happen entirely on the server: every inference request
made a synchronous ``/token-auth`` call that both identified the caller and
decided whether they were allowed through. That is why a server restart took
inference down with it. ``gpustack-ext-auth`` splits the two --

* **authentication** is a pure function of the credential, so it moves here:
  the plugin verifies a generated key's ``secret_key_digest`` locally in a few
  microseconds and never calls out;
* **authorization** spans key scope, per-user RBAC and the route's access
  policy, none of which have a single invalidation signal, so it stays on the
  server and the plugin only ever caches its *output*.

This module owns the CR's shape. What goes *in* the key tables is decided by
:mod:`gpustack.server.gateway_auth_reconciler`, which recomputes them from the
database; the two halves meet at :func:`ext_auth_spec`.

The CR is a **hybrid** resource: the static base (endpoint, timeouts, signing
key) comes from ``cfg`` and is rewritten at every startup, while ``local_auth``
and the PUBLIC route rules come from the database and must survive that
rewrite -- see :func:`ext_auth_init_spec_diff`.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from gpustack import envs
from gpustack.api.auth import GATEWAY_AUTH_TOKEN_HEADER
from gpustack.config.config import Config
from gpustack.gateway.client import (
    McpBridgeRegistry,
    WasmPluginMatchRule,
    WasmPluginSpec,
)
from gpustack.gateway.plugins import plugin_entry, plugin_spec_overrides
from gpustack.gateway.utils import model_route_ingress_prefix
from gpustack.security import AUTH_CACHE_HEADER

logger = logging.getLogger(__name__)

# K8s resource name. Deliberately the same slot the upstream ext-auth plugin
# occupied, so switching is an in-place URL + config swap on one CR. Creating a
# second CR and deleting the old one would leave a window where a request is
# authorized twice, or none at all.
ext_auth_resource_name = "gpustack-llm-ext-auth"

ext_auth_plugin_name = "gpustack-ext-auth"

# Access-policy value the plugin understands. Lower-case on purpose: it mirrors
# ``AccessPolicyEnum.PUBLIC``'s wire value, which is what the reconciler
# compares against.
ACCESS_POLICY_PUBLIC = "public"


class LocalAuthOverride(BaseModel):
    """Operator-settable part of ``local_auth``.

    ``keys`` and ``refs`` are absent on purpose and must stay absent: they are
    the authentication table, so a config file able to write them would be a
    config file able to mint credentials. Leaving them undeclared makes that a
    startup error rather than a rule somebody has to remember.
    """

    model_config = ConfigDict(extra="forbid")

    # Kill switch. Off, the plugin forwards every credential to the server and
    # the whole thing degrades to the behavior before authentication moved to
    # the gateway -- an incident lever, not a tuning knob.
    enabled: bool = True


class AuthzOverride(BaseModel):
    """Operator-settable part of ``authz``.

    ``endpoint`` is not settable: it points at gpustack's own ``/token-auth``
    and is derived from the service registry. ``authorization_request`` is not
    settable either -- the gateway token it carries is what the server checks
    before honouring an asserted identity.
    """

    model_config = ConfigDict(extra="forbid")

    # 30s is empirical, and low values have been tried and reverted: this began
    # as a hardcoded 1s, which produced enough reports to be made configurable
    # (af1fe1b9), and 3s was no better. The call looks like it should take
    # milliseconds -- indexed lookups, a cached key read -- but the number that
    # matters is how long a saturated Python server takes to get round to
    # answering, and under load that routinely exceeds seconds.
    #
    # Lowering it to shorten the stall during a server restart is the obvious
    # trade to reach for and is not worth making. A server that is down refuses
    # the connection, which Envoy turns into a 503 straight away, so the failure
    # mode engages without waiting for this at all; the timeout only governs the
    # narrower case of a server that accepts the connection and is too busy to
    # answer -- and there, waiting is the correct behaviour, because the answer
    # is coming. Cutting it would convert a load spike into "authorization
    # skipped for everyone the gateway can name", which is a security
    # degradation triggered by traffic rather than by an outage.
    #
    # Defaults to the deprecated GPUSTACK_HIGRESS_EXT_AUTH_TIMEOUT_MS when that
    # is set, so an upgrade does not silently revert a value someone chose.
    timeout: int = Field(default_factory=envs.deprecated_ext_auth_timeout_ms, ge=1)


class ExtAuthOverride(BaseModel):
    """``gateway_plugin["gpustack-ext-auth"].config``.

    Shaped like the CR's ``defaultConfig`` so field paths transcribe straight
    out of the plugin's README, and validated field by field so the shape does
    not turn into a general-purpose way to write that CR. Everything left
    undeclared is refused, which is what keeps these out of reach:

    * ``access_policy`` -- declaring it here makes *every* route public; it is
      only meaningful on a match rule, one route at a time;
    * ``route_match_regexes`` -- the gate deciding which routes this plugin
      touches at all, so widening it reaches other tenants on a shared gateway;
    * ``local_auth.keys`` / ``refs`` -- the authentication table itself;
    * ``auth_cache`` -- the marker signing key is derived from
      ``jwt_secret_key``, and the header name is a contract with the server;
    * ``upstream_request.headers_to_remove`` -- dropping ``cookie`` or
      ``x-api-key`` from it sends client credentials on to the model.

    ``failure_mode_allow`` is also deliberately absent. It is a real upstream
    knob, but it admits requests carrying no credential at all, so on an
    internet-facing gateway it turns a server outage into an open inference
    proxy; ``failure_mode_allow_authenticated`` covers the actual need.
    """

    model_config = ConfigDict(extra="forbid")

    # ``default_factory`` rather than an instance: a nested model written as
    # ``authz: AuthzOverride = AuthzOverride()`` is constructed once, at import,
    # which freezes any default that is itself read at construction time -- here
    # the deprecated timeout variable.
    local_auth: LocalAuthOverride = Field(default_factory=LocalAuthOverride)
    authz: AuthzOverride = Field(default_factory=AuthzOverride)
    # Let a caller the gateway authenticated itself through while the server is
    # unreachable. See the comment where it is rendered.
    #
    # The plugin ships this off, on the grounds that a plugin should default to
    # fail closed. GPUStack overrides it deliberately: the gateway is here to
    # keep inference serving through a server restart, and defaulting to closed
    # would leave that undelivered until an operator found the switch. Changing
    # the plugin's own default would not change what deployments get -- this
    # value is written into the CR unconditionally.
    failure_mode_allow_authenticated: bool = True
    # What the client sees when the authorization service itself fails. Bounded
    # to the 4xx/5xx range so a typo cannot render as a success; 401 is a poor
    # choice in particular, since SDKs read it as "refresh the token and retry"
    # and would hammer a server that is already failing.
    status_on_error: int = Field(default=403, ge=400, le=599)


def ext_auth_override(cfg: Config) -> ExtAuthOverride:
    """The operator's settings for this plugin, or the defaults.

    Validated here rather than in ``config.py`` so the schema lives with the
    code that renders it: adding a knob is a change to this module alone. A
    bad value fails gateway initialization with the offending path named,
    which is a better outcome than a plugin quietly running on defaults
    nobody chose.
    """
    entry = plugin_entry(ext_auth_plugin_name, cfg)
    if entry is None or not entry.config:
        return ExtAuthOverride()
    try:
        return ExtAuthOverride.model_validate(entry.config)
    except ValidationError as e:
        raise ValueError(
            f"Invalid gateway_plugin.{ext_auth_plugin_name}.config: {e}"
        ) from e


def ext_auth_module_source(cfg: Config) -> Dict[str, Any]:
    """Where Envoy pulls the module from, as ``WasmPluginSpec`` fields.

    Resolved from the bundled plugin package at whatever version that package
    ships, unless ``gateway_plugin.gpustack-ext-auth.url`` overrides it. Nothing
    is defaulted when neither yields a URL -- a package carrying no such plugin
    at all. The alternative failure is Envoy unable to pull the module, which
    means the filter does not load, which under ``FAIL_OPEN`` serves every
    inference route unauthenticated and silently. A refused startup is the loud
    form of the same problem.
    """
    try:
        return plugin_spec_overrides(ext_auth_plugin_name, cfg=cfg)
    except ValueError as e:
        raise ValueError(
            f"{e} Upgrade the gpustack-higress-plugins package, or point "
            f"gateway_plugin.{ext_auth_plugin_name}.url (and .sha256) at a build "
            "of it in config.yaml."
        ) from e


def model_route_prefix(cfg: Config) -> str:
    """Route-name prefix covering every gpustack inference route.

    Higress qualifies a route name with ``<namespace>/`` when the ingresses
    live outside the gateway's own namespace. Getting this wrong does not fail
    loudly -- the matcher simply returns nothing, and under ``FAIL_OPEN`` the
    inference APIs are then served unauthenticated.
    """
    namespace = cfg.get_namespace()
    service_namespace_prefix = (
        f"{namespace}/" if namespace and namespace != cfg.gateway_namespace else ""
    )
    return f"{service_namespace_prefix}{model_route_ingress_prefix}"


def route_match_regexes(cfg: Config) -> List[str]:
    """Which route names this plugin owns, as the plugin's own safety gate.

    Checked against Envoy's ``route_name`` before anything else, which is what
    keeps everything in ``defaultConfig`` off routes this deployment does not
    own -- other tenants' included, on a shared gateway. It cannot be left to
    the rule matcher: that falls back to the global config when no rule
    matches, so "no rule matched" and "not ours" are indistinguishable there.

    Anchored explicitly. The plugin does not anchor patterns for you, and an
    unanchored prefix would match a route that merely contains it somewhere.
    Empty matches nothing, so an over-narrow pattern fails closed for this
    plugin -- but under ``FAIL_OPEN`` that means the inference routes are
    served unauthenticated, not that they are refused.
    """
    return [f"^{re.escape(model_route_prefix(cfg))}"]


def public_route_match_rules(
    public_route_ingresses: Optional[List[List[str]]] = None,
) -> Optional[List[WasmPluginMatchRule]]:
    """One CR match rule per PUBLIC route, keyed by ingress.

    This is the only thing that needs a rule at all: a route that is not PUBLIC
    takes the global config, never sees an ``access_policy``, and so keeps
    authorizing per request. The catch-all that used to occupy
    ``defaultConfig._rules_`` is gone -- ``route_match_regexes`` covers it, and
    a hand-written ``_rules_`` could not coexist with these rules anyway, since
    the Higress controller overwrites it wholesale once ``matchRules`` is
    non-empty.

    Each rule lists both the main and the ``.fallback`` ingress. The fallback
    trip is a fresh pass over the whole filter chain against a rewritten
    request; listing only the main name would drop it to the global config
    mid-request, at the exact moment ``ai-proxy`` has replaced the credential
    it would then need.

    ``access_policy`` belongs here and nowhere else: in ``defaultConfig`` it
    would declare every route public.

    No PUBLIC route yields ``None`` rather than ``[]``, and the difference is
    not cosmetic. Some API servers store this field through a typed decoder
    that omits an empty array, so a CR written with ``matchRules: []`` reads
    back with no ``matchRules`` at all -- while ``ensure_wasm_plugin`` compares
    with ``exclude_none=True``, which drops ``None`` but keeps ``[]``. The two
    then never compare equal, and a deployment with no public routes rewrites
    this CR on every reconcile tick: an xDS push, and a wasm VM rebuilt on
    every gateway pod, every 30 seconds, forever. Rendering the empty case as
    absent makes both sides agree whichever way the store normalizes it.

    Removal still propagates: going from one rule to none leaves the live CR
    holding a ``matchRules`` this returns nothing for, which is a difference,
    and the replace that follows carries no such field -- so the rule is gone
    from the CR. Only "empty here, empty there" is collapsed.
    """
    rules = [
        WasmPluginMatchRule(
            ingress=list(ingress_names),
            config={"access_policy": ACCESS_POLICY_PUBLIC},
        )
        for ingress_names in public_route_ingresses or []
    ]
    return rules or None


def ext_auth_default_config(
    cfg: Config,
    registry: McpBridgeRegistry,
    keys: Optional[Dict[str, Any]] = None,
    refs: Optional[Dict[str, Any]] = None,
    public_route_ingresses: Optional[List[List[str]]] = None,
) -> Dict[str, Any]:
    override = ext_auth_override(cfg)
    return {
        # The gate, checked before anything else below is allowed to apply.
        "route_match_regexes": route_match_regexes(cfg),
        "local_auth": {
            # Kill switch. Turning it off drops the plugin back to forwarding
            # every credential to the server, i.e. exactly today's behavior.
            "enabled": override.local_auth.enabled,
            # Generated keys with a digest: the plugin verifies these itself.
            "keys": keys if keys is not None else {},
            # Everything else the server would accept, as a validity index
            # only -- a ref cannot verify a credential, it can only confirm
            # that an identity the server already handed back is still good.
            "refs": refs if refs is not None else {},
        },
        "authz": {
            "endpoint": {
                "path": "/token-auth",
                "request_method": "GET",
                "service_name": registry.get_service_name(),
                "service_port": registry.port,
            },
            "endpoint_mode": "forward_auth",
            "timeout": override.authz.timeout,
            "authorization_request": {
                "allowed_headers": [
                    {"exact": "X-Real-IP"},
                    {"exact": "X-Forwarded-For"},
                    {"exact": "x-higress-llm-model"},
                    {"exact": "x-api-key"},
                    {"exact": "cookie"},
                    # The plugin forwards this one anyway, but the server's own
                    # marker branch is still what carries a non-PUBLIC fallback
                    # trip: on that pass ``Authorization`` has been replaced by
                    # ai-proxy, and the marker the server signed is the only
                    # thing left that identifies the caller. Declaring it costs
                    # nothing and does not depend on plugin behavior.
                    {"exact": AUTH_CACHE_HEADER},
                    {"exact": GATEWAY_AUTH_TOKEN_HEADER},
                ],
                # The asserted-identity headers are absent on purpose: their
                # values are computed per request from what the plugin resolved,
                # which neither a static allow-list nor headers_to_add can do.
                "headers_to_add": {
                    GATEWAY_AUTH_TOKEN_HEADER: cfg.get_derived_gateway_token(),
                },
            },
            # No authorization_response block: each header the server returns
            # has one fixed meaning, and a list of "which are allowed" only
            # creates a way to misconfigure it. X-Mse-Consumer is injected
            # unconditionally under a hard-coded name (every plugin in the
            # Higress ecosystem that reads it does so by constant), the marker
            # is passed back unconditionally, X-GPUStack-Key-Ref is for the
            # plugin alone and must never reach the model, and cookies are
            # handled below.
            #
            # No match_list / match_type either: omitted, the defaults are
            # whitelist + empty list, which authenticates every path -- the
            # same fail-closed outcome as the previous blacklist + "/" prefix.
        },
        "upstream_request": {
            # Client credentials that must not travel on to the model.
            #
            # ``cookie`` replaces the server's ``cookie: dummy=dummy`` response
            # header: ext-auth can only *set* response headers, never delete
            # them, so the server used to overwrite the client's cookie with
            # junk. A wasm plugin can just remove it.
            #
            # ``x-api-key`` closes a leak that predates this plugin. ai-proxy's
            # openai provider only ever *overwrites* ``Authorization``
            # (``OverwriteRequestAuthorizationHeader``); it reads ``x-api-key``
            # as a fallback source and never deletes it, so a client
            # authenticating with that header had its GPUStack key forwarded
            # verbatim -- to a worker for a self-hosted model, and to a third
            # party for an external openai-compatible provider. Its anthropic
            # provider does delete it (``claude.go``), which is why the leak is
            # specific to the openai path. Removing it here is safe because
            # ai-proxy runs later (priority 100 against this plugin's 360) and
            # takes its credential from ``apiTokens`` first, which is populated
            # for both self-hosted deployments and external providers.
            #
            # This sits outside ``authz`` because it has to happen whether or
            # not an authorization call was made -- on a PUBLIC route there is
            # no response to carry it.
            "headers_to_remove": ["cookie", "x-api-key"],
        },
        # Supplying a signing key is what turns markers on, and markers are what
        # let a fallback trip re-establish the caller locally: ai-proxy has
        # replaced ``Authorization`` with the provider credential by then, so on
        # a PUBLIC route -- where the server is never called -- there would
        # otherwise be nothing left to identify anyone by.
        #
        # The cost, accepted rather than solved: a marker rides the *upstream*
        # request, so the model backend receives it, and for a route pointing at
        # a third-party provider that means off-premises. It cannot be stripped
        # before the upstream call, because the fallback trip is an internal
        # redirect that replays the request as ai-proxy left it -- a marker
        # removed on the way out is a marker the fallback trip does not have
        # either, which trades an outbound bearer for a fallback that stops
        # working without a live server.
        #
        # What a leaked one permits is bounded by the same properties that make
        # it useful: acting as that caller, against that one model, on this
        # gateway, for at most five minutes. It carries no key material, so the
        # API key behind it stays secret; the server does not accept it (it
        # signs its own with a different key); and revocation reaches it, since
        # the identity a marker names is re-checked against ``keys`` / ``refs``
        # before use.
        "auth_cache": {
            "header": AUTH_CACHE_HEADER,
            # Dedicated derivation, never ``jwt_secret_key`` itself: that key
            # also signs user session JWTs, and this value travels to every
            # gateway pod via xDS. Not the gateway token either -- that one is
            # designed to be sent, a signing key is designed never to be.
            "signing_key": cfg.get_derived_auth_cache_key(),
        },
        "status_on_error": override.status_on_error,
        # What happens on a non-PUBLIC route while the server is unreachable --
        # a rolling upgrade included. On by default: a caller the plugin
        # authenticated itself is let through, and only that caller. An unknown
        # key or no credential at all is still rejected, which is what separates
        # this from the blanket ``failure_mode_allow`` that would turn an outage
        # into an open inference proxy.
        #
        # The price is that authorization is skipped for the duration, so during
        # the outage a key can reach a model its scope or allowed_model_names
        # would normally refuse -- bounded by how long the server is down.
        #
        # PUBLIC routes never reach this: they do not call the server at all.
        "failure_mode_allow_authenticated": override.failure_mode_allow_authenticated,
    }


def ext_auth_spec(
    cfg: Config,
    registry: McpBridgeRegistry,
    keys: Optional[Dict[str, Any]] = None,
    refs: Optional[Dict[str, Any]] = None,
    public_route_ingresses: Optional[List[List[str]]] = None,
) -> WasmPluginSpec:
    return WasmPluginSpec(
        defaultConfig=ext_auth_default_config(
            cfg=cfg,
            registry=registry,
            keys=keys,
            refs=refs,
            public_route_ingresses=public_route_ingresses,
        ),
        matchRules=public_route_match_rules(public_route_ingresses),
        defaultConfigDisable=False,
        # Unchanged from the upstream plugin, and the exposure it carries is a
        # real one: a module Envoy cannot load means this filter is absent, and
        # absent means every inference route is served with no authentication,
        # silently.
        #
        # It stays open because the alternative is worse where it matters. The
        # supported deployments serve modules from their own workload, so a
        # gateway pod can load them whether or not the server is up (see
        # ``get_plugin_url_prefix``); what FAIL_CLOSE would add there is a total
        # inference outage on any transient fetch failure. Where the server does
        # serve them -- embedded mode, or ``gateway_plugin_server_url`` unset --
        # FAIL_CLOSE would couple every gateway pod restart to the server being
        # up, which is the coupling this whole design exists to remove.
        #
        # Worth revisiting only alongside a guarantee that the module is always
        # loadable, which is a property of how it is distributed rather than of
        # this field.
        failStrategy="FAIL_OPEN",
        phase="AUTHN",
        priority=360,
        **ext_auth_module_source(cfg),
    )


def _rule_access_policy(rule: Any) -> Any:
    """The access policy on a match rule, whichever shape it arrived in.

    Reading the CR back yields parsed ``WasmPluginMatchRule`` objects, but this
    accepts a plain dict too: the alternative is silently dropping every PUBLIC
    rule on a restart if that ever stops being true, which would quietly put
    every public route back on a live server.
    """
    config = rule.config if isinstance(rule, WasmPluginMatchRule) else None
    if config is None and isinstance(rule, dict):
        config = rule.get("config")
    return (config or {}).get("access_policy")


def _database_owned_parts(
    current_spec: Optional[WasmPluginSpec],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[WasmPluginMatchRule]]:
    """``(keys, refs, public_rules)`` as they currently stand in the CR.

    Everything the reconciler owns, read back off the live spec so a restart
    does not blank it out. Anything unrecognizable is treated as absent: the
    reconciler recomputes the whole set within one interval anyway, so the
    cost of guessing wrong here is bounded, while raising would leave the CR
    permanently unreconciled.
    """
    default_config = getattr(current_spec, "defaultConfig", None) or {}
    local_auth = default_config.get("local_auth")
    keys, refs = {}, {}
    if isinstance(local_auth, dict):
        keys = (
            local_auth.get("keys") if isinstance(local_auth.get("keys"), dict) else {}
        )
        refs = (
            local_auth.get("refs") if isinstance(local_auth.get("refs"), dict) else {}
        )
    rules = getattr(current_spec, "matchRules", None)
    public_rules = [
        rule
        for rule in (rules if isinstance(rules, list) else [])
        # A rule that authorizes local allow-through is exactly one naming an
        # access policy. Nothing else has any business being in here, but
        # filtering on the field rather than taking the list wholesale means a
        # rule somebody added by hand cannot ride along through a restart.
        if _rule_access_policy(rule)
    ]
    return keys, refs, public_rules


def ext_auth_init_spec_diff(
    current_spec: Optional[WasmPluginSpec],
    expected_spec: WasmPluginSpec,
) -> WasmPluginSpec:
    """Startup diff: refresh the static base, keep what the database owns.

    Runs on every server start, when the reconciler has not produced anything
    yet. Dropping the key tables here would be safe but wasteful -- every key
    would fall back to asking the server until the first reconcile -- and
    dropping the PUBLIC rules would take PUBLIC routes off local allow-through
    for the same window. Both come back on their own; carrying them over just
    avoids the gap, and avoids two CR writes (and two xDS pushes) per restart.
    """
    if current_spec is None:
        return expected_spec
    keys, refs, public_rules = _database_owned_parts(current_spec)
    default_config = dict(expected_spec.defaultConfig or {})
    local_auth = dict(default_config.get("local_auth") or {})
    local_auth["keys"] = keys
    local_auth["refs"] = refs
    default_config["local_auth"] = local_auth
    return expected_spec.model_copy(
        # ``or None`` for the same reason :func:`public_route_match_rules` ends
        # that way: an empty array here is what a store that omits one turns
        # into a permanent difference.
        update={"defaultConfig": default_config, "matchRules": public_rules or None}
    )


def ext_auth_reconcile_spec_diff(
    current_spec: Optional[WasmPluginSpec],
    keys: Dict[str, Any],
    refs: Dict[str, Any],
    public_route_ingresses: List[List[str]],
    cfg: Config,
    registry: McpBridgeRegistry,
) -> WasmPluginSpec:
    """Reconciler diff: replace the database-owned parts, keep the base.

    A missing CR is rebuilt in full rather than left alone. Elsewhere in this
    codebase a reconciler returns ``None`` for a plugin somebody deleted by
    hand, on the grounds that recreating it guesses at intent -- but that
    convention cannot be borrowed here twice over. ``ensure_wasm_plugin``
    passes the result straight to *create* when the CR is absent, so returning
    ``None`` would create one carrying no spec at all; and for this plugin
    specifically, absent means no authentication filter on any inference route,
    which ``failStrategy: FAIL_OPEN`` then serves unauthenticated and silently.
    Every input needed to rebuild it is already in hand, so rebuilding is both
    possible and plainly better than either alternative.

    The tables are replaced wholesale rather than merged. They are a full
    recomputation from the database every time, and a merge could only ever
    resurrect a key that the recomputation decided to drop -- which is the one
    failure this design cannot tolerate, since a stale ``keys`` entry is a
    revoked credential still being let through locally.
    """
    # Mutated in place, where the startup diff returns a ``model_copy``. Safe
    # because ``ensure_wasm_plugin`` hands each diff a deepcopy of the live
    # spec; the difference is that this one owns almost all of what it touches,
    # while the startup diff builds on a freshly rendered spec it must not
    # disturb for the next caller.
    if current_spec is None:
        logger.warning(
            f"WasmPlugin {ext_auth_resource_name} is missing its spec or the "
            "resource itself; rebuilding it. Inference routes are unauthenticated "
            "until this lands."
        )
        return ext_auth_spec(
            cfg=cfg,
            registry=registry,
            keys=keys,
            refs=refs,
            public_route_ingresses=public_route_ingresses,
        )
    default_config = dict(current_spec.defaultConfig or {})
    # Anything unrecognizable is treated as absent, as in _database_owned_parts:
    # a hand-edited CR holding a non-dict here would otherwise raise on every
    # pass, and a reconciler that cannot complete is one that stops propagating
    # revocations.
    live_local_auth = default_config.get("local_auth")
    local_auth = dict(live_local_auth) if isinstance(live_local_auth, dict) else {}
    local_auth["keys"] = keys
    local_auth["refs"] = refs
    default_config["local_auth"] = local_auth
    # Rewritten alongside the rules even though it is part of the static base:
    # the two have to agree, and a CR whose gate went missing would carry
    # PUBLIC rules for routes the plugin then ignores entirely. Deriving it
    # from cfg makes writing it here idempotent.
    default_config["route_match_regexes"] = route_match_regexes(cfg)
    current_spec.defaultConfig = default_config
    current_spec.matchRules = public_route_match_rules(public_route_ingresses)
    return current_spec
