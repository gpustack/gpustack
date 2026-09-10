"""The ext-auth CR's database-owned half.

Two properties carry most of the weight here and are worth stating plainly,
because both fail silently rather than loudly:

* a key that should be gone must not survive into ``keys`` -- there it is
  authenticated locally, and on a PUBLIC route nothing behind the plugin will
  catch it;
* a rule must list the fallback ingress name too, or the fallback trip drops to
  the catch-all rule at the exact moment ``ai-proxy`` has replaced the
  credential it would need there;
* ``unrestricted`` must never be set on a key whose scope or
  ``allowed_model_names`` the server would still have checked -- on an AUTHED
  route the flag is what removes that check.
"""

import logging
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gpustack import envs
from gpustack.config.config import GatewayPluginEntry
from gpustack.gateway.client import WasmPluginMatchRule, WasmPluginSpec
from gpustack.gateway.ext_auth import (
    ext_auth_init_spec_diff,
    ext_auth_reconcile_spec_diff,
    route_match_rules,
    route_match_regexes,
)
from gpustack.schemas.api_keys import PermissionScope
from gpustack.schemas.model_routes import AccessPolicyEnum
from gpustack.schemas.principals import PrincipalType
from gpustack.server.bus import EventType
from gpustack.server.gateway_auth_reconciler import (
    KEY_ENTRY_BYTES,
    MATCH_RULE_BYTES,
    GatewayAuthReconciler,
    authed_rules_budget,
    build_local_auth_tables,
    build_skippable_routes,
    split_cr_budget,
    gateway_digest_publishable,
    gateway_key_unrestricted,
    gateway_ref_eligible,
    gateway_ref_indexable,
    route_rule_ingresses,
)


def _cfg(namespace="default", gateway_namespace="higress-system"):
    cfg = MagicMock()
    # The plugin package carries no entry for this plugin, so a build has to be
    # named in config before a spec can be rendered at all.
    cfg.gateway_plugin = {
        "gpustack-ext-auth": GatewayPluginEntry(url="oci://example.com/ext-auth:1")
    }
    cfg.get_namespace.return_value = namespace
    cfg.gateway_namespace = gateway_namespace
    cfg.get_derived_gateway_token.return_value = "token"
    cfg.get_derived_auth_cache_key.return_value = "auth-cache-key"
    return cfg


class _FakeSession:
    """Returns a fixed row set for whatever is asked, keeping the statement so
    a test can assert on the filtering the query itself does."""

    def __init__(self, rows):
        self._rows = rows
        self.statement = None

    async def exec(self, statement):
        self.statement = statement
        return SimpleNamespace(all=lambda: self._rows)


# As the server stores it, and as gateway_digest() renders it for the config.
_STORED_DIGEST = (
    "sha256$4f3c2a1b9e8d7c6b5a4938271605f4e3"
    "$7ca1547a67b46a04ee5dc4ff669ae460680a8f82e1714a679c086cc401b5748f"
)
_CONFIG_DIGEST = "s128$4f3c2a1b9e8d7c6b5a4938271605f4e3$fKFUeme0agTuXcT_ZprkYA"


def _key(**overrides):
    fields = {
        "id": 1,
        "access_key": "3192253c1f4a9b7e",
        # A real stored value: the reconciler derives the config's truncated
        # form from it, so a placeholder would simply be dropped as unusable.
        "secret_key_digest": _STORED_DIGEST,
        "expires_at": None,
        "user_id": 7,
        "is_custom": False,
        "deleted_at": None,
        # What the create endpoint defaults to, so the common row is one that
        # earns ``unrestricted``.
        "scope": [PermissionScope.ALL],
        "allowed_model_names": None,
        "principal_kind": PrincipalType.USER,
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _principal(kind=PrincipalType.USER, is_active=True):
    return SimpleNamespace(id=7, kind=kind, is_active=is_active)


def _row(**overrides):
    """One row in the shape and order the column select returns.

    Principal state is only partly here: activity and soft-deletion are
    filtered in SQL, so a row reaching the loop has already passed those, while
    ``kind`` is selected because ``gateway_key_unrestricted`` needs the value
    rather than the exclusion. What the loop still decides -- which table a row
    belongs in -- is covered by ``test_ref_indexability_is_one_predicate``.
    """
    k = _key(**overrides)
    return (
        k.id,
        k.access_key,
        k.secret_key_digest,
        k.expires_at,
        k.user_id,
        k.is_custom,
        k.scope,
        k.allowed_model_names,
        k.principal_kind,
    )


@pytest.mark.asyncio
async def test_a_digest_puts_a_key_in_keys_whether_or_not_it_is_custom():
    """ "Hit in keys" always means "has a digest, go verify it" -- there is no
    such thing as a hit without one. How the secret was chosen does not enter
    into it: the plugin runs the same comparison either way, and whether a
    custom key was given a digest at all was settled before this ran."""
    rows = [
        _row(id=1, access_key="ak-generated"),
        _row(id=2, access_key="ak-custom", is_custom=True),
    ]

    keys, refs = await build_local_auth_tables(_FakeSession(rows))

    assert list(keys) == ["ak-generated", "ak-custom"]
    assert refs == {}


@pytest.mark.asyncio
async def test_refs_holds_a_custom_key_only_where_none_may_be_published(monkeypatch):
    """``refs`` is a validity index that cannot verify anything, so it earns its
    place only for a key that can never reach ``keys``. Turning the switch off
    is what makes that permanent -- see
    ``test_a_custom_key_awaiting_its_digest_is_in_neither_table`` for the case
    that looks the same and is not."""
    monkeypatch.setattr(envs, "GATEWAY_AUTH_ALLOW_CUSTOM_KEYS", False)
    rows = [
        _row(id=1, access_key="ak-generated"),
        _row(id=2, access_key="ak-custom", is_custom=True),
    ]

    keys, refs = await build_local_auth_tables(_FakeSession(rows))

    assert list(keys) == ["ak-generated"]
    assert list(refs) == ["2"]
    # Keyed by api_keys.id, and deliberately carrying no user_id: a custom key's
    # consumer embeds its access_key, which is itself a hash of the secret and
    # so is exactly what the switch withholds. user_id alone could not rebuild
    # the consumer anyway. ``unrestricted`` is the one thing a refs entry does
    # share with a keys entry -- a marker or a cache hit can name a ref, and
    # both may then skip an authed route's call.
    assert refs["2"] == {"unrestricted": True}
    assert "user_id" not in refs["2"]


@pytest.mark.asyncio
async def test_expired_keys_are_excluded_by_the_query():
    """They would be rejected by the plugin and by the server either way, so
    keeping them buys nothing -- and the cap drops rows in id order, i.e. keeps
    the oldest, so a long-dead key would sit in the table while a newly created
    live one is the thing dropped.

    Asserted on the statement because the filtering happens in SQL. It has to:
    the column is naive in PostgreSQL and the comparison only works through the
    column's own type, so an expression built any other way fails at the driver
    rather than filtering.
    """
    session = _FakeSession([])

    await build_local_auth_tables(session)

    where = str(session.statement.whereclause)
    assert "expires_at IS NULL" in where and "expires_at >" in where


@pytest.mark.asyncio
async def test_a_key_expiring_later_still_carries_its_exp():
    """The query only sweeps up what is already dead; a key expiring between
    two passes is the plugin's own clock to catch, which is what makes that
    immediate rather than a wait of up to one interval."""
    future = datetime(2030, 1, 1, tzinfo=timezone.utc)

    keys, _ = await build_local_auth_tables(_FakeSession([_row(expires_at=future)]))

    assert keys["3192253c1f4a9b7e"]["exp"] == int(future.timestamp())


@pytest.mark.asyncio
@pytest.mark.parametrize("is_custom", [False, True])
async def test_a_key_awaiting_its_digest_is_in_neither_table(is_custom):
    """It reaches ``keys`` on its first use, in the same CR write that would
    otherwise have removed it from ``refs`` -- so listing it saves no write and
    only leaves the plugin a shared-data entry it can never read again. The
    server must withhold its key ref to match; see
    ``test_key_ref_is_withheld_from_a_key_the_gateway_cannot_index``.

    Custom or generated makes no difference. A custom key predating this feature
    is in the same position as a generated key predating the digest column, and
    treating it as a permanent refs entry would mean writing the CR twice for
    every such key the moment it is used.
    """
    keys, refs = await build_local_auth_tables(
        _FakeSession([_row(secret_key_digest=None, is_custom=is_custom)])
    )

    assert keys == {} and refs == {}


@pytest.mark.parametrize(
    "overrides,principal,allowed,indexable",
    [
        # The switch off is the whole of the rule: a custom key qualifies then,
        # and its digest column is not consulted -- a stale value from when the
        # switch was on must not strand it.
        ({"is_custom": True, "secret_key_digest": None}, _principal(), False, True),
        (
            {"is_custom": True, "secret_key_digest": _STORED_DIGEST},
            _principal(),
            False,
            True,
        ),
        # With it on, a custom key is on its way to ``keys`` or already there.
        ({"is_custom": True, "secret_key_digest": None}, _principal(), True, False),
        (
            {"is_custom": True, "secret_key_digest": _STORED_DIGEST},
            _principal(),
            True,
            False,
        ),
        # A generated key is never a refs entry, whatever the switch says.
        ({"is_custom": False, "secret_key_digest": None}, _principal(), False, False),
        (
            {"is_custom": False, "secret_key_digest": _STORED_DIGEST},
            _principal(),
            False,
            False,
        ),
        (
            {"is_custom": True, "secret_key_digest": None},
            _principal(kind=PrincipalType.SYSTEM),
            False,
            False,
        ),
        (
            {"is_custom": True, "secret_key_digest": None},
            _principal(is_active=False),
            False,
            False,
        ),
        (
            {"is_custom": True, "secret_key_digest": None, "deleted_at": object()},
            _principal(),
            False,
            False,
        ),
    ],
)
def test_ref_indexability_is_one_predicate(
    monkeypatch, overrides, principal, allowed, indexable
):
    """Shared with ``/token-auth``: whatever is not indexable here must not be
    handed a key ref there, or the plugin ends up minting a marker it will
    later refuse."""
    monkeypatch.setattr(envs, "GATEWAY_AUTH_ALLOW_CUSTOM_KEYS", allowed)
    assert gateway_ref_indexable(_key(**overrides), principal) is indexable


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed", [True, False])
async def test_the_custom_key_switch_is_re_read_on_every_pass(monkeypatch, allowed):
    """A digest outlives the switch that allowed it.

    Turning ``GATEWAY_AUTH_ALLOW_CUSTOM_KEYS`` off is what an operator does
    after deciding their users' custom keys are too weak to publish -- and the
    keys they mean are precisely the ones already carrying a digest. Deciding
    only at creation time would give that action no effect on any of them, and
    the row itself never changes to say otherwise, so nothing later would
    correct it.
    """
    monkeypatch.setattr(envs, "GATEWAY_AUTH_ALLOW_CUSTOM_KEYS", allowed)
    rows = [_row(id=2, access_key="ak-custom", is_custom=True)]

    keys, refs = await build_local_auth_tables(_FakeSession(rows))

    if allowed:
        assert list(keys) == ["ak-custom"] and refs == {}
    else:
        # refs, not dropped: it carries an id and an expiry, nothing derived
        # from the secret, so it withholds what the switch is about while still
        # letting the key be revoked while the server is down.
        assert keys == {} and list(refs) == ["2"]


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed", [True, False])
async def test_the_query_narrows_to_what_the_switch_admits(monkeypatch, allowed):
    """Both predicates are stated in SQL as well as in the loop, so
    ``limit(max_entries + 1)`` counts only rows that will be used. With the
    switch on a custom key awaiting its digest is in no table, so admitting it
    here would let a backlog of them spend the entry budget on nothing."""
    monkeypatch.setattr(envs, "GATEWAY_AUTH_ALLOW_CUSTOM_KEYS", allowed)
    session = _FakeSession([])

    await build_local_auth_tables(session)

    where = str(session.statement.whereclause)
    assert "secret_key_digest IS NOT NULL" in where
    assert ("is_custom IS true" in where) is not allowed


@pytest.mark.parametrize("allowed", [True, False])
def test_the_two_tables_never_claim_the_same_key(monkeypatch, allowed):
    """``build_local_auth_tables`` and ``/token-auth`` must place a key the same
    way, and neither table may hold what the other does. The switch moves custom
    keys across that boundary at runtime, which is the window where a mismatch
    would otherwise appear."""
    monkeypatch.setattr(envs, "GATEWAY_AUTH_ALLOW_CUSTOM_KEYS", allowed)
    key = _key(is_custom=True, secret_key_digest=_STORED_DIGEST)

    in_keys = gateway_digest_publishable(key.is_custom, key.secret_key_digest)

    assert in_keys is allowed
    assert gateway_ref_indexable(key, _principal()) is not in_keys


@pytest.mark.parametrize("allowed", [True, False])
def test_the_switch_does_not_reach_generated_keys(monkeypatch, allowed):
    """It is about secrets a user chose. A generated one is 128 bits of CSPRNG
    output either way, and searching for it is infeasible whoever can read the
    CR."""
    monkeypatch.setattr(envs, "GATEWAY_AUTH_ALLOW_CUSTOM_KEYS", allowed)

    assert gateway_digest_publishable(False, _STORED_DIGEST) is True
    assert gateway_digest_publishable(False, None) is False
    assert gateway_ref_eligible(False) is False


@pytest.mark.asyncio
async def test_keys_carry_the_user_id_for_local_consumer_rebuilding():
    keys, _ = await build_local_auth_tables(_FakeSession([_row(user_id=7)]))

    assert keys["3192253c1f4a9b7e"] == {
        "digest": _CONFIG_DIGEST,
        "user_id": 7,
        "unrestricted": True,
    }


@pytest.mark.asyncio
async def test_expiry_is_unix_seconds_and_omitted_when_absent():
    """The plugin compares against proxy-wasm's integer clock; making it parse
    RFC 3339 inside wasm would be a cost with no upside."""
    expires_at = datetime(2026, 8, 10, tzinfo=timezone.utc)
    rows = [
        _row(id=1, access_key="ak-a", expires_at=expires_at),
        _row(id=2, access_key="ak-b", expires_at=None),
    ]

    keys, _ = await build_local_auth_tables(_FakeSession(rows))

    assert keys["ak-a"]["exp"] == int(expires_at.timestamp())
    assert "exp" not in keys["ak-b"]


@pytest.mark.asyncio
async def test_a_digest_eligible_key_without_an_access_key_is_dropped():
    """The legacy cluster token's row stores the deployment-provided token
    under an empty access key. It belongs to a SYSTEM principal and the query
    filters it out; this is the second line of defense, because there is no
    key to index it under and an empty-string entry would match nothing at
    best."""
    keys, refs = await build_local_auth_tables(_FakeSession([_row(access_key="")]))

    assert keys == {} and refs == {}


@pytest.mark.asyncio
async def test_a_key_entry_always_carries_a_digest():
    """The plugin rejects a ``keys`` entry without one, and that error fails
    the entire config -- so a single malformed entry would take every other key
    down with it, not just itself. The digest predicate is what guarantees this
    can't happen: no digest means the key went to ``refs``."""
    keys, _ = await build_local_auth_tables(_FakeSession([_row()]))

    assert all(entry.get("digest") for entry in keys.values())


# --- unrestricted -----------------------------------------------------------
#
# The flag stands in for both checks /token-auth would have run after
# authentication on an AUTHED route: inference_scope, and the
# allowed_model_names half of model_allowed_for_user. Setting it where either
# would have refused is an authorization bypass, so every way of *not* earning
# it is enumerated below.


@pytest.mark.parametrize(
    "scope,allowed_model_names,kind,expected",
    [
        # Earns it: the two scopes that admit inference, and no model list.
        ([PermissionScope.ALL], None, PrincipalType.USER, True),
        ([PermissionScope.INFERENCE], None, PrincipalType.USER, True),
        ([PermissionScope.INFERENCE], [], PrincipalType.USER, True),
        (
            [PermissionScope.MANAGEMENT, PermissionScope.INFERENCE],
            None,
            PrincipalType.USER,
            True,
        ),
        # inference_scope would have refused: no scope admitting inference.
        ([PermissionScope.MANAGEMENT], None, PrincipalType.USER, False),
        ([], None, PrincipalType.USER, False),
        # A NULL scope column is a broken row, and "ask the server" is the
        # right answer for one.
        (None, None, PrincipalType.USER, False),
        # model_allowed_for_user would have intersected against this list, and
        # the route may not be in it.
        ([PermissionScope.ALL], ["llama"], PrincipalType.USER, False),
        (
            [PermissionScope.INFERENCE],
            ["llama", "qwen"],
            PrincipalType.USER,
            False,
        ),
        # Only a USER-principal is covered by the argument that makes the RBAC
        # term constant-true: non_admin_user_models filters u.kind = 'USER',
        # and a non-admin principal of any other kind falls through it to an
        # empty accessible set, i.e. the server refuses.
        ([PermissionScope.ALL], None, PrincipalType.ORG, False),
        ([PermissionScope.ALL], None, PrincipalType.GROUP, False),
        ([PermissionScope.ALL], None, PrincipalType.SYSTEM, False),
    ],
)
def test_unrestricted_is_both_key_side_checks_and_nothing_else(
    scope, allowed_model_names, kind, expected
):
    assert gateway_key_unrestricted(scope, allowed_model_names, kind) is expected


@pytest.mark.asyncio
async def test_a_key_naming_models_carries_no_flag_at_all():
    """Absent, not ``false``. The plugin reads absence as "ask the server", so
    the negative case costs ~20 bytes of a shared budget to say what silence
    already says."""
    rows = [
        _row(id=1, access_key="ak-open"),
        _row(id=2, access_key="ak-narrowed", allowed_model_names=["llama"]),
    ]

    keys, _ = await build_local_auth_tables(_FakeSession(rows))

    assert keys["ak-open"]["unrestricted"] is True
    assert "unrestricted" not in keys["ak-narrowed"]


@pytest.mark.asyncio
async def test_a_management_only_key_carries_no_flag():
    """``inference_scope`` would have refused it outright, so a skip would let
    a management-only key reach a model."""
    rows = [_row(scope=[PermissionScope.MANAGEMENT])]

    keys, _ = await build_local_auth_tables(_FakeSession(rows))

    assert "unrestricted" not in keys["3192253c1f4a9b7e"]


@pytest.mark.asyncio
async def test_the_flag_rides_the_entry_it_belongs_to():
    """Withdrawing it has to be the same CR write, on the same immediate flush,
    as withdrawing the entry -- on a skipped route nothing else asks the server
    whether the key still qualifies. A table of its own would have made those
    two separate writes."""
    session = _FakeSession([_row()])

    keys, _ = await build_local_auth_tables(session)

    assert keys["3192253c1f4a9b7e"] == {
        "digest": _CONFIG_DIGEST,
        "user_id": 7,
        "unrestricted": True,
    }


@pytest.mark.asyncio
async def test_the_entry_cap_drops_rather_than_truncates_silently(caplog):
    rows = [_row(id=i, access_key=f"ak-{i}") for i in range(1, 6)]

    keys, refs = await build_local_auth_tables(_FakeSession(rows), max_entries=2)

    assert len(keys) + len(refs) == 2
    assert "at its 2 key entry budget" in caplog.text


@pytest.mark.asyncio
async def test_rules_list_the_fallback_ingress_too():
    # The query selects (id, access_policy), so a row is that pair.
    routes = await build_skippable_routes(
        _FakeSession([(42, AccessPolicyEnum.PUBLIC), (7, AccessPolicyEnum.AUTHED)])
    )

    assert route_rule_ingresses(routes, _cfg()) == [
        (
            [
                "default/ai-route-route-42.internal",
                "default/ai-route-route-42.fallback.internal",
            ],
            "public",
        ),
        (
            [
                "default/ai-route-route-7.internal",
                "default/ai-route-route-7.fallback.internal",
            ],
            "authed",
        ),
    ]


@pytest.mark.asyncio
async def test_public_routes_are_ordered_ahead_of_authed_ones():
    """The order is what the budget truncates against, and the two are not
    worth the same: a PUBLIC rule delivers a skip on its own, while an AUTHED
    one delivers nothing unless the caller's key also made it into the tables
    the same budget pays for."""
    rows = [
        (1, AccessPolicyEnum.AUTHED),
        (2, AccessPolicyEnum.PUBLIC),
        (3, AccessPolicyEnum.AUTHED),
        (4, AccessPolicyEnum.PUBLIC),
    ]

    routes = await build_skippable_routes(_FakeSession(rows))

    # PUBLIC first, ascending id preserved inside each group so two passes over
    # the same rows render byte-identical rules.
    assert routes == [(2, "public"), (4, "public"), (1, "authed"), (3, "authed")]


@pytest.mark.asyncio
async def test_only_the_policies_the_gateway_can_act_on_are_queried():
    """``ALLOWED_PRINCIPALS`` turns on per-principal grants that are published
    nowhere near the edge, so it takes the global config and keeps authorizing
    per request."""
    session = _FakeSession([])

    await build_skippable_routes(session)

    where = str(session.statement.whereclause)
    assert "access_policy IN" in where
    assert "ALLOWED_PRINCIPALS" not in where


def test_a_rule_carries_its_route_policy_verbatim():
    """The plugin's own constants are ``AccessPolicyEnum``'s wire values, so a
    mapping on this side would only be a second place for the two to drift."""
    rules = route_match_rules(
        [
            (["ns/route-42", "ns/route-42.fallback"], "public"),
            (["ns/route-43", "ns/route-43.fallback"], "authed"),
        ]
    )

    assert [rule.ingress for rule in rules] == [
        ["ns/route-42", "ns/route-42.fallback"],
        ["ns/route-43", "ns/route-43.fallback"],
    ]
    assert [rule.config for rule in rules] == [
        {"access_policy": "public"},
        {"access_policy": "authed"},
    ]
    # Absent, not empty -- see the next test for what an empty array costs.
    assert route_match_rules([]) is None


def test_the_route_gate_is_anchored():
    """Unanchored, the pattern would also match a route that merely contains
    the prefix somewhere -- including one belonging to another tenant."""
    assert route_match_regexes(_cfg()) == [r"^default/ai\-route\-route\-"]


def test_startup_keeps_the_tables_the_database_owns():
    live = WasmPluginSpec(
        defaultConfig={
            "local_auth": {"enabled": True, "keys": {"ak": {}}, "refs": {"7": {}}},
            "status_on_error": 500,
        },
        matchRules=[
            WasmPluginMatchRule(ingress=["ns/r"], config={"access_policy": "public"}),
            WasmPluginMatchRule(ingress=["ns/hand-written"], config={}),
        ],
    )
    expected = WasmPluginSpec(
        defaultConfig={
            "local_auth": {"enabled": True, "keys": {}, "refs": {}},
            "route_match_regexes": ["^ns/ai-route-route-"],
            "status_on_error": 403,
        },
        matchRules=[],
    )

    merged = ext_auth_init_spec_diff(live, expected)

    # Carried over: re-deriving them costs a window where every key falls back
    # to the server and PUBLIC routes lose local allow-through.
    assert merged.defaultConfig["local_auth"]["keys"] == {"ak": {}}
    assert merged.defaultConfig["local_auth"]["refs"] == {"7": {}}
    assert [rule.ingress for rule in merged.matchRules] == [["ns/r"]]
    # Refreshed from cfg.
    assert merged.defaultConfig["status_on_error"] == 403
    assert merged.defaultConfig["route_match_regexes"] == ["^ns/ai-route-route-"]


def test_startup_strips_the_authorization_skip_flag():
    """Everything else carried across a restart is bounded by something -- a
    key entry still has to verify a secret, a rule still has to name a caller.
    ``unrestricted`` bounds nothing: it is the whole authorization check on an
    AUTHED route. Republishing a stale one lets a key that has since been
    narrowed reach a model its ``allowed_model_names`` no longer admits, for
    the whole of a server boot, since this runs before the schema is even
    migrated."""
    live = WasmPluginSpec(
        defaultConfig={
            "local_auth": {
                "enabled": True,
                "keys": {
                    "ak": {
                        "digest": _CONFIG_DIGEST,
                        "user_id": 7,
                        "exp": 4102444800,
                        "unrestricted": True,
                    }
                },
                "refs": {"7": {"unrestricted": True}},
            }
        },
        matchRules=[
            WasmPluginMatchRule(ingress=["ns/r"], config={"access_policy": "authed"})
        ],
    )
    expected = WasmPluginSpec(
        defaultConfig={"local_auth": {"enabled": True, "keys": {}, "refs": {}}},
        matchRules=[],
    )

    merged = ext_auth_init_spec_diff(live, expected)

    local_auth = merged.defaultConfig["local_auth"]
    assert "unrestricted" not in local_auth["keys"]["ak"]
    assert "unrestricted" not in local_auth["refs"]["7"]
    # Only that key is dropped -- authentication still happens locally from the
    # same entry, which is the whole reason the table is carried over at all.
    assert local_auth["keys"]["ak"] == {
        "digest": _CONFIG_DIGEST,
        "user_id": 7,
        "exp": 4102444800,
    }
    # The rule stays. With no entry carrying the flag nothing qualifies for the
    # skip, so it is inert until the reconciler republishes from live rows.
    assert [rule.config for rule in merged.matchRules] == [{"access_policy": "authed"}]


def test_startup_survives_an_unrecognizable_live_config():
    live = WasmPluginSpec(defaultConfig={"local_auth": "nonsense"}, matchRules=None)
    expected = WasmPluginSpec(
        defaultConfig={"local_auth": {"enabled": True, "keys": {}, "refs": {}}},
        matchRules=[],
    )

    merged = ext_auth_init_spec_diff(live, expected)

    assert merged.defaultConfig["local_auth"] == {
        "enabled": True,
        "keys": {},
        "refs": {},
    }
    assert merged.matchRules is None


def _spec_dump(spec):
    """``ensure_wasm_plugin``'s comparison, spelled the same way it spells it."""
    return spec.model_dump(exclude_none=True) if spec else {}


def test_a_deployment_with_no_public_routes_stops_rewriting_the_cr():
    """The pass runs every interval whether or not anything changed, so a
    difference that cannot converge is a CR write -- an xDS push, and a wasm VM
    rebuilt on every gateway pod -- on every tick, forever.

    An API server that stores this field through a typed decoder omits an empty
    array, so a CR written with ``matchRules: []`` reads back without the field.
    ``exclude_none=True`` keeps ``[]`` and drops ``None``, which is what turned
    that into a permanent difference.
    """
    live = WasmPluginSpec(
        defaultConfig={
            "local_auth": {"enabled": True, "keys": {}, "refs": {}},
            "route_match_regexes": [r"^default/ai\-route\-route\-"],
        },
        # As the store hands it back: no matchRules at all.
        matchRules=None,
    )

    updated = ext_auth_reconcile_spec_diff(
        live.model_copy(deep=True),
        keys={},
        refs={},
        route_rules=[],
        cfg=_cfg(),
        registry=MagicMock(),
    )

    assert _spec_dump(updated) == _spec_dump(live)


def test_the_last_public_route_leaving_still_reaches_the_cr():
    """The other half of the rule above: collapsing empty and absent must not
    also collapse "had a rule, should not any more". That rule is a standing
    authorization to serve a route without asking the server."""
    live = WasmPluginSpec(
        defaultConfig={"local_auth": {"enabled": True, "keys": {}, "refs": {}}},
        matchRules=[
            WasmPluginMatchRule(ingress=["ns/r"], config={"access_policy": "public"})
        ],
    )

    updated = ext_auth_reconcile_spec_diff(
        live.model_copy(deep=True),
        keys={},
        refs={},
        route_rules=[],
        cfg=_cfg(),
        registry=MagicMock(),
    )

    assert updated.matchRules is None
    # A difference, so the write happens -- and it carries no matchRules, which
    # is what takes the rule off the CR.
    assert _spec_dump(updated) != _spec_dump(live)
    assert "matchRules" not in _spec_dump(updated)


def test_reconcile_replaces_the_tables_wholesale():
    """A merge could only ever resurrect a key the recomputation dropped, and
    that key is a revoked credential still being allowed through locally."""
    live = WasmPluginSpec(
        defaultConfig={
            "local_auth": {"enabled": True, "keys": {"stale": {}}, "refs": {"9": {}}},
            "authz": {"timeout": 30000},
            "route_match_regexes": ["^stale/"],
        }
    )

    updated = ext_auth_reconcile_spec_diff(
        live,
        keys={"fresh": {"digest": _CONFIG_DIGEST, "user_id": 7}},
        refs={},
        route_rules=[(["default/ai-route-route-42.internal"], "public")],
        cfg=_cfg(),
        registry=MagicMock(),
    )

    default_config = updated.defaultConfig
    assert default_config["local_auth"]["keys"] == {
        "fresh": {"digest": _CONFIG_DIGEST, "user_id": 7}
    }
    assert default_config["local_auth"]["refs"] == {}
    assert default_config["local_auth"]["enabled"] is True
    # The static base is left alone.
    assert default_config["authz"] == {"timeout": 30000}
    assert [rule.config for rule in updated.matchRules] == [{"access_policy": "public"}]
    # The gate is rewritten with the rules: PUBLIC rules pointing at routes the
    # plugin has been told to ignore would be inert.
    assert default_config["route_match_regexes"] == [r"^default/ai\-route\-route\-"]


def test_a_missing_cr_is_rebuilt_in_full():
    """``ensure_wasm_plugin`` hands whatever this returns straight to *create*
    when the CR is absent, so returning None would create one with no spec --
    which for this plugin means no authentication filter on any inference
    route, served open and silent under FAIL_OPEN."""
    registry = MagicMock()
    registry.get_service_name.return_value = "gpustack.static"
    registry.port = 80

    spec = ext_auth_reconcile_spec_diff(
        None,
        keys={"ak": {"digest": _CONFIG_DIGEST, "user_id": 7}},
        refs={"58": {}},
        route_rules=[(["default/ai-route-route-42.internal"], "authed")],
        cfg=_cfg(),
        registry=registry,
    )

    assert spec is not None
    assert spec.defaultConfig["local_auth"]["keys"] == {
        "ak": {"digest": _CONFIG_DIGEST, "user_id": 7}
    }
    assert spec.defaultConfig["local_auth"]["refs"] == {"58": {}}
    assert [rule.config for rule in spec.matchRules] == [{"access_policy": "authed"}]
    # The static base is present too, not just the tables.
    assert spec.defaultConfig["authz"]["endpoint"]["path"] == "/token-auth"
    assert spec.phase == "AUTHN" and spec.priority == 360


# --- The ModelRoute event filter ------------------------------------------
#
# ModelRoute rows are rewritten constantly (targets / ready_targets restamp on
# every target state transition) while the rules derive from (id,
# access_policy) alone. The filter drops the events that cannot matter. It is
# allowed to be wrong only in the direction that costs a redundant pass.


def _reconciler(applied_policies):
    reconciler = GatewayAuthReconciler.__new__(GatewayAuthReconciler)
    reconciler._applied_route_policies = applied_policies
    return reconciler


def _route_event(**fields):
    return SimpleNamespace(type=EventType.UPDATED, data=SimpleNamespace(**fields))


def test_a_route_leaving_the_skippable_policies_still_flushes():
    """The case the whole applied-state invariant exists for. The payload
    already reads the policy that gets no rule, so only "the CR has a rule for
    this route" can catch it -- and missing it would leave the route skipped
    locally with no second gate."""
    reconciler = _reconciler({42: "authed"})

    event = _route_event(id=42, access_policy=AccessPolicyEnum.ALLOWED_PRINCIPALS)

    assert reconciler._model_route_may_change_rules(event) is True


def test_a_route_gaining_a_skippable_policy_still_flushes():
    reconciler = _reconciler({})

    for policy in (AccessPolicyEnum.PUBLIC, AccessPolicyEnum.AUTHED):
        event = _route_event(id=42, access_policy=policy)

        assert reconciler._model_route_may_change_rules(event) is True


@pytest.mark.parametrize(
    "applied,policy",
    [
        ("public", AccessPolicyEnum.AUTHED),
        ("authed", AccessPolicyEnum.PUBLIC),
    ],
    ids=["public-to-authed", "authed-to-public"],
)
def test_a_route_crossing_between_the_two_policies_flushes(applied, policy):
    """The rule stays but its value changes, so tracking ids alone would miss
    it -- and the route would keep being skipped under the policy it just left
    until the periodic pass caught up."""
    reconciler = _reconciler({42: applied})

    event = _route_event(id=42, access_policy=policy)

    assert reconciler._model_route_may_change_rules(event) is True


def test_a_deleted_route_always_flushes():
    """A delete carries the row as it was, so its policy still matches what the
    last pass recorded -- the comparison would read the removal as "nothing
    moved" and leave the rule on the CR outliving the route it names."""
    reconciler = _reconciler({42: "authed"})

    event = SimpleNamespace(
        type=EventType.DELETED,
        data=SimpleNamespace(id=42, access_policy=AccessPolicyEnum.AUTHED),
    )

    assert reconciler._model_route_may_change_rules(event) is True


def test_churn_on_a_route_that_has_no_rule_is_dropped():
    """The high-frequency case: a target came up, ready_targets was restamped.
    The route's policy gets no rule and it has none in the CR, so the rules
    array cannot move."""
    reconciler = _reconciler({7: "public"})

    event = _route_event(id=42, access_policy=AccessPolicyEnum.ALLOWED_PRINCIPALS)

    assert reconciler._model_route_may_change_rules(event) is False


@pytest.mark.parametrize(
    "policy",
    [AccessPolicyEnum.PUBLIC, AccessPolicyEnum.AUTHED],
    ids=["public", "authed"],
)
def test_churn_on_a_route_whose_rule_is_unchanged_is_dropped(policy):
    """AUTHED is the default policy, so nearly every route now carries a rule.
    Matching on the id alone would make this both the common case and the case
    the filter never suppresses -- every target state transition in the
    deployment would cost a pass."""
    reconciler = _reconciler({42: policy.value})

    event = _route_event(id=42, access_policy=policy)

    assert reconciler._model_route_may_change_rules(event) is False


@pytest.mark.parametrize(
    "policy",
    [
        AccessPolicyEnum.PUBLIC,
        "public",
        "PUBLIC",
        AccessPolicyEnum.AUTHED,
        "authed",
        "AUTHED",
    ],
    ids=[
        "public-enum",
        "public-value",
        "public-column-name",
        "authed-enum",
        "authed-value",
        "authed-column-name",
    ],
)
def test_a_skippable_policy_is_recognised_in_every_shape_the_payload_may_carry(
    policy,
):
    """Hydrated model, JSON round-trip, and the name the column stores. Read
    against an empty applied state, so recognising the policy at all is the
    whole of what makes this reconcile."""
    assert _reconciler({})._model_route_may_change_rules(
        _route_event(id=42, access_policy=policy)
    )


def test_nothing_applied_yet_means_nothing_can_be_ruled_out():
    assert _reconciler(None)._model_route_may_change_rules(
        _route_event(id=42, access_policy=AccessPolicyEnum.ALLOWED_PRINCIPALS)
    )


@pytest.mark.parametrize(
    "data",
    [
        {"id": 42},  # distributed mode: id-only dict, no policy to read
        SimpleNamespace(id=42),  # a model that never loaded the column
        None,
        {},
    ],
    ids=["id-only-dict", "no-policy-attr", "no-payload", "empty-dict"],
)
def test_an_unreadable_payload_reconciles_anyway(data):
    reconciler = _reconciler({7: "public"})

    event = SimpleNamespace(type=EventType.UPDATED, data=data)

    assert reconciler._model_route_may_change_rules(event) is True


def test_a_dict_payload_is_read_like_a_model():
    reconciler = _reconciler({7: "public"})

    dropped = SimpleNamespace(
        type=EventType.UPDATED, data={"id": 42, "access_policy": "allowed_principals"}
    )
    kept = SimpleNamespace(
        type=EventType.UPDATED, data={"id": 42, "access_policy": "authed"}
    )

    assert reconciler._model_route_may_change_rules(dropped) is False
    assert reconciler._model_route_may_change_rules(kept) is True


@pytest.fixture
def stub_reconcile_inputs(monkeypatch):
    """Point a reconciler's queries and its CR write at doubles."""
    from contextlib import asynccontextmanager

    state = {"routes": [(42, "public")], "applied": []}

    @asynccontextmanager
    async def _session():
        yield object()

    async def _tables(session, max_entries=0):
        return {}, {}

    async def _routes(session):
        return state["routes"]

    async def _ensure(**kwargs):
        if state.get("fail"):
            raise RuntimeError("API server said no")
        state["applied"].append(kwargs)

    monkeypatch.setattr(
        "gpustack.server.gateway_auth_reconciler.async_session", _session
    )
    monkeypatch.setattr(
        "gpustack.server.gateway_auth_reconciler.build_local_auth_tables", _tables
    )
    monkeypatch.setattr(
        "gpustack.server.gateway_auth_reconciler.build_skippable_routes", _routes
    )
    monkeypatch.setattr(
        "gpustack.server.gateway_auth_reconciler.ensure_wasm_plugin", _ensure
    )
    return state


def _live_reconciler():
    reconciler = GatewayAuthReconciler.__new__(GatewayAuthReconciler)
    reconciler._config = _cfg()
    reconciler._budget = 10_000_000
    reconciler._extensions_api = object()
    reconciler._registry = MagicMock()
    reconciler._applied_route_policies = None
    reconciler._applied_state = None
    return reconciler


@pytest.mark.asyncio
async def test_the_applied_set_records_what_was_written(stub_reconcile_inputs):
    reconciler = _live_reconciler()

    await reconciler.reconcile()

    assert reconciler._applied_route_policies == {42: "public"}


@pytest.mark.asyncio
async def test_a_failed_write_leaves_the_applied_set_alone(stub_reconcile_inputs):
    """Otherwise the filter would start believing a CR state that was never
    applied, and would drop the very events that would have retried it -- the
    route would stay allowed locally until the periodic pass."""
    stub_reconcile_inputs["fail"] = True
    reconciler = _live_reconciler()

    with pytest.raises(RuntimeError):
        await reconciler.reconcile()

    assert reconciler._applied_route_policies is None


# --- The shared byte budget --------------------------------------------------
#
# Keys and match rules live in one CR and therefore under one etcd object limit.
# Capping them separately would let the sum overrun it, and overrunning it is
# not a partial failure: the write is refused, the tables freeze, and
# revocations stop propagating.
#
# Three claims, served PUBLIC rules -> key entries -> AUTHED rules.


def test_everything_together_stays_inside_the_budget():
    for public_routes in (0, 10, 1000, 100_000):
        for keys in (0, 500, 100_000):
            public_rules, entries = split_cr_budget(1_100_000, public_routes)
            published = min(keys, entries)
            authed = authed_rules_budget(1_100_000, public_rules, published)

            used = (
                public_rules + authed
            ) * MATCH_RULE_BYTES + published * KEY_ENTRY_BYTES
            assert used <= 1_100_000, f"{public_routes} routes / {keys} keys overran"
            assert public_rules <= public_routes


def test_public_rules_squeeze_the_key_table():
    """The interaction the budget exists to make explicit, rather than caps
    that each look fine on their own."""
    _, without = split_cr_budget(1_100_000, 0)
    _, with_routes = split_cr_budget(1_100_000, 1000)

    assert with_routes < without
    # The rules' bytes come out of the same pool, give or take the one entry
    # lost to flooring.
    displaced = (without - with_routes) * KEY_ENTRY_BYTES
    assert abs(displaced - 1000 * MATCH_RULE_BYTES) < KEY_ENTRY_BYTES


def test_keys_are_served_before_authed_rules():
    """The ordering the whole split exists for. A route past the budget loses
    only its authorization skip -- its callers still authenticate locally. A
    key past it loses local authentication too, so every one of its requests
    carries a credential to the server. Serving all rules first, as this did
    when PUBLIC was the only policy that got one, would invert that."""
    public_rules, entries = split_cr_budget(1_100_000, 0)

    # Every key the cap admits is published, and only then do AUTHED rules get
    # a look in.
    assert authed_rules_budget(1_100_000, public_rules, entries) == 0
    assert entries > 7_000


def test_authed_rules_get_what_the_keys_did_not_use():
    """The cap is a ceiling, not a reservation: a deployment with few keys
    spends the remainder on rules rather than holding it for keys that do not
    exist."""
    public_rules, entries = split_cr_budget(1_100_000, 0)

    assert authed_rules_budget(1_100_000, public_rules, 100) > 6_000
    assert authed_rules_budget(1_100_000, public_rules, entries // 2) > 3_000


def test_public_rules_are_served_before_keys():
    """The one exception to keys-first: a public route with no rule needs a
    live server even for anonymous traffic, since nothing else in the chain can
    name that caller."""
    public_rules, entries = split_cr_budget(1_100_000, 500)

    assert public_rules == 500
    assert entries > 0


def test_a_budget_the_public_rules_exhaust_leaves_no_keys():
    """Not a hypothetical shape -- it is what a deployment with far more public
    routes than budget looks like, and every key then authenticates at the
    server exactly as it did before any of this existed."""
    public_rules, entries = split_cr_budget(MATCH_RULE_BYTES * 5, 100)

    assert public_rules == 5  # capped by the budget, not by the 100 asked for
    assert entries == 0
    assert authed_rules_budget(MATCH_RULE_BYTES * 5, public_rules, 0) == 0


@pytest.mark.asyncio
async def test_a_tight_budget_drops_authed_rules_and_keeps_public_ones(
    stub_reconcile_inputs, caplog
):
    """The ordering, end to end. Room for two rules and no keys: both PUBLIC
    routes keep theirs and both AUTHED routes lose theirs, rather than the
    array being truncated in the order the query returned it."""
    stub_reconcile_inputs["routes"] = [
        (1, "public"),
        (2, "public"),
        (3, "authed"),
        (4, "authed"),
    ]
    reconciler = _live_reconciler()
    reconciler._budget = MATCH_RULE_BYTES * 2

    with caplog.at_level(logging.WARNING):
        await reconciler.reconcile()

    # Recorded for every skippable route, including the two the budget
    # dropped: the filter asks whether an event changes the input, and a route
    # dropped for want of budget has the same input as one that fits.
    assert reconciler._applied_route_policies == {
        1: "public",
        2: "public",
        3: "authed",
        4: "authed",
    }
    assert "2 routes left out" in caplog.text


@pytest.mark.asyncio
async def test_churn_on_a_route_the_budget_dropped_is_still_suppressed(
    stub_reconcile_inputs,
):
    """The filter has to keep working past the budget. A route left out for
    want of it has the same input as one that fits -- re-running the pass would
    drop it again -- so recording only what was written would turn every
    ``targets`` restamp on those routes into a reconcile, failing exactly where
    the deployment is already large enough to need the suppression."""
    stub_reconcile_inputs["routes"] = [(1, "public"), (2, "authed")]
    reconciler = _live_reconciler()
    reconciler._budget = MATCH_RULE_BYTES  # room for the PUBLIC rule alone

    await reconciler.reconcile()

    dropped_route_churn = _route_event(id=2, access_policy=AccessPolicyEnum.AUTHED)
    assert reconciler._model_route_may_change_rules(dropped_route_churn) is False


@pytest.mark.asyncio
async def test_an_unchanged_pass_says_nothing(stub_reconcile_inputs, caplog):
    """The pass runs every interval regardless, so a line on each one is a
    heartbeat that says nothing about the config -- and cannot even be read as
    one, since it looks the same whether the CR was rewritten or diffed away.
    Its presence has to mean something moved."""
    reconciler = _live_reconciler()

    with caplog.at_level(
        logging.DEBUG, logger="gpustack.server.gateway_auth_reconciler"
    ):
        await reconciler.reconcile()
        first = caplog.text
        caplog.clear()
        await reconciler.reconcile()

    assert "skippable routes" in first, "the first pass established the state"
    assert caplog.text == ""


@pytest.mark.asyncio
async def test_a_changed_pass_says_so(stub_reconcile_inputs, caplog):
    reconciler = _live_reconciler()

    with caplog.at_level(
        logging.DEBUG, logger="gpustack.server.gateway_auth_reconciler"
    ):
        await reconciler.reconcile()
        stub_reconcile_inputs["routes"] = [(42, "public"), (7, "authed")]
        caplog.clear()
        await reconciler.reconcile()

    assert "2 skippable routes (1 public)" in caplog.text


@pytest.mark.asyncio
async def test_a_zero_cap_publishes_nothing():
    """Zero is a real cap, not "no cap". ``split_cr_budget`` returns it for a
    budget the public routes have already spent, and reading that as unlimited
    would publish every key in the deployment -- the exact opposite."""
    rows = [_row(id=i, access_key=f"ak-{i}") for i in range(1, 4)]

    keys, refs = await build_local_auth_tables(_FakeSession(rows), max_entries=0)

    assert (keys, refs) == ({}, {})


@pytest.mark.asyncio
async def test_no_cap_is_spelled_none():
    rows = [_row(id=i, access_key=f"ak-{i}") for i in range(1, 4)]
    session = _FakeSession(rows)

    keys, _ = await build_local_auth_tables(session, max_entries=None)

    assert len(keys) == 3
    assert session.statement._limit_clause is None


@pytest.mark.asyncio
async def test_a_custom_key_in_keys_carries_what_rebuilds_its_consumer():
    """Once in ``keys`` a custom key is indistinguishable from a generated one:
    same digest comparison, and the same ``user_id`` the plugin joins to the
    access key to name the caller without asking the server."""
    rows = [_row(id=2, access_key="ak-custom", is_custom=True)]

    keys, refs = await build_local_auth_tables(_FakeSession(rows))

    assert list(keys) == ["ak-custom"]
    assert refs == {}
    assert keys["ak-custom"]["user_id"] == 7


def test_the_server_withholds_a_ref_wherever_refs_would_be_empty(monkeypatch):
    """A ref names an entry in ``refs``. With the switch on there are none, so
    handing one out would have the plugin mint a marker against an id that
    validates against nothing -- and refuse its own marker a pass later."""
    monkeypatch.setattr(envs, "GATEWAY_AUTH_ALLOW_CUSTOM_KEYS", True)
    for digest in (_STORED_DIGEST, None):
        assert not gateway_ref_indexable(
            _key(is_custom=True, secret_key_digest=digest), _principal()
        )
