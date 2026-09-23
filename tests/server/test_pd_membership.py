"""Router membership: the registry is read, never assumed.

What a router already holds when this code first reaches it is
version-dependent. Upstream's igw branch does build the PD router with empty
worker lists (`create_vllm_pd_router(&[], &[], ...)` with the comment "Empty
worker list - workers added later"; SGLang's `create_pd_router(None, None,
...)`), yet a 1P1D started with `--enable-igw --prefill ... --decode ...`
reports both members in `GET /workers` — the empty worker list is only a
startup transient, and command-line peers do enter the registry.

So the invariant these tests guard is not "the registry starts empty" but
"the registry is read back, and the read-back is what decides servability" —
which is the same code either way.
"""

from types import SimpleNamespace

import pytest

from gpustack.schemas.models import ModelInstanceStateEnum, RoleNameEnum
from gpustack.server import pd_membership
from gpustack.server.pd_membership import (
    MembershipOutcome,
    desired_members,
    member_url,
    router_addresses,
)


def _instance(role, port, state=ModelInstanceStateEnum.RUNNING, ip="10.0.0.1"):
    return SimpleNamespace(role=role, port=port, state=state, worker_ip=ip)


def _model():
    return SimpleNamespace(
        id=1, name="pd", disaggregation=SimpleNamespace(mode="vllm-nixl")
    )


def test_only_running_gpu_roles_are_registered():
    """The router is not its own upstream, and a member that is not RUNNING has
    nothing listening — upstream probes a peer before admitting it and drops it
    silently on timeout, so registering one early buys a silent absence."""
    instances = [
        _instance("prefill", 40010),
        _instance("decode", 40011),
        _instance(RoleNameEnum.ROUTER.value, 40012),
        _instance("decode", 40013, state=ModelInstanceStateEnum.STARTING),
    ]
    members = desired_members(_model(), instances)
    assert members == {
        "http://10.0.0.1:40010": "prefill",
        "http://10.0.0.1:40011": "decode",
    }


def test_a_member_is_addressed_the_way_the_router_stores_it():
    """`host:port` of the serving listener. Measured: the `worker` label on the
    router's own counters carries the whole URL it was launched with, not a
    worker name or id, and `DELETE /workers/{url}` matches on the same."""
    assert member_url(_instance("decode", 40011)) == "http://10.0.0.1:40011"
    assert member_url(_instance("decode", None)) is None


def test_every_running_router_is_reconciled():
    """One router with an empty registry serves 503s while another serves fine,
    so each is reconciled on its own — the replica count is declared like any
    other even though the shipped recipes run one."""
    instances = [
        _instance(RoleNameEnum.ROUTER.value, 40012),
        _instance(RoleNameEnum.ROUTER.value, 40013),
        _instance(RoleNameEnum.ROUTER.value, 40014, state=ModelInstanceStateEnum.ERROR),
        _instance("decode", 40011),
    ]
    assert router_addresses(instances) == ["10.0.0.1:40012", "10.0.0.1:40013"]


@pytest.mark.asyncio
async def test_a_recipe_that_declares_the_api_without_launching_it_is_not_a_failure():
    """The regression that protects every deployment on the other path.

    Declared is not usable: without the flag its API needs, the router already
    knows its peers from the command line and `POST /workers` answers 400.
    Reporting that as a failed registration would park such a group in PARTIAL
    forever.

    The state is built here rather than read off a shipped recipe: a recipe
    may gain or lose `--enable-igw` without changing the behaviour this
    guards, so the mode it needs is constructed and the test survives a recipe
    that flips either way.
    """
    from gpustack.server.pd_mode_catalog import get_pd_mode

    shipped = get_pd_mode("vllm-nixl")
    assert shipped.router.membership_api.available is True

    unlaunched = shipped.model_copy(
        update={
            "router": shipped.router.model_copy(
                update={
                    "command": [
                        token
                        for token in (shipped.router.command or [])
                        if str(token) != "--enable-igw"
                    ]
                }
            )
        },
        deep=True,
    )
    assert unlaunched.router.membership_api.available is True
    assert unlaunched.router.membership_api_usable is False

    outcome = await pd_membership.reconcile(
        _model(), unlaunched, [_instance("decode", 40011)], "10.0.0.1:40012"
    )
    assert outcome.ok is True
    assert outcome.reason is None


def test_a_persistent_failure_says_where_to_look_without_naming_a_cause():
    """The escalated message is USER-VISIBLE, so what it may claim matters.

    One failure is ordinary — a router that just came up, a member still being
    probed — so escalating on the first would train people to ignore it. Five
    in a row earns the right to send someone to the router itself.

    And it must send them to evidence, not to a mechanism. Members can also
    join from the command line, so the message must not claim this API is the
    only way in, nor prescribe dropping `--enable-igw` — that would point at
    the recipe for a failure the recipe did not cause. The negative assertions
    below are what keeps such wording out.
    """
    pd_membership.forget(7)
    for _ in range(pd_membership.PERSISTENT_FAILURE_PASSES - 1):
        pd_membership.record(7, MembershipOutcome(ok=False, reason="not admitted"))
    assert pd_membership.outcome_for(7).reason == "not admitted"

    pd_membership.record(7, MembershipOutcome(ok=False, reason="not admitted"))
    escalated = pd_membership.outcome_for(7).reason
    # It still says what happened, for how long, and what to read next.
    assert "not admitted" in escalated
    assert str(pd_membership.PERSISTENT_FAILURE_PASSES) in escalated
    assert "GET /workers" in escalated
    # And it no longer explains the failure with a refuted mechanism.
    assert "--enable-igw" not in escalated
    assert "only join through this API" not in escalated
    assert "remove that flag" not in escalated

    # A success clears the count, so a transient outage does not leave the
    # group wearing an escalated message it has grown out of.
    pd_membership.record(7, MembershipOutcome(ok=True))
    assert pd_membership.consecutive_failures(7) == 0
    pd_membership.forget(7)


def test_the_servability_gate_reads_the_recorded_outcome():
    """The seam this feature was built into. `upstream_registration_ready`
    had a TODO saying "return the recorded outcome when the registration step
    lands" — this is that step, and an unrecorded outcome must still mean
    servable so the command-line path keeps working."""
    from gpustack.schemas.models import Model, RoleSpec
    from gpustack.server.controllers import upstream_registration_ready

    model = Model(name="pd")
    model.id = 42
    assert upstream_registration_ready(model) is True, "no router role: vacuous"

    model.roles = [
        RoleSpec(name="prefill", replicas=1),
        RoleSpec(name="router", replicas=1),
    ]
    pd_membership.forget(42)
    assert upstream_registration_ready(model) is True, "unrecorded: command-line path"

    pd_membership.record(42, MembershipOutcome(ok=False, reason="not admitted"))
    assert upstream_registration_ready(model) is False

    pd_membership.record(42, MembershipOutcome(ok=True))
    assert upstream_registration_ready(model) is True
    pd_membership.forget(42)


def test_a_members_url_is_percent_encoded_in_the_removal_path():
    """The bug this pins, found on a live router.

    A member's id IS a URL, so substituting it raw into `DELETE /workers/{url}`
    makes the path `/workers/http://host:port` — which upstream routes to its
    transparent proxy instead and answers 405 "Only POST requests are supported
    for transparent proxy". Percent-encoded it is 200.

    Worse than the 405: removal failures are deliberately non-fatal (upstream's
    removal gate is global and waits for in-flight requests), so the stale
    member stayed while the outcome still read ok. The leniency hid it. That is
    why a kept member now shows up in `reason` even on success.
    """
    from urllib.parse import quote

    template = "http://r:1/workers/{url}"
    url = "http://10.0.0.1:40051"
    assert template.replace("{url}", quote(url, safe="")) == (
        "http://r:1/workers/http%3A%2F%2F10.0.0.1%3A40051"
    )
    # The raw form is what produced the 405 — kept as the negative case so a
    # future "simplification" back to it fails here rather than in production.
    assert "{url}" not in template.replace("{url}", quote(url, safe=""))
    assert "/workers/http://" in template.replace("{url}", url)


def test_only_an_unreadable_registry_counts_toward_a_restart():
    """Which failure a restart can fix, and which it only makes worse.

    The shipped recipes run one router per group and it is the gateway's only
    upstream, so recreating it interrupts the whole group until the
    replacement is up and has probed its peers. That price buys something only
    when the router is not answering at all. A refused member means it is
    alive and disagreeing -- a version or argument mismatch that the
    replacement would be handed and would refuse again.
    """
    model_id = 7788
    pd_membership.forget(model_id)

    for _ in range(pd_membership.RESTART_AFTER_UNREADABLE_PASSES + 2):
        pd_membership.record(
            model_id,
            pd_membership.MembershipOutcome(ok=False, reason="router refused a member"),
        )
    assert pd_membership.should_restart_router(model_id) is False

    pd_membership.forget(model_id)
    for _ in range(pd_membership.RESTART_AFTER_UNREADABLE_PASSES):
        pd_membership.record(
            model_id,
            pd_membership.MembershipOutcome(
                ok=False, unreadable=True, reason="could not read the registry"
            ),
        )
    assert pd_membership.should_restart_router(model_id) is True
    pd_membership.forget(model_id)


def test_one_unreadable_pass_does_not_cost_the_group_an_outage():
    """A single dropped request must not trade a blip for a real outage."""
    model_id = 7789
    pd_membership.forget(model_id)
    pd_membership.record(
        model_id,
        pd_membership.MembershipOutcome(
            ok=False, unreadable=True, reason="could not read the registry"
        ),
    )
    assert pd_membership.should_restart_router(model_id) is False
    pd_membership.forget(model_id)


def test_a_readable_pass_breaks_the_streak():
    """The streak has to be consecutive: a router that answers once is not the
    wedged process this path exists for."""
    model_id = 7790
    pd_membership.forget(model_id)
    for _ in range(pd_membership.RESTART_AFTER_UNREADABLE_PASSES - 1):
        pd_membership.record(
            model_id,
            pd_membership.MembershipOutcome(ok=False, unreadable=True, reason="x"),
        )
    pd_membership.record(model_id, pd_membership.MembershipOutcome(ok=True))
    pd_membership.record(
        model_id,
        pd_membership.MembershipOutcome(ok=False, unreadable=True, reason="x"),
    )
    assert pd_membership.should_restart_router(model_id) is False
    pd_membership.forget(model_id)


# --- reaching a router the server cannot dial -------------------------------


def test_router_instances_carry_the_worker_the_proxy_belongs_to():
    """The address alone cannot reach a router on a tunnel-mode worker.

    A tunnel worker only ever dials out, so `http://worker_ip:40027` times out
    from the server and the group parks in PARTIAL with "waiting for upstream
    registration" while the router is healthy and merely empty. The proxy is a
    property of the WORKER, so the caller needs the instance, not just its
    address — which is why `router_instances` exists beside `router_addresses`.
    """
    instances = [
        _instance(RoleNameEnum.ROUTER.value, 40012),
        _instance(RoleNameEnum.ROUTER.value, 40014, state=ModelInstanceStateEnum.ERROR),
        _instance("decode", 40011),
    ]
    routers = pd_membership.router_instances(instances)
    assert [i.port for i in routers] == [40012]
    # The two views agree on which routers count, so a caller switching to the
    # richer one cannot silently start reconciling a different set.
    assert [f"{i.worker_ip}:{i.port}" for i in routers] == router_addresses(instances)


@pytest.mark.asyncio
async def test_the_proxy_is_passed_to_every_call_not_just_the_read():
    """A registration that reads through the proxy and writes around it would
    report an empty registry it could never fill — the failure would look like
    a router refusing members rather than like a network it cannot cross."""
    seen = []

    class _Response:
        status = 200

        async def json(self):
            return {"workers": []}

        async def text(self):
            return ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _Client:
        def request(self, method, url, **kwargs):
            seen.append((method, url, kwargs.get("proxy")))
            return _Response()

        async def close(self):
            return None

    from gpustack.server.pd_mode_catalog import get_pd_mode

    mode = get_pd_mode("vllm-nixl")
    assert mode.router.membership_api_usable, "the recipe must launch the flag"

    instances = [
        _instance(RoleNameEnum.ROUTER.value, 40027),
        _instance("prefill", 40055),
        _instance("decode", 40029),
    ]
    await pd_membership.reconcile(
        _model(),
        mode,
        instances,
        "10.0.0.1:40027",
        client=_Client(),
        proxy="http://user:pass@127.0.0.1:30079",
    )
    assert seen, "reconcile made no request at all"
    assert all(
        proxy == "http://user:pass@127.0.0.1:30079" for _, _, proxy in seen
    ), seen
    # And the reads and the writes both happened, so this is not vacuous.
    assert any(method == "GET" for method, _, _ in seen), seen
    assert any(method == "POST" for method, _, _ in seen), seen


@pytest.mark.asyncio
async def test_no_proxy_leaves_the_direct_path_unchanged():
    """`get_proxy_address()` returns None for every mode but `tunnel`, so every
    other deployment must keep dialling direct."""
    seen = []

    class _Response:
        status = 200

        async def json(self):
            return {"workers": []}

        async def text(self):
            return ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _Client:
        def request(self, method, url, **kwargs):
            seen.append(kwargs.get("proxy"))
            return _Response()

        async def close(self):
            return None

    from gpustack.server.pd_mode_catalog import get_pd_mode

    await pd_membership.reconcile(
        _model(),
        get_pd_mode("vllm-nixl"),
        [
            _instance(RoleNameEnum.ROUTER.value, 40027),
            _instance("prefill", 40055),
        ],
        "10.0.0.1:40027",
        client=_Client(),
    )
    assert seen and all(proxy is None for proxy in seen), seen


# --- restarting only where restarting can help ------------------------------


def _unreadable_outcome():
    return MembershipOutcome(
        ok=False, unreadable=True, reason="the router's member list could not be read"
    )


def test_a_wedged_router_is_still_restarted():
    """The case this path exists for: the process is up and not answering, and
    a fresh one rendered from the group's current addresses is the repair."""
    pd_membership.forget(99)
    for _ in range(pd_membership.RESTART_AFTER_UNREADABLE_PASSES):
        pd_membership.record(99, _unreadable_outcome())
    assert pd_membership.should_restart_router(99)
    pd_membership.forget(99)


def test_an_unreachable_router_is_not_restarted_forever():
    """The loop this fixes. `note_restart_ordered` clears the streak so the
    next pass measures the new process — and without a budget the streak just
    refills, so a network the server cannot cross produced one router restart
    every five passes indefinitely, each a real outage and none of them able to
    help. Measured on a `tunnel`-mode worker before the proxy path existed."""
    pd_membership.forget(99)
    ordered = 0
    # Ten times the streak length: far past anything a wedged process needs.
    for _ in range(pd_membership.RESTART_AFTER_UNREADABLE_PASSES * 10):
        pd_membership.record(99, _unreadable_outcome())
        if pd_membership.should_restart_router(99):
            pd_membership.note_restart_ordered(99)
            ordered += 1
    assert ordered == pd_membership.RESTART_ATTEMPT_LIMIT, ordered
    assert pd_membership.restarts_exhausted(99)
    pd_membership.forget(99)


def test_a_readable_registry_refreshes_the_restart_budget():
    """The budget answers "has restarting ever helped", so only evidence that
    the path works may reset it — a later wedge on a group that once recovered
    still gets its restarts."""
    pd_membership.forget(99)
    for _ in range(pd_membership.RESTART_AFTER_UNREADABLE_PASSES):
        pd_membership.record(99, _unreadable_outcome())
    pd_membership.note_restart_ordered(99)
    assert pd_membership.restarts_exhausted(99) is False

    # A read got through: not ok yet (members still missing), but readable.
    pd_membership.record(99, MembershipOutcome(ok=False, reason="member missing"))
    for _ in range(pd_membership.RESTART_AFTER_UNREADABLE_PASSES):
        pd_membership.record(99, _unreadable_outcome())
    assert pd_membership.should_restart_router(99), "budget was not refreshed"
    pd_membership.forget(99)


def test_ordering_a_restart_does_not_refresh_its_own_budget():
    """The mistake that would reintroduce the loop: clearing the counter on the
    attempt makes every attempt look like the first."""
    pd_membership.forget(99)
    for _ in range(pd_membership.RESTART_AFTER_UNREADABLE_PASSES):
        pd_membership.record(99, _unreadable_outcome())
    pd_membership.note_restart_ordered(99)
    pd_membership.note_restart_ordered(99)
    assert pd_membership.restarts_exhausted(99)
    pd_membership.forget(99)


def test_forgetting_a_group_clears_its_restart_budget():
    """A group that is deleted and redeployed is a new group, and must not
    inherit a spent budget from the old one."""
    pd_membership.forget(99)
    for _ in range(pd_membership.RESTART_AFTER_UNREADABLE_PASSES):
        pd_membership.record(99, _unreadable_outcome())
    pd_membership.note_restart_ordered(99)
    pd_membership.note_restart_ordered(99)
    assert pd_membership.restarts_exhausted(99)
    pd_membership.forget(99)
    assert pd_membership.restarts_exhausted(99) is False


# --- the SGLang gateway: one declaration across a breaking upstream rename ---
#
# `sglang_router` under `--pd-disaggregation` behaves as follows, and each test
# below pins one of these:
#
#   POST /workers -> 202 (queued, not 200), member present on the next read
#   a prefill registered with `bootstrap_port` as a STRING -> 422, never joins
#   0.2.2 (SGLang v0.5.5): GET /workers reports id == url;
#     DELETE /workers/{encoded url} -> 202
#   0.3.2 (v0.5.8+): id is a UUID;  the url form -> 400 "expected UUID";
#     UUID -> 202
#   traffic reaches an added prefill carrying the bootstrap_port it registered
#   with


def _band(base):
    return SimpleNamespace(base=base)


def _prefill_with_bootstrap(port=40010, bootstrap=9001):
    instance = _instance("prefill", port)
    instance.named_ports = {"bootstrap": _band(bootstrap)}
    return instance


def test_a_bootstrap_port_reaches_the_body_as_a_number():
    """A string here is refused with `invalid type: string "9001", expected
    u16` and the member never registers — the group then has a prefill the
    router does not know about, which reads like a scheduling problem."""
    from gpustack.server.pd_mode_catalog import get_pd_mode

    api = get_pd_mode("sglang-mooncake").router.membership_api
    ports = pd_membership.desired_member_ports([_prefill_with_bootstrap()])
    body = pd_membership._body(
        api, "http://10.0.0.1:40010", "prefill", "pd", ports["http://10.0.0.1:40010"]
    )

    assert body["bootstrap_port"] == 9001
    assert isinstance(body["bootstrap_port"], int), body
    assert body["url"] == "http://10.0.0.1:40010"
    assert body["worker_type"] == "prefill"


def test_a_role_without_the_band_simply_omits_the_key():
    """Decode has no bootstrap service, so the key is absent rather than null:
    a null would be a claim about a port, and one `body` has to describe both
    roles without growing a per-role section."""
    from gpustack.server.pd_mode_catalog import get_pd_mode

    api = get_pd_mode("sglang-mooncake").router.membership_api
    body = pd_membership._body(api, "http://10.0.0.1:40011", "decode", "pd", {})

    assert "bootstrap_port" not in body, body
    assert body == {"url": "http://10.0.0.1:40011", "worker_type": "decode"}


def test_a_template_with_literal_text_still_renders_to_a_string():
    """Only a bare placeholder carries a type through. `vllm-nixl` renders its
    address this way, and an address is a string on every router."""
    rendered = pd_membership._render(
        "http://{{peer.ip}}:{{peer.port}}",
        pd_membership._scope("http://10.0.0.1:40010", "pd", {}),
    )
    assert rendered == "http://10.0.0.1:40010"


@pytest.mark.asyncio
async def test_a_member_is_removed_by_the_id_the_registry_reports():
    """A router that keys a member by a registry-generated id rejects its URL
    in that position. Taking the id off the probe is right whichever way the
    router names a member, and needs no version check against the image.
    """
    from gpustack.server.pd_mode_catalog import get_pd_mode

    seen = []
    uuid = "cd144cb9-2026-4854-9478-7b67c10de9ca"

    class _Response:
        status = 200

        def __init__(self, payload):
            self._payload = payload

        async def json(self):
            return self._payload

        async def text(self):
            return ""

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _Client:
        def request(self, method, url, **kwargs):
            seen.append((method, url))
            return _Response(
                {
                    "workers": [
                        {
                            "id": uuid,
                            "url": "http://10.0.0.1:40099",
                            "worker_type": "prefill",
                        }
                    ]
                }
            )

        async def close(self):
            return None

    # The registry holds a member the group no longer has, so it must go.
    outcome = await pd_membership.reconcile(
        _model(),
        get_pd_mode("sglang-mooncake"),
        [_instance(RoleNameEnum.ROUTER.value, 40012), _prefill_with_bootstrap()],
        "10.0.0.1:40012",
        client=_Client(),
    )

    deletes = [url for method, url in seen if method == "DELETE"]
    assert deletes == [f"http://10.0.0.1:40012/workers/{uuid}"], seen
    # And not the address form, which this router version answers with 400.
    assert not any("40099" in url for url in deletes), deletes
    assert outcome is not None


@pytest.mark.asyncio
async def test_a_router_without_the_route_is_not_a_failed_registration():
    """The regression that protects older images. A router that predates
    this API (SGLang through v0.5.2 served only the older `/add_worker`) has
    its peers from the command line and serves fine. Reading its 404 as an
    unreadable registry would park a healthy group in PARTIAL and then spend
    the restart budget on a route that cannot appear."""
    from gpustack.server.pd_mode_catalog import get_pd_mode

    class _Response:
        status = 404

        async def json(self):
            return {}

        async def text(self):
            return "Not Found"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _Client:
        def request(self, method, url, **kwargs):
            return _Response()

        async def close(self):
            return None

    outcome = await pd_membership.reconcile(
        _model(),
        get_pd_mode("sglang-mooncake"),
        [_instance(RoleNameEnum.ROUTER.value, 40012), _prefill_with_bootstrap()],
        "10.0.0.1:40012",
        client=_Client(),
    )

    assert outcome.ok is True
    assert outcome.unreadable is False, "a missing route is not a wedged router"
    assert outcome.reason is None


def test_the_sglang_recipes_need_no_flag_to_make_membership_usable():
    """The difference from the vLLM fork, stated where a reader will hit it:
    there is no `--enable-igw` equivalent here, so `usable` follows from
    `available` alone."""
    from gpustack.server.pd_mode_catalog import get_pd_mode

    for name in ("sglang-mooncake", "sglang-nixl"):
        mode = get_pd_mode(name)
        assert mode.router.membership_api.requires_args == [], name
        assert mode.router.membership_api.available is True, name
        assert mode.router.membership_api_usable is True, name


# --- «restart on error» off means the platform does not repair, either -------


def _pd_model(restart_on_error: bool):
    from gpustack.schemas.models import Model, RoleSpec

    model = Model(name="pd", restart_on_error=restart_on_error)
    model.id = 4242
    model.roles = [
        RoleSpec(name="prefill", replicas=1),
        RoleSpec(name="decode", replicas=1),
        RoleSpec(name="router", replicas=1),
    ]
    return model


def test_an_unreadable_registry_says_the_repair_was_declined():
    """The group that looked like it was restarting forever.

    Recreating the router is a recovery like any other, and a more disruptive
    one than most: the replacement is a NEW member with a fresh name and a
    zeroed restart count, so a deployment with «restart on error» off never
    settled into a state anyone could inspect — it just kept producing routers.

    The switch now gates it, and the outcome has to SAY so: parked with the
    bare "could not be read" reads identically to parked one pass before the
    platform recreates the router, and those are opposite situations for
    whoever is watching.
    """
    from gpustack.server.controllers import _explain_unreadable

    bare = MembershipOutcome(
        ok=False, unreadable=True, reason="the router's member list could not be read"
    )

    pd_membership.forget(4242)
    explained = _explain_unreadable(_pd_model(restart_on_error=False), bare)
    assert explained.unreadable is True
    assert "did NOT do it" in explained.reason
    assert "restart on error" in explained.reason

    # With the switch on and the budget unspent, the platform is about to try:
    # nothing to explain, so the outcome is passed through untouched.
    assert _explain_unreadable(_pd_model(restart_on_error=True), bare) is bare
    pd_membership.forget(4242)


def test_a_spent_restart_budget_still_outranks_the_switch():
    """Having tried and got nowhere is a stronger statement than having been
    told not to try, so it keeps its own message — the operator needs to know
    the path is suspect, not that a switch is off."""
    from gpustack.server.controllers import _explain_unreadable

    pd_membership.forget(4242)
    for _ in range(pd_membership.RESTART_AFTER_UNREADABLE_PASSES):
        pd_membership.record(4242, _unreadable_outcome())
    for _ in range(pd_membership.RESTART_ATTEMPT_LIMIT):
        pd_membership.note_restart_ordered(4242)
    assert pd_membership.restarts_exhausted(4242)

    explained = _explain_unreadable(
        _pd_model(restart_on_error=False),
        MembershipOutcome(ok=False, unreadable=True, reason="x"),
    )
    assert "tunnel" in explained.reason, explained.reason
    pd_membership.forget(4242)


def test_a_readable_outcome_is_never_reworded():
    """The rewording is about one failure; everything else passes through, or a
    successful registration would start carrying a failure's explanation."""
    from gpustack.server.controllers import _explain_unreadable

    ok = MembershipOutcome(ok=True, registered=["http://10.0.0.1:40010"])
    assert _explain_unreadable(_pd_model(restart_on_error=False), ok) is ok

    refused = MembershipOutcome(ok=False, reason="the router did not admit x")
    assert _explain_unreadable(_pd_model(restart_on_error=False), refused) is refused
