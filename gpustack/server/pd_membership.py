"""Telling a PD router who its members are, over HTTP.

Two facts decide the shape of everything here, and both come from the behaviour
of a live 1P1D rather than from docs:

**1. Without `--enable-igw` there is no membership API at all.** `POST /workers`
returns 400 `"PD router requires specific add_prefill_server or
add_decode_server methods"` — the single-router path drops `worker_type` before
the PD router sees it. Confirmed in upstream source: the non-igw branch builds
the router through `create_router()`, which reads `prefill_urls`/`decode_urls`
off the command line.

**2. What the registry already holds is version-dependent, so it is READ and
never assumed.** This is the fact that shapes the module. The source reads as
if the command-line peers never reach the registry under `--enable-igw` — the
igw branch builds the PD router with hard-coded empty worker lists (vLLM's
`create_vllm_pd_router(&[], &[], ...)` with the upstream comment
`// Empty worker list - workers added later`, SGLang's
`create_pd_router(None, None, ...)`), which would put a structural 503 `"No
available workers"` window in front of every router start.

Against the `vllm-router` shipped with vLLM 0.20.2 the CLI peers **do** enter
the registry under igw: a 1P1D started with `--enable-igw --prefill ...
--decode ...` reports both members in `GET /workers` with the right
`worker_type`, answers `/v1/models` 200, and serves a two-hop request. The
`total: 0` visible before that is the startup transient, while the router is
still probing its peers. Whether the source reading holds on some other build
is unknown — hence version-dependent, and hence read rather than assumed
either way.

⇒ So this module assumes nothing about who is already in: it reads the
registry, adds only what is missing, removes only what is stale, and reads
back. That diff is correct whether a member arrived from the command line or
from this API. It is also why a group is not reported servable until the
read-back agrees — a registry that is merely late looks exactly like one that
is empty, and only the read tells them apart.

**A persistent failure is stated, not repaired.** Nothing here can change
how the router was launched, and restarting it re-runs the same call against
the same endpoint. So a failure that outlives `PERSISTENT_FAILURE_PASSES` says
what did not happen and where to look, and the group stays PARTIAL rather than
claiming to serve.

**3. Fact 1 does not generalise to every router either, and this module must
not assume it does.** Fact 1 describes the vLLM fork. On the gateway that fork
descends from (`sglang_router` wheels 0.2.2 and 0.3.2, `--pd-disaggregation`,
no `--enable-igw`):

- the command-line peers **do** enter the registry — `GET /workers` reports
  them with the right `worker_type` and the prefill's `bootstrap_port`;
- `POST /workers` is accepted (**202**, an async job, not 200) and the member
  is serving on the next read, with no flag to enable and no restart;
- with the command-line prefill removed, a request routed through the
  dynamically added one, carrying the `bootstrap_port` it was registered with.

So the shape of the answer, not the answer itself, is what this module encodes:
send explicitly, **read back**, and let the read-back decide. The one thing
that is read as "not applicable" rather than as a failure is a router that has
no such route at all — see `_read_registry`.
"""

import asyncio
import logging
import re
from urllib.parse import quote, urlsplit
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import aiohttp

from gpustack.schemas.models import Model, ModelInstance, RoleNameEnum
from gpustack.schemas.pd_modes import PDMembershipAPI, PDMode

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 10.0
_RECONCILE_DEADLINE_SECONDS = 30.0

# How long the read-back may wait for members that were just sent.
#
# Because an add is not always synchronous. The vLLM fork answers `POST
# /workers` with 200 once the member is in; the gateway it forks from answers
# **202** and does the health-probe and the registration on a queue, so a member
# can be absent from one read and serving a few seconds later. Reading back
# immediately would therefore
# report a false "not admitted" on every scale-out on that family, park the
# group in PARTIAL for a cycle, and put a misleading reason in the log.
#
# Costs nothing where the add was synchronous: the first read already agrees
# and the loop returns without sleeping.
_ADMISSION_WINDOW_SECONDS = 6.0
_ADMISSION_POLL_SECONDS = 0.5


class MembershipOutcome:
    """What one reconcile attempt achieved, in the terms the caller acts on.

    `ok` gates the group's servability, so it is deliberately pessimistic: it
    is True only when the router's own read-back agrees with the members we
    believe exist. An accepted `POST` is not enough — upstream admits a peer
    only after probing it, and drops it silently on timeout with a default
    window tuned for small models. Without the read-back, a scale-out that
    quietly failed is indistinguishable from one that worked.
    """

    def __init__(
        self,
        ok: bool,
        reason: Optional[str] = None,
        registered: Optional[Sequence[str]] = None,
        unreadable: bool = False,
    ):
        self.ok = ok
        self.reason = reason
        self.registered = list(registered or [])
        # Separate from `ok` because only one kind of failure is worth
        # restarting the router over. A refused `POST` means the router is
        # alive and disagrees -- a version or argument mismatch that a restart
        # repeats rather than fixes, from a registry that at best comes back
        # the way it went out. A registry that cannot be READ is the other
        # thing: the process itself is not answering, and restarting is the
        # only move left.
        self.unreadable = unreadable

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"MembershipOutcome(ok={self.ok}, reason={self.reason!r})"


def member_url(instance: ModelInstance) -> Optional[str]:
    """The address the router knows a member by.

    `host:port` of the serving listener, which is what upstream stores and
    what `DELETE /workers/{url}` matches on — measured: the `worker` label on
    the router's own request counters carries the whole URL it was launched
    with, not a worker name or id.
    """
    if not instance.worker_ip or not instance.port:
        return None
    return f"http://{instance.worker_ip}:{instance.port}"


def desired_members(model: Model, instances: Sequence[ModelInstance]) -> Dict[str, str]:
    """`{url: role}` for every member that should be in the router's registry.

    Only RUNNING GPU roles. The router is excluded because it is not its own
    upstream, and a member that is not RUNNING has nothing listening — adding
    it would be admitted only to fail the router's own probe, and upstream
    drops such a peer silently.

    **A draining member is excluded while still RUNNING**, and that is the
    whole of soft scale-down's first step. It is deliberately not a state
    change: the container has to keep serving the decodes that are already
    pulling KV from it, and only *new* work has to stop arriving. Removing the
    address is exactly that distinction, and it is measured at 18ms with
    requests in flight — so changing the ratio does not interrupt them.
    """
    from gpustack.schemas.models import ModelInstanceStateEnum

    members: Dict[str, str] = {}
    for instance in instances:
        if instance.role == RoleNameEnum.ROUTER.value or not instance.role:
            continue
        if instance.state != ModelInstanceStateEnum.RUNNING:
            continue
        if getattr(instance, "draining_since", None) is not None:
            continue
        url = member_url(instance)
        if url:
            members[url] = instance.role
    return members


def desired_member_ports(
    instances: Sequence[ModelInstance],
) -> Dict[str, Dict[str, int]]:
    """`{url: {band name: base port}}` for the members that declare any.

    Separate from `desired_members` because only one router needs it and most
    members have no named band at all. It exists because a membership body can
    legitimately carry more than an address: SGLang's prefill registers with a
    `bootstrap_port`, and a prefill added without one is admitted, routed to,
    and then hangs — the router injects `"bootstrap_port": null` into the
    request and decode has nowhere to rendezvous. The band is the same one the
    peer flag renders from, so the two paths cannot drift apart.
    """
    ports: Dict[str, Dict[str, int]] = {}
    for instance in instances:
        url = member_url(instance)
        if not url:
            continue
        bands = getattr(instance, "named_ports", None) or {}
        named = {
            name: int(band.base)
            for name, band in bands.items()
            if getattr(band, "base", None) is not None
        }
        if named:
            ports[url] = named
    return ports


class RegistryEntry(NamedTuple):
    """One row of the router's own member list.

    `id` is separate from the URL because a router may name a member by
    something only its registry knows — a generated UUID rather than the
    address it was added with. Reading the id back off the probe is what lets
    one declaration address either kind, since the removal path uses whatever
    this says.
    """

    role: str
    id: Optional[str] = None


class RegistryRead(NamedTuple):
    """The three answers a probe can give, kept apart on purpose.

    `entries` None means *unreadable* — we cannot tell, so the group stays
    PARTIAL and the restart path is allowed to consider it. An empty mapping
    means the router genuinely has no members, which is a fact to act on.
    `absent` is the third: this router serves no such route, so there is
    nothing to reconcile and nothing is wrong.
    """

    entries: Optional[Dict[str, RegistryEntry]] = None
    absent: bool = False


def _endpoint(
    api: PDMembershipAPI, base: str, spec: Optional[str]
) -> Optional[Tuple[str, str]]:
    """A declared `"METHOD /path"` -> `(method, absolute url)`.

    The method is part of the declaration because the two shapes upstream
    ships differ in it: a REST resource (`POST /workers`) versus an action
    endpoint (`POST /instances/add`).
    """
    if not spec:
        return None
    parts = spec.split(None, 1)
    if len(parts) != 2:
        logger.warning("Malformed membership endpoint %r; ignoring", spec)
        return None
    method, path = parts[0].upper(), parts[1]
    return method, f"{base.rstrip('/')}{path}"


_PLACEHOLDER = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")

# A template whose value cannot be resolved for this member. Not an error and
# not an empty string: the key is left out of the body entirely.
_MISSING = object()


def _scope(url: str, model_name: str, ports: Dict[str, int]) -> Dict[str, object]:
    """What a body template may refer to, for one member.

    The same names the peer flags use (`peer.ip`, `peer.port`,
    `peer.ports.<band>`), because a body and a peer flag describe the same
    member to the same router and having two vocabularies for that would be a
    second thing to keep in step.
    """
    split = urlsplit(url)
    scope: Dict[str, object] = {"peer.url": url, "model_name": model_name}
    if split.hostname:
        scope["peer.ip"] = split.hostname
    if split.port:
        scope["peer.port"] = split.port
    for name, base in (ports or {}).items():
        scope[f"peer.ports.{name}"] = base
    return scope


def _render(template: str, scope: Dict[str, object]) -> object:
    """One body value, with its type preserved when the template is just a name.

    The type matters and cost a 422 to learn: `bootstrap_port` is a `u16`
    upstream, and a body carrying `"9002"` is rejected with
    `invalid type: string "9002", expected u16` — the whole member never
    registers. So `"{{peer.ports.bootstrap}}"` renders to the integer 9002,
    while `"http://{{peer.ip}}:{{peer.port}}"` renders to a string, which is
    what each of them has to be.

    An unknown name yields `_MISSING` rather than an empty string, so a body
    key that does not apply to this member is simply absent: decode declares no
    bootstrap band, and sending it `"bootstrap_port": null` would be a claim
    about a port rather than silence about one.
    """
    whole = _PLACEHOLDER.fullmatch(template.strip())
    if whole:
        return scope.get(whole.group(1), _MISSING)

    missing = False

    def _substitute(match: "re.Match[str]") -> str:
        nonlocal missing
        value = scope.get(match.group(1), _MISSING)
        if value is _MISSING:
            missing = True
            return ""
        return str(value)

    rendered = _PLACEHOLDER.sub(_substitute, template)
    return _MISSING if missing else rendered


def _body(
    api: PDMembershipAPI,
    url: str,
    role: str,
    model_name: str,
    ports: Optional[Dict[str, int]] = None,
) -> dict:
    """The request body for one member.

    `role_field` is named in the declaration rather than assumed, because it
    is the field the single-router path drops — so a consumer has to know what
    to look for in the read-back to tell "accepted" from "actually joined".
    """
    scope = _scope(url, model_name, ports or {})
    payload: Dict[str, object] = {}
    for key, template in (api.body or {}).items():
        value = _render(str(template), scope)
        if value is not _MISSING:
            payload[key] = value
    # Set last and unconditionally: every shipped router keys a member on its
    # address, so this one field is an invariant of the call rather than a
    # detail of the declaration.
    payload["url"] = url
    if api.role_field:
        payload[api.role_field] = (api.role_values or {}).get(role, role)
    return payload


async def _read_registry(
    client: aiohttp.ClientSession,
    api: PDMembershipAPI,
    base: str,
    proxy: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
) -> RegistryRead:
    """What the router itself says its members are.

    Unreadable, empty and absent are three different answers — see
    `RegistryRead`.

    **404 and 405 mean absent, and that is not a failure.** A router old
    enough to predate this API serves no `/workers` at all (SGLang through
    v0.5.2 shipped only the older `/add_worker` query-param form, and the PD
    path refused even that), while its command-line peers are in place and the
    group is serving. Reading that as an unreadable registry would park a
    healthy group in PARTIAL and then spend the restart budget on a route that
    cannot appear, whereas the version that has the API answers this probe on
    the first pass.
    """
    resolved = _endpoint(api, base, api.probe)
    if resolved is None:
        return RegistryRead()
    method, url = resolved
    try:
        async with client.request(
            method,
            url,
            proxy=proxy,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=_TIMEOUT_SECONDS),
        ) as response:
            if response.status in (404, 405):
                logger.debug(
                    "The router at %s serves no membership API (%s); "
                    "its peers come from the command line",
                    url,
                    response.status,
                )
                return RegistryRead(absent=True)
            if response.status != 200:
                return RegistryRead()
            payload = await response.json()
    except Exception as e:
        logger.debug("Could not read the router's registry at %s: %s", url, e)
        return RegistryRead()

    rows = payload.get("workers") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return RegistryRead()
    registry: Dict[str, RegistryEntry] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        peer = row.get("url")
        if not peer:
            continue
        role = row.get(api.role_field) if api.role_field else None
        member_id = row.get("id")
        registry[str(peer)] = RegistryEntry(
            role=str(role or ""),
            id=str(member_id) if member_id else None,
        )
    return RegistryRead(entries=registry)


async def _add_missing(
    client: aiohttp.ClientSession,
    api: PDMembershipAPI,
    base: str,
    model_name: str,
    wanted: Dict[str, str],
    current: Dict[str, RegistryEntry],
    ports: Optional[Dict[str, Dict[str, int]]] = None,
    proxy: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
) -> None:
    """Add every wanted member the router does not already have.

    Failures are logged rather than raised: the read-back at the end of
    `reconcile` is what decides the outcome, so a refusal here shows up there
    as a missing member with the router's own words attached. Raising would
    skip the members after it for no gain.

    An accepted add is not necessarily a 200: the gateway this catalog's
    routers descend from queues the registration and answers **202**, which is
    why nothing here treats a particular success code as the signal. The
    read-back does.
    """
    add = _endpoint(api, base, api.add)
    if add is None:
        return
    method, endpoint = add
    for url, role in wanted.items():
        if url in current:
            continue
        try:
            async with client.request(
                method,
                endpoint,
                json=_body(api, url, role, model_name, (ports or {}).get(url)),
                proxy=proxy,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=_TIMEOUT_SECONDS),
            ) as response:
                if response.status >= 400:
                    body = (await response.text())[:200]
                    logger.warning("Router refused member %s (%s): %s", url, role, body)
        except Exception as e:
            logger.warning("Could not add member %s: %s", url, e)


async def _remove_stale(
    client: aiohttp.ClientSession,
    api: PDMembershipAPI,
    base: str,
    wanted: Dict[str, str],
    current: Dict[str, RegistryEntry],
    proxy: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Drop members the group no longer has, and report what would not go.

    A failure here is NOT fatal: upstream's removal gate is global -- it waits
    for the in-flight count to reach zero -- so a member can legitimately
    linger while traffic continues, and it takes no new requests meanwhile.
    Parking a serving group in PARTIAL over such an entry would be worse.

    The kept entries are returned rather than only logged, because that
    leniency is what hides a broken call: an unencoded URL in the path gets
    405 and the stale member stays, while the outcome still reads ok.
    """
    remove = _endpoint(api, base, api.remove)
    kept: List[str] = []
    if remove is None:
        return kept
    method, template = remove
    for url, entry in current.items():
        if url in wanted:
            continue
        # Percent-encoded, and that matters: on the routers that
        # key a member by its address, the member's id IS a URL, so
        # substituting it raw makes the path `/workers/http://host:port`,
        # which upstream routes to its transparent proxy instead -- 405 "Only
        # POST requests are supported for transparent proxy". Encoded it is
        # 200. `safe=""` because the `:` and `/` are exactly what must escape.
        #
        # `{id}` is the other half: a router that keys a member by a
        # registry-generated id rejects its URL in this position. Taking the id
        # from the probe rather than from a version check is correct either
        # way, because a router that keys on the URL reports the URL as the id.
        endpoint = template.replace("{url}", quote(url, safe="")).replace(
            "{id}", quote(entry.id or url, safe="")
        )
        try:
            async with client.request(
                method,
                endpoint,
                proxy=proxy,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=_TIMEOUT_SECONDS),
            ) as response:
                if response.status >= 400:
                    kept.append(f"{url} (status {response.status})")
                    logger.info(
                        "Router kept stale member %s (status %s); its removal "
                        "gate waits for in-flight requests",
                        url,
                        response.status,
                    )
        except Exception as e:
            kept.append(f"{url} ({e})")
            logger.info("Could not remove stale member %s: %s", url, e)
    return kept


async def _await_registration(
    client: aiohttp.ClientSession,
    api: PDMembershipAPI,
    base: str,
    wanted: Dict[str, str],
    proxy: Optional[str] = None,
    headers: Optional[Dict[str, str]] = None,
) -> RegistryRead:
    """Read the registry back, giving a queued registration time to land.

    Polls rather than sleeping a fixed amount: a router that admits members
    synchronously answers on the first read and pays nothing, and one that
    queues them is given `_ADMISSION_WINDOW_SECONDS` before its answer is taken
    as final. The last read is what is returned either way — a member still
    missing when the window closes is missing, and that is the answer the
    outcome is built from.
    """
    deadline = asyncio.get_event_loop().time() + _ADMISSION_WINDOW_SECONDS
    while True:
        read = await _read_registry(client, api, base, proxy=proxy, headers=headers)
        if read.absent or read.entries is None:
            return read
        if not set(wanted) - set(read.entries):
            return read
        if asyncio.get_event_loop().time() >= deadline:
            return read
        await asyncio.sleep(_ADMISSION_POLL_SECONDS)


async def reconcile(  # noqa: C901
    model: Model,
    mode: Optional[PDMode],
    instances: Sequence[ModelInstance],
    router_address: Optional[str],
    client: Optional[aiohttp.ClientSession] = None,
    proxy: Optional[str] = None,
    proxy_token: Optional[str] = None,
) -> MembershipOutcome:
    """Make the router's registry match the group's members, and say whether it does.

    Idempotent by construction: it diffs what the router reports against what
    the group has, so running it on every pass costs one read when nothing
    changed. That matters because this is the same call used at group
    formation and at scale-out — one code path, so the startup case cannot
    drift away from the steady-state case.

    `proxy` is what makes this work on a worker the server cannot dial.
    Every call here targets the ROUTER's own port, not the worker API, and a
    `tunnel`-mode worker only ever dials out — so `http://worker_ip:40027`
    times out from the server and the group parks in PARTIAL with
    "waiting for upstream registration" while the router is in fact healthy
    and merely empty. Measured on a tunnel worker: the registration path was
    the one thing in the product still assuming it could reach a worker
    directly, because the gateway reaches model instances through the same
    proxy and this is not a model instance. `Worker.get_proxy_address()`
    returns None for every other proxy mode, so the direct path is unchanged.

    `proxy_token` is not optional in practice once `proxy` is set: the
    server's forward proxy authenticates every request that is not
    `GET /metrics`, so an unauthenticated hop answers 401 and the read looks
    like an unreadable registry — which then trips the router restart, and no
    restart can supply a missing credential. It rides `Proxy-Authorization`
    rather than `Authorization`: the token authorises the HOP, and the router
    has its own opinion about `Authorization`.
    """
    if mode is None or not mode.router.membership_api_usable:
        # The recipe does not launch the flag the API needs. Nothing to do,
        # and specifically NOT a failure: the group is on the command-line
        # path, where the router already knows its peers.
        return MembershipOutcome(ok=True, reason=None)

    if not router_address:
        return MembershipOutcome(
            ok=False, reason="the router's address is not known yet"
        )

    api = mode.router.membership_api
    wanted = desired_members(model, instances)
    ports = desired_member_ports(instances)
    if not wanted:
        return MembershipOutcome(
            ok=False, reason="no member is running yet, so none can be registered"
        )

    base = f"http://{router_address}"
    # Only when a proxy is in play: on the direct path there is no hop to
    # authorise, and handing the worker's token to the router would be giving
    # a credential to something that never asked for one.
    proxy_headers = (
        {"Proxy-Authorization": f"Bearer {proxy_token}"}
        if proxy and proxy_token
        else None
    )
    owned = client is None
    try:
        if owned:
            # `force_close` because these requests may ride a forward proxy
            # to a tunnel worker, and that hop does not keep the connection
            # alive between requests. Measured: the first `POST /workers`
            # succeeded and the second failed with
            # `Can not write request body` — aiohttp had reused a socket the
            # proxy had already closed, so exactly one of two members got
            # registered and the group parked in PARTIAL with a message that
            # blamed registration rather than the connection.
            client = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(force_close=True)
            )

        async def _run() -> MembershipOutcome:
            read = await _read_registry(
                client, api, base, proxy=proxy, headers=proxy_headers
            )
            if read.absent:
                # This build serves no membership API, so its peers are the
                # ones it was launched with and there is nothing to reconcile.
                # Same answer as an undeclared API, for the same reason: the
                # group is on the command-line path and it works.
                return MembershipOutcome(ok=True, reason=None)
            current = read.entries
            if current is None:
                return MembershipOutcome(
                    ok=False,
                    unreadable=True,
                    reason=(
                        "the router's member list could not be read, so whether "
                        "it can serve is unknown"
                    ),
                )

            await _add_missing(
                client,
                api,
                base,
                model.name,
                wanted,
                current,
                ports=ports,
                proxy=proxy,
                headers=proxy_headers,
            )

            # Stale entries are removed, and a failure to remove one is NOT
            # fatal: upstream's removal gate is global (it waits for the
            # in-flight count to reach zero), so a member can legitimately
            # linger while traffic continues. Treating that as broken would
            # park a serving group in PARTIAL over an entry that no longer
            # takes new requests.
            #
            # That leniency is also what hides a broken call — an unencoded
            # URL in the path gets 405 and the stale member stays, while the
            # outcome still reads ok. So the failure is logged at INFO with the
            # status, not swallowed: "lenient" has to still be visible.
            stale_kept = await _remove_stale(
                client, api, base, wanted, current, proxy=proxy, headers=proxy_headers
            )

            # The read-back, and the whole reason `ok` is trustworthy. An
            # accepted POST is not a joined member: upstream probes the peer
            # first and drops it silently on timeout — and on one family the
            # POST only queues the work, which is why this waits a bounded
            # while rather than reading once.
            final = (
                await _await_registration(
                    client, api, base, wanted, proxy=proxy, headers=proxy_headers
                )
            ).entries
            if final is None:
                return MembershipOutcome(
                    ok=False,
                    unreadable=True,
                    reason="the router's member list became unreadable",
                )
            missing = sorted(set(wanted) - set(final))
            if missing:
                return MembershipOutcome(
                    ok=False,
                    reason=(
                        "the router did not admit "
                        f"{', '.join(missing)} within "
                        f"{int(_ADMISSION_WINDOW_SECONDS)}s — it health-checks "
                        "a peer before admitting it and drops it on timeout"
                    ),
                    registered=sorted(set(wanted) & set(final)),
                )
            extra = sorted(set(final) - set(wanted))
            if extra:
                # Servable, so `ok` stays True — every wanted member is in.
                # But the reason carries what did not leave, because a
                # silently-lingering member is how the encoding bug above went
                # unnoticed.
                return MembershipOutcome(
                    ok=True,
                    reason=(
                        "still registered after removal: "
                        f"{', '.join(stale_kept or extra)}"
                    ),
                    registered=sorted(final),
                )
            return MembershipOutcome(ok=True, registered=sorted(final))

        return await asyncio.wait_for(_run(), timeout=_RECONCILE_DEADLINE_SECONDS)
    except asyncio.TimeoutError:
        return MembershipOutcome(
            ok=False, reason="the router did not answer within the deadline"
        )
    except (aiohttp.ClientError, OSError) as e:
        return MembershipOutcome(
            ok=False,
            reason=f"the router is unreachable: {str(e) or e.__class__.__name__}",
        )
    finally:
        if owned and client is not None:
            await client.close()


# ---------------------------------------------------------------------------
# The recorded outcome, which is what `upstream_registration_ready` reads.
# ---------------------------------------------------------------------------

_outcomes: Dict[int, MembershipOutcome] = {}
_failures: Dict[int, int] = {}
_unreadable: Dict[int, int] = {}
# Restarts already ordered for this group without the registry ever becoming
# readable afterwards. Cleared by a successful READ, not by ordering a
# restart -- see `should_restart_router`.
_restarts: Dict[int, int] = {}

PERSISTENT_FAILURE_PASSES = 5

# How many consecutive passes the registry must be unreadable before the
# router is restarted.
#
# Not one. The shipped recipes run ONE router per group and it is the gateway's
# only upstream, so restarting it is a full-group interruption for as long as
# the replacement takes to come up and finish probing its peers -- a window
# that exists on every build, whichever way that build populates its registry
# (see fact 2 at the top of this module: that is read, not assumed). A single
# dropped request must not buy that.
#
# Five passes of a reconcile that runs on every model pass is long enough that
# a router which was merely busy, or still coming up, has answered, and short
# enough that a wedged one is not left serving nothing for minutes.
RESTART_AFTER_UNREADABLE_PASSES = 5

# How many times a group's router may be restarted over an unreadable
# registry before this stops trying.
#
# A restart repairs exactly one cause: a router process that is up but
# wedged. It cannot repair a network path that does not exist -- and that is
# the other thing "unreadable" means. On a `tunnel`-mode worker the server
# cannot dial the router at all, so the streak refills after every restart and
# an unbounded policy would churn one router per five passes indefinitely, each
# restart costing a real interruption (one router, the gateway's only upstream)
# and none of them able to help.
#
# Two, not one: the first covers the wedged process this path exists for,
# the second covers losing that race against a router that was still coming
# up. A third has nothing left to prove.
RESTART_ATTEMPT_LIMIT = 2

"""Consecutive failed reconciles before the message says where to look.

What changes at this point is not the outcome — it was already False — but
what the message is allowed to claim. One failure is ordinary and says nothing
about a cause: a router that has just come up, a member still being probed.
Five in a row is a router that is not going to admit its members on its own,
and only then is it honest to send whoever is watching to the router itself.

The escalation names evidence, not a mechanism. What the registry holds is
version-dependent (fact 2 at the top of this module), and the two causes that
survive five passes — a router that disagrees with the call, and a router this
server cannot reach — are told apart by the router's own `GET /workers` and
its log, not by anything this code can infer. In particular the message must
not claim that members can only ever join through this API — they can also
arrive from the command line — because stating a cause we cannot observe sends
people to fix the wrong thing.

And it is stated, not performed. The router's command is rendered from
`pd-modes.yaml`, so any change to how it was launched is an operator action —
this code cannot un-launch a flag on a running process, and restarting the
router only repeats the same call against the same endpoint."""


def record(model_id: int, outcome: MembershipOutcome) -> None:
    # Tracked apart from `_failures`: that counter drives the wording of a
    # persistent failure, this one drives an action.
    if outcome.unreadable:
        _unreadable[model_id] = _unreadable.get(model_id, 0) + 1
    else:
        _unreadable.pop(model_id, None)
        # A read that got through is proof the path works, so whatever the
        # earlier restarts were fighting is over and the budget is fresh.
        # Deliberately reset HERE and not when a restart is ordered: the
        # question the budget answers is "has restarting ever helped", and
        # clearing it on the attempt would make every attempt look like the
        # first one -- which is precisely the loop.
        _restarts.pop(model_id, None)

    if outcome.ok:
        _failures.pop(model_id, None)
    else:
        count = _failures.get(model_id, 0) + 1
        _failures[model_id] = count
        if count >= PERSISTENT_FAILURE_PASSES and outcome.reason:
            outcome = MembershipOutcome(
                ok=False,
                reason=(
                    f"{outcome.reason} (unchanged for {count} passes, so the "
                    "member management API is failing consistently rather "
                    "than transiently. Read the router's own member list — "
                    "`GET /workers` on the router's address — and the "
                    "router's log: which members it already holds, and what "
                    "it answered for the ones it would not take, is what "
                    "tells a router that disagrees with the call apart from "
                    "a router this server cannot reach)"
                ),
                registered=outcome.registered,
            )
    _outcomes[model_id] = outcome


def outcome_for(model_id: int) -> Optional[MembershipOutcome]:
    return _outcomes.get(model_id)


def forget(model_id: int) -> None:
    _outcomes.pop(model_id, None)
    _failures.pop(model_id, None)
    _restarts.pop(model_id, None)


def consecutive_failures(model_id: int) -> int:
    return _failures.get(model_id, 0)


def should_restart_router(model_id: int) -> bool:
    """Whether the router has been unreachable long enough to be worth losing.

    Two conditions, and the second is what keeps this from looping.

    `RESTART_AFTER_UNREADABLE_PASSES` in a row, and only for the unreadable
    case: a router that refuses a member is answering, and a restart hands it
    the same members to refuse again — a version or argument mismatch survives
    the new process.

    And `RESTART_ATTEMPT_LIMIT` restarts not yet spent. A restart repairs a
    wedged process; it cannot repair a network the server cannot cross, and
    "unreadable" covers both. Without this the streak simply refills after
    each restart and the group churns a router every five passes forever —
    each one a real interruption, none of them able to help. Measured on a
    `tunnel`-mode worker before the proxy path existed.
    """
    if _unreadable.get(model_id, 0) < RESTART_AFTER_UNREADABLE_PASSES:
        return False
    return _restarts.get(model_id, 0) < RESTART_ATTEMPT_LIMIT


def restarts_exhausted(model_id: int) -> bool:
    """Whether restarting has been tried and stopped helping.

    The caller turns this into the group's message, because at this point the
    honest statement changes: it is no longer "the router has not registered
    its members yet" but "the router cannot be reached from here, and
    restarting it did not change that".
    """
    return _restarts.get(model_id, 0) >= RESTART_ATTEMPT_LIMIT


def note_restart_ordered(model_id: int) -> None:
    """Forget the streak once a restart has been ordered, so the next pass
    measures the new process rather than re-ordering against the old count —
    and spend one of the restart budget, so a path that restarting cannot fix
    stops being restarted.
    """
    _unreadable.pop(model_id, None)
    _restarts[model_id] = _restarts.get(model_id, 0) + 1


def router_instances(instances: Sequence[ModelInstance]) -> List[ModelInstance]:
    """Every running router of the group.

    A list because the group's router count is a declared replica count like
    any other, even though the shipped recipes run one: a second router with
    an empty registry would serve 503s while the first served fine, so each
    has to be reconciled on its own.
    """
    from gpustack.schemas.models import ModelInstanceStateEnum

    return [
        i
        for i in instances
        if i.role == RoleNameEnum.ROUTER.value
        and i.state == ModelInstanceStateEnum.RUNNING
        and i.worker_ip
        and i.port
    ]


def router_addresses(instances: Sequence[ModelInstance]) -> List[str]:
    """Every running router's `host:port`.

    The address alone is not enough to REACH a router on a tunnel-mode
    worker — see `reconcile`'s `proxy` — so the caller wants
    `router_instances` and the worker behind each. Kept because the address is
    still the identity the router is known by, and reading it out of an
    instance twice would be two places to get it wrong.
    """
    return [f"{i.worker_ip}:{i.port}" for i in router_instances(instances)]
