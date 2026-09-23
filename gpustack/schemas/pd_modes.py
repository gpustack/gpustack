import re
from enum import Enum
from typing import Any, Dict, Iterator, List, Literal, Optional, Union

from pydantic import BaseModel, model_validator

from gpustack.schemas.gpu_filters import GPUFilters

_PLACEHOLDER_OCCURRENCE = re.compile(r"\{\{.*?\}\}")
"""Any {{...}} run, valid or not — the validator's discovery pass."""

_PLACEHOLDER = re.compile(
    r"^\{\{[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*\}\}$"
)
"""A well-formed placeholder: dotted identifiers, no inner spaces. The
renderer's pattern takes no spaces either, so "{{ worker_ip }}" would reach
the container verbatim (M0: a literal "{{worker_ip}}" in
VLLM_NIXL_SIDE_CHANNEL_HOST became `ZMQError: No such device`). Rejecting the
spaced form at load time is cheaper than debugging it on a worker."""


def iter_placeholders(value: Any) -> Iterator[str]:
    """Yield every {{...}} occurrence in a declaration subtree.

    Templates are nested freely (a connector descriptor carries dicts of
    dicts), so the walk is structural rather than per-field.
    """
    if isinstance(value, str):
        for match in _PLACEHOLDER_OCCURRENCE.finditer(value):
            yield match.group(0)
    elif isinstance(value, BaseModel):
        yield from iter_placeholders(value.model_dump())
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_placeholders(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_placeholders(item)


class PDInjectTargetEnum(str, Enum):
    """Where an injected value lands in the engine's launch.

    The three targets are the ``injection`` vocabulary of
    ``cache-providers.yaml``, word for word. They are not
    interchangeable per engine: NIXL's side-channel port is an env var,
    Mooncake's kv_port is a field of the connector descriptor (rendered
    into ``--kv-transfer-config``, hence ``args``), and Mooncake's
    transfer-engine config is read *only* from the JSON file
    MOONCAKE_CONFIG_PATH points at, so nothing but ``files`` can carry it.
    """

    ENV = "env"
    ARGS = "args"
    FILES = "files"


class PDPortScopeEnum(str, Enum):
    """Who a named port band belongs to (RBG's PodScoped / RoleScoped).

    hostNetwork is the locked-in default shape, so every band shipped today
    is per-instance: two members of one role on one host must not share a
    band or they collide on bind.
    """

    INSTANCE = "instance"
    ROLE = "role"


class PDPeerStyleEnum(str, Enum):
    """How a router's command line takes the group's peer addresses.

    A new engine that reuses one of these styles is YAML only. A genuinely
    new wire style (e.g. TRT-LLM's delegate protocol) needs a renderer
    branch, which is why an unknown value fails at load time instead of
    being accepted and silently rendered wrong.
    """

    REPEATED_FLAG = "repeated_flag"
    """--prefill http://a:1 --prefill http://b:2 (vLLM router, SGLang router)."""

    PARALLEL_LISTS = "parallel_lists"
    """--prefiller-hosts a b --prefiller-ports 1 2 (vllm-ascend's proxy example)."""

    USER_PROVIDED = "user_provided"
    """The user supplies image and command on the role itself."""


class PDRouterProtocolEnum(str, Enum):
    TWO_HOP = "two_hop"
    """Router calls prefill, then carries kv_transfer_params to decode."""

    SAME_DISPATCH = "same_dispatch"
    """Router dispatches to both sides and the engines rendezvous over
    their own bootstrap handshake."""

    USER_PROVIDED = "user_provided"


class PDKVLeaseTargetEnum(str, Enum):
    """Where a connector's KV lease / abort window is configured.

    Deliberately a different vocabulary from ``PDInjectTargetEnum``: the
    window is a connector-level knob, and per connector it is a field of
    ``kv_connector_extra_config``, an env var, or nothing at all.
    """

    CONNECTOR_EXTRA_CONFIG = "connector_extra_config"
    ENV = "env"
    ARGS = "args"
    FILES = "files"
    NONE = "none"
    """Not configurable — a hardcoded engine constant, or absent entirely.
    Declared anyway so the window is on the record."""


class PDKVLease(BaseModel):
    """One connector's KV lease / abort window.

    The parameter name, the default and the injection target all differ per
    connector, and the windows differ by two orders of magnitude (NIXL 30s,
    Mooncake 480s, MoRIIO 3600s), so this cannot be a platform-wide
    constant. Whether the expiry is observable differs too: NIXL exports a
    Prometheus counter, Mooncake exports none.
    """

    connector: str
    """Connector identity, and the key modes reference. Unqualified ids are
    vLLM ``kv_connector`` implementations; ``sglang-*`` ids are SGLang
    ``--disaggregation-transfer-backend`` values, whose KV lifecycle is a
    different mechanism (bootstrap timeout, not a lease)."""

    param: Optional[str] = None
    """The knob's name: a connector-config field, or an env var. None when
    the connector has no knob at all."""

    inject_to: PDKVLeaseTargetEnum = PDKVLeaseTargetEnum.NONE

    settable: bool = True
    """False for a hardcoded constant. GPUStack injects nothing and the
    declaration is accounting only."""

    engine_default: Optional[int] = None
    """The engine's own default, in seconds. None when it is not in the
    measured record — nothing is injected on a guess."""

    gpustack_default: Optional[int] = None
    """The platform's default, in seconds: one window across connectors
    instead of inheriting a 16x spread. None leaves the engine default
    alone."""

    expired_metric: Optional[str] = None
    """Prometheus metric counting expiries, if the connector exports one.
    None means the event is invisible to Prometheus and only diagnosable
    from the engine log."""

    description: Optional[str] = None

    @model_validator(mode="after")
    def check_target_agrees_with_settable(self) -> "PDKVLease":
        if self.settable and self.inject_to == PDKVLeaseTargetEnum.NONE:
            raise ValueError(
                f"kv_lease '{self.connector}' is settable but declares no "
                "injection target"
            )
        if self.settable and not self.param:
            raise ValueError(
                f"kv_lease '{self.connector}' is settable but names no parameter"
            )
        if not self.settable and self.inject_to != PDKVLeaseTargetEnum.NONE:
            raise ValueError(
                f"kv_lease '{self.connector}' is not settable, so its "
                f"inject_to must be 'none', not '{self.inject_to.value}'"
            )
        if not self.settable and self.gpustack_default is not None:
            raise ValueError(
                f"kv_lease '{self.connector}' is not settable, so a "
                "gpustack_default would never be applied"
            )
        return self


class PDPairingToggle(BaseModel):
    """An engine switch both roles of a pair must land on the same way.

    Distinct from the parameters compared in ``server/pd_pairing.py``: those
    carry a value, this is a boolean argparse spells as a pair of mutually
    exclusive flags, so what gets compared is *which flag came last*, not a
    value. That difference is why it could not go in that table.
    """

    backend: str
    key: str
    """Short name for the switch, used in the mismatch message."""

    enable_flag: str
    disable_flag: str

    default_enabled: bool = False
    """What the engine does when neither flag appears."""

    description: Optional[str] = None
    """Why the two sides diverging matters. Written for the operator who has
    to act on the mismatch."""


class PDCacheRefusal(BaseModel):
    """Roles that may not take a shared cache on this engine, and why.

    Named roles rather than the connector's ``kv_role``, because every mode has
    roles while only some declare a kv_role — the SGLang recipes configure
    their transport through ``args`` and leave the connector descriptor empty.
    A rule keyed on a field two of five shipped modes do not set would go quiet
    exactly where it cannot be seen doing so, and quiet here means the cache is
    attached to the side that dies on a hit.

    It reads directly too: this list next to the mode's own role names answers
    "may prefill take a cache, may decode" with nothing to cross-reference.
    """

    roles: List[str] = []
    """Empty refuses nothing, which is also what an absent declaration means."""

    reference: Optional[str] = None
    description: Optional[str] = None
    """Why, in the words the operator sees on the degraded instance."""


class PDComposedCache(BaseModel):
    """What an engine requires once a shared cache is folded in beside the PD
    connector, keyed by the engine that does the folding.

    A member that takes both lands them in one ``--kv-transfer-config``, and
    that composition has requirements neither side owns: not the mode's, since
    every recipe on the same engine shares them, and not the cache provider's,
    since swapping providers does not change whether the engine can compose at
    all. Declared once here rather than spelled into each mode.

    **Absence is the answer for engines that never compose.** SGLang attaches
    its cache through ``--enable-lmcache`` and a config file, so no entry is
    declared for it and nothing is enforced — rather than an entry whose
    fields are permissive enough to always pass, which would read as "checked"
    when the check is meaningless.
    """

    backend: str
    """The inference backend, spelled as ``BackendEnum`` does."""

    min_version: Optional[str] = None
    """Below this the composition is broken in a way the engine does not
    report — see ``description``. Compared with the tolerant range check, so
    an unparseable engine version fails open."""

    reference: Optional[str] = None
    """Where ``min_version`` comes from, e.g. an upstream PR."""

    description: Optional[str] = None
    """What goes wrong below ``min_version``. Surfaced in the degradation
    reason, so it is written for whoever reads that."""

    top_level_only: List[str] = []
    """Descriptor keys the engine reads only off the *top* level.

    Folding pushes each original descriptor down a level, so a key listed here
    has to be lifted back onto the composite or it silently reverts to the
    engine's default — the setting is still in the launch command, one level
    too deep to be read. Declared rather than hard-coded because it is a fact
    about the engine's config parser, and the same list decides whether a mode
    may accept the setting at all."""

    refuse_cache_on: Optional[PDCacheRefusal] = None
    """Roles that may not take a shared cache. Absent means every role may."""


class PDTransferMetrics(BaseModel):
    """What one connector's KV transfer means, not what it is called.

    **The metric NAMES are not here.** They live in `metrics_config.yaml`,
    where the worker's aggregator normalizes every engine's own spelling — and
    its own units — onto one set of `gpustack:pd_*` names. A second copy here
    would be two files to edit for one rename, and one of them read by nothing.

    What is here is the part that is genuinely about the connector and cannot
    be normalized away, because it is semantics rather than spelling.
    """

    connector: str
    """The key modes resolve through, the same id ``kv_leases`` uses."""

    read_from_role: str = "decode"
    """Which side of the pair owns the transfer counters. **Both values are
    in use — this is not a formality with one real answer.**

    The rule is "whichever side moves the bytes", and the two engines differ:

    - vLLM's NIXL *pulls*: decode reads from prefill, so decode counts.
      Measured on a working 1P1D — prefill's transfer count stayed at 0.0 for
      the whole run while decode's rose with every request.
    - SGLang *pushes*: prefill writes into slots decode registered through the
      bootstrap service, so prefill counts. Its byte and speed counters exist
      only on prefill for exactly this reason.

    Get it backwards and a healthy pair reports "no KV ever moved" — the exact
    failure this metric exists to detect, fired at a deployment that is fine.
    The default is ``decode`` only because NIXL came first, not because it is
    the normal case.

    This governs the **connector's own** counters and nothing else. The
    engine's token accounting is read on the receiving side no matter who
    moved the bytes, which coincides with this only because every connector
    that exports the token split happens to pull."""


class PDPortSpec(BaseModel):
    """A named port band a role needs allocated.

    ``[kv_side_channel]`` is shorthand for
    ``{name: kv_side_channel, count: 1, inject_to: env}``.
    """

    name: str

    count: Union[int, str] = 1
    """Band width: an integer, or a single placeholder resolved at
    allocation time. The width is decided by the connector, not by a
    platform formula — Mooncake's kv_port is a base address and the connector
    binds one port per *worker rank*, so the band is the member's card count
    (``{{accelerator_count}}``; measured TP8/DP1 -> 41100-41107 and DP2xTP2 ->
    20001-20004), while NIXL's side channel is offset per DP index instead."""

    inject_to: PDInjectTargetEnum = PDInjectTargetEnum.ENV
    """Which injection channel carries the band's base. Cross-checked at
    load time against where {{ports.<name>}} actually appears, so a wrong
    declaration cannot sit there looking plausible."""

    scope: PDPortScopeEnum = PDPortScopeEnum.INSTANCE

    @model_validator(mode="before")
    @classmethod
    def expand_shorthand(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"name": value}
        return value

    @model_validator(mode="after")
    def check_count(self) -> "PDPortSpec":
        if isinstance(self.count, str):
            if not _PLACEHOLDER.match(self.count):
                raise ValueError(
                    f"port '{self.name}' count '{self.count}' is neither an "
                    "integer nor a single placeholder"
                )
        elif self.count < 1:
            raise ValueError(f"port '{self.name}' count must be >= 1")
        return self


class PDModeRole(BaseModel):
    """What one role of a mode needs allocated and injected.

    ``connector`` is the only part that is not a cache-provider-shaped
    injection: it stays a structured descriptor because that is already the
    shape the extended-KV-cache assembler consumes, so the PD side needs no
    string surgery. Everything in it renders into ``--kv-transfer-config``,
    which is why a port carried in the descriptor declares
    ``inject_to: args``.
    """

    ports: List[PDPortSpec] = []
    connector: Dict[str, Any] = {}
    env: Dict[str, str] = {}
    args: List[str] = []
    files: Dict[str, str] = {}
    host_mounts: List[str] = []
    """Host paths the engine container needs read-only, same path inside.

    A fifth injection channel, and the one the other four cannot stand in
    for: a transport that reads a host file the accelerator runtime does not
    inject needs that file *present*, and `files` writes contents GPUStack
    composes rather than passing through something only the host knows.

    On Ascend, vllm-ascend's Mooncake uses AscendDirectTransport, which
    resolves a peer's host address to the per-card RoCE addresses through
    `/etc/hccn.conf`. The Ascend container runtime injects the devices and the
    driver but not that file, so without the mount cross-host
    `batch_transfer_sync_read` fails on every rank while same-host transfers
    succeed — and the connector logs the exception, sends its done signal
    anyway and lets decode answer 200 with garbage.
    vllm-ascend's own multi-node guide says it plainly: *"Ensure that the
    hccn.conf file exists in the environment. If using Docker, mount it into
    the container."*

    Declared per recipe rather than mounted for every Ascend deployment: a
    non-PD model on the same cards needs nothing of the kind, and the reason
    this path needs it belongs next to the transport that reads it.
    """

    @model_validator(mode="after")
    def check_port_names_unique(self) -> "PDModeRole":
        names = [port.name for port in self.ports]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate named ports: {duplicates}")
        return self

    def channel(self, target: PDInjectTargetEnum) -> Any:
        """The declaration subtree an inject target renders into. ``args``
        covers the connector descriptor as well: it is rendered into one."""
        if target == PDInjectTargetEnum.ENV:
            return self.env
        if target == PDInjectTargetEnum.FILES:
            return self.files
        return [self.args, self.connector]

    @property
    def port_names(self) -> List[str]:
        return [port.name for port in self.ports]


class PDRouterPeers(BaseModel):
    """How the group's peer addresses reach the router's command line."""

    style: PDPeerStyleEnum

    prefill: Dict[str, str] = {}
    """Style-specific rendering: repeated_flag takes {flag, value} with
    {{peer.ip}} / {{peer.port}} per peer; parallel_lists takes {host_flag,
    port_flag}."""

    decode: Dict[str, str] = {}


class PDRouterCapabilities(BaseModel):
    """What the router actually serves.

    Every field defaults to False: an undeclared endpoint must be treated
    as absent, not assumed present. vllm-ascend's proxy example serves neither
    /metrics nor /v1/models, and polling them produces a ~1/s 404 storm in the
    router log plus a permanent false alarm.

    No shipped mode declares these False: every catalog recipe launches a
    first-party router that serves both. The default stays False for `custom`,
    where the router is the user's and nothing is known about it.
    """

    metrics: bool = False
    """Serves a Prometheus exposition. False means health checks degrade to
    process liveness plus a port probe, and the PD-effectiveness ratio
    loses its denominator."""

    models_endpoint: bool = False
    """Serves /v1/models."""

    kv_expired_metric: bool = False
    """The engine side of this mode exports a KV-lease-expiry counter.
    Mirrors ``kv_lease.expired_metric``; the loader asserts they agree."""


class PDMembershipAPI(BaseModel):
    """How a router is told its peer list changed, without restarting it.

    Why this is worth a schema at all: the router is a single replica and the
    gateway's only upstream, so restarting it to re-render peers is a full
    outage for the group. Every change of ratio would cost one. An HTTP
    membership call turns that into no interruption at all.

    **Every field is optional, and all-empty is a legitimate declaration** —
    it means "this router has no such API, fall back to re-render and restart".
    Reading a missing API as a bug would make the fallback path look like a
    failure.

    **Declaring it is not the same as it working**, which is why
    `requires_args` exists. `POST /workers` carries `worker_type`, but the
    single-router path drops that field before it reaches the PD router, which
    then returns a hardcoded "requires specific add_prefill_server or
    add_decode_server methods"; `--enable-igw` routes the call through the
    manager that dispatches on `worker_type`.

    **The flag is in the shipped recipes.** Under `--enable-igw`, `POST
    /workers` with `worker_type` enters the registry and takes traffic, and
    `DELETE` removes a member with a request in flight. A consumer should still
    treat a failed call as "fall back to restart" rather than "the group is
    broken" — that is a property of a single-replica router.

    **An empty `requires_args` is a finding of its own, not an omission.**
    The gateway that fork descends from needs no such flag: its command-line
    peers and its API go through the same registration job, and its PD router
    reads the shared registry per request, so membership works from the start.
    Two routers of shared ancestry disagreeing about this is exactly why the
    declaration is
    per-recipe rather than one rule in code.
    """

    add: Optional[str] = None
    """`"METHOD /path"`, e.g. `"POST /workers"`. Method included because the
    two shapes upstream ships differ in it: a REST resource (`POST /workers`)
    versus an action endpoint (`POST /instances/add`)."""

    remove: Optional[str] = None
    """`"METHOD /path"`, with `{url}` or `{id}` substituted for the peer being
    removed. Both are percent-encoded by the consumer.

    **`{id}` is the one that works whichever way the router names a member.**
    It takes whatever the `probe` reported for that member — the URL where the
    router keys on the address, a registry-generated UUID where it does not —
    so one declaration covers both, and a router of the second kind rejects a
    URL in this position. Use `{url}` only where the router is known to address
    members that way for good."""

    probe: Optional[str] = None
    """Read-back for reconciliation. The one field that is not optional in
    practice: an accepted `add` does not mean the member is in the registry —
    upstream implementations poll the peer first and drop it silently on
    timeout, and the default windows are tuned for small models. Without a
    read-back, a scale-out that quietly failed looks identical to one that
    worked."""

    body: Optional[Dict[str, str]] = None
    """Body template for `add`. Values may carry the same `{{...}}`
    placeholders the rest of the catalog uses — `{{peer.url}}`, `{{peer.ip}}`,
    `{{peer.port}}`, `{{peer.ports.<band>}}`, `{{model_name}}`.

    Two rules the consumer applies, both of which a router refuses the member
    over if they are got wrong:

    * **A bare placeholder keeps its type.** `"{{peer.ports.bootstrap}}"`
      becomes the integer 9002, because upstream declares that field a `u16`
      and answers `invalid type: string "9002", expected u16` otherwise. A
      template with any literal text around it renders to a string, as it must.
    * **A placeholder that does not resolve drops its key.** A decode has no
      bootstrap band, and sending `"bootstrap_port": null` would be a claim
      about a port rather than silence about one. So one `body` can describe
      both roles without a per-role section."""

    role_field: Optional[str] = None
    """Which body key carries the role. Named rather than assumed because it
    is the field the single-router path drops (see the class note), so a
    consumer needs to know what to check for in the read-back."""

    role_values: Optional[Dict[str, str]] = None
    """GPUStack role name -> the value this router expects. Not an identity
    map: a router may call them `prefill`/`decode` or something else, and the
    router role itself has no membership at all."""

    requires_args: List[str] = []
    """Launch flags without which the API exists but does not work. Empty is
    the normal case; a non-empty list means the recipe's `command` must already
    carry them, and a mismatch is a catalog bug rather than a runtime one."""

    @property
    def available(self) -> bool:
        """Whether scale-out can go through the API at all.

        Both `add` and `probe`: an add with no read-back cannot be reconciled,
        and the design's own rule is "send explicitly, read back, fall back to
        restart on failure" — two of those three need this pair.
        """
        return bool(self.add and self.probe)


class PDTunableArg(BaseModel):
    """One router flag the deployment may legitimately change.

    Declared rather than inferred because the alternative — a hardcoded list
    of "flags a user may touch" — has to be edited in Python every time a
    recipe is added, which is the thing the catalog exists to avoid.

    ``options`` and ``min`` / ``max`` are for rendering, not enforcement: they
    let a form offer a select or a bounded number field instead of a bare text
    box. A value outside them is still submitted — the two shipped routers
    disagree about their own strategy sets between wheel and repository at the
    same version number, so a closed list here would forbid a flag the
    installed binary accepts.
    """

    flag: str
    """Long form with the dashes, e.g. ``--decode-policy``."""
    default: Optional[str] = None
    """What the catalog passes when the deployment says nothing. Rendered into
    the command, so it is also what the read-only view shows."""
    options: List[str] = []
    """Known-good values, for a select. Advisory — see the class docstring."""
    value_type: Literal["string", "int", "float"] = "string"
    min: Optional[float] = None
    max: Optional[float] = None
    description: Optional[str] = None

    @property
    def tokens(self) -> List[str]:
        """The flag and its default, as command tokens. Empty when there is no
        default: a declared knob with nothing to pass is a knob, not an
        argument."""
        return [self.flag, self.default] if self.default is not None else []


class PDRouter(BaseModel):
    """The router role of a mode. There is no universal router — the
    catalog format is what generalizes, not the binary.

    **The invocation is declared in three parts, not one string.** They mean
    three different things to the deployment, and a single ``command`` list
    could not say which was which:

    ``entrypoint``
        Which executable inside the image. Same image as the model's — the
        difference between a router and an engine is the binary, not the
        image — so this is the one line that says what actually runs.
    ``connection_args``
        Addresses, ports and the transport handshake. Rendered from placement
        facts the deployment does not have, so GPUStack fills them in — but a
        deployment may now override one by setting the same flag, which is
        appended after the declared command and wins by last-wins, exactly as
        for ``tunable_args``. What stays refused is a peer flag: ``--prefill``
        and ``--decode`` are ``action="append"`` in both shipped routers, so a
        second one does not replace the injected peer, it adds a phantom one
        the router then forwards to.
    ``tunable_args``
        Strategy and resilience defaults. Overridable, because repeated flags
        are last-wins for every one of them (verified against both wheels:
        ``--decode-policy round_robin --decode-policy cache_aware`` parses to
        ``cache_aware``). A deployment's own parameters are appended after
        these, which is what makes "append" and "override" the same gesture.

    ``command`` stays readable as the whole invocation — it is composed from
    the three parts at load time, so every existing consumer (the renderer, the
    read-only view) keeps seeing one list.
    """

    protocol: PDRouterProtocolEnum

    image: Optional[str] = None
    ports: List[PDPortSpec] = []
    entrypoint: List[str] = []
    connection_args: List[str] = []
    tunable_args: List[PDTunableArg] = []
    command: List[str] = []
    """The full invocation. Composed from the three parts above unless given
    directly; giving both is refused at load time, because then two places
    would describe the same command line and only one of them would be read."""
    env: Dict[str, str] = {}
    """Environment a router needs in order to start at all.

    Not an injection channel like a role's — see ``channel`` — but part of the
    invocation, in the same sense the command is: values that decide whether
    the binary comes up, not values that configure KV transfer. vllm-ascend's
    proxy is the case that forced it. Its first import reaches
    ``vllm.logger``, which pulls in torch and then torch_npu, and torch_npu
    raises without an accelerator visible:

        RuntimeError: Failed to load the backend extension: torch_npu

    A router is ``cpu_only`` by design, so this lands on exactly the role that
    must not hold a card. ``TORCH_DEVICE_BACKEND_AUTOLOAD=0`` is the upstream
    escape hatch and was measured to fix it. Declaring it here keeps
    "which router needs what to boot" in the same file as the rest of that
    router's contract; the alternative was a user typing it on a role whose
    other fields GPUStack fills in.
    """

    peers: Optional[PDRouterPeers] = None
    capabilities: PDRouterCapabilities = PDRouterCapabilities()
    membership_api: PDMembershipAPI = PDMembershipAPI()

    @property
    def membership_api_usable(self) -> bool:
        """Whether scale-out can go through the API *as this recipe launches it*.

        Deliberately not an assertion in the loader. `membership_api` records
        the shape upstream serves; `command` records what we actually start.
        Those two legitimately disagree while a flag is declared but unverified
        — which is today's state for `--enable-igw` — and a loader that refused
        the mismatch would force the choice between deleting the knowledge and
        enabling an untested flag.

        So the mismatch gets a name instead. A consumer asking "can I add a
        member without an outage" reads this and gets False, and it flips to
        True the moment the flag is added to the recipe, with nothing else to
        change.
        """
        if not self.membership_api.available:
            return False
        rendered = " ".join(self.command or [])
        return all(flag in rendered for flag in self.membership_api.requires_args)

    health_path: Optional[str] = None
    """None means the router serves no health endpoint, so readiness falls
    back to process liveness."""

    @property
    def platform_owned_flags(self) -> List[str]:
        """The flag names a deployment may not set — the blacklist, derived.

        **Only the peer flags.** They live in ``peers`` rather than in
        ``connection_args`` because the renderer appends one per member after
        the declared command, and they are here because they are the one part
        a user value cannot override: ``--prefill`` and ``--decode`` are
        ``action="append"`` in both shipped routers, so a second one does not
        replace the injected peers — it adds one the router forwards to and
        cannot reach, and that member simply never gets traffic.

        ``connection_args`` is deliberately not in here, for the same reason
        ``tunable_args`` is not: repeated flags are last-wins for all of them,
        a deployment's own parameters are appended after the declared command,
        and so setting one simply overrides it. Listing them would make the
        router's connection flags the only injected parameters a user cannot
        edit, while the rest of the form says "GPUStack fills this in, and you
        may change it". Breaking the router by mis-editing ``--port`` is the
        user's to own.

        Only long-form tokens would have counted anyway: a value that happens
        to start with ``--`` would be a value, not a flag.
        """
        flags: List[str] = []
        if self.peers is not None:
            for spec in (self.peers.prefill, self.peers.decode):
                for key in ("flag", "host_flag", "port_flag"):
                    value = spec.get(key)
                    if value and value.startswith("--"):
                        flags.append(value)
        return flags

    @model_validator(mode="after")
    def compose_command(self) -> "PDRouter":
        """Build ``command`` from the three declared parts.

        Runs before ``check_user_provided`` reads ``command``, which is why the
        composition lives in its own validator rather than inside that one:
        field validators run in declaration order, and the check needs a
        composed value to check.
        """
        parts = self.entrypoint + self.connection_args
        for arg in self.tunable_args:
            parts.extend(arg.tokens)
        # Equal is not "both": `model_dump()` emits the parts *and* the
        # composed command, and the endpoint serves that dump back — so
        # re-validating one's own output has to be a no-op. Only a `command`
        # that disagrees with the parts is two descriptions of one command
        # line, and only that is worth refusing.
        if parts and self.command and self.command != parts:
            raise ValueError(
                "a router's `command` disagrees with its entrypoint / "
                "connection_args / tunable_args — declare the invocation in "
                "one place, because only one of the two would be read"
            )
        if parts:
            self.command = parts
        flags = [arg.flag for arg in self.tunable_args]
        duplicates = sorted({flag for flag in flags if flags.count(flag) > 1})
        if duplicates:
            raise ValueError(f"duplicate tunable flags on the router: {duplicates}")
        overlap = sorted(set(flags) & set(self.platform_owned_flags))
        if overlap:
            raise ValueError(
                f"flags declared both platform-owned and tunable: {overlap} — "
                "a flag the platform renders from placement facts cannot also "
                "be offered for editing"
            )
        return self

    @model_validator(mode="after")
    def check_user_provided(self) -> "PDRouter":
        user_provided = self.protocol == PDRouterProtocolEnum.USER_PROVIDED
        if user_provided and (self.command or self.image or self.ports):
            raise ValueError(
                "a user_provided router takes its image, command and ports "
                "from the role spec, so the catalog must not declare them"
            )
        if not user_provided and not self.command:
            raise ValueError(f"a {self.protocol.value} router needs a command")
        if not user_provided and self.peers is None:
            raise ValueError(
                f"a {self.protocol.value} router needs a peers declaration"
            )
        names = [port.name for port in self.ports]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate named ports on the router: {duplicates}")
        return self

    def channel(self, target: PDInjectTargetEnum) -> Any:
        """A router is mostly a command line: the peer templates are part of
        the args, and it writes no files. Its ``env`` is answered here so that
        a router port band declared ``inject_to: env`` is cross-checked like
        any other — the env reaches the container through the router's
        workload rather than through the injector, but a band that names
        itself there is still consumed there."""
        if target == PDInjectTargetEnum.ENV:
            return self.env
        if target == PDInjectTargetEnum.FILES:
            return {}
        return [self.command, self.peers]

    @property
    def port_names(self) -> List[str]:
        return [port.name for port in self.ports]


class PDTensorParallelPairingEnum(str, Enum):
    """Which way a prefill/decode tensor-parallel mismatch is allowed to go.

    The direction is the connector's, not PD's. vLLM's NixlConnector maps
    each decode rank onto a slice of one prefill rank and asserts "Decode TP
    cannot be smaller than prefill TP" (measured: it surfaces as an
    IndexError inside decode). vllm-ascend's Mooncake connector carries both
    sides' tp/dp in its config precisely so a decode rank can gather from
    several prefill ranks — Huawei's reference deployment is prefill TP4 /
    decode TP1, the exact shape the NIXL rule forbids. One rule for every
    recipe therefore rejects a working Ascend deployment, so the rule is
    declared per recipe.
    """

    DECODE_GE_PREFILL = "decode_ge_prefill"
    """decode's tensor parallelism must be at least prefill's (NIXL)."""
    PREFILL_GE_DECODE = "prefill_ge_decode"
    """prefill's tensor parallelism must be at least decode's."""
    ANY = "any"
    """No constraint GPUStack knows of; the engine is the judge."""


class PDPairing(BaseModel):
    """Cross-role constraints the recipe's connector imposes.

    Only the ones that differ between connectors live here. Factors every
    connector hashes on contact (dtype, block size, KV layout) and the one
    nothing checks (`max_model_len`) are the same for all recipes and stay in
    the validation code.
    """

    tensor_parallel: PDTensorParallelPairingEnum = (
        PDTensorParallelPairingEnum.DECODE_GE_PREFILL
    )
    """Default is the NIXL rule, which every shipped recipe satisfies; a recipe
    whose connector differs declares so."""


class PDNetDevicePlaneEnum(str, Enum):
    """What the NIC `{{net_device}}` names actually carries in this recipe.

    The placeholder is one name injected into variables of two different
    natures, and the right answer differs by *kind*, not by degree:

    - **data** — `UCX_NET_DEVICES` (both NIXL recipes). This NIC moves the KV
      bytes. The management NIC is usually the wrong fabric, and picking it
      silently fails at the far end of the handshake with `NIXL_ERR_BACKEND`,
      so a multi-NIC host must refuse to guess and ask for `kv_ifname`.
    - **control** — `HCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` /
      `TP_SOCKET_IFNAME` (`vllm-ascend-mooncake`). These carry the handshake
      sockets only; the data plane rides the cards' own RoCE ports, which the
      host stack never sees. The answer is not a guess at all: `HCCL_IF_IP` is
      the worker's registration IP, HCCL requires it to sit on the NIC named by
      `HCCL_SOCKET_IFNAME`, and `Worker.ifname` is *by definition* the NIC that
      carries worker->server traffic. Refusing to derive it there costs an
      operator a mandatory manual value AND adds a failure mode — a hand-typed
      NIC that does not hold `HCCL_IF_IP` is harder to diagnose than a wrong
      card. On an Ascend host the 25G ports physically on the RDMA card look
      like the knowledgeable choice, while the right answer is the management
      NIC.

    Declared per recipe rather than branched on the mode name in code: the next
    Ascend-family recipe would be missed by an if-else and silently inherit the
    refusal.
    """

    DATA = "data"
    CONTROL = "control"


class PDMode(BaseModel):
    """One disaggregation recipe: engine, KV connector and router.

    The user picks one of these from a single dropdown; every
    connection-state parameter (connector, ports, handshake variables)
    comes from here and none of it reaches the form.
    """

    name: str
    """Must equal a ``PDModeEnum`` value verbatim — the catalog is looked up
    by it. The loader asserts the two sets are equal at load time."""

    display_name: Optional[str] = None
    description: Optional[str] = None

    backends: List[str] = []
    """Inference backends this recipe may be injected into, in
    ``BackendEnum`` spelling. A recipe expands into one engine's connector
    config, so a role on another engine would be handed configuration it
    cannot read. Empty means unconstrained (only ``custom``, which injects
    nothing). ``PD_MODE_BACKENDS`` must agree; the loader asserts it."""

    backend_versions: Optional[str] = None
    """Engine versions this recipe is known to work against, in
    ``version_in_range`` spelling (e.g. ``">=0.5.7"``).

    **Reported, never refused** — unlike a cache provider's ``versions``, which
    is ENFORCED, with a 400 at ``create_model``. That one can be, because
    an out-of-range engine there is handed injected args it cannot parse and
    never starts, so the refusal costs a deployment that was not going to run.
    A recipe's floor is different in kind: the group runs either way, and the
    number may belong to a self-built image whose private version carries the
    fix. So a version outside this range sets the
    ``engine_version_below_recipe_floor`` degradation on the model and nothing
    is rejected.

    It is not decorative either. The ``>=0.5.7`` on the SGLang recipes exists
    because a member's id became a registry-minted UUID at that version, so an
    older build answers 400 to ``DELETE /workers/{url}`` and a scaled-down
    member keeps taking traffic.

    Unparseable and unpinned versions fail open, the same way the cache check
    does: the marker is set only for a version positively shown to be out of
    range. None means the recipe declares no floor, never "any version is
    fine"."""

    gpu_filters: Optional[GPUFilters] = None
    """Which accelerators this recipe may be injected into.

    A positive declaration is the only shape that expresses both directions.
    An "unconstrained or not declared" sentinel could say "Ascend only" but not
    "NVIDIA only", and since no AMD recipe ships that gap is a real defect:
    three NVIDIA recipes would be offered on AMD clusters.

    ``None`` is reserved for ``custom``, which injects nothing and must stay
    selectable on every accelerator -- an unsupported pair means "no built-in
    recipe", never "no PD".
    """

    transport: Optional[str] = None
    """The KV transport this recipe uses, e.g. "NIXL" or "Mooncake".

    Separate from ``display_name`` because the two are read in different
    places. The picker lists recipes for several engines side by side, so
    there it has to say *which* NIXL ("vLLM + NIXL" vs "SGLang + NIXL"). The
    derived one-liner is shown after the engine has already been chosen and
    named, so repeating it there is noise -- the only new fact is the
    transport.

    None for ``custom``, which has no transport of its own: the user supplies
    the connector.
    """

    preferred: bool = False
    """Pick this one when several recipes fit the same engine and accelerator.

    Exactly one cell needs it today: SGLang on NVIDIA, where Mooncake and NIXL
    both work and Mooncake is the answer. Keeping the tie-break in the catalog
    means neither the API nor the UI has to hold a "which one is better" rule.
    """

    roles: Dict[str, PDModeRole] = {}
    """Engine roles by name. Empty injects nothing (``custom``). Role names
    are deliberately unconstrained here: the API validation layer decides
    which roles are admissible, so opening up encoder / draft later stays a
    validation change, not a catalog schema change."""

    router: Optional[PDRouter] = None

    kv_lease: Optional[PDKVLease] = None
    """Resolved by the loader from the catalog's per-connector registry.
    None means GPUStack configures no window (``custom``)."""

    transfer_metrics: Optional[PDTransferMetrics] = None
    """Resolved by the loader from the same connector id ``kv_lease``
    names — the connector identifies the transport, and the transport is
    what decides both the lease window and the counters. None means the
    mode has no connector GPUStack knows (``custom``), so PD effectiveness
    is not measurable for it."""

    pairing: PDPairing = PDPairing()
    """Connector-specific cross-role constraints, read by admission. Absent
    means the NIXL defaults — see ``PDPairing``."""

    net_device_plane: PDNetDevicePlaneEnum = PDNetDevicePlaneEnum.DATA
    """Which plane this recipe's ``{{net_device}}`` rides — see
    ``PDNetDevicePlaneEnum``.

    The default is ``data`` because that is the stricter of the two: a recipe
    that forgets to declare its plane keeps the multi-NIC refusal, which costs
    an operator one ``kv_ifname``. The reverse default would put KV bytes on
    the management NIC of every unclassified recipe."""

    def role(self, name: str) -> Optional[PDModeRole]:
        return self.roles.get(name)

    @model_validator(mode="after")
    def check_templates(self) -> "PDMode":
        """Reject a declaration whose placeholders cannot resolve.

        Structural only, on purpose: there is no allowlist of placeholder
        names, because a new engine needing a new variable must stay a YAML
        change. What is checked is that every placeholder is well formed,
        that {{ports.<name>}} names a band the same declaration allocates
        and appears in the channel the band declares, and that
        {{roles.<role>...}} names a role the mode declares.
        """
        declared_roles = set(self.roles)
        allocated = {
            port.name for holder in self._holders_only() for port in holder.ports
        }
        for scope, holder in self._holders():
            for occurrence in iter_placeholders(holder):
                if not _PLACEHOLDER.match(occurrence):
                    raise ValueError(
                        f"{scope}: malformed placeholder {occurrence} "
                        "(dotted identifiers only, no inner spaces)"
                    )
                parts = occurrence[2:-2].split(".")
                if parts[0] == "roles":
                    if len(parts) != 3:
                        raise ValueError(
                            f"{scope}: cross-role reference {occurrence} must be "
                            "{{roles.<role>.<field>}}"
                        )
                    if parts[1] not in declared_roles:
                        raise ValueError(
                            f"{scope}: cross-role reference {occurrence} names "
                            f"undeclared role '{parts[1]}'"
                        )
                elif parts[0] == "ports":
                    if len(parts) not in (2, 3) or (
                        len(parts) == 3 and parts[2] != "count"
                    ):
                        raise ValueError(
                            f"{scope}: port reference {occurrence} must be "
                            "{{ports.<name>}} or {{ports.<name>.count}}"
                        )
                    # A router reads another role's band (SGLang's router
                    # takes prefill's bootstrap port as a positional
                    # argument), so the band only has to exist somewhere in
                    # the mode.
                    if parts[1] not in allocated:
                        raise ValueError(
                            f"{scope}: {occurrence} references a port band "
                            "nothing in this mode allocates"
                        )
            # And every allocated band must be consumed where it says it is:
            # an unreferenced band is a port taken out of a 64-port pool for
            # nothing, and a band referenced from the wrong channel is the
            # NIXL-vs-Mooncake mistake this schema exists to prevent.
            for spec in holder.ports:
                references = {
                    "{{ports." + spec.name + "}}",
                    "{{ports." + spec.name + ".count}}",
                }
                found = set(iter_placeholders(holder.channel(spec.inject_to)))
                if not references & found:
                    raise ValueError(
                        f"{scope}: port band '{spec.name}' declares inject_to "
                        f"'{spec.inject_to.value}' but nothing there references it"
                    )
        return self

    @model_validator(mode="after")
    def check_membership_body_bands(self) -> "PDMode":
        """A membership body may only name a port band some role allocates.

        Checked here rather than left to runtime because this is the one
        template whose failure is *silent*. Everywhere else an unresolved
        placeholder reaches the process verbatim and the engine dies with it in
        the message; a membership value that cannot resolve is dropped from the
        body instead, deliberately — that is what lets one ``body`` serve a
        prefill that has a bootstrap band and a decode that has none. So a
        typo'd band name would not fail, it would register a prefill without
        its ``bootstrap_port``, and every request routed to that member would
        hang with nothing anywhere saying why.
        """
        if self.router is None or not self.router.membership_api.body:
            return self
        allocated = {
            port.name for holder in self._holders_only() for port in holder.ports
        }
        for key, template in self.router.membership_api.body.items():
            for occurrence in iter_placeholders(str(template)):
                parts = occurrence[2:-2].split(".")
                if parts[:2] != ["peer", "ports"]:
                    continue
                if len(parts) != 3 or parts[2] not in allocated:
                    raise ValueError(
                        f"router membership_api.body['{key}']: {occurrence} "
                        "references a port band nothing in this mode allocates"
                    )
        return self

    def _holders(self):
        for name, role in self.roles.items():
            yield f"role '{name}'", role
        if self.router is not None:
            yield "router", self.router

    def _holders_only(self):
        return [holder for _, holder in self._holders()]


class PDModeCatalog(BaseModel):
    """The whole catalog document.

    ``kv_leases`` is keyed by connector and is the source each mode's
    resolved ``kv_lease`` came from. It survives resolution because an entry
    can legitimately have no mode: MoRIIO's window is on the record even
    though no shipped recipe uses it and GPUStack cannot change it.
    """

    kv_leases: Dict[str, PDKVLease] = {}
    composed_cache: Dict[str, PDComposedCache] = {}
    """Keyed by backend. An engine with no entry composes without a declared
    requirement — see `PDComposedCache` for why absence is deliberate."""

    pairing_toggles: List[PDPairingToggle] = []
    """Boolean engine switches both roles must agree on, across every mode of
    the backend that declares them."""

    kv_transfer_metrics: Dict[str, PDTransferMetrics] = {}
    """Keyed by the same connector id as ``kv_leases``, and required to
    cover every connector that appears there: an entry whose fields are all
    null declares "this connector exports nothing", which is a measured
    fact about Mooncake, while a *missing* entry would be an oversight that
    reads identically at runtime."""

    modes: List[PDMode] = []

    def mode(self, name: str) -> Optional[PDMode]:
        for mode in self.modes:
            if mode.name.lower() == (name or "").lower():
                return mode
        return None
