import json
import re
from enum import Enum
from functools import cmp_to_key
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, model_validator

from gpustack.utils.version import compare_versions, pick_runtime_version

CUSTOM_VERSION = "custom"
"""Reserved provider_version identifier: the service pins a user-supplied
container image (config.image) instead of a declared version."""

CPU_BACKEND = "cpu"
"""The accelerator key of a node that has none, spelled as the rest of the
platform spells it — the framework token a worker with no GPU is matched
against when its inference backend picks an image."""

LocalizedText = Union[str, Dict[str, str]]
"""A user-facing string, either bare or keyed by locale.

A bare string is the text in every locale. A mapping carries one entry per
locale plus the required "default" fallback:

    description:
      default: XSKY MeshFusion Store distributed storage as the L2 tier.
      zh-CN: 以 XSKY MeshFusion Store 分布式存储作为 L2 层。

Only display text is localizable. Identity a declaration renders or
validates against — field ``name``, ``options`` values, catalog keys — stays
verbatim, so a translated catalog produces byte-identical connector config.

The API serves the mapping as declared and the UI resolves it against the
locale the user picked, which the request carries no reliable signal of.
"""

DEFAULT_LOCALE = "default"
"""Locale key every localized mapping must carry: what the UI falls back to
for a locale the declaration does not translate, and the text non-UI
consumers (catalog search, logs) read."""

# BCP 47 shape, loose enough to admit any real tag: a two- or three-letter
# primary subtag (ISO 639-1 "zh", 639-2/3 "yue") plus any number of script,
# region or variant subtags. Its job is to catch a typo like "ZH_cn", which
# would otherwise be text that renders for nobody — a tag it rejects costs
# the whole provider, so it must not reject one a UI could legitimately ask
# for.
LOCALE_PATTERN = re.compile(r"^(default|[a-z]{2,3}(-[A-Za-z0-9]+)*)$")


def localized_default(value: Optional[LocalizedText]) -> Optional[str]:
    """The declaration's fallback text: the bare string, or the mapping's
    "default" entry."""
    if isinstance(value, dict):
        return value.get(DEFAULT_LOCALE)
    return value


def localized_values(value: Optional[LocalizedText]) -> List[str]:
    """Every translation of a localized string, for consumers matching
    against text in a locale they do not know (catalog search)."""
    if isinstance(value, dict):
        return [text for text in value.values() if text]
    return [value] if value else []


class CacheProviderSourceEnum(str, Enum):
    BUILT_IN = "built_in"
    COMMUNITY = "community"
    PARTNER = "partner"


class CacheProviderLink(BaseModel):
    """A brand link (docs, product page, support) rendered on the
    provider's catalog card."""

    label: LocalizedText
    url: str


DEFAULT_PORT_NAME = "port"
"""Name of the port a component answers on when it declares none: the
one every cache server has, the one its address is built from."""

DEFAULT_METRICS_PORT_NAME = "metrics"
"""Name of the port a Prometheus exposition conventionally sits on, and
the second port a component that declares none is given."""

IMPLICIT_PORT_NAMES = (DEFAULT_PORT_NAME, DEFAULT_METRICS_PORT_NAME)
"""What a component binds without saying so: a service port and a
metrics port, which is what a cache server usually is. A component that
declares ``ports`` replaces this list outright — that is how a role with
no HTTP surface stops holding a metrics listener on every worker it runs
on, and how one with three listeners describes all three."""


class CacheProviderHealthCheck(BaseModel):
    scheme: str = "tcp"
    """Probe scheme: "tcp" (connect check) or "http" (GET on path)."""

    path: Optional[str] = None
    """HTTP path for scheme "http". Ignored for "tcp"."""

    target: Optional[str] = None
    """Name of the port the probe hits, one the component declares; None
    hits the port the component is addressed by. Declared where readiness
    does not live where the component serves — LMCache's /healthcheck is
    on its HTTP frontend, not on the ZMQ port engines use."""


class CacheProviderVersionConfig(BaseModel):
    image: Optional[str] = None
    """Container image for the managed cache server: the one a declaration
    names when it has a single build for everything, and what a node with
    no accelerator runs when runtime_images has no "cpu" entry. A declared
    version resolving to no image at all is a catalog error; a version
    derived from the runner images has none, since every image there is
    built for an accelerator and the device-free one is a runtime_images
    key like the rest."""

    runtime_images: Dict[str, Dict[str, str]] = {}
    """Images keyed by accelerator backend (e.g. "cuda") then runtime
    version (a major like "12" or a full "12.6"). The worker resolves per
    node at instance start with the platform-wide runtime-match rule
    (see pick_runtime_version), so a heterogeneous per_node fleet mixes
    images correctly. All entries must be command-compatible with
    run_command.

    A backend whose SoC generations need their own build takes a
    "<backend>-<variant>" key (e.g. "cann-910b"), matching the runner
    catalog's backend/backend_variant pair; the bare family key serves
    every variant that has none of its own.

    Together with image this forms the version's image layout, inherited
    from the provider's defaults as a unit: a version declaring either
    field owns both."""

    run_command: Optional[str] = None
    """Argument-vector template with {{placeholder}} substitution, taking
    the image's ENTRYPOINT slot — it names the executable, not just its
    flags. For an image whose entrypoint already starts the cache server
    (and may do setup around it) declare run_args instead, which keeps
    that entrypoint."""

    run_args: Optional[str] = None
    """Argument template appended to the image's own entrypoint, for
    images that start the cache server themselves. Same substitution as
    run_command; the two are alternatives — a version declares at most
    one, since a command and its arguments concatenate into the same
    vector either way."""

    env: Optional[Dict[str, str]] = None
    """Env template for the managed container. Values support {{placeholder}}."""

    metrics: Optional["CacheProviderMetrics"] = None
    """Overrides the provider-level metrics declaration for this version,
    whole — declared when the version's exposition renames metrics (an
    exporter change typically renames a family at once, so per-key
    inheritance would hide half the picture). Undeclared versions read
    the provider default."""

    def _backend_images(
        self, backend: Optional[str], variant: Optional[str]
    ) -> Dict[str, str]:
        """The images this node's accelerator may run: the variant's own
        entry where the catalog distinguishes one, otherwise the family's.

        A family whose SoC generations are not interchangeable is built per
        variant (Ascend's 910b, a3, 950, 310p are four images), and a
        derived version keys them apart — a 310p node handed the 910b build
        runs the wrong one, and where the package is in one build and not
        another it runs one that cannot serve at all. A declaration naming
        the family alone means one image for all of it, which is what a
        provider with no per-variant build has.
        """
        if variant:
            images = self.runtime_images.get(f"{backend}-{variant}")
            if images:
                return images
        return self.runtime_images.get(backend) or {}

    def supports_runtime(
        self, backend: Optional[str], variant: Optional[str] = None
    ) -> bool:
        """Whether the version can run on the node's accelerator.
        runtime_images doubles as the support matrix: a node is only
        served when its backend has an entry — an image built for
        another accelerator family would just crash-loop. A version
        declaring no runtime_images has one build for everything and
        serves every node.

        A node with no accelerator asks under the "cpu" key, and the
        plain image answers for it where there is none: that is the
        single build a declaration names for everything, so it runs
        there as much as anywhere. It answers for that node alone — on
        an accelerator whose family it was not built for, serving it is
        the crash-loop this matrix exists to prevent."""
        if not backend:
            return bool(self.image)
        if not self.runtime_images or self._backend_images(backend, variant):
            return True
        return backend == CPU_BACKEND and bool(self.image)

    def resolve_image(
        self,
        backend: Optional[str] = None,
        runtime_version: Optional[str] = None,
        variant: Optional[str] = None,
    ) -> str:
        """The image for a node's accelerator runtime, matched with the
        same rule inference-backend runners use (newest declared version
        <= the host runtime, with same-major and oldest fallbacks).
        Accelerator-less nodes and backends without entries get the
        plain image."""
        if not backend:
            return self.image
        by_version = self._backend_images(backend, variant)
        picked = pick_runtime_version(list(by_version), runtime_version)
        if picked is None:
            return self.image
        return by_version[picked]


class CacheProviderKVTransferConfig(BaseModel):
    """Structured form of a vLLM-style single-slot connector argument.

    The engine accepts exactly one value for this flag while several
    parties may want to write it (this cache integration, PD transfer
    connectors, the user); a slot like that must be assembled by one
    owner. Declaring the payload structured — instead of a pre-rendered
    JSON string in args — keeps the platform able to compose it (e.g.
    into a MultiConnector) and to detect user takeover of the flag."""

    flag: str = "--kv-transfer-config"
    """Engine argument that carries the serialized payload."""

    kv_connector: str
    kv_connector_module_path: Optional[str] = None
    """Optional Python module path for engines that load the connector
    implementation outside their default registry."""

    kv_role: str = "kv_both"
    kv_connector_extra_config: Dict[str, Any] = {}
    """Connector-specific settings. String values support {{placeholder}};
    a value that is exactly one placeholder keeps the parameter's type
    (e.g. a port renders as a JSON number, not a string)."""


class CacheProviderInjection(BaseModel):
    """Connector config injected into an inference engine that attaches to a cache service."""

    env: Dict[str, str] = {}
    """Env template. Values support {{placeholder}}; entries rendering empty are dropped."""

    kv_transfer_config: Optional[CacheProviderKVTransferConfig] = None
    """The engine's connector slot, rendered ahead of args as
    "<flag> <compact JSON>"."""

    args: List[str] = []
    """Extra command args for the inference engine. Items support {{placeholder}}."""

    files: Dict[str, str] = {}
    """Config files written inside the engine container before it starts,
    keyed by absolute path; contents support {{placeholder}}. For
    connectors that read a config file instead of env/args (e.g.
    a connector that reads its settings from a path an env var
    points at)."""

    locality_params: Dict[str, Dict[str, Any]] = {}
    """Placeholder defaults keyed by the engine-to-instance placement the
    resolver derives ("node_local" | "remote"). Lets a declaration vary
    connector config by placement (e.g. a same-node zero-copy transfer
    mode) in its own vocabulary; the platform only supplies the fact."""


class CacheProviderIntegration(BaseModel):
    backend: str
    """Inference backend name this provider can attach to (e.g. "vLLM")."""

    frameworks: Optional[List[str]] = None
    """Accelerator frameworks this entry is scoped to, in the
    runtime_images key vocabulary (e.g. "cuda", "cann"). None makes the
    entry generic: it serves every framework no scoped entry claims.
    Lets a provider vary the attach contract per accelerator (e.g.
    vllm-ascend trails vLLM releases and may need a different connector
    config). Entries are selected whole — the chosen entry's versions
    AND injection apply; scoped entries do not merge with the generic
    one."""

    versions: Optional[str] = None
    """Compatible backend version range (e.g. ">=0.25.0"). Enforced when
    the engine version is known: model validation rejects a pinned
    backend_version outside the range, and the injection resolver
    degrades instead of injecting args the engine may not accept.
    Unparseable values fail open."""

    injection: CacheProviderInjection = CacheProviderInjection()


class CacheProviderResourceProfile(BaseModel):
    """How capacity config maps to per-instance host resource claims.
    ram_gib is a template over the declared field values (e.g.
    "{{ram_size}}"); the service form's placement pre-flight renders it
    to warn about workers that cannot hold an instance. The scheduler
    does not enforce it."""

    ram_gib: Optional[str] = None
    cpu: Optional[float] = None


class CacheProviderMetricValue(BaseModel):
    """How to extract one semantic metric value from the provider's
    Prometheus exposition. At most one of the forms is set (validated —
    the query builder would otherwise silently pick one of several).
    Consumed by the cache-service metrics endpoint, which translates the
    form into a PromQL query over the service's scrape series."""

    gauge: Optional[str] = None
    """Gauge metric name; charted as-is."""

    rate: Optional[str] = None
    """Counter metric name; charted as its per-second rate over the
    chart's rate window (e.g. lookup traffic in tokens per second)."""

    ratio: Optional[Dict[str, str]] = None
    """{"numerator": counter, "denominator": counter}: the ratio of the
    two counters' increases over the chart's rate window."""

    gauge_ratio: Optional[Dict[str, str]] = None
    """{"numerator": gauge, "denominator": gauge}: the instantaneous ratio
    of two gauges (e.g. allocated / capacity)."""

    histogram_avg: Optional[str] = None
    """Histogram base name: increase(_sum) / increase(_count) over the
    rate window, i.e. the average observed value."""

    aggregate: Optional[str] = None
    """How gauge values combine into the service-level series: "sum"
    (default — capacities, byte counts) or "avg" (ratios). Only valid
    with the gauge form: the other forms aggregate naturally (operands
    sum before dividing, weighting instances by their actual traffic)."""

    @model_validator(mode="after")
    def _validate_forms(self):
        forms = [
            name
            for name in ("gauge", "rate", "ratio", "gauge_ratio", "histogram_avg")
            if getattr(self, name)
        ]
        if len(forms) > 1:
            raise ValueError(
                f"metric rule sets multiple extraction forms: {', '.join(forms)}"
            )
        if self.aggregate is not None:
            if self.aggregate not in ("sum", "avg"):
                raise ValueError(
                    f"aggregate must be 'sum' or 'avg', got '{self.aggregate}'"
                )
            if not self.gauge:
                raise ValueError("aggregate applies only to the gauge form")
        return self


class CacheProviderMetrics(BaseModel):
    """Where a cache service's Prometheus exposition is scraped, and how
    its semantic metrics are extracted from it."""

    path: str = "/metrics"
    """HTTP path of the Prometheus exposition on the metrics port."""

    mappings: Dict[str, CacheProviderMetricValue] = {}
    """Semantic key -> extraction rule. Keys use the platform's tier
    vocabulary — L1 is the memory (near) tier, L2 the capacity tier
    (disk/remote) — regardless of the provider's own naming: hit_rate,
    l1_usage_bytes, l1_usage_ratio, l2_usage_bytes. A provider with
    several L2 backends keeps them apart by series label, not by key."""

    throughput: Dict[str, CacheProviderMetricValue] = {}
    """Named throughput series (unit: GB/s) -> extraction rule."""


# CacheProviderVersionConfig.metrics forward-references this module's
# tail; resolve it now that the metrics classes exist.
CacheProviderVersionConfig.model_rebuild()


class CacheProviderL2Field(BaseModel):
    """One configurable parameter of an L2 storage backend."""

    name: str
    label: Optional[LocalizedText] = None
    """UI label; defaults to name."""

    description: Optional[LocalizedText] = None
    """What the value does, for a parameter whose label does not say it —
    the backend's own description covers the field set as a whole, not
    the one knob whose effect an operator cannot guess."""

    type: str = "string"
    """Value type: "string" | "number" | "boolean" | "password"."""

    required: bool = False
    default: Optional[Any] = None

    env_name: Optional[str] = None
    """When set, the value is delivered to the managed container via this
    env var instead of the adapter JSON (keeps secrets off the command line)."""

    metrics_target: bool = False
    """When set, the value is an additional Prometheus scrape address
    (host:port or URL) for this storage backend, added to the service's
    scrape targets. Never rendered into the adapter JSON or env."""


class CacheProviderL2Backend(BaseModel):
    """A storage backend the provider's L2 adapter can spill KV cache to."""

    display_name: Optional[LocalizedText] = None
    description: Optional[LocalizedText] = None
    icon: Optional[str] = None
    """Logo URL for brand display; the UI falls back to a generic icon."""

    adapter_flag_optional: bool = False
    """Whether the UI should offer a separate switch for enabling the
    provider's ``l2_adapter_flag`` for this backend."""

    adapter_flag_default: bool = True
    """Default state of the optional adapter-flag switch."""

    adapter_flag_label: Optional[LocalizedText] = None
    """Label for the optional adapter-flag switch."""

    adapter_type: Optional[str] = None
    """JSON ``type`` emitted for this backend; defaults to its catalog key."""

    adapter_backend: Optional[str] = None
    """Optional JSON ``backend`` value for adapters with a second type."""

    adapter_params: Dict[str, str] = {}
    """Mapping of nested ``backend_params`` keys to declared field names.

    When set, field values are emitted under ``backend_params`` instead of
    being placed directly on the adapter object.
    """

    fields: List[CacheProviderL2Field] = []


class CacheProviderFieldOption(BaseModel):
    """A choice of an options field whose display text differs from the
    stored value."""

    value: str
    label: Optional[LocalizedText] = None
    description: Optional[LocalizedText] = None
    """One-line explanation rendered under the label in the dropdown."""


class CacheProviderField(BaseModel):
    """A managed-mode configuration value promoted to a structured field
    in the service form's advanced section. The field carries no
    destination of its own: it adds a {{name}} placeholder to the
    template namespace, and the version's run_command/env templates
    decide where the value lands. A flag whose placeholder renders empty
    is dropped with its value, and user-supplied free-form parameters
    still override any flag the templates produce."""

    name: str
    """Placeholder name; must not collide with the reserved platform
    placeholders (host/port/metrics_port/service_id)."""

    label: Optional[LocalizedText] = None
    description: Optional[LocalizedText] = None

    placeholder: Optional[str] = None
    """Sample value shown in the empty input — the shape of the value
    where prose cannot convey it (an endpoint list, a device name)."""

    type: str = "string"
    """Value type: "string" | "number" | "boolean" (booleans render as
    "true"/"false")."""

    default: Optional[Any] = None

    required: bool = False
    """Managed creation rejects a blank value (a declared default
    satisfies it), and the form marks the input accordingly."""

    options: Optional[List[Union[str, "CacheProviderFieldOption"]]] = None
    """When set, the UI offers a fixed choice. An entry is either the
    value itself or {value, label} when the display text differs from
    the stored value (e.g. "Standalone Store" over standalone-store)."""

    visible_by: Optional[str] = None
    """Name of another declared field this one's visibility follows; the
    field renders only while that field equals visible_when (e.g. the
    RDMA device only matters on the rdma protocol). Value resolution
    honors the gate only when gated_default is declared; a plain
    default still renders while hidden."""

    def option_values(self) -> List[str]:
        return [
            option if isinstance(option, str) else option.value
            for option in (self.options or [])
        ]

    visible_when: Optional[Any] = None
    """Value of the visible_by field that shows this one."""

    framework_defaults: Optional[Dict[str, Any]] = None
    """Default per accelerator framework of the workers the service will
    run on (the runtime_images key vocabulary: cuda, cann, ...), falling
    back to ``default`` for the rest. For a value the hardware decides
    rather than the operator — a transport that is the accelerator's own
    on NPU nodes and nothing else works there."""

    gated_default: Optional[Any] = None
    """Value the field resolves to while its visible_by gate does not
    match. The form never submits a hidden field, but its plain default
    would still render — e.g. the engine's segment contribution must
    render 0 while a standalone store owns the pool."""

    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    """Numeric bounds and stepper increment for number-typed fields;
    the UI control and the API validation both honor them."""


class CacheProviderComponentPort(BaseModel):
    """A port a component binds, optionally only for the configurations
    that need it."""

    name: str
    enabled_by: Optional[str] = None
    """Name of a declared field that turns this port on; None means
    always. Without enabled_when the field reads as a boolean."""

    enabled_when: Optional[Any] = None


class CacheProviderComponent(BaseModel):
    """One process role of a multi-component managed provider (e.g. a
    coordinating master and the per-node stores holding its capacity).
    A provider without ``components`` is single-component: the
    provider-level topology and the version launch templates describe
    its one process, and nothing changes for it."""

    topology: str = "replicas"
    """Instance layout of this component: "replicas" runs ``replicas``
    scheduler-placed instances spread across matching workers (sticky to
    the workers they already run on), "per_node" runs one per matching
    cluster worker."""

    replicas: int = 1
    """Instance count for the "replicas" topology. Fewer matching
    workers than replicas deploys what fits (a smaller pool beats
    parking the service). Ignored by "per_node"."""

    replicas_by: Optional[str] = None
    """Name of a number declared field whose configured value overrides
    ``replicas`` (e.g. a user-sized store fleet); None keeps the declared
    count."""

    depends_on: Optional[str] = None
    """Name of a component whose instances must be RUNNING (with ports
    known) before this component's instances are created — e.g. stores
    need the master's address. The dependency must be addressable: either
    a single fixed replica, or a component declaring an
    ``address_template``."""

    address_template: Optional[str] = None
    """How clients address this component when an indirection stands in
    for one instance's host:port — HA masters that elect a leader
    that clients discover through the coordination backend
    (``etcd://{{ha_backend_connstring}}``). Engines attaching to the
    component and dependents rendering {{component.<name>.address}} use
    it while every placeholder it references has a value, and fall back
    to the resolved instance address otherwise. A component sized by a
    field (``replicas_by``) must declare one to be addressable at all."""

    run_command: Optional[str] = None
    run_args: Optional[str] = None
    """Launch template of this component, same semantics as the version
    slots (a command takes the entrypoint, args ride the image's own).
    Components own their launch: version-level launch templates apply
    only to single-component providers, since one template cannot serve
    two roles. {{component.<name>.address}} resolves to a
    single-replica component's host:port."""

    ports: List[Union[str, "CacheProviderComponentPort"]] = []
    """Every port this component binds — the one it serves on, a metrics
    or admin listener, the handshake socket a peer-to-peer transfer
    channel needs. An entry is a bare name, or {name, enabled_by,
    enabled_when} for a port only some configurations need — a list
    holding one of the latter writes all of its entries that way, so a
    reader compares like with like. Empty means IMPLICIT_PORT_NAMES; a
    declared list replaces them.

    The worker allocates one port per enabled name, records them on the
    instance so a restart keeps the ports it already published to peers,
    and renders each as {{ports.<name>}}, plus {{ports.<name>.url}} for
    the worker-routable host:port a peer would dial. Both render empty
    while the port is not allocated, so a flag carrying one drops with
    it. {{port}} and {{metrics_port}} address the same allocation by
    role — the component's address port and its metrics port — which is
    the spelling templates outside a component (a version's launch
    template, an engine's injection) have to use."""

    address_port: Optional[str] = None
    """Which declared port the component answers on: its instance
    address, what {{component.<name>.address}} and {{port}} render and
    what engines attaching to it dial. None takes the port named "port"
    if there is one, else the first declared."""

    def enabled_ports(
        self, config_fields: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Names of the ports to allocate for the given field values.

        The values are read as given: callers deciding what to allocate
        pass what a launch would render (see
        ``CacheProvider.enabled_port_names``), so a port follows exactly
        the value its component's gate did. Passing raw request values
        here asks a different question — which ports that configuration
        alone turns on — and the declaration check below is the one place
        that wants it.
        """
        if not self.ports:
            return list(IMPLICIT_PORT_NAMES)
        values = config_fields or {}
        names: List[str] = []
        for entry in self.ports:
            if isinstance(entry, str):
                names.append(entry)
                continue
            if entry.enabled_by is None:
                names.append(entry.name)
                continue
            value = values.get(entry.enabled_by)
            wanted = entry.enabled_when
            if (value == wanted) if wanted is not None else bool(value):
                names.append(entry.name)
        return names

    def declared_ports(self) -> List[str]:
        """Every port name the component may bind, gated or not."""
        if not self.ports:
            return list(IMPLICIT_PORT_NAMES)
        return [entry if isinstance(entry, str) else entry.name for entry in self.ports]

    def address_port_name(self) -> str:
        """The port name the component's address is built from."""
        if self.address_port:
            return self.address_port
        declared = self.declared_ports()
        if DEFAULT_PORT_NAME in declared:
            return DEFAULT_PORT_NAME
        return declared[0] if declared else DEFAULT_PORT_NAME

    env: Dict[str, str] = {}
    """Env template for this component's container; values support
    {{placeholder}} including cross-component addresses."""

    data_dirs: List[str] = []
    """Directories this component keeps its data in (templates, e.g. a
    configured disk-tier path). The worker creates each before the
    container starts — a server told to keep data somewhere expects the
    directory to exist — on the filesystem the cache container shares
    with it. An entry whose placeholders have no value renders empty and
    is skipped, so a path only some configurations use costs nothing when
    unused."""

    health_check: Optional[CacheProviderHealthCheck] = None
    """Probe for this component's instances; None inherits the
    provider-level health_check."""

    common_parameters: List[str] = []
    """Flags the UI offers as completion hints in this component's
    parameters editor. Each role runs its own binary with its own flags,
    so the provider-level list — which describes the one engines attach
    to — is no help to the others. Excludes what the platform injects
    (host, ports, capacity), which a hand-written copy would fight."""

    metrics_port: Optional[str] = None
    """Which declared port carries the exposition the provider's metrics
    declaration describes: what {{metrics_port}} renders and what the
    platform scrapes. None means this component is not scraped at all —
    scrape targets are built from the components naming one (a
    coordinating master, not the stores behind it). A listener that serves something else
    belongs in ``ports`` under its own name."""

    attach_endpoint: bool = False
    """Whether engines attach to this component's address (exactly one
    component of a multi-component provider declares it — a master,
    say, where the stores behind it are internal). It must be addressable (one
    fixed replica or an address_template) and cannot be gated by
    enabled_by."""

    enabled_by: Optional[str] = None
    """Name of a declared field that turns this component on; None means
    always on. Without enabled_when the field reads as a boolean; with
    it, the component is on while the field equals that value (e.g.
    a pool's stores exist while its mode is "standalone-store"). A
    disabled component keeps no instances."""

    enabled_when: Optional[Any] = None
    """Value of the enabled_by field that turns this component on."""

    gpu_access: bool = True
    """Whether this component's container mounts the node's GPUs. LMCache
    needs a CUDA context for its IPC transport; a pure-RAM component
    opts out and saves the per-GPU context memory."""

    resource_profile: Optional[CacheProviderResourceProfile] = None
    """Per-instance host resource claim of this component (e.g. the
    store's segment size), same template semantics as the provider-level
    profile — which describes the single-component case only and does
    not apply to components."""

    def addressable_alone(self) -> bool:
        """Whether one instance of this component is the whole address:
        true only for a single fixed replica, since a field-sized or
        per-node component has several endpoints."""
        return (
            self.topology == "replicas" and self.replicas == 1 and not self.replicas_by
        )

    @model_validator(mode="after")
    def _one_launch_slot(self):
        if self.run_command and self.run_args:
            raise ValueError("a component declares run_command or run_args, not both")
        return self


IMPLICIT_COMPONENT = CacheProviderComponent(
    ports=list(IMPLICIT_PORT_NAMES),
    metrics_port=DEFAULT_METRICS_PORT_NAME,
    attach_endpoint=True,
)
"""The one process a provider without ``components`` describes: it
serves engines on the platform's port, exposes its metrics on the
metrics port, and its launch templates live at the version level."""


def _validate_port_gates(
    name: str, component: CacheProviderComponent, fields: List["CacheProviderField"]
) -> None:
    """A port gated on a field nobody declares would never be bound, and
    every flag carrying {{ports.<name>}} would drop with it — silently,
    at launch, where a misspelled gate looks exactly like a feature that
    is off."""
    declared = {field.name for field in fields}
    for entry in component.ports:
        if isinstance(entry, str) or entry.enabled_by is None:
            continue
        if entry.enabled_by not in declared:
            raise ValueError(
                f"component '{name}' gates port '{entry.name}' on "
                f"'{entry.enabled_by}', which is not a declared field"
            )


def _validate_component_shape(name: str, component: CacheProviderComponent) -> None:
    """Check what a component declares about itself, independent of how it
    relates to the others."""
    if component.topology not in ("replicas", "per_node"):
        raise ValueError(
            f"component '{name}' declares unknown topology " f"'{component.topology}'"
        )
    if component.replicas < 1:
        raise ValueError(
            f"component '{name}' declares replicas "
            f"{component.replicas}; at least one is required"
        )
    declared = component.declared_ports()
    for port_name in declared:
        if not port_name.isidentifier():
            raise ValueError(
                f"component '{name}' declares port '{port_name}', "
                "which is not a valid placeholder name"
            )
    if len(set(declared)) != len(declared):
        raise ValueError(f"component '{name}' declares a port name twice")

    # A role naming a port the component never binds resolves to nothing
    # at runtime — an address with no port, a scrape target that is never
    # built, a probe that can never pass.
    for role, port_name in (
        ("address_port", component.address_port),
        ("metrics_port", component.metrics_port),
        (
            "health_check target",
            component.health_check.target if component.health_check else None,
        ),
    ):
        if port_name is not None and port_name not in declared:
            raise ValueError(
                f"component '{name}' points {role} at port '{port_name}', "
                f"which it does not declare (declares: {', '.join(declared)})"
            )

    # The address has to hold for every configuration the component runs
    # in. A port gated on nothing always does; one gated exactly as the
    # component is does too, since the configurations that close it are
    # the ones with no instances to address. Anything else leaves the
    # component running without an address for some configuration.
    address_port = component.address_port_name()
    gate = next(
        (
            entry
            for entry in component.ports
            if not isinstance(entry, str) and entry.name == address_port
        ),
        None,
    )
    if gate is not None and gate.enabled_by is not None:
        follows_component = (
            gate.enabled_by == component.enabled_by
            and gate.enabled_when == component.enabled_when
        )
        if not follows_component:
            raise ValueError(
                f"component '{name}' takes its address from port "
                f"'{address_port}', which is gated on "
                f"'{gate.enabled_by}': a configuration closing that gate "
                "leaves the component running with no address"
            )

    # A component rendering {{metrics_port}} without naming one renders
    # nothing — a flag silently dropped from the launch, which is how a
    # listener a probe depends on goes missing.
    if component.metrics_port is None:
        for template in (
            component.run_command,
            component.run_args,
            *component.env.values(),
        ):
            if _references_placeholder(template, "metrics_port"):
                raise ValueError(
                    f"component '{name}' renders {{{{metrics_port}}}} but "
                    "declares no metrics_port; a listener it is not scraped "
                    "on belongs in ports under its own name"
                )


class CacheProvider(BaseModel):
    name: str
    display_name: Optional[LocalizedText] = None
    source: CacheProviderSourceEnum = CacheProviderSourceEnum.BUILT_IN
    description: Optional[LocalizedText] = None
    icon: Optional[str] = None

    links: List[CacheProviderLink] = []
    """Brand links (docs, product page) rendered on the catalog card."""

    unavailable_reason: Optional[LocalizedText] = None
    """Why this installation cannot run the provider, which is also what
    marks it unavailable: the catalog lists it so the choice is visible,
    the form does not offer it, and a service naming it is refused. A
    declaration an extension registers over this one carries no reason and
    the provider becomes usable — the placeholder holds the card's place
    until whatever it needs is installed, and describes nothing else."""

    dashboard_uid: Optional[str] = None
    """UID of a provider-specific Grafana dashboard provisioned alongside
    the generic cache-service one; the service's Grafana entry points
    redirect to it. None falls back to the generic dashboard."""

    topology: str = "replicas"
    """Instance layout of a single-component provider:
    "replicas" runs one scheduler-placed instance (pinned when the
    service names a worker_id); "per_node" runs one instance per active
    worker of the service's cluster, following workers as they join and
    leave. Multi-component providers declare topology per component
    instead."""

    attach_locality: str = "cluster"
    """Where an engine may attach from: "node_local" means the connector
    only works against a cache server on the engine's own node (e.g.
    LMCache MP's CUDA-IPC transport), so remote fallback and multi-worker
    instances degrade; "cluster" (default) means the endpoint is
    network-reachable from any worker. Deliberately separate from
    ``topology``: placement and attach contract only coincide for
    LMCache-style providers — a distributed pool may run per-node data
    components while engines attach its cluster-wide endpoint."""

    management_url: bool = False
    """Whether the engine ships its own management UI worth linking to:
    the service form then offers a management_url config field, rendered
    as a link beside the service name."""

    components: Dict[str, CacheProviderComponent] = {}
    """Managed-mode process roles, keyed by component name. Empty means
    single-component (the provider-level topology and version launch
    templates describe the one process). Declared components each own
    their topology, launch and env; the shared image layout still comes
    from the version."""

    default_version: Optional[str] = None
    versions: Dict[str, CacheProviderVersionConfig] = {}

    runner_dependency: Optional[str] = None
    """Name of the package whose presence in a runner image makes that image a
    version of this provider, as the runner catalog's ``dependencies`` spells it
    (e.g. "mooncake-transfer-engine").

    Declaring it replaces ``versions`` entirely: the release line, its images
    and the default version are read off the images the installation actually
    has. A version then names a release some image really carries, which a
    hand-written one could not be held to — it appears when such an image does,
    with no edit here, and an accelerator whose images were never probed offers
    no version at all rather than one that cannot run.

    What a version names is the wheel, not a build of one: the same string is
    compiled into several images, and nothing here pairs the build a cache
    server runs with the build inside an engine attaching to it. Where that
    matters — a provider whose wire format breaks across builds — this narrows
    the gap rather than closing it; the engine's side is gated by the
    integration's version range alone.

    A provider whose images are not runner images (a partner's own registry)
    declares ``versions`` by hand instead. The two are exclusive to an author,
    and the check sits on the document rather than here: once the derivation
    has run this model carries both, which is what a reader of the catalog —
    and the row it was materialized into — is given."""

    runner_frameworks: List[str] = []
    """Accelerator families the derived release line is built from. Empty
    takes every family whose images were probed to carry the package.

    An image holding the package is not the same as the provider working on
    that accelerator: the probe reads a version off a wheel, while whether the
    engine attaches and the transport moves a block there is something someone
    has to run. Naming the families keeps a version off an accelerator nobody
    has run it on, instead of offering one and finding out at the first
    attach. A family joins the list once it has been.

    Named by family, so "cann" covers each of its per-SoC builds. Only a
    provider reading its release line off the runner images has one."""

    default_run_command: Optional[str] = None
    default_run_args: Optional[str] = None
    """Launch template shared by versions that declare none of their own.
    A provider whose CLI is stable across its release line states it once
    here; a version departing from it declares its own run_command or
    run_args, and one that must run the image entrypoint bare opts out
    with an empty run_command. The pair is inherited as a unit — a
    version declaring either owns its launch — and resolved into each
    version at model construction, so every consumer (including the
    catalog API) reads the effective launch off the version config."""

    default_image: Optional[str] = None
    default_runtime_images: Dict[str, Dict[str, str]] = {}
    """Image layout shared by versions that declare none of their own, in
    the same shape as a version's image / runtime_images. {{version}}
    stands for the version key, so a provider whose tags embed the
    version declares the layout once and each version is just its key —
    the version string is stated once instead of copied into every tag.
    The placeholder is optional: a provider whose versions all share one
    image states it here without it. A version whose images depart from
    the layout (a one-off registry, a build only it has) declares its
    own, which takes over the layout whole."""

    custom_version: bool = False
    """Whether a service may pin a user-supplied container image instead of
    a declared version; the default version's run command and env templates
    still apply, so the image must be command-compatible."""

    fields: List[CacheProviderField] = []
    """Managed-mode configuration values promoted to structured form
    fields (e.g. the eviction policy), wired into the runtime config by
    the version templates via {{name}}; everything else stays reachable
    through the free-form parameters editor."""

    resource_profile: Optional[CacheProviderResourceProfile] = None
    health_check: CacheProviderHealthCheck = CacheProviderHealthCheck()
    default_metrics: Optional[CacheProviderMetrics] = None
    """The all-version default declaration, named like the other
    provider-level defaults (default_image, default_run_command). Do not
    read it directly for a service — a version may carry its own metrics
    block; metrics_for() resolves the effective one."""

    inference_backend_integrations: List[CacheProviderIntegration] = []

    common_parameters: List[str] = []
    """Flags the UI offers as completion hints in the extra-parameters
    editor. Excludes flags GPUStack injects itself (host/ports/capacity/
    L2 adapter), which would conflict with the structured config."""

    l2_adapter_flag: Optional[str] = None
    """Command-line flag that carries the L2 adapter JSON
    (e.g. "--l2-adapter"); None means the provider has no L2 support."""

    l2_backends: Dict[str, CacheProviderL2Backend] = {}
    """Adapter type identifier (the "type" value in the adapter JSON)
    -> backend declaration."""

    def component_layouts(self) -> Dict[str, str]:
        """Component name -> topology. A single-component provider maps
        {"": topology} — the empty string is the stored component value
        of its instance rows (a real column value, not NULL, so the
        (service, worker, component) uniqueness holds on every database:
        NULLs compare distinct inside unique constraints)."""
        if self.components:
            return {name: c.topology for name, c in self.components.items()}
        return {"": self.topology}

    def get_component(self, name: str) -> Optional[CacheProviderComponent]:
        if not name:
            return None
        return self.components.get(name)

    def _port_layout(self, component: Optional[str]) -> CacheProviderComponent:
        """The declaration answering port questions for a component —
        IMPLICIT_COMPONENT for the single process a provider without
        ``components`` describes, which declares nothing about itself."""
        return self.components.get(component or "") or IMPLICIT_COMPONENT

    def enabled_port_names(
        self,
        component: Optional[str],
        config_fields: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """Ports to allocate for an instance of this component under the
        given configuration. The gates read the values a launch would
        render, as the component's own gate does: a port whose gate sits
        behind a closed one must not be bound from a stale value."""
        return self._port_layout(component).enabled_ports(
            resolved_field_values(self.fields, config_fields or {})
        )

    def declared_port_names(self, component: Optional[str]) -> List[str]:
        """Every port name an instance of this component may bind, gated
        or not — the placeholders its templates can reference."""
        return self._port_layout(component).declared_ports()

    def address_port_name(self, component: Optional[str]) -> str:
        """Port name an instance of this component is addressed by."""
        return self._port_layout(component).address_port_name()

    def metrics_port_name(self, component: Optional[str]) -> Optional[str]:
        """Port name this component serves the declared exposition on;
        None when it is not scraped."""
        return self._port_layout(component).metrics_port

    def health_check_for(self, component: Optional[str]) -> CacheProviderHealthCheck:
        """Probe for a component's instances: its own declaration, else
        the provider-level one — an HTTP metrics endpoint for one role where
        another answers on a plain port."""
        spec = self.get_component(component or "")
        if spec is not None and spec.health_check is not None:
            return spec.health_check
        return self.health_check or CacheProviderHealthCheck()

    def probe_port_name(self, component: Optional[str]) -> str:
        """Port name a component's probe hits: what its health check
        targets, else the port it is addressed by."""
        return self.health_check_for(component).target or self.address_port_name(
            component
        )

    @model_validator(mode="after")
    def _validate_field_gates(self) -> "CacheProvider":
        """Every visibility gate names a field that exists.

        A gate naming nothing resolves to None, which matches no
        ``visible_when``, so the field it guards reads as ungated and renders
        its plain default — the failure is a value quietly not being what the
        declaration says, which nothing downstream can report.
        """
        declared = {field.name for field in self.fields}
        for field in self.fields:
            gate = field.visible_by
            if gate is not None and gate not in declared:
                raise ValueError(
                    f"provider '{self.name}' field '{field.name}' is gated by "
                    f"'{gate}', which it does not declare"
                )
        return self

    @model_validator(mode="after")
    def _validate_runner_frameworks(self) -> "CacheProvider":
        """Only a release line read off the runner images can be scoped to
        accelerator families — there is nothing else for the list to narrow,
        and one on a declaration that names its own versions would read as a
        restriction while changing nothing about them."""
        if self.runner_frameworks and not self.runner_dependency:
            raise ValueError(
                f"Cache provider '{self.name}' declares runner_frameworks "
                f"without runner_dependency: the list scopes a release line "
                f"read off the runner images, and this one declares its "
                f"versions"
            )
        return self

    @model_validator(mode="after")
    def _validate_single_process_probe(self) -> "CacheProvider":
        """A provider running one process declares no ports of its own: it
        binds the implicit pair, so a probe naming anything else names a port
        nothing allocates and fails for the life of the service. The component
        case is checked against declared ports in ``_validate_components``."""
        if self.components or self.health_check is None:
            return self
        target = self.health_check.target
        if target is not None and target not in IMPLICIT_PORT_NAMES:
            raise ValueError(
                f"provider '{self.name}' health check probes port '{target}', "
                f"which a single-process provider does not bind (it binds: "
                f"{', '.join(IMPLICIT_PORT_NAMES)})"
            )
        return self

    @model_validator(mode="after")
    def _validate_components(self) -> "CacheProvider":
        for name, component in self.components.items():
            _validate_component_shape(name, component)
            # A component without its own probe inherits the provider's,
            # whose target has to name a port this component binds.
            if component.health_check is None and self.health_check is not None:
                declared = component.declared_ports()
                if (
                    self.health_check.target is not None
                    and self.health_check.target not in declared
                ):
                    raise ValueError(
                        f"component '{name}' inherits the provider health "
                        f"check, which probes port '{self.health_check.target}'; "
                        f"the component declares: {', '.join(declared)}"
                    )
            _validate_port_gates(name, component, self.fields)
            dep_name = component.depends_on
            if dep_name is None:
                continue
            dependency = self.components.get(dep_name)
            if dependency is None or dep_name == name:
                raise ValueError(
                    f"component '{name}' depends on unknown component " f"'{dep_name}'"
                )
            if not (dependency.addressable_alone() or dependency.address_template):
                raise ValueError(
                    f"component '{name}' depends on '{dep_name}', which is "
                    "neither a single-replica component nor declares an "
                    "address_template: a dependent needs one address"
                )
            if dependency.depends_on:
                raise ValueError(
                    f"component '{name}' depends on '{dep_name}', which has "
                    "its own dependency: chains are not supported"
                )
        if self.components:
            attach = [
                name
                for name, component in self.components.items()
                if component.attach_endpoint
            ]
            if len(attach) != 1:
                raise ValueError(
                    "a multi-component provider declares exactly one "
                    "attach_endpoint component; engines need one address"
                )
            spec = self.components[attach[0]]
            # Addressable means an engine can name one endpoint. A single
            # replica is one; an address_template stands in for one; and
            # a per_node component is one per consumer, which only holds
            # while the provider says engines attach node-locally.
            per_node_local = (
                spec.topology == "per_node" and self.attach_locality == "node_local"
            )
            if not (
                spec.addressable_alone() or spec.address_template or per_node_local
            ):
                raise ValueError(
                    f"attach_endpoint component '{attach[0]}' must be "
                    "addressable: a single replica, an address_template, or "
                    "per_node with node_local attach"
                )
            if spec.enabled_by:
                raise ValueError(
                    f"attach_endpoint component '{attach[0]}' cannot be "
                    "gated by enabled_by: the attach address must always "
                    "exist"
                )
            declared_fields = {field.name for field in self.fields}
            for name, component in self.components.items():
                if component.enabled_by and component.enabled_by not in declared_fields:
                    raise ValueError(
                        f"component '{name}' is enabled by undeclared "
                        f"declared field '{component.enabled_by}'"
                    )
                if (
                    component.replicas_by
                    and component.replicas_by not in declared_fields
                ):
                    raise ValueError(
                        f"component '{name}' sizes replicas by undeclared "
                        f"declared field '{component.replicas_by}'"
                    )
        return self

    def attach_component(self) -> str:
        """Name of the component engines attach to ("" for
        single-component providers)."""
        for name, component in self.components.items():
            if component.attach_endpoint:
                return name
        return ""

    def component_enabled(
        self, name: str, config_fields: Optional[Dict[str, Any]]
    ) -> bool:
        """Whether the component should have instances for a service
        configured with ``config_fields``.

        The gate reads the value the launch would render, not the one the
        request carried: a field behind a closed gate of its own resolves
        to its gated default, and a value left over from when that gate
        was open must not keep a component alive."""
        spec = self.get_component(name)
        if spec is None or not spec.enabled_by:
            return True
        value = resolved_field_values(self.fields, config_fields or {}).get(
            spec.enabled_by
        )
        if spec.enabled_when is not None:
            return value == spec.enabled_when
        return bool(value)

    def metrics_for(self, version: Optional[str]) -> Optional[CacheProviderMetrics]:
        """The effective metrics declaration for a service pinned to
        ``version``. Resolution rides get_version_config, so None falls
        back to the default version like everywhere else (a managed
        service created without an explicit version stores None). A
        resolved version owns its block whole; versions without one —
        and the custom version, whose image ships unknown metrics — fall
        back to the provider default (best effort, and a mismatch
        surfaces as a reasoned all-queries-failed degradation rather
        than silently empty charts)."""
        config, _ = self.get_version_config(version)
        if config is not None and config.metrics is not None:
            return config.metrics
        return self.default_metrics

    @model_validator(mode="after")
    def resolve_version_defaults(self) -> "CacheProvider":
        """Fold the provider-level templates into each version so that a
        version config is self-contained: the worker, the validators and
        the catalog API all read one effective image and command off it,
        with no second lookup on the provider."""
        for version, config in self.versions.items():
            # run_command and run_args are two slots of one launch, so a
            # version opts out of both together: a version declaring args
            # for its own entrypoint must not also inherit a command that
            # replaces that entrypoint.
            if config.run_command is None and config.run_args is None:
                config.run_command = self.default_run_command
                config.run_args = self.default_run_args
            if config.run_command and config.run_args:
                raise ValueError(
                    f"Cache provider '{self.name}' version '{version}' "
                    "declares both run_command and run_args: a command and "
                    "its arguments form one vector, so state it as whichever "
                    "one the image's entrypoint calls for"
                )
            # image and runtime_images describe one image layout, so a
            # version opts out of both together: inheriting half a layout
            # would serve some accelerators from the version's own
            # registry and the rest from the provider's.
            if config.image is None and not config.runtime_images:
                config.image = self.default_image
                config.runtime_images = {
                    backend: dict(images)
                    for backend, images in self.default_runtime_images.items()
                }
            params = {"version": version}
            if config.image:
                config.image = render_template(config.image, params)
            config.runtime_images = {
                backend: {
                    runtime: render_template(image, params)
                    for runtime, image in images.items()
                }
                for backend, images in config.runtime_images.items()
            }
            # A derived version may legitimately have none: the images it is
            # built from are each for an accelerator, and the plain slot is
            # filled only by a family that also runs without a device. Absent,
            # the version serves accelerator nodes alone, which
            # ``supports_runtime`` reports rather than leaving to a container
            # that cannot load. A declared version has no such excuse.
            if not config.image and not self.runner_dependency:
                raise ValueError(
                    f"Cache provider '{self.name}' version '{version}' "
                    "resolves to no image: it must declare one, or declare "
                    "neither image nor runtime_images and inherit the "
                    "provider's default_image"
                )
        if not self.versions:
            # A provider with no release line to declare (an image that is
            # not published, so every service names its own) still needs a
            # way to reach an image: the custom version is it. A
            # declaration holding a card's place launches nothing at all,
            # so it is held to none of this — and neither is one whose
            # release line is read off the runner catalog, which has none
            # to show until the catalog is in hand.
            if (
                not self.custom_version
                and not self.unavailable_reason
                and not self.runner_dependency
            ):
                raise ValueError(
                    f"Cache provider '{self.name}' declares no versions: it "
                    "then resolves no image at all unless it allows the "
                    "custom version"
                )
            if self.default_run_command and self.default_run_args:
                raise ValueError(
                    f"Cache provider '{self.name}' declares both "
                    "default_run_command and default_run_args: a command and "
                    "its arguments form one vector, so state it as whichever "
                    "one the image's entrypoint calls for"
                )
        return self

    def custom_version_config(self) -> Optional[CacheProviderVersionConfig]:
        """The launch template a service pinning the reserved "custom"
        version runs with: the default version's config, or — for a provider
        declaring no versions at all — one built from the provider-level
        launch defaults, the only launch declaration such a catalog entry
        has. None when the provider declares versions but no usable default,
        which leaves the custom version nothing to template."""
        config, _ = self.get_version_config(None)
        if config is not None:
            return config
        if self.versions:
            return None
        return CacheProviderVersionConfig(
            run_command=self.default_run_command,
            run_args=self.default_run_args,
        )

    def get_version_config(
        self, version: Optional[str] = None
    ) -> Tuple[Optional[CacheProviderVersionConfig], Optional[str]]:
        """Resolve a version config, falling back to the default version."""
        target = version or self.default_version
        if target and target in self.versions:
            return self.versions[target], target
        return None, target

    def integration_for(
        self, backend_name: str, framework: Optional[str] = None
    ) -> Optional[CacheProviderIntegration]:
        """
        Pick the integration entry for an inference backend, preferring
        one scoped to the engine worker's accelerator ``framework`` over
        the generic (unscoped) entry. With framework unknown (None, e.g.
        validation before scheduling) a scoped-only declaration still
        answers "attachable" through any entry for the backend.
        """
        matches = [
            entry
            for entry in self.inference_backend_integrations
            if entry.backend.lower() == (backend_name or "").lower()
        ]
        if framework:
            for entry in matches:
                if entry.frameworks and framework in entry.frameworks:
                    return entry
        generic = next((c for c in matches if not c.frameworks), None)
        if generic is not None or framework:
            return generic
        return matches[0] if matches else None


# Dots namespace cross-component placeholders (component.master.address);
# a trailing |filter converts the value on the way out.
_TEMPLATE_PATTERN = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_.]*)(?:\|([a-z_]+))?\}\}")


def _references_placeholder(template: Optional[str], name: str) -> bool:
    """Whether a template renders the named placeholder, with or without
    a filter."""
    if not template:
        return False
    return any(match.group(1) == name for match in _TEMPLATE_PATTERN.finditer(template))


_GIB_BYTES = 1024**3


def _gib_to_bytes(value: Any) -> Any:
    """A size a field states in GiB, as the byte count a program that
    takes no unit wants."""
    try:
        return int(float(value) * _GIB_BYTES)
    except (TypeError, ValueError):
        return value


TEMPLATE_FILTERS = {"gib_to_bytes": _gib_to_bytes}
"""Conversions a placeholder may name (``{{cap_gb|gib_to_bytes}}``), for
values a declaration states in the unit a user thinks in and a program
reads in another."""


def _render_value(value: Any, filter_name: Optional[str] = None) -> str:
    """One resolved value as a template renders it. Booleans render
    lowercase: that is the literal JSON accepts and gflags parses, so one
    declared boolean serves a config file and a command-line flag
    alike."""
    if filter_name:
        conversion = TEMPLATE_FILTERS.get(filter_name)
        if conversion is None:
            # Caught at load time by ``validate_template_filters``; a
            # declaration that reaches here naming an unknown one is a bug in
            # that check, and a KeyError mid-render says nothing about which
            # declaration carried it.
            raise ValueError(f"unknown template filter '{filter_name}'")
        value = conversion(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def render_template(value: str, params: Dict[str, Any]) -> str:
    """
    Substitute {{placeholder}} occurrences with values from params.
    Placeholders whose value is None render as an empty string; unknown
    placeholders are left unchanged.
    """

    def replace_var(match):
        var_name = match.group(1)
        if var_name in params:
            resolved = params[var_name]
            return "" if resolved is None else _render_value(resolved, match.group(2))
        return match.group(0)

    return _TEMPLATE_PATTERN.sub(replace_var, value)


def render_argument(value: str, params: Dict[str, Any]) -> str:
    """Render one launch argument or env value. A placeholder the params
    know but have no value for empties the whole token, not just its own
    span: a flag reading "http://{{addr}}" has to disappear with its
    address rather than carry a bare scheme. A placeholder the params do
    not know at all is left as written, so a typo in a declaration fails
    loudly instead of silently dropping the flag it was meant to fill."""
    empty = False

    def replace_var(match):
        nonlocal empty
        name = match.group(1)
        if name not in params:
            return match.group(0)
        resolved = params[name]
        if resolved is None or resolved == "":
            empty = True
            return ""
        return _render_value(resolved, match.group(2))

    rendered = _TEMPLATE_PATTERN.sub(replace_var, value)
    return "" if empty else rendered


def render_optional_template(
    value: Optional[str], params: Dict[str, Any]
) -> Optional[str]:
    """Render a template that only means something once every placeholder
    it references has a value: an unset one makes the whole rendering
    None rather than a string with a hole in it ("etcd://" for an unset
    connection string). Same idiom as a flag dropped with its empty
    value."""
    if not value:
        return None
    missing = False

    def replace_var(match):
        nonlocal missing
        name = match.group(1)
        resolved = params.get(name)
        if resolved is None or resolved == "":
            missing = True
            return ""
        return _render_value(resolved, match.group(2))

    rendered = _TEMPLATE_PATTERN.sub(replace_var, value)
    return None if missing else rendered


def _coerce_l2_field_value(field: CacheProviderL2Field, value: Any) -> Any:
    """
    Normalize a field value for the adapter JSON. Number fields render as
    JSON integers when integral (a port must serialize as 6379, not 6379.0);
    boolean fields render as JSON booleans.
    """
    if field.type == "number":
        number = float(value)
        return int(number) if number.is_integer() else number
    if field.type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    return value


def _l2_adapter_output_name(
    backend_spec: CacheProviderL2Backend, field_name: str
) -> Optional[str]:
    output_name = next(
        (
            name
            for name, mapped_field in backend_spec.adapter_params.items()
            if mapped_field == field_name
        ),
        None,
    )
    if output_name is not None:
        return output_name
    if field_name in backend_spec.adapter_params:
        return backend_spec.adapter_params[field_name]
    if backend_spec.adapter_backend is not None and not backend_spec.adapter_params:
        return field_name
    return None


def _render_l2_adapter_fields(
    backend_spec: CacheProviderL2Backend,
    params: Dict[str, Any],
    adapter: Dict[str, Any],
    nested_values: Optional[Dict[str, Any]],
) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for field in backend_spec.fields:
        if field.metrics_target:
            continue
        value = params.get(field.name, field.default)
        if value is None or value == "":
            continue
        value = _coerce_l2_field_value(field, value)
        if field.env_name:
            env[field.env_name] = str(value)
            continue
        if nested_values is None:
            adapter[field.name] = value
            continue

        output_name = _l2_adapter_output_name(backend_spec, field.name)
        if output_name is not None:
            # NIXL plugin backend_params are string-valued, even for values
            # that look numeric (for example capacity in GiB).
            nested_values[output_name] = str(value)
    return env


def _attach_l2_backend_params(
    backend_spec: CacheProviderL2Backend,
    adapter: Dict[str, Any],
    nested_values: Optional[Dict[str, Any]],
) -> None:
    if nested_values is None:
        return
    if backend_spec.adapter_params:
        adapter["backend_params"] = {
            name: nested_values[name]
            for name in backend_spec.adapter_params
            if name in nested_values
        }
    else:
        adapter["backend_params"] = nested_values


def render_l2_adapter(
    provider: CacheProvider,
    backend: str,
    params: Dict[str, Any],
    adapter_flag_enabled: Optional[bool] = None,
) -> Tuple[List[str], Dict[str, str]]:
    """
    Build the (command args, container env) that configure a managed cache
    server's L2 storage backend. Fields declaring env_name are delivered via
    env so secrets stay off the command line; the rest form the adapter JSON
    together with the backend's type identifier. Unset optional fields are
    omitted from both. Raises ValueError when the provider has no L2 support
    or does not declare the backend.
    """
    backend_spec = provider.l2_backends.get(backend)
    if backend_spec is None:
        raise ValueError(
            f"Cache provider '{provider.name}' has no L2 storage "
            f"backend '{backend}'"
        )
    if not provider.l2_adapter_flag:
        raise ValueError(
            f"Cache provider '{provider.name}' does not support L2 storage"
        )

    flag_enabled = (
        backend_spec.adapter_flag_default
        if adapter_flag_enabled is None
        else bool(adapter_flag_enabled)
    )
    if backend_spec.adapter_flag_optional and not flag_enabled:
        return [], {}

    adapter: Dict[str, Any] = {
        "type": backend_spec.adapter_type or backend,
    }
    if backend_spec.adapter_backend is not None:
        adapter["backend"] = backend_spec.adapter_backend
    nested_values: Optional[Dict[str, Any]] = (
        {}
        if (backend_spec.adapter_backend is not None or backend_spec.adapter_params)
        else None
    )
    env = _render_l2_adapter_fields(backend_spec, params, adapter, nested_values)
    _attach_l2_backend_params(backend_spec, adapter, nested_values)

    args = [provider.l2_adapter_flag, json.dumps(adapter, separators=(",", ":"))]
    return args, env


def resolved_field_values(
    fields: List["CacheProviderField"], values: Dict[str, Any]
) -> Dict[str, Any]:
    """Field values as the templates should see them: the configured
    value falling back to the declared default — except that a field
    whose visible_by gate does not match resolves to its gated_default
    when one is declared.

    A gate is read resolved, not raw, so gates chain: a field behind a
    switch that is itself behind a mode closes with the mode, however the
    switch was left when the mode last offered it."""
    declared = {field.name: field for field in fields}
    resolved: Dict[str, Any] = {}
    resolving: Set[str] = set()

    def resolve(field: "CacheProviderField") -> Any:
        if field.name in resolved:
            return resolved[field.name]
        value = values.get(field.name, field.default)
        if field.visible_by and field.gated_default is not None:
            gate_field = declared.get(field.visible_by)
            if gate_field is not None and field.name not in resolving:
                resolving.add(field.name)
                gate_value = resolve(gate_field)
                resolving.discard(field.name)
            else:
                gate_value = values.get(field.visible_by)
            if gate_value != field.visible_when:
                value = field.gated_default
        resolved[field.name] = value
        return value

    for field in fields:
        resolve(field)
    return resolved


RESERVED_INJECTION_PLACEHOLDERS = frozenset(
    {
        "host",
        "port",
        "metrics_port",
        "service_id",
        "local_hostname",
        "master_server_address",
        "locality",
    }
)
"""Placeholders the platform itself supplies to injection rendering."""


def validate_injection_templates(provider: "CacheProvider") -> List[str]:
    """
    Check a provider's injection templates against the placeholder
    contract; returns human-readable violations (empty when clean).

    Enforced at load time because the failure mode is silent at runtime:
    a placeholder that nothing resolves renders literally into connector
    config, corrupting it. Every referenced placeholder must be a
    reserved platform placeholder, a declared field, or a key present in
    every locality bucket.
    """
    errors: List[str] = []
    declared = {field.name for field in provider.fields}
    for integration in provider.inference_backend_integrations:
        injection = integration.injection
        buckets = [set(bucket) for bucket in injection.locality_params.values()]
        locality_common = set.intersection(*buckets) if buckets else set()
        templates: List[str] = []
        templates.extend(injection.env.values())
        templates.extend(injection.args)
        templates.extend(injection.files.values())
        if injection.kv_transfer_config:
            templates.extend(
                value
                for value in injection.kv_transfer_config.kv_connector_extra_config.values()
                if isinstance(value, str)
            )
        referenced = {
            match.group(1)
            for template in templates
            for match in _TEMPLATE_PATTERN.finditer(template)
        }
        prefix = f"'{provider.name}' integration '{integration.backend}'"
        allowed = RESERVED_INJECTION_PLACEHOLDERS | declared | locality_common
        for name in sorted(referenced - allowed):
            errors.append(
                f"{prefix} references placeholder '{name}', which is not a "
                "reserved placeholder, a declared field, or a key present "
                "in every locality bucket"
            )
    return errors


def _localized_violations(value: Any, where: str) -> List[str]:
    """Check one localized slot against the mapping contract."""
    if not isinstance(value, dict):
        return []
    if not value:
        return [f"{where} is an empty locale mapping"]
    errors: List[str] = []
    if not value.get(DEFAULT_LOCALE):
        # Without a fallback, a locale the declaration skips has nothing
        # to render and the slot reads as untranslated rather than as
        # the author's canonical text.
        errors.append(f"{where} has no '{DEFAULT_LOCALE}' entry")
    for locale in sorted(value):
        if not LOCALE_PATTERN.match(locale):
            errors.append(f"{where} has invalid locale key '{locale}'")
    return errors


def validate_template_filters(provider: "CacheProvider") -> List[str]:
    """
    Check every ``{{name|filter}}`` a declaration carries against the filters
    that exist; returns human-readable violations (empty when clean).

    Enforced at load time because the failure is both late and loud in the
    wrong place: a misspelled filter renders fine through every check, then
    raises while a launch command is being built — far from the document that
    named it, and only for the configurations that reach that template.

    Walks the whole declaration rather than named slots: a filter may appear in
    a run command, a resource claim, a data directory or an injection alike,
    and a check that knows which fields hold templates goes stale the moment
    one is added.
    """
    errors: List[str] = []
    prefix = f"'{provider.name}'"
    unknown = set()

    def walk(value: Any) -> None:
        if isinstance(value, str):
            for match in _TEMPLATE_PATTERN.finditer(value):
                filter_name = match.group(2)
                if filter_name and filter_name not in TEMPLATE_FILTERS:
                    unknown.add(filter_name)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(provider.model_dump(mode="json"))
    known = ", ".join(sorted(TEMPLATE_FILTERS))
    for filter_name in sorted(unknown):
        errors.append(
            f"{prefix} references template filter '{filter_name}', which does "
            f"not exist (known filters: {known})"
        )
    return errors


def validate_localized_text(provider: "CacheProvider") -> List[str]:
    """
    Check every localizable slot of a provider against the LocalizedText
    contract; returns human-readable violations (empty when clean).

    Enforced at load time because the UI resolves these mappings itself:
    a slot missing its "default" degrades to whatever key the fallback
    chain lands on, differently per locale, and a typo'd locale key is
    text that simply never appears for anyone.
    """
    errors: List[str] = []
    prefix = f"'{provider.name}'"
    errors.extend(
        _localized_violations(provider.display_name, f"{prefix} display_name")
    )
    errors.extend(_localized_violations(provider.description, f"{prefix} description"))
    errors.extend(
        _localized_violations(
            provider.unavailable_reason, f"{prefix} unavailable_reason"
        )
    )
    for index, link in enumerate(provider.links):
        errors.extend(
            _localized_violations(link.label, f"{prefix} link #{index} label")
        )
    for field in provider.fields:
        where = f"{prefix} declared field '{field.name}'"
        errors.extend(_localized_violations(field.label, f"{where} label"))
        errors.extend(_localized_violations(field.description, f"{where} description"))
        for option in field.options or []:
            if isinstance(option, str):
                continue
            at = f"{where} option '{option.value}'"
            errors.extend(_localized_violations(option.label, f"{at} label"))
            errors.extend(
                _localized_violations(option.description, f"{at} description")
            )
    for key, backend in provider.l2_backends.items():
        where = f"{prefix} l2 backend '{key}'"
        errors.extend(
            _localized_violations(backend.display_name, f"{where} display_name")
        )
        errors.extend(
            _localized_violations(backend.description, f"{where} description")
        )
        errors.extend(
            _localized_violations(
                backend.adapter_flag_label, f"{where} adapter_flag_label"
            )
        )
        for field in backend.fields:
            at = f"{where} field '{field.name}'"
            errors.extend(_localized_violations(field.label, f"{at} label"))
            errors.extend(_localized_violations(field.description, f"{at} description"))
    return errors


def render_typed_template(value: Any, params: Dict[str, Any]) -> Any:
    """
    Render a template value preserving parameter types: a string that is
    exactly one known placeholder substitutes to the parameter's value
    as-is (an int stays an int), anything else renders as a string.
    Non-string values pass through untouched.
    """
    if not isinstance(value, str):
        return value
    match = _TEMPLATE_PATTERN.fullmatch(value)
    # a filtered placeholder has converted its value, so it renders as
    # the string the conversion produced rather than passing through
    if match and not match.group(2) and match.group(1) in params:
        return params[match.group(1)]
    return render_template(value, params)


def render_kv_transfer_config(
    config: CacheProviderKVTransferConfig, params: Dict[str, Any]
) -> List[str]:
    """Serialize a structured connector slot into its argument pair:
    [flag, compact JSON payload]."""
    payload: Dict[str, Any] = {
        "kv_connector": config.kv_connector,
    }
    if config.kv_connector_module_path:
        payload["kv_connector_module_path"] = render_typed_template(
            config.kv_connector_module_path, params
        )
    payload["kv_role"] = config.kv_role
    if config.kv_connector_extra_config:
        payload["kv_connector_extra_config"] = {
            key: render_typed_template(value, params)
            for key, value in config.kv_connector_extra_config.items()
        }
    return [config.flag, json.dumps(payload, separators=(",", ":"))]


def render_injection(
    integration: CacheProviderIntegration, params: Dict[str, Any]
) -> Tuple[Dict[str, str], List[str], Dict[str, str]]:
    """
    Render an integration entry's injection templates into
    (env, args, files). The structured connector slot (if declared)
    renders ahead of the free-form args. Env entries whose rendered
    value is empty are dropped so that unset optional parameters (e.g.
    chunk_size) don't produce invalid engine config; file contents keep
    empty renderings — a config file's schema decides what an empty
    field means.
    """
    env: Dict[str, str] = {}
    for key, value in (integration.injection.env or {}).items():
        rendered = render_template(value, params)
        if rendered:
            env[key] = rendered
    args: List[str] = []
    if integration.injection.kv_transfer_config is not None:
        args.extend(
            render_kv_transfer_config(integration.injection.kv_transfer_config, params)
        )
    args.extend(
        render_template(arg, params) for arg in (integration.injection.args or [])
    )
    files = {
        path: render_template(content, params)
        for path, content in (integration.injection.files or {}).items()
    }
    return env, args, files


_DEVICE_FREE_IMAGE_BACKENDS = ("cuda", "rocm")
"""Image families whose builds run on a node with no accelerator, in the order
a derived version borrows one from.

Runner images are built per accelerator, and the catalog publishes no "cpu"
one, so what a node without a device runs has to come from a family whose
binaries load without a driver. Measured, not inferred: the cache server starts
under ``docker run`` with no device present in a cuda image and in a rocm one,
and fails in a cann image with ``libascend_hal.so: cannot open shared object
file`` — that family links the host's driver library outright, and the host
mounts it only where a device was asked for. A family stays out of this tuple
until it has been run that way.

A version carrying none of them offers nothing to such a node, which the
support matrix reports where the instance would start rather than leaving it to
a container that cannot load. Little is lost: what lands there holds no cache —
a coordinator, or a store on a RAM-rich host — and a component that needs a
device is placed where its engines are.
"""


def derive_runner_versions(
    provider: CacheProvider, runners: List[Any]
) -> Dict[str, CacheProviderVersionConfig]:
    """The release line a provider's ``runner_dependency`` produces from the
    runner catalog: one version per distinct package version, carrying the
    images that were probed to hold it.

    Images are keyed by the runner's own backend version rather than its major,
    which ``resolve_image`` already matches the way an inference backend does —
    so a node runs the newest image at or below its runtime instead of whatever
    the major happens to name.

    Skipped, and why none of them is the same as "unsupported":

    - ``deprecated`` rows, which are images an installation still has but
      should stop starting.
    - an accelerator family the declaration does not name, whatever its images
      hold.
    - rows whose ``dependencies`` is None, meaning the image was never probed,
      and rows whose probe ran and does not list the package. The two mean
      different things — unknown, and absent — and yield the same nothing: an
      image nobody has looked inside is not one to build a release line from.
    """
    families = {name.lower() for name in provider.runner_frameworks}
    by_version: Dict[str, Dict[str, Dict[str, str]]] = {}
    engine_of: Dict[Tuple[str, str, str], str] = {}
    for runner in runners:
        if getattr(runner, "deprecated", False):
            continue
        # An accelerator the declaration does not name is left out whatever
        # its images hold: the probe finds the package, and the provider
        # answers for whether it works there.
        if families and runner.backend.lower() not in families:
            continue
        dependencies = getattr(runner, "dependencies", None)
        if not dependencies:
            continue
        found = dependencies.get(provider.runner_dependency)
        if not found:
            continue
        # A variant is a build of its own, so it gets a key of its own: the
        # four Ascend SoC generations are four images, and they do not all
        # carry the same packages — 310p has no Mooncake where 910b does.
        # Collapsing them onto the family would hand a node an image built
        # for another generation, or one the probe says lacks the package.
        backend = runner.backend
        if runner.backend_variant:
            backend = f"{backend}-{runner.backend_variant}"
        # Several engine releases can carry one package version, and the
        # platform's own images repeat across host architectures. Both land
        # on this coordinate, so the newest engine build takes it: the
        # package is the same either way, and that is the build the fleet's
        # engines are moving to. Left to arrival order, the pick would be
        # whatever the catalog happens to list last.
        coordinate = (found, backend, runner.backend_version)
        incumbent = engine_of.get(coordinate)
        if incumbent and compare_versions(runner.service_version, incumbent) <= 0:
            continue
        engine_of[coordinate] = runner.service_version
        backends = by_version.setdefault(found, {})
        backends.setdefault(backend, {})[runner.backend_version] = runner.docker_image

    versions: Dict[str, CacheProviderVersionConfig] = {}
    for version, runtime_images in by_version.items():
        # A node with no accelerator is one more entry in the matrix rather
        # than a case beside it, and nothing publishes an image for it, so it
        # borrows one from a family that runs without a device. Keyed by the
        # version it was borrowed under, which says where it came from and
        # leaves the map one shape; a version with no such family gets no
        # entry, and the matrix then reports the node as unserved.
        if CPU_BACKEND not in runtime_images:
            for family in _DEVICE_FREE_IMAGE_BACKENDS:
                by_runtime = runtime_images.get(family)
                if not by_runtime:
                    continue
                newest = max(by_runtime, key=cmp_to_key(compare_versions))
                runtime_images[CPU_BACKEND] = {newest: by_runtime[newest]}
                break
        versions[version] = CacheProviderVersionConfig(runtime_images=runtime_images)
    return versions


def with_runner_versions(
    providers: List[CacheProvider], runners: Optional[List[Any]]
) -> List[CacheProvider]:
    """The catalog with every derived provider's release line filled in.

    Rebuilt through the model rather than assigned onto it, so a derived
    provider is held to the same rules as a declared one — the checks that read
    versions run on what will actually serve.

    A provider whose dependency no runner image carries keeps no versions, which
    is how an accelerator with nothing to run it says so: the card is listed and
    offers no version, rather than offering one that is not there.
    """
    if not runners:
        return providers
    resolved: List[CacheProvider] = []
    for provider in providers:
        if not provider.runner_dependency:
            resolved.append(provider)
            continue
        versions = derive_runner_versions(provider, runners)
        data = provider.model_dump()
        data["versions"] = {
            name: config.model_dump() for name, config in versions.items()
        }
        # Tolerant ordering, as everywhere a version string off a probe is
        # compared: these are whatever a wheel calls itself, and one that is
        # not PEP 440 would take the whole catalog build down with it.
        data["default_version"] = (
            max(versions, key=cmp_to_key(compare_versions)) if versions else None
        )
        resolved.append(CacheProvider(**data))
    return resolved
