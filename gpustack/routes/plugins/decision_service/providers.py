"""The decision-service plugin's providers catalogue: how ``type=gpustack-lb-typesafe``
ModelProviders become the wasm plugin's ``defaultConfig.providers`` entries.

One entry per decision-service ModelProvider, id ``provider-{id}`` (the same
registry-name rule ai-proxy uses, so the McpBridge registry
``provider_registry()`` already maintains is the cluster behind it).

Every entry carries an explicit ``endpoint`` — a custom base url when the
provider config sets one, otherwise the TypeSafe hosted default
(``https://api.typesafe.ai``) — and the plugin-side ``type`` is uniformly
``systemone`` (the generic implementation; per review the plugin does not
distinguish flavors yet).

Every entry also carries an explicit ``clusters`` list: the plugin derives
``outbound|<port>||<host>`` from the endpoint only when no cluster is
configured, which matches a registry-generated name only when the registry
name equals the endpoint host — never the case for ``provider-{id}`` names.
The registry-generated form is ``outbound|<port>||<name>.<static|dns>``.

The catalogue is maintained per entry, the ai-proxy discipline: each half
has exactly one writer (Model events own ``gpustack-model-{id}`` entries,
ModelProvider events own ``provider-{id}`` ones), and a reconcile merges
its own entry into the live CR while every other entry, the static half
and the matchRules carry over untouched. A scoped sync never creates the
CR — creation belongs to the route plugin's create base, which renders the
full catalogue.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.ext.asyncio import AsyncSession

from gpustack.config.config import Config
from gpustack.gateway.client.extensions_higress_io_v1_api import WasmPluginSpec
from gpustack.schemas.model_provider import (
    ModelProvider,
    ModelProviderTypeEnum,
    TypesafeConfig,
)
from gpustack.schemas.models import (
    Model,
    ModelInstance,
    ModelInstanceStateEnum,
    is_decision_model,
    servable_instances,
)
from gpustack.schemas.clusters import Cluster
from gpustack.schemas.users import Principal, is_default_cluster_principal
from gpustack.schemas.workers import Worker

if TYPE_CHECKING:
    from gpustack.gateway.utils import McpBridgeRegistry

logger = logging.getLogger(__name__)

# The wasm plugin ships as gpustack-lb-decision-service: CR name, module
# path and gateway_plugin override key all follow it.
DECISION_SERVICE_CR_NAME = "gpustack-lb-decision-service"

# The single provider type that feeds the decision-service plugin today.
# Kept as an explicit set (not a naming-prefix convention) plus one
# predicate so every consumer — discovery categorisation, the test
# endpoints, route-target validation, catalogue generation and event
# handling — answers "is this a decision provider" through the same place;
# a future decision-engine type joins this set instead of duplicating the
# checks.
DECISION_PROVIDER_TYPES = frozenset({ModelProviderTypeEnum.GPUSTACK_LB_TYPESAFE})


def is_decision_config(config: Any) -> bool:
    """Whether a provider config feeds the decision-service plugin (and is
    therefore NOT an inference upstream)."""
    return isinstance(config, TypesafeConfig)


class DecisionServiceOverride(BaseModel):
    """Deployment-level knobs of the plugin's defaultConfig — the only place
    they are configurable. Route sections never override them; operators
    inject them through ``gateway_plugin["gpustack-lb-decision-service"].config``."""

    model_config = ConfigDict(extra="forbid")

    decisionTimeoutMs: Optional[int] = Field(default=None, ge=1)
    maxStateBytes: Optional[int] = Field(default=None, ge=1)
    maxBodyBytes: Optional[int] = Field(default=None, ge=1)


def decision_default_config(cfg: Optional[Config]) -> Dict[str, Any]:
    """The CR defaultConfig's static half: the inert switch (the plugin is
    instantiated on every route but does nothing without a matchRule) plus
    the decision knobs, with operator overrides layered on top."""
    from gpustack.gateway.plugins import plugin_entry

    config: Dict[str, Any] = {
        "enabled": False,
        "decisionTimeoutMs": 3000,
        "maxStateBytes": 65536,
        "maxBodyBytes": 104857600,
    }
    entry = plugin_entry(DECISION_SERVICE_CR_NAME, cfg)
    if entry is not None and entry.config:
        try:
            override = DecisionServiceOverride.model_validate(entry.config)
        except Exception as e:
            logger.error(
                "Invalid gateway_plugin.%s.config; using defaults: %s",
                DECISION_SERVICE_CR_NAME,
                e,
            )
        else:
            config.update(override.model_dump(exclude_none=True))
    return config


def decision_provider_entry(provider: ModelProvider) -> Optional[Dict[str, Any]]:
    """One ModelProvider's catalogue entry.

    The McpBridge registry the provider machinery maintains
    (``provider-{id}``) supplies the cluster. The endpoint is the provider's
    custom base url, or the TypeSafe hosted default when none is set; the
    plugin-side ``type`` is uniformly ``systemone``.
    """
    from gpustack.gateway import utils as gateway_utils

    config = provider.config
    if not isinstance(config, TypesafeConfig):
        return None
    entry: Dict[str, Any] = {
        "id": gateway_utils.provider_registry_name(provider.id),
        "type": "systemone",
        "endpoint": config.get_base_url(),
    }
    if config.model:
        entry["model"] = config.model
    # Canonical credential form: the plugin tries the keys in order and
    # retries the callout with the next one on 401/403/429 (invalid or
    # quota-exhausted key); the legacy singular apiToken is deprecated.
    if provider.api_tokens:
        entry["apiTokens"] = list(provider.api_tokens)
    # The cluster always comes from the provider's own ``provider-{id}``
    # registry — the registry name never equals the endpoint host, so the
    # endpoint-derived name the plugin would fall back to never resolves.
    registry = gateway_utils.provider_registry(provider)
    if registry is None:
        # Without a registry the plugin would derive the cluster from the
        # endpoint host, which never matches a provider-{id} registry name
        # -- the entry could not resolve, so it is skipped rather than
        # deployed broken.
        logger.warning(
            "decision-service ModelProvider %s (id %s) has no registry; skipping "
            "the decision-service catalogue entry",
            provider.name,
            provider.id,
        )
        return None
    entry["clusters"] = [f"outbound|{registry.port}||{registry.get_service_name()}"]
    return entry


async def instance_workers_by_id(
    session: AsyncSession, instances: List[ModelInstance]
) -> Dict[int, Worker]:
    """The workers the given instances run on, keyed by id.

    The gateway registers each instance's registry with its worker (see
    ``ModelController._ensure_model_mcp_bridge``): under the WORKER/TUNNEL
    proxy modes the registered address is the worker's, so the catalogue
    must resolve the same worker-aware registry or its clusters reference
    registrations the bridge never made."""
    worker_ids = {i.worker_id for i in instances if i.worker_id}
    if not worker_ids:
        return {}
    workers = await Worker.all_by_fields(
        session,
        extra_conditions=[
            Worker.id.in_(worker_ids),
            # A soft-deleted worker is a gone worker: resolving it would
            # render a cluster against a bridge registration that no
            # longer exists, defeating the stale-instance skip below.
            Worker.deleted_at.is_(None),
        ],
    )
    return {worker.id: worker for worker in workers}


def decision_model_entry(
    model: Model,
    instances: List[ModelInstance],
    workers_by_id: Optional[Dict[int, Worker]] = None,
    api_tokens: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """One locally deployed decision Model's catalogue entry.

    Every servable RUNNING instance rides the entry as one cluster in the
    ``clusters`` list — the explicit form, which the plugin prefers over
    any endpoint-derived name — and it picks one uniformly at random per
    callout, so the decision traffic spreads across the replicas. The
    ``endpoint`` (:authority source) is still required by the plugin, and
    ``clusterNameHeader`` needs wasm 0.2.1+ (wheel 0.5.1+); older builds
    ignore it, and their callouts through a shared worker-proxy entrypoint
    cannot select the member — the wheel pin is the real floor.

    ``api_tokens`` is the deployment cluster's registration token: the
    clusters may point at worker-proxy entrypoints (the WORKER/TUNNEL
    proxy modes, a remote cluster gateway) that authenticate with it —
    the same credential ai-proxy attaches for the deployment's inference
    traffic (``cluster_registration_tokens``). A cluster without a token
    renders no ``apiTokens``, the anonymous callout a DIRECT instance
    (nothing between the gateway and the inference server) accepts.
    """
    from gpustack.gateway import utils as gateway_utils

    resolved = []
    for instance in instances:
        address = instance.worker_advertise_address or instance.worker_ip
        if not address or instance.port is None:
            continue
        # Worker-aware on purpose: the gateway registers each instance's
        # registry with its worker, and the WORKER/TUNNEL proxy modes point
        # it at the worker's address (a tunnel without its proxy address
        # registers nothing at all) — a worker-blind registry here would
        # name a cluster the bridge never registered.
        worker = None
        if instance.worker_id is not None:
            worker = (workers_by_id or {}).get(instance.worker_id)
            if worker is None:
                # A gone worker row means the instance is stale (a deleted
                # worker takes its instances with it): the worker-blind
                # fallback would name a dead address, so the instance is
                # skipped like any other unrenderable member.
                continue
        registry = gateway_utils.model_instance_registry(instance, worker=worker)
        if registry is None:
            continue
        resolved.append((instance, address, registry))
    if not resolved:
        return None

    def _cluster_name(registry) -> str:
        return f"outbound|{registry.port}||{registry.get_service_name()}"

    first_instance, first_address, _ = resolved[0]
    # The served model name is the natural decision-engine alias for a
    # local engine (vLLM/SGLang accept any name and ignore the
    # distinction); a route's decisionModel overrides it.
    entry: Dict[str, Any] = {
        "id": gateway_utils.model_ai_proxy_provider_id(model.id),
        "type": "systemone",
        "endpoint": f"http://{first_address}:{first_instance.port}",
        "model": model.name,
        "clusters": [_cluster_name(registry) for _, _, registry in resolved],
        # The worker proxy behind a shared entrypoint (WORKER/TUNNEL proxy
        # modes: every instance of a worker rides the one registry this
        # cluster names) selects the member by this header. The plugin
        # derives the value from the cluster it picks per callout — the
        # same segment-4 service name gpustack-set-header-pre-route
        # writes for routed traffic — so the value follows the pick by
        # construction and needs no per-cluster mapping here. Harmless —
        # ignored by the server — where the cluster points straight at
        # the instance.
        "clusterNameHeader": gateway_utils.router_header_key,
    }
    if api_tokens:
        # The token authenticates the callout at the worker-proxy
        # entrypoints the clusters may name; the plugin retries the
        # callout with the next entry on 401/403/429.
        entry["apiTokens"] = list(api_tokens)
    return entry


def _remote_cluster_entry(
    model: Model, registry: "McpBridgeRegistry", api_tokens: Optional[List[str]]
) -> Dict[str, Any]:
    """The entry a remote-cluster deployment renders: one cluster — the
    cluster gateway entrypoint its inference traffic is routed through —
    carrying the cluster's registration token. Shared by the single-model
    render and the batch render so the two cannot drift."""
    from gpustack.gateway import utils as gateway_utils

    entry: Dict[str, Any] = {
        "id": gateway_utils.model_ai_proxy_provider_id(model.id),
        "type": "systemone",
        "endpoint": f"http://{registry.domain}",
        "model": model.name,
        "clusters": [f"outbound|{registry.port}||{registry.get_service_name()}"],
    }
    if api_tokens:
        entry["apiTokens"] = list(api_tokens)
    return entry


async def decision_cluster_context(
    session: AsyncSession, cluster_ids: set
) -> Tuple[Dict[int, List[str]], Dict[int, "McpBridgeRegistry"]]:
    """One Cluster read for both halves of the cluster context: each
    cluster's registration token (the worker-proxy entrypoints and a
    remote gateway entry authenticate with it) and the gateway entrypoint
    of each *remote* cluster (a cluster whose system principal is the
    default one is local — its deployments are reached through their own
    instance registries, not a gateway). Two queries total, however many
    clusters are named."""
    from gpustack.gateway import utils as gateway_utils

    ids = [cluster_id for cluster_id in cluster_ids if cluster_id is not None]
    if not ids:
        return {}, {}
    clusters = await Cluster.all_by_fields(
        session,
        extra_conditions=[Cluster.id.in_(ids)],
    )
    tokens = {
        cluster.id: [cluster.registration_token]
        for cluster in clusters
        if cluster.registration_token
    }
    with_principal = [
        cluster for cluster in clusters if cluster.system_principal_id is not None
    ]
    principals = []
    if with_principal:
        principals = await Principal.all_by_fields(
            session,
            extra_conditions=[
                Principal.id.in_([c.system_principal_id for c in with_principal])
            ],
        )
    default_principal_ids = {
        p.id for p in principals if is_default_cluster_principal(p)
    }
    registries = {}
    for cluster in with_principal:
        if cluster.system_principal_id in default_principal_ids:
            continue
        # None for a remote cluster without a gateway endpoint: the
        # deployment renders through its instances instead.
        registry = gateway_utils.cluster_registry(cluster)
        if registry is not None:
            registries[cluster.id] = registry
    return tokens, registries


def decision_model_entries(
    models: List[Model],
    instances: List[ModelInstance],
    workers_by_id: Optional[Dict[int, Worker]] = None,
    api_tokens_by_cluster_id: Optional[Dict[int, List[str]]] = None,
    cluster_registry_by_id: Optional[Dict[int, Any]] = None,
) -> List[Dict[str, Any]]:
    """The locally deployed half of the catalogue: one entry per decision
    Model that has a RUNNING servable instance, model-id-sorted for a
    byte-stable CR. A model whose ``cluster_id`` maps to a cluster gateway
    in ``cluster_registry_by_id`` renders the gateway entry instead (the
    caller resolves those, so the batch stays synchronous); passing no map
    treats every model as local.

    A model whose instances are all down is absent rather than present with
    dead clusters: routes selecting it fail open (their rule is stripped)
    until an instance is back, which the model-event-driven catalogue sync
    re-renders. Every servable RUNNING instance enters the entry's
    ``clusters`` list (instance-id order, byte-stable) — the plugin spreads
    the callouts across them — with the lowest-id instance also serving the
    ``endpoint``.
    """
    instances_by_model: Dict[int, List[ModelInstance]] = {}
    for instance in instances:
        if instance.model_id is None:
            continue
        if instance.state != ModelInstanceStateEnum.RUNNING:
            continue
        if getattr(instance, "deleted_at", None) is not None:
            continue
        instances_by_model.setdefault(instance.model_id, []).append(instance)

    tokens_by_cluster = api_tokens_by_cluster_id or {}
    registries_by_cluster = cluster_registry_by_id or {}
    entries: List[Dict[str, Any]] = []
    for model in sorted(models, key=lambda m: m.id):
        if model.deleted_at is not None or not is_decision_model(model):
            continue
        registry = registries_by_cluster.get(model.cluster_id)
        if registry is not None:
            entries.append(
                _remote_cluster_entry(
                    model, registry, tokens_by_cluster.get(model.cluster_id)
                )
            )
            continue
        running = servable_instances(model, instances_by_model.get(model.id, []))
        candidates = sorted(
            running,
            key=lambda inst: inst.id if inst.id is not None else 0,
        )
        if not candidates:
            continue
        try:
            entry = decision_model_entry(
                model,
                candidates,
                workers_by_id,
                api_tokens=(api_tokens_by_cluster_id or {}).get(model.cluster_id),
            )
        except Exception as e:
            logger.warning(
                "decision-service catalogue entry for model %s (id %s) "
                "could not be built; skipping it: %s",
                model.name,
                model.id,
                e,
            )
            continue
        if entry is not None:
            entries.append(entry)
    return entries


def decision_model_referencing_route_ids(
    policies: List[Any], model_id: int
) -> "set[int]":
    """Route ids whose stored decision-service policy pins ``model_id`` —
    the local-target twin of ``decision_referencing_route_ids``: the routes
    a model event (instances up/down, category moved) has to re-reconcile."""
    return {
        policy.route_id
        for policy in policies or []
        if (policy.config or {}).get("modelId") == model_id
    }


def decision_provider_entries(providers: List[ModelProvider]) -> List[Dict[str, Any]]:
    """The full catalogue: one entry per decision-service ModelProvider,
    id-sorted for a byte-stable CR.

    One malformed provider must not block the rest: the catalogue spans
    every organization, so a bad entry is skipped with a warning and the
    valid endpoints and credentials still sync.

    There is deliberately NO synthetic hosted-default entry: every entry
    names a ModelProvider, whose endpoint is that provider's custom base
    url or the hosted default — one entry per configured provider, nothing
    invented.
    """
    entries: List[Dict[str, Any]] = []
    for provider in sorted(providers, key=lambda p: p.id):
        if not is_decision_config(provider.config):
            continue
        try:
            entry = decision_provider_entry(provider)
        except Exception as e:
            logger.warning(
                "decision-service catalogue entry for provider %s (id %s) "
                "could not be built; skipping it: %s",
                provider.name,
                provider.id,
                e,
            )
            continue
        if entry is not None:
            entries.append(entry)
    return entries


def decision_module_available(cfg: Optional[Config]) -> bool:
    """Whether the shipped plugins manifest can deploy the module. The
    module ships from gpustack-higress-plugins 0.5.1 (wasm 0.2.1); older
    installs keep the catalogue sync silent rather than failing every
    provider reconcile."""
    from gpustack.gateway.plugins import plugin_spec_overrides

    try:
        plugin_spec_overrides(DECISION_SERVICE_CR_NAME, cfg=cfg)
    except ValueError:
        return False
    return True


def _catalogue_sort_key(entry: Dict[str, Any]) -> Tuple[int, int, str]:
    """Order entries the way a full render does — provider entries first,
    model entries second, each by numeric id — so scoped writes and the
    route plugin's recreate base (the one remaining full render) converge
    on the same list instead of fighting over its order."""
    entry_id = str(entry.get("id", ""))
    for bucket, prefix in enumerate(("provider-", "gpustack-model-")):
        if entry_id.startswith(prefix):
            suffix = entry_id[len(prefix) :]
            return (bucket, int(suffix) if suffix.isdigit() else 0, entry_id)
    return (2, 0, entry_id)


def merge_decision_catalogue_entry(
    current: WasmPluginSpec,
    owned_id: str,
    entry: Optional[Dict[str, Any]],
) -> Tuple[WasmPluginSpec, bool]:
    """Replace ``owned_id``'s catalogue entry with ``entry`` (``None``
    removes it), keeping every other entry, the static half
    (url/sha256/phase/priority) and the matchRules untouched — the decision
    twin of ``compare_and_append_default_proxy_config``.

    Unlike the ai-proxy merge there is deliberately NO
    rules-referenced garbage collection: the CR's matchRules belong to the
    route plugin and a catalogue entry with no route referencing it yet is
    the normal, catalogue-first state. An owner removes its entry itself.

    Returns the merged spec and whether the providers list actually
    changed — the caller's route re-notification gate.

    There is deliberately no ``activeProviderId`` in a merged
    defaultConfig: an id that names no catalogue entry makes the plugin's
    config parse fail outright, and the routes select their provider
    through the matchRule's activeProviderId instead.
    """
    if current.defaultConfig is None:
        current.defaultConfig = {}
    providers = list(current.defaultConfig.get("providers") or [])
    merged = [e for e in providers if e.get("id") != owned_id]
    if entry is not None:
        merged.append(entry)
    merged.sort(key=_catalogue_sort_key)
    current.defaultConfig["providers"] = merged
    return current, merged != providers


async def _ensure_decision_entry(
    cfg: Optional[Config],
    extensions_api: Any,
    owned_id: str,
    entry: Optional[Dict[str, Any]],
) -> bool:
    """Merge one owned entry into the live CR, reporting whether it
    changed. An absent CR stays absent: the scoped syncs never create it
    (a spec without the static half is not deployable — the route plugin's
    create base owns creation, rendering the full catalogue)."""
    from gpustack.gateway import utils as gateway_utils

    changed = False

    def _diff(current_spec: Optional[WasmPluginSpec]) -> Optional[WasmPluginSpec]:
        nonlocal changed
        if current_spec is None:
            return None
        _, changed = merge_decision_catalogue_entry(current_spec, owned_id, entry)
        return current_spec

    # Positional binding: ensure_wasm_plugin invokes spec_diff(current_spec)
    # positionally, and a keyword-bound first argument collides with it.
    await gateway_utils.ensure_wasm_plugin(
        api=extensions_api,
        name=DECISION_SERVICE_CR_NAME,
        namespace=cfg.gateway_namespace,
        spec_diff=_diff,
    )
    return changed


async def render_decision_model_entry(
    session: AsyncSession,
    model_id: int,
) -> Optional[Dict[str, Any]]:
    """The DB half of the model entry sync: render ``gpustack-model-{id}``
    from the model's own rows.

    A deployment in a remote cluster renders one cluster — the cluster
    gateway entrypoint, exactly where inference traffic for that
    deployment is routed: its workers are reachable only through it, so a
    per-instance registry would name an unreachable private address while
    ordinary inference keeps working. The gateway routes on its side and
    the cluster's registration token authenticates at the entrypoint.

    Locally, every servable RUNNING instance contributes its registry to
    the ``clusters`` list. A model that is gone, soft-deleted, no longer a
    decision deployment, or with no renderable target renders to ``None``
    — removal — absent rather than present with dead clusters, the rule
    the full render applies too. Whether the row was a decision model
    does not have to be known up front: removing an id the catalogue never
    held is a no-op, which is what keeps an id-only DELETE at one entry
    instead of a whole-catalogue rebuild over the surviving rows."""
    model = await Model.one_by_id(session, model_id)
    if model is None or model.deleted_at is not None or not is_decision_model(model):
        return None
    # One Cluster read carries both halves of the context: the gateway
    # entrypoint (remote) and the registration token (either side).
    tokens, registries = await decision_cluster_context(session, {model.cluster_id})
    cluster_registry = registries.get(model.cluster_id)
    if cluster_registry is not None:
        return _remote_cluster_entry(
            model, cluster_registry, tokens.get(model.cluster_id)
        )
    instances = await ModelInstance.all_by_fields(
        session,
        fields={"model_id": model.id, "deleted_at": None},
    )
    workers_by_id = await instance_workers_by_id(session, instances)
    rendered = decision_model_entries([model], instances, workers_by_id, tokens)
    return rendered[0] if rendered else None


async def render_all_decision_model_entries(
    session: AsyncSession,
) -> List[Dict[str, Any]]:
    """Render every decision model's catalogue entry — the recreate base's
    full render. Bounded queries (one batch per data source, however many
    decision models exist), each entry rendered through the same helpers
    the single-model render uses so the batch and the scoped syncs cannot
    drift."""
    models = await Model.all_by_field(session, "deleted_at", None)
    decision_models = [m for m in models if is_decision_model(m)]
    if not decision_models:
        return []
    cluster_ids = {m.cluster_id for m in decision_models}
    tokens, registries = await decision_cluster_context(session, cluster_ids)
    # Remote deployments render their gateway entry without instance rows
    # (their workers are not in this database), so only the local ones
    # contribute to the instance fetch.
    local_model_ids = [m.id for m in decision_models if m.cluster_id not in registries]
    instances = []
    if local_model_ids:
        instances = await ModelInstance.all_by_fields(
            session,
            fields={"deleted_at": None},
            extra_conditions=[ModelInstance.model_id.in_(local_model_ids)],
        )
    workers_by_id = await instance_workers_by_id(session, instances)
    return decision_model_entries(models, instances, workers_by_id, tokens, registries)


async def ensure_decision_model_entry(
    cfg: Optional[Config],
    extensions_api: Any,
    model_id: int,
    entry: Optional[Dict[str, Any]],
) -> bool:
    """The CR half of the model entry sync: merge the rendered entry under
    ``gpustack-model-{id}`` and report whether it changed."""
    from gpustack.gateway import utils as gateway_utils

    if not decision_module_available(cfg):
        return False
    return await _ensure_decision_entry(
        cfg,
        extensions_api,
        gateway_utils.model_ai_proxy_provider_id(model_id),
        entry,
    )


async def sync_decision_model_entry(
    cfg: Optional[Config],
    session: AsyncSession,
    extensions_api: Any,
    model_id: int,
) -> bool:
    """Render and merge ONE model's catalogue entry in one call — the
    composition of :func:`render_decision_model_entry` and
    :func:`ensure_decision_model_entry`. The Model controller calls the
    halves separately so the route re-notification can run between them."""
    entry = await render_decision_model_entry(session, model_id)
    return await ensure_decision_model_entry(cfg, extensions_api, model_id, entry)


async def sync_decision_provider_entry(
    cfg: Optional[Config],
    session: AsyncSession,
    extensions_api: Any,
    provider_id: int,
) -> bool:
    """Reconcile ONE ``gpustack-lb-typesafe`` ModelProvider's catalogue
    entry under ``provider-{id}`` — the provider-events twin of
    :func:`sync_decision_model_entry`. A provider that is gone,
    soft-deleted, moved off the decision types, or without a renderable
    registry merges to removal."""
    from gpustack.gateway import utils as gateway_utils

    if not decision_module_available(cfg):
        return False
    entry = None
    provider = await ModelProvider.one_by_id(session, provider_id)
    if provider is not None and provider.deleted_at is None:
        if is_decision_config(provider.config):
            # None for a decision provider whose registry cannot resolve:
            # the entry could not be reached, so the catalogue must not
            # hold it.
            entry = decision_provider_entry(provider)
    return await _ensure_decision_entry(
        cfg,
        extensions_api,
        gateway_utils.provider_registry_name(provider_id),
        entry,
    )
