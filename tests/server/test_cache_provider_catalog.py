import json

import pytest
from pydantic import ValidationError

from gpustack.schemas.cache_providers import (
    CacheProvider,
    CacheProviderL2Backend,
    CacheProviderL2Field,
    CacheProviderVersionConfig,
    localized_default,
    localized_values,
    render_argument,
    render_l2_adapter,
    render_template,
    render_typed_template,
    resolved_field_values,
    validate_injection_templates,
    validate_localized_text,
)
from gpustack.server import cache_provider_catalog
from gpustack.server.cache_provider_catalog import (
    asset_providers,
    render_injection,
)


def _asset_provider(name: str):
    """One declaration out of what this installation carries.

    These tests read the packaged catalog, not what a cluster serves — the
    serving catalog is a table the leader materializes, and its contents are
    whatever document an admin configured.
    """
    wanted = name.lower()
    return next(
        (provider for provider in asset_providers() if provider.name.lower() == wanted),
        None,
    )


def test_catalog_asset_loads():
    providers = asset_providers()
    assert providers, "bundled cache-providers.yaml should yield at least one provider"


def test_malformed_entry_costs_only_its_own_provider(monkeypatch):
    """A declaration the model rejects — including one that is not a
    mapping at all — is skipped on its own; the rest of the catalog still
    serves, so a bad edit degrades one provider instead of every cache
    service in the deployment."""
    asset = (
        "- just a string\n"
        "- name: Broken\n"
        "  versions:\n"
        "    \"v1.0\": {}\n"  # resolves to no image
        "- name: Good\n"
        "  default_image: \"repo/cache:{{version}}\"\n"
        "  versions:\n"
        "    \"v1.0\": {}\n"
    )

    class _Asset:
        def is_file(self):
            return True

        def read_text(self, encoding=None):
            return asset

    try:
        monkeypatch.setattr(cache_provider_catalog, "files", lambda _package: _Asset())
        monkeypatch.setattr(_Asset, "joinpath", lambda self, _name: self, raising=False)
        providers = asset_providers()
        assert [provider.name for provider in providers] == ["Good"]
    finally:
        monkeypatch.undo()


def test_a_placeholder_provider_is_listed_with_nothing_to_launch():
    """A provider this installation does not carry still has a card that
    reads like any other — what it is, what it attaches to, where to read
    about it — and nothing behind it: no version, no image, no component.
    A declaration that launches nothing is held to none of the rules
    about the two."""
    provider = _asset_provider("Mooncake")
    assert localized_default(provider.unavailable_reason)
    assert localized_default(provider.description)
    # the card's accelerator tags: with no images to read them off, the
    # scope of its engine integrations is the claim
    assert [
        integration.frameworks
        for integration in provider.inference_backend_integrations
    ] == [["cuda"], ["cann"]]

    assert provider.versions == {}
    assert provider.custom_version is False
    assert provider.components == {}


def test_a_plugin_asset_replaces_the_placeholder_in_place(monkeypatch):
    """A plugin carrying what a provider needs ships its declaration,
    which replaces the placeholder of the same name where it already sits
    — the catalog's order is the order the cards are read in, and it must
    not shuffle because a plugin is installed."""
    asset = (
        "- name: Mooncake\n"
        '  default_image: "repo/mooncake:{{version}}"\n'
        "  versions:\n"
        '    "v1.0": {}\n'
    )

    class _Asset:
        def is_file(self):
            return True

        def read_text(self, encoding=None):
            return asset

        def joinpath(self, _name):
            return self

    class _Plugin:
        @classmethod
        def cache_provider_assets(cls):
            return [("plugin.assets", "cache-providers.yaml")]

    bundled_files = cache_provider_catalog.files

    def fake_files(package):
        if package == "plugin.assets":
            return _Asset()
        return bundled_files(package)

    try:
        monkeypatch.setattr(cache_provider_catalog, "files", fake_files)
        monkeypatch.setattr(
            "gpustack.extension.iter_plugin_classes",
            lambda: iter([("plugin", _Plugin)]),
        )
        providers = asset_providers()
        names = [provider.name for provider in providers]
        assert names.index("Mooncake") == 1
        mooncake = _asset_provider("Mooncake")
        assert mooncake.unavailable_reason is None
        assert list(mooncake.versions) == ["v1.0"]
    finally:
        monkeypatch.undo()


def test_provider_defaults_fold_into_every_version():
    """The provider-level templates are resolved at construction, so a
    version config is self-contained: consumers read one effective image
    and command off it, and {{version}} keeps the version string stated
    once instead of copied into every tag."""
    provider = CacheProvider(
        name="Templated",
        default_image="registry/cache:{{version}}",
        default_runtime_images={"cuda": {"12": "registry/cache:{{version}}-cu12"}},
        default_run_command="cache serve --port {{port}}",
        versions={
            "v1.0": {},
            # A version departing from the layout keeps its own images,
            # and its declared map replaces the default whole.
            "v2.0": {
                "image": "other/cache:2.0",
                "runtime_images": {"cann": {"8": "other/cache:2.0-cann"}},
            },
        },
    )

    templated = provider.versions["v1.0"]
    assert templated.image == "registry/cache:v1.0"
    assert templated.runtime_images == {"cuda": {"12": "registry/cache:v1.0-cu12"}}
    assert templated.run_command == "cache serve --port {{port}}"

    explicit = provider.versions["v2.0"]
    assert explicit.image == "other/cache:2.0"
    assert explicit.runtime_images == {"cann": {"8": "other/cache:2.0-cann"}}
    # runtime_images doubles as the support matrix, so a replaced map
    # narrows the accelerators the version serves.
    assert explicit.supports_runtime("cann") is True
    assert explicit.supports_runtime("cuda") is False

    # Resolution is per version: the default map is copied, never shared.
    assert templated.runtime_images is not provider.default_runtime_images


def test_own_image_takes_over_the_layout_whole():
    """image and runtime_images are one layout: a version off the
    provider's tag scheme must not serve some accelerators from its own
    registry and the rest from the provider's template."""
    provider = CacheProvider(
        name="Templated",
        default_image="registry/cache:{{version}}",
        default_runtime_images={"cuda": {"12": "registry/cache:{{version}}-cu12"}},
        versions={"v1.0": {"image": "vendor/cache:one-off"}},
    )

    version = provider.versions["v1.0"]
    assert version.image == "vendor/cache:one-off"
    assert version.runtime_images == {}
    # With no layout of its own, every node runs the declared image.
    assert version.resolve_image("cuda", "12.8") == "vendor/cache:one-off"


def test_version_without_any_image_is_rejected():
    """An image is the one thing a managed version cannot do without;
    silently declaring none would only surface as a container that never
    starts."""
    with pytest.raises(ValidationError):
        CacheProvider(name="Imageless", versions={"v1.0": {}})


def test_provider_without_versions_must_allow_the_custom_version():
    """A provider declaring no release line resolves no image of its own;
    without the custom version its services could never start."""
    with pytest.raises(ValidationError):
        CacheProvider(name="Versionless")

    provider = CacheProvider(
        name="Versionless",
        custom_version=True,
        default_run_args="--port {{port}}",
    )
    # The provider-level launch declaration is what the service's own
    # image runs on, reached through the same version-config contract.
    template = provider.custom_version_config()
    assert template.run_args == "--port {{port}}"
    assert template.run_command is None


def test_own_launch_takes_over_the_pair_whole():
    """run_command and run_args are two slots of one launch: a version
    supplying arguments for the image's own entrypoint must not also
    inherit a command that replaces that entrypoint."""
    provider = CacheProvider(
        name="Launched",
        default_image="registry/cache:{{version}}",
        default_run_command="cache serve --port {{port}}",
        versions={
            "v1.0": {},
            "v2.0": {"run_args": "--port {{port}}"},
        },
    )

    inherited = provider.versions["v1.0"]
    assert inherited.run_command == "cache serve --port {{port}}"
    assert inherited.run_args is None

    own = provider.versions["v2.0"]
    assert own.run_command is None
    assert own.run_args == "--port {{port}}"


def test_version_declaring_both_launch_slots_is_rejected():
    """A command and its arguments concatenate into one vector either
    way, so declaring both states the same launch twice — and only one of
    them can decide whether the image's entrypoint survives."""
    with pytest.raises(ValidationError):
        CacheProvider(
            name="Ambiguous",
            default_image="registry/cache:v1",
            versions={"v1.0": {"run_command": "cache serve", "run_args": "--port 1"}},
        )


def test_component_declarations_validate():
    """Multi-component providers declare per-role topology and launch;
    dependencies must point at single-replica components (the only ones
    with one addressable endpoint) and cannot chain."""
    from gpustack.schemas.cache_providers import CacheProviderComponent

    provider = CacheProvider(
        name="Pool",
        default_image="repo/pool:{{version}}",
        versions={"v1.0": {}},
        components={
            "master": CacheProviderComponent(
                topology="replicas",
                run_command="pool-master --port {{port}}",
                metrics_port="metrics",
                attach_endpoint=True,
                gpu_access=False,
            ),
            "store": CacheProviderComponent(
                topology="per_node",
                depends_on="master",
                run_command="pool-store --port {{port}}",
                enabled_by="standalone_store",
                gpu_access=False,
            ),
        },
        fields=[{"name": "standalone_store", "type": "boolean", "default": False}],
    )
    assert provider.component_layouts() == {
        "master": "replicas",
        "store": "per_node",
    }
    assert provider.get_component("store").depends_on == "master"
    assert provider.attach_component() == "master"
    # enabled_by gates on the field value, falling back to its declared
    # default
    assert provider.component_enabled("master", None) is True
    assert provider.component_enabled("store", None) is False
    assert provider.component_enabled("store", {"standalone_store": True}) is True

    # single-component providers map the "" component — the stored
    # column value of their instance rows
    single = CacheProvider(
        name="Solo",
        topology="per_node",
        default_image="repo/solo:{{version}}",
        versions={"v1.0": {}},
    )
    assert single.component_layouts() == {"": "per_node"}
    assert single.get_component("") is None

    with pytest.raises(ValidationError):
        CacheProvider(
            name="Dangling",
            default_image="repo/x:{{version}}",
            versions={"v1.0": {}},
            components={
                "store": CacheProviderComponent(
                    topology="per_node", depends_on="ghost"
                ),
                "master": CacheProviderComponent(attach_endpoint=True),
            },
        )
    with pytest.raises(ValidationError):
        # a per_node dependency has no single address to hand out
        CacheProvider(
            name="FanDep",
            default_image="repo/x:{{version}}",
            versions={"v1.0": {}},
            components={
                "a": CacheProviderComponent(topology="per_node"),
                "b": CacheProviderComponent(
                    topology="replicas", depends_on="a", attach_endpoint=True
                ),
            },
        )
    with pytest.raises(ValidationError):
        # neither does a multi-replica one (HA addresses the leader
        # through the backend URI instead)
        CacheProvider(
            name="WideDep",
            default_image="repo/x:{{version}}",
            versions={"v1.0": {}},
            components={
                "a": CacheProviderComponent(topology="replicas", replicas=3),
                "b": CacheProviderComponent(
                    topology="replicas", depends_on="a", attach_endpoint=True
                ),
            },
        )
    with pytest.raises(ValidationError):
        CacheProviderComponent(run_command="x", run_args="y")


def test_component_port_roles_must_name_declared_ports():
    """Every role a component assigns — its address, its exposition, what
    its probe hits — points at a port it binds. A role naming a port that
    does not exist resolves to nothing at runtime: an address without a
    port, a scrape target never built, a probe that can never pass."""
    from gpustack.schemas.cache_providers import (
        CacheProviderComponent,
        CacheProviderHealthCheck,
    )

    def _provider(**component_fields) -> CacheProvider:
        return CacheProvider(
            name="Pool",
            default_image="repo/pool:{{version}}",
            versions={"v1.0": {}},
            components={
                "master": CacheProviderComponent(
                    attach_endpoint=True, **component_fields
                )
            },
        )

    # A component that declares nothing binds a service port and a
    # metrics port, which is what a cache server usually is.
    provider = _provider()
    assert provider.enabled_port_names("master") == ["port", "metrics"]
    assert provider.address_port_name("master") == "port"

    # A declared list replaces them outright, so a role with no
    # exposition holds no metrics listener.
    provider = _provider(ports=["api"], address_port="api")
    assert provider.enabled_port_names("master") == ["api"]
    assert provider.metrics_port_name("master") is None

    with pytest.raises(ValidationError):
        _provider(ports=["api"], metrics_port="metrics")
    with pytest.raises(ValidationError):
        _provider(ports=["api"], address_port="rpc")
    with pytest.raises(ValidationError):
        _provider(
            ports=["api"],
            health_check=CacheProviderHealthCheck(target="metrics"),
        )
    with pytest.raises(ValidationError):
        # a configuration closing that gate would leave the component
        # running with no address
        _provider(
            ports=[{"name": "api", "enabled_by": "p2p"}],
            address_port="api",
        )
    with pytest.raises(ValidationError):
        # rendering {{metrics_port}} without declaring one drops the flag
        # it rides on
        _provider(ports=["port", "http"], run_command="x --http {{metrics_port}}")


def test_lmcache_provider_declaration():
    provider = _asset_provider("LMCache")
    assert provider is not None
    # Managed only: LMCache is the single-container engine GPUStack runs
    # itself; reference-only distributed caches are what external is for.
    # The MP server keeps KV transfers node-local, so managed deployments
    # run one instance per worker of the cluster; attach_locality is the
    # declared contract the resolver's node-local rules key on —
    # deliberately separate from topology (a distributed pool may run
    # per-node data components while engines attach its cluster-wide
    # endpoint).
    assert provider.topology == "per_node"
    assert provider.attach_locality == "node_local"
    # /healthcheck verifies engine readiness (503 until initialized) and
    # lives on the HTTP frontend — the metrics port in our port model.
    assert provider.health_check.scheme == "http"
    assert provider.health_check.path == "/healthcheck"
    assert provider.health_check.target == "metrics"

    # One declared version, the release the runner images bundle; a
    # service may still pin its own image via the reserved "custom"
    # version.
    assert provider.default_version == "v0.5.3"
    assert set(provider.versions) == {"v0.5.3"}
    assert provider.custom_version is True

    version_config, version = provider.get_version_config()
    assert version_config is not None
    assert version == provider.default_version
    # The vLLM runners rather than upstream's image: P2P needs nixl,
    # which upstream ships as an optional extra and does not bundle. The
    # worker resolves per node, so a heterogeneous per_node fleet mixes
    # images; unknown runtimes and accelerator-less workers get the plain
    # one.
    assert version_config.image == "gpustack/runner:cuda12.9-vllm0.27.1"
    assert (
        version_config.resolve_image("cuda", "13.0")
        == "gpustack/runner:cuda13.0-vllm0.27.1"
    )
    assert (
        version_config.resolve_image("cuda", "12.8")
        == "gpustack/runner:cuda12.9-vllm0.27.1"
    )
    assert version_config.resolve_image(None, None) == (
        "gpustack/runner:cuda12.9-vllm0.27.1"
    )
    # Two components: the cache servers engines attach to, and the peer
    # registry P2P needs. A component owns its launch, so the version
    # slots carry none.
    assert set(provider.components) == {"server", "coordinator"}
    for declared in provider.versions.values():
        assert declared.run_command is None
        assert declared.run_args is None
    server = provider.components["server"]
    coordinator = provider.components["coordinator"]
    # The full CLI entry: the HTTP frontend on --http-port serves
    # /metrics (same registry as the standalone exposition) plus
    # /healthcheck and the admin APIs; --prometheus-port is ignored
    # there, so the frontend port doubles as the metrics port.
    assert server.run_command == (
        "lmcache server --host {{host}} --port {{port}} "
        "--l1-size-gb {{ram_size}} --chunk-size {{chunk_size}} "
        "--http-host {{host}} --http-port {{metrics_port}} "
        "--supported-transfer-mode auto --worker-reap-timeout-seconds 60 "
        "--eviction-policy {{eviction_policy}} "
        "--eviction-trigger-watermark {{eviction_trigger_watermark}} "
        "--eviction-ratio {{eviction_ratio}} --l1-align-bytes 65536 "
        "--coordinator-url http://{{component.coordinator.address}} "
        # the worker's own address, not the outbound IP the server would
        # guess: a peer lookup that cannot reach its target is a silent miss
        "--coordinator-advertise-ip {{worker_ip}} "
        "--p2p-advertise-url {{ports.p2p.url}}"
    )
    # Engines attach per node, and the servers hold the capacity.
    assert server.topology == "per_node"
    assert server.attach_endpoint is True
    assert server.metrics_port == "metrics"
    assert server.resource_profile.ram_gib == "{{ram_size}}"
    # The registry exists only with P2P, holds no cache and takes no GPU;
    # so does the port its peers dial, which is why both P2P flags above
    # render empty and drop while the feature is off.
    assert coordinator.enabled_by == "enable_p2p"
    assert coordinator.gpu_access is False
    assert server.depends_on == "coordinator"
    assert server.enabled_ports({}) == ["port", "metrics"]
    assert server.enabled_ports({"enable_p2p": True}) == ["port", "metrics", "p2p"]
    # A registry with nothing to scrape binds one port, not the metrics
    # port every cache server is otherwise given.
    assert coordinator.enabled_ports({"enable_p2p": True}) == ["port"]
    assert coordinator.metrics_port is None
    # The coordinator speaks HTTP, so its flag carries a scheme the
    # stamped address does not — and the whole token has to vanish with
    # the address, not leave a bare scheme behind.
    assert render_argument("http://{{component.coordinator.address}}", {}) == (
        "http://{{component.coordinator.address}}"
    )
    assert (
        render_argument(
            "http://{{component.coordinator.address}}",
            {"component.coordinator.address": None},
        )
        == ""
    )
    # Capacity, chunking and the eviction knobs are all ordinary declared
    # fields wired into the run command through their placeholders; the
    # platform reserves only host/port/metrics_port for itself.
    fields = {field.name: field for field in provider.fields}
    # Order is the form's layout: capacity, then the knobs that free it,
    # then what a reader leaves alone unless they know why — chunking
    # falls through to the engine, and P2P is off.
    assert [field.name for field in provider.fields] == [
        "ram_size",
        "eviction_policy",
        "eviction_trigger_watermark",
        "eviction_ratio",
        "chunk_size",
        "enable_p2p",
    ]
    assert fields["enable_p2p"].default is False
    # capacity always renders (required guards a cleared value, the
    # default seeds the form); chunking may fall through to the engine
    assert fields["ram_size"].required and fields["ram_size"].default == 20
    assert not fields["chunk_size"].required
    # the pre-flight sizes an instance by the capacity field
    assert provider.resource_profile.ram_gib == "{{ram_size}}"
    assert fields["eviction_policy"].default == "LRU"
    assert fields["eviction_policy"].options == ["LRU", "IsolatedLRU", "noop"]
    assert fields["eviction_trigger_watermark"].type == "number"
    # Curated defaults from the upstream deployment recipes, deliberately
    # not the CLI code defaults (0.8/0.2): retain more, evict gentler.
    assert fields["eviction_trigger_watermark"].default == 0.85
    assert fields["eviction_ratio"].default == 0.1
    # Both are 0-1 fractions: without declared bounds and a fractional
    # step, the UI stepper walks 0.8 to -0.2 in one click.
    for name in ("eviction_trigger_watermark", "eviction_ratio"):
        assert fields[name].min == 0
        assert fields[name].max == 1
        assert fields[name].step == 0.05
    # Every declared field earns its place: it either fills a placeholder
    # somewhere in the declaration or gates something (a component, a
    # port). And none shadows a reserved platform placeholder.
    declaration = provider.model_dump_json()
    gates = {component.enabled_by for component in provider.components.values()} | {
        entry.enabled_by
        for component in provider.components.values()
        for entry in component.ports
        if not isinstance(entry, str)
    }
    for name in fields:
        assert f"{{{{{name}}}}}" in declaration or name in gates
    assert not set(fields) & {"host", "port", "metrics_port"}
    # Capacity flows through --l1-size-gb on the command line, not env.
    assert not version_config.env
    assert not server.env

    compat = provider.integration_for("vLLM")
    assert compat is not None


def test_metrics_for_resolves_version_override():
    """A version carrying its own metrics block owns it whole; versions
    without one and the custom version read the provider default — and
    a service stored without an explicit version (None) resolves through
    the default version like every other version lookup, so an override
    on the default version reaches the services actually running it."""
    from gpustack.schemas.cache_providers import (
        CacheProvider,
        CacheProviderMetrics,
        CacheProviderMetricValue,
        CacheProviderVersionConfig,
    )

    default = CacheProviderMetrics(
        mappings={"hit_rate": CacheProviderMetricValue(gauge="old_name")}
    )
    renamed = CacheProviderMetrics(
        mappings={"hit_rate": CacheProviderMetricValue(gauge="new_name")}
    )
    provider = CacheProvider(
        name="X",
        default_version="v2",
        versions={
            "v1": CacheProviderVersionConfig(image="img:v1"),
            "v2": CacheProviderVersionConfig(image="img:v2", metrics=renamed),
        },
        default_metrics=default,
    )

    assert provider.metrics_for("v1").mappings["hit_rate"].gauge == "old_name"
    assert provider.metrics_for("v2").mappings["hit_rate"].gauge == "new_name"
    assert provider.metrics_for("custom").mappings["hit_rate"].gauge == "old_name"
    assert provider.metrics_for("v9-unknown").mappings["hit_rate"].gauge == "old_name"
    assert provider.metrics_for(None).mappings["hit_rate"].gauge == "new_name"


def test_lmcache_metrics_declaration():
    provider = _asset_provider("LMCache")
    assert provider is not None

    metrics = provider.default_metrics
    assert metrics is not None
    assert metrics.path == "/metrics"

    hit_rate = metrics.mappings["hit_rate"]
    assert hit_rate.ratio == {
        "numerator": "lmcache_mp_lookup_hit_tokens_total",
        "denominator": "lmcache_mp_lookup_requested_tokens_total",
    }
    assert (
        metrics.mappings["l1_usage_bytes"].gauge == "lmcache_mp_l1_memory_usage_bytes"
    )
    assert metrics.mappings["l1_usage_ratio"].gauge == "lmcache_mp_l1_usage_ratio"
    assert metrics.mappings["l2_usage_bytes"].gauge == "lmcache_mp_l2_usage_bytes"

    assert set(metrics.throughput) == {
        "l0_l1_store",
        "l0_l1_load",
        "l2_store",
        "l2_load",
    }
    for rule in metrics.throughput.values():
        assert rule.histogram_avg
        assert rule.gauge is None and rule.ratio is None
    # The OTel Prometheus exporter appends the histograms' "GB/s" unit to
    # the exported name; the declaration must carry the exported form.
    assert (
        metrics.throughput["l0_l1_store"].histogram_avg
        == "lmcache_mp_l0_l1_store_throughput_GB_per_second"
    )


def test_lmcache_l2_declaration():
    provider = _asset_provider("LMCache")
    assert provider is not None
    assert provider.l2_adapter_flag == "--l2-adapter"
    assert set(provider.l2_backends) == {"fs_native", "resp", "s3"}

    fs = provider.l2_backends["fs_native"]
    fs_fields = {field.name: field for field in fs.fields}
    assert set(fs_fields) == {
        "base_path",
        "max_capacity_gb",
        "num_workers",
        "use_odirect",
    }
    assert fs_fields["base_path"].required is True
    # seeded into the form so a plain "add Local Filesystem" works
    # without inventing a path; lands in the platform data dir, which
    # the mirrored deployment mounts from the host
    assert fs_fields["base_path"].default == "/var/lib/gpustack/cache/lmcache/l2"
    assert fs_fields["max_capacity_gb"].type == "number"
    assert fs_fields["num_workers"].type == "number"
    assert fs_fields["use_odirect"].type == "boolean"
    # the backend description covers the path and the cap; the two knobs
    # whose effect a label cannot carry explain themselves
    assert fs_fields["num_workers"].description
    assert fs_fields["use_odirect"].description
    # fs_native fields all ride in the adapter JSON.
    assert all(field.env_name is None for field in fs.fields)

    resp = provider.l2_backends["resp"]
    resp_fields = {field.name: field for field in resp.fields}
    assert set(resp_fields) == {
        "host",
        "port",
        "username",
        "password",
        "max_capacity_gb",
    }
    assert resp_fields["host"].required is True
    assert resp_fields["port"].required is True
    assert resp_fields["port"].type == "number"
    assert resp_fields["max_capacity_gb"].type == "number"
    # Credentials reach the server via env, keeping them off the command line.
    assert resp_fields["username"].env_name == "LMCACHE_RESP_USERNAME"
    assert resp_fields["password"].type == "password"
    assert resp_fields["password"].env_name == "LMCACHE_RESP_PASSWORD"

    s3 = provider.l2_backends["s3"]
    s3_fields = {field.name: field for field in s3.fields}
    assert set(s3_fields) == {
        "s3_endpoint",
        "s3_region",
        "aws_access_key_id",
        "aws_secret_access_key",
        "disable_tls",
        "max_capacity_gb",
    }
    # Virtual-hosted addressing needs both pieces to sign requests.
    assert s3_fields["s3_endpoint"].required is True
    assert s3_fields["s3_region"].required is True
    assert s3_fields["disable_tls"].type == "boolean"
    assert s3_fields["max_capacity_gb"].type == "number"
    # Credentials ride in env (resolved via the boto3 default chain),
    # keeping them off the command line like the resp backend's.
    assert s3_fields["aws_access_key_id"].env_name == "AWS_ACCESS_KEY_ID"
    assert s3_fields["aws_secret_access_key"].type == "password"
    assert s3_fields["aws_secret_access_key"].env_name == "AWS_SECRET_ACCESS_KEY"


def test_meshfusion_provider_is_a_branded_lmcache_clone():
    meshfusion = _asset_provider("XSKY MeshFusion")
    lmcache = _asset_provider("LMCache")
    assert meshfusion is not None and lmcache is not None

    # XSKY partner branding. The catalog carries the public product name
    # only; AKV-Cache and XDFS are XSKY internal component names. No
    # provider-specific dashboard this version — the engine exposes
    # LMCache's lmcache_mp_* metrics, so both providers fall back to the
    # generic cache-service dashboard.
    assert meshfusion.source.value == "partner"
    assert meshfusion.icon == "/static/catalog_icons/xsky.png"
    assert meshfusion.dashboard_uid is None
    assert lmcache.dashboard_uid is None

    # Functionally an LMCache fork: the runtime contract matches LMCache
    # apart from the branding fields and the XSKY-specific L2 storage.
    brand_fields = {
        "name",
        "display_name",
        "source",
        "icon",
        "description",
        "links",
        "dashboard_uid",
        # whether a vendor ships its own management UI is branding, not
        # an engine trait the clone would inherit
        "management_url",
    }
    diverging_fields = {
        # P2P is declared for LMCache alone until XSKY confirms their
        # image ships the coordinator CLI, so only LMCache splits into
        # components — and carries the field that gates them. Both are
        # checked below rather than left unchecked.
        "components",
        "fields",
        "l2_backends",
        "versions",
        "default_version",
        # MeshFusion images are not published, so it declares no image
        # layout at all; the custom version carries the service's own.
        "default_image",
        "default_runtime_images",
        # The two launch through different slots: MeshFusion's image is
        # expected to start the cache server itself.
        "default_run_command",
        "default_run_args",
        "inference_backend_integrations",
    }
    meshfusion_dump = meshfusion.model_dump()
    lmcache_dump = lmcache.model_dump()
    differing = {
        key
        for key in meshfusion_dump
        if key not in brand_fields | diverging_fields
        and meshfusion_dump[key] != lmcache_dump[key]
    }
    assert differing == set()

    # No release line to declare: services name the image themselves under
    # the reserved custom version, which runs on the provider-level launch
    # arguments — the entry's only launch declaration.
    assert meshfusion.versions == {}
    assert meshfusion.default_version is None
    assert meshfusion.custom_version is True
    custom_config = meshfusion.custom_version_config()
    assert custom_config.run_command is None
    assert custom_config.run_args == meshfusion.default_run_args
    assert "--supported-transfer-mode auto" not in custom_config.run_args
    assert lmcache.versions

    # Every integration is framework-scoped — the catalog is the single
    # accelerator gate. MeshFusion diverges from LMCache only by the
    # extra cann-scoped vLLM entry (an assumed placeholder for XSKY;
    # vllm-ascend trails vLLM, so its attachable range is declared
    # separately). The connector settings mirror LMCache's, while MeshFusion
    # omits the non-hybrid manager flag because its image owns that setup.
    vllm_entries = [
        c for c in meshfusion.inference_backend_integrations if c.backend == "vLLM"
    ]
    assert [(c.frameworks, c.versions) for c in vllm_entries] == [
        (["cuda"], ">=0.25.0"),
        (["cann"], ">=0.25.0"),
    ]
    lm_vllm = lmcache.integration_for("vLLM", "cuda")
    assert lm_vllm.frameworks == ["cuda"]
    assert [
        entry.injection.locality_params["node_local"]["mp_transfer_mode"]
        for entry in vllm_entries
    ] == ["auto", "engine_driven"]

    # The two diverge on P2P alone: LMCache splits into components and
    # declares the switch that gates them, while every other declared
    # field stays in step. Sizing stays shared — LMCache's server
    # component repeats it, but the provider-level profile both read is
    # the same.
    assert set(meshfusion.components) == set()
    assert [field.name for field in meshfusion.fields] == [
        field.name for field in lmcache.fields if field.name != "enable_p2p"
    ]
    assert [
        field.model_dump() for field in lmcache.fields if field.name != "enable_p2p"
    ] == [field.model_dump() for field in meshfusion.fields]
    for entry in vllm_entries:
        mesh_injection = entry.injection.model_dump()
        lm_injection = lm_vllm.injection.model_dump()
        mesh_injection.pop("locality_params", None)
        lm_injection.pop("locality_params", None)
        mesh_kv_config = mesh_injection["kv_transfer_config"]
        lm_kv_config = lm_injection["kv_transfer_config"]
        mesh_kv_config.pop("kv_connector_module_path", None)
        lm_kv_config.pop("kv_connector_module_path", None)
        assert {
            key: value for key, value in mesh_injection.items() if key != "args"
        } == {key: value for key, value in lm_injection.items() if key != "args"}
        assert entry.injection.args == ["--shutdown-timeout", "20"]
        assert (
            entry.injection.kv_transfer_config.kv_connector_module_path
            == "lmcache.integration.vllm.lmcache_mp_connector"
        )
    sglang_entries = [
        c for c in meshfusion.inference_backend_integrations if c.backend == "SGLang"
    ]
    assert sglang_entries == [lmcache.integration_for("SGLang", "cuda")]
    # Framework routing: cuda/cann engine workers each get their scoped
    # entry; an unknown framework (pre-scheduling validation) still
    # answers "attachable"; an undeclared framework gets no contract —
    # for vLLM and SGLang alike (no worker-side half-injection).
    assert meshfusion.integration_for("vLLM", "cuda") is vllm_entries[0]
    assert meshfusion.integration_for("vLLM", "cann") is vllm_entries[1]
    assert meshfusion.integration_for("vLLM") is vllm_entries[0]
    assert meshfusion.integration_for("vLLM", "rocm") is None
    assert meshfusion.integration_for("SGLang", "cann") is None
    assert all(c.frameworks == ["cuda"] for c in lmcache.inference_backend_integrations)

    # XSKY's store (catalog key "xdfs", rendered as the NIXL dynamic
    # adapter and branded with the XSKY icon) is the only L2 tier
    # MeshFusion is deployed with, and LMCache has none of it.
    assert "xdfs" not in lmcache.l2_backends
    assert set(meshfusion.l2_backends) == {"xdfs"}
    xdfs = meshfusion.l2_backends["xdfs"]
    assert xdfs.icon == "/static/catalog_icons/xsky.png"
    assert xdfs.adapter_flag_optional is True
    assert xdfs.adapter_flag_default is False
    assert localized_default(xdfs.adapter_flag_label) == "Enable L2 Adapter Flag"
    assert xdfs.adapter_type == "nixl_store_dynamic"
    assert xdfs.adapter_backend == "XDFS_KV"
    assert xdfs.adapter_params == {
        "conf": "conf",
        "params_file": "params_file",
        "tenant_id": "tenant_id",
        "max_capacity_gb": "max_capacity_gb",
    }
    xdfs_fields = {field.name for field in xdfs.fields}
    # MeshFusion supplies the plugin files from its image; the service form
    # exposes only the tenant override.
    assert xdfs_fields == {"tenant_id"}
    assert next(f for f in xdfs.fields if f.name == "tenant_id").default == "nixl"

    args, env = render_l2_adapter(
        meshfusion,
        "xdfs",
        {},
        adapter_flag_enabled=False,
    )
    # MeshFusion Store is configured by the image itself; the backend must
    # not emit the generic LMCache adapter flag or JSON payload.
    assert args == []
    assert env == {}

    args, env = render_l2_adapter(
        meshfusion,
        "xdfs",
        {
            "tenant_id": "glmint4mix-1787763619",
        },
        adapter_flag_enabled=True,
    )
    assert args == [
        "--l2-adapter",
        '{"type":"nixl_store_dynamic","backend":"XDFS_KV",'
        '"backend_params":{"tenant_id":"glmint4mix-1787763619"}}',
    ]
    assert env == {}


def test_render_l2_adapter_stringifies_nested_backend_params():
    provider = CacheProvider(
        name="nested-adapter",
        custom_version=True,
        l2_adapter_flag="--l2-adapter",
        l2_backends={
            "store": CacheProviderL2Backend(
                adapter_type="dynamic",
                adapter_backend="STORE",
                adapter_params={"capacity": "max_capacity_gb"},
                fields=[CacheProviderL2Field(name="max_capacity_gb", type="number")],
            )
        },
    )

    args, env = render_l2_adapter(provider, "store", {"max_capacity_gb": 1048576})

    assert json.loads(args[1])["backend_params"] == {"capacity": "1048576"}
    assert env == {}


def test_provider_brand_links():
    def labels(provider):
        return {localized_default(link.label) for link in provider.links}

    lmcache = _asset_provider("LMCache")
    assert labels(lmcache) == {"Documentation", "GitHub"}
    assert all(link.url.startswith("https://") for link in lmcache.links)

    mooncake = _asset_provider("Mooncake")
    assert labels(mooncake) == {"Documentation", "GitHub"}

    meshfusion = _asset_provider("XSKY MeshFusion")
    assert meshfusion.links, "partner card needs at least one brand link"


def test_version_config_resolves_runtime_image_by_platform_rule():
    cfg = CacheProviderVersionConfig(
        image="repo/x:v1",
        runtime_images={"cuda": {"12.9": "repo/x:v1-cu129", "12": "repo/x:v1-cu12"}},
    )
    # Newest declared version <= the host runtime wins (the rule shared
    # with inference-backend runners); a host older than every declared
    # build gets the oldest one — the closest guess, not the plain
    # (newest-CUDA) image. Other backends fall back to the plain image.
    assert cfg.resolve_image("cuda", "12.9") == "repo/x:v1-cu129"
    assert cfg.resolve_image("cuda", "12.4") == "repo/x:v1-cu12"
    assert cfg.resolve_image("cuda", "11.8") == "repo/x:v1-cu12"
    assert cfg.resolve_image("rocm", "6.1") == "repo/x:v1"


def test_version_config_runtime_support_matrix():
    cfg = CacheProviderVersionConfig(
        image="repo/x:v1",
        runtime_images={"cuda": {"13": "repo/x:v1"}},
    )
    # runtime_images doubles as the support matrix: a foreign
    # accelerator (e.g. Ascend's cann) is rejected instead of falling
    # back to an image built for another family; accelerator-less nodes
    # run the plain image CPU-only.
    assert cfg.supports_runtime("cuda") is True
    assert cfg.supports_runtime("cann") is False
    assert cfg.supports_runtime(None) is True
    unconstrained = CacheProviderVersionConfig(image="repo/x:v1")
    assert unconstrained.supports_runtime("cann") is True


def test_provider_lookup_is_case_insensitive():
    assert _asset_provider("lmcache") is not None
    assert _asset_provider("no-such-provider") is None


def test_render_injection_substitutes_host_and_port():
    provider = _asset_provider("LMCache")
    rendered = render_injection(
        provider,
        "vLLM",
        {
            "host": "10.0.0.5",
            "port": 9000,
            "chunk_size": 256,
            "ram_size": 8,
            "locality": "node_local",
        },
    )
    assert rendered is not None
    env, args, files = rendered
    # The MP connector carries the endpoint in the transfer config and no
    # config file; the only env is the pinned hash seed keeping chunk keys
    # consistent across engine processes on the builtin-hash fallback path.
    assert env == {"PYTHONHASHSEED": "0"}
    assert files == {}
    assert args[0] == "--kv-transfer-config"
    assert '"kv_connector":"LMCacheMPConnector"' in args[1]
    assert '"lmcache.mp.host":"tcp://10.0.0.5"' in args[1]
    assert '"lmcache.mp.port":9000' in args[1]
    # The declaration's locality_params map the resolver's neutral
    # placement fact to LMCache's transfer-mode vocabulary: node-local
    # attachments may negotiate CUDA IPC (auto), remote ones stay on
    # engine-driven copies since IPC handles cannot cross hosts.
    assert '"lmcache.mp.mp_transfer_mode":"auto"' in args[1]
    # Nothing in the injection touches the hybrid KV cache manager: the
    # connector advertises hybrid support, and an engine told to disable it
    # cannot start a model whose Mamba and full-attention layers need
    # different cache specs.
    assert not any("hybrid-kv-cache-manager" in arg for arg in args)


def test_meshfusion_vllm_injection_includes_connector_module_path():
    provider = _asset_provider("XSKY MeshFusion")
    rendered = render_injection(
        provider,
        "vLLM",
        {"host": "127.0.0.1", "port": 5556, "locality": "node_local"},
    )
    assert rendered is not None
    _, args, _ = rendered
    payload = json.loads(args[1])
    assert payload["kv_connector"] == "LMCacheMPConnector"
    assert (
        payload["kv_connector_module_path"]
        == "lmcache.integration.vllm.lmcache_mp_connector"
    )
    assert payload["kv_role"] == "kv_both"
    assert payload["kv_connector_extra_config"]["lmcache.mp.host"] == "tcp://127.0.0.1"
    assert payload["kv_connector_extra_config"]["lmcache.mp.port"] == 5556
    assert args[2:] == ["--shutdown-timeout", "20"]

    rendered_cann = render_injection(
        provider,
        "vLLM",
        {"host": "127.0.0.1", "port": 5556, "locality": "node_local"},
        framework="cann",
    )
    assert rendered_cann is not None
    _, cann_args, _ = rendered_cann
    cann_payload = json.loads(cann_args[1])
    assert (
        cann_payload["kv_connector_extra_config"]["lmcache.mp.mp_transfer_mode"]
        == "engine_driven"
    )


def test_kv_transfer_config_renders_structured_slot_with_types():
    """The connector slot is declared structured (one owner assembles the
    single-value engine flag) and placeholder types survive into the
    JSON payload — the port must be a number, not a string."""
    provider = _asset_provider("LMCache")
    integration = provider.integration_for("vLLM", "cuda")
    slot = integration.injection.kv_transfer_config
    assert slot is not None
    assert slot.flag == "--kv-transfer-config"
    assert slot.kv_connector == "LMCacheMPConnector"

    rendered = render_injection(
        provider,
        "vLLM",
        {"host": "10.0.0.5", "port": 9000, "locality": "node_local"},
    )
    assert rendered is not None
    _, args, _ = rendered
    assert args[0] == "--kv-transfer-config"
    payload = json.loads(args[1])
    extra = payload["kv_connector_extra_config"]
    assert extra["lmcache.mp.host"] == "tcp://10.0.0.5"
    assert extra["lmcache.mp.port"] == 9000
    assert isinstance(extra["lmcache.mp.port"], int)
    assert extra["lmcache.mp.mp_transfer_mode"] == "auto"
    # Free-form args follow the slot: the graceful-shutdown window that
    # lets the engine tear its CUDA IPC handles down.
    assert args[2:] == ["--shutdown-timeout", "20"]


def test_injection_backstops_a_gated_field_with_its_gated_default():
    """A field behind a closed gate renders the value the launch would use, not
    the plain default — an engine's contribution has to read 0 while something
    else owns the pool, and the declared default says otherwise."""
    provider = CacheProvider(
        name="Gated",
        default_image="demo:v1",
        versions={"v1.0": {}},
        default_run_command="demo",
        fields=[
            {"name": "pool_mode", "type": "select", "options": ["shared", "own"]},
            {
                "name": "segment_size",
                "type": "number",
                "default": 32,
                "visible_by": "pool_mode",
                "visible_when": "shared",
                "gated_default": 0,
            },
        ],
        inference_backend_integrations=[
            {
                "backend": "vLLM",
                "injection": {"env": {"SEGMENT": "{{segment_size}}"}},
            }
        ],
    )

    env, _, _ = render_injection(provider, "vLLM", {"pool_mode": "own"})
    assert env["SEGMENT"] == "0"

    # Gate open: the declared default is what the field is worth.
    env, _, _ = render_injection(provider, "vLLM", {"pool_mode": "shared"})
    assert env["SEGMENT"] == "32"


def test_render_injection_maps_node_local_locality_to_auto():
    """Engines attach node-local only (the resolver degrades instead of
    crossing nodes), so the declaration maps the sole placement fact to
    the auto-negotiated zero-copy path."""
    provider = _asset_provider("LMCache")
    rendered = render_injection(
        provider,
        "vLLM",
        {"host": "10.0.0.5", "port": 9000, "locality": "node_local"},
    )
    assert rendered is not None
    _, args, _ = rendered
    assert '"lmcache.mp.mp_transfer_mode":"auto"' in args[1]


def test_render_injection_explicit_param_beats_locality_default():
    provider = _asset_provider("LMCache")
    rendered = render_injection(
        provider,
        "vLLM",
        {
            "host": "10.0.0.5",
            "port": 9000,
            "locality": "node_local",
            "mp_transfer_mode": "engine_driven",
        },
    )
    assert rendered is not None
    _, args, _ = rendered
    assert '"lmcache.mp.mp_transfer_mode":"engine_driven"' in args[1]


def test_render_injection_returns_none_for_incompatible_backend():
    provider = _asset_provider("LMCache")
    rendered = render_injection(
        provider,
        "no-such-backend",
        {"host": "10.0.0.5", "port": 9000},
    )
    assert rendered is None


def test_resolve_image_matches_minor_version_keys():
    """runtime_images accepts full-version keys with the same match rule
    inference-backend runners use: a 12.8 host takes the 12.6 build
    instead of silently falling back to the plain (newest-CUDA) image."""
    version_config = CacheProviderVersionConfig(
        image="cache:latest",
        runtime_images={"cuda": {"12.6": "cache:cu126", "12": "cache:cu12"}},
    )
    assert version_config.resolve_image("cuda", "12.8") == "cache:cu126"
    assert version_config.resolve_image("cuda", "12.3") == "cache:cu12"
    # accelerator-less nodes and undeclared backends keep the plain image
    assert version_config.resolve_image(None, None) == "cache:latest"
    assert version_config.resolve_image("rocm", "6.3") == "cache:latest"


def test_lmcache_sglang_injection_renders_config_file():
    """SGLang attaches through --enable-lmcache with a YAML config file
    carrying the MP server address; the adapter pulls the chunk size from
    the server, so host/port is the whole contract."""
    provider = _asset_provider("LMCache")
    compat = provider.integration_for("SGLang")
    assert compat is not None
    # LMCache MP support landed in sglang v0.5.13 (PR #24089).
    assert compat.versions == ">=0.5.13"
    rendered = render_injection(
        provider,
        "SGLang",
        {"host": "10.0.0.5", "port": 9000, "locality": "node_local"},
    )
    assert rendered is not None
    env, args, files = rendered
    assert env == {"PYTHONHASHSEED": "0"}
    assert args == [
        "--enable-lmcache",
        "--lmcache-config-file",
        "/tmp/gpustack-lmcache-sgl.yaml",
    ]
    config = files["/tmp/gpustack-lmcache-sgl.yaml"]
    assert 'mp_host: "10.0.0.5"' in config
    assert "mp_port: 9000" in config
    assert "{{" not in config


def test_integration_for_framework_scoping():
    """A scoped-only declaration attaches on its named frameworks only;
    an unknown framework (validation before scheduling) still answers
    "attachable" so a scoped-only provider is not rejected up front."""
    provider = CacheProvider(
        name="scoped-only",
        custom_version=True,
        inference_backend_integrations=[
            {"backend": "vLLM", "frameworks": ["cann"], "versions": ">=1"},
        ],
    )
    scoped = provider.inference_backend_integrations[0]
    assert provider.integration_for("vLLM", "cann") is scoped
    assert provider.integration_for("vLLM", "cuda") is None
    assert provider.integration_for("vLLM") is scoped
    assert provider.integration_for("SGLang", "cann") is None


def test_bundled_catalog_passes_injection_contract():
    """Every shipped provider must satisfy the placeholder contract the
    loader enforces (a violating provider is excluded at load time)."""
    for provider in asset_providers():
        assert validate_injection_templates(provider) == []


def test_injection_contract_flags_violations():
    provider = CacheProvider(
        name="bad-provider",
        custom_version=True,
        inference_backend_integrations=[
            {
                "backend": "vLLM",
                "injection": {
                    "args": ["--peer", "{{undeclared_thing}}"],
                    "locality_params": {
                        "node_local": {"mode": "auto"},
                        # "mode" missing here: not common to all buckets
                        "remote": {"other": "x"},
                    },
                    "files": {"/tmp/x": "mode={{mode}}"},
                },
            }
        ],
    )
    errors = validate_injection_templates(provider)
    joined = "\n".join(errors)
    assert "placeholder 'undeclared_thing'" in joined
    # "mode" is not present in every locality bucket, so it is
    # unresolvable on the remote path and must be flagged.
    assert "placeholder 'mode'" in joined


def test_bare_string_and_locale_mapping_are_both_accepted():
    """A declaration written before the catalog had locales stays valid:
    a bare string is the text in every locale, so translating a slot is
    additive rather than a rewrite."""
    provider = CacheProvider(
        name="localized",
        custom_version=True,
        display_name="Localized",
        description={"default": "A cache", "zh-CN": "一个缓存"},
        links=[{"label": {"default": "Docs", "zh-CN": "文档"}, "url": "https://x"}],
        fields=[{"name": "size", "label": {"default": "Size", "ja-JP": "サイズ"}}],
    )
    assert validate_localized_text(provider) == []
    assert localized_default(provider.display_name) == "Localized"
    assert localized_default(provider.description) == "A cache"
    assert localized_values(provider.display_name) == ["Localized"]
    assert sorted(localized_values(provider.description)) == ["A cache", "一个缓存"]


def test_localized_slot_without_a_default_is_a_violation():
    """Every locale mapping needs the fallback entry: without it a locale
    the declaration skips has no text to render, so the slot reads as
    untranslated instead of as the author's canonical wording."""
    provider = CacheProvider(
        name="no-default",
        custom_version=True,
        display_name={"zh-CN": "只有中文"},
        description={},
        l2_backends={
            "b": CacheProviderL2Backend(
                description={"zh-CN": "只有中文"},
                fields=[CacheProviderL2Field(name="x", label={"zh-CN": "只有中文"})],
            )
        },
    )
    joined = "\n".join(validate_localized_text(provider))
    assert "display_name has no 'default' entry" in joined
    assert "description is an empty locale mapping" in joined
    assert "l2 backend 'b' description has no 'default' entry" in joined
    assert "l2 backend 'b' field 'x' label has no 'default' entry" in joined


def test_invalid_locale_key_is_a_violation():
    """A key that no locale resolves to is text that renders for nobody;
    it is caught at load time rather than silently never appearing.

    Rejection costs the whole provider, so the check admits every real tag
    shape: a script or region suffix, and the three-letter primary subtags
    of ISO 639-2/3 alongside 639-1's two."""
    provider = CacheProvider(
        name="bad-locale",
        custom_version=True,
        fields=[
            {
                "name": "protocol",
                "label": {
                    "default": "Protocol",
                    "ZH_cn": "协议",
                    "zh-Hant": "協議",
                    "yue": "協議",
                    "fil-PH": "Protocol",
                },
            }
        ],
    )
    joined = "\n".join(validate_localized_text(provider))
    assert "invalid locale key 'ZH_cn'" in joined
    for valid in ("zh-Hant", "yue", "fil-PH"):
        assert valid not in joined


def test_localized_violation_costs_only_its_own_provider(monkeypatch):
    """The localized-text contract is enforced with the same blast radius
    as the injection contract: the offending provider drops out, the rest
    of the catalog still serves."""
    asset = (
        "- name: Broken\n"
        '  default_image: "repo/cache:{{version}}"\n'
        "  description:\n"
        "    zh-CN: 没有默认文案\n"
        "  versions:\n"
        '    "v1.0": {}\n'
        "- name: Good\n"
        '  default_image: "repo/cache:{{version}}"\n'
        "  description:\n"
        "    default: A cache\n"
        "    zh-CN: 一个缓存\n"
        "  versions:\n"
        '    "v1.0": {}\n'
    )

    class _Asset:
        def is_file(self):
            return True

        def read_text(self, encoding=None):
            return asset

    try:
        monkeypatch.setattr(cache_provider_catalog, "files", lambda _package: _Asset())
        monkeypatch.setattr(_Asset, "joinpath", lambda self, _name: self, raising=False)
        providers = asset_providers()
        assert [provider.name for provider in providers] == ["Good"]
    finally:
        monkeypatch.undo()


def test_every_form_field_declares_a_label():
    """The UI humanizes a missing label from the field name, which is an
    English identifier: a field without a label is a slot that stays
    English in every other locale."""
    missing = []
    for provider in asset_providers():
        for field in provider.fields:
            if field.label is None:
                missing.append(f"{provider.name} declared field '{field.name}'")
        for key, backend in provider.l2_backends.items():
            for field in backend.fields:
                if field.label is None:
                    missing.append(f"{provider.name} l2 '{key}' field '{field.name}'")
    assert not missing, "fields missing a label: " + ", ".join(missing)


def test_every_declared_description_is_translated():
    """A label may be a bare string — a brand or a protocol name reads
    the same in every locale — but a description is a sentence, and one
    left untranslated shows up as English in the middle of a translated
    form."""
    untranslated = []
    for provider in asset_providers():
        slots = [(f"{provider.name}", provider.description)]
        for field in provider.fields:
            where = f"{provider.name} field '{field.name}'"
            slots.append((where, field.description))
            for option in field.options or []:
                if not isinstance(option, str):
                    slots.append(
                        (f"{where} option '{option.value}'", option.description)
                    )
        for key, backend in provider.l2_backends.items():
            where = f"{provider.name} l2 '{key}'"
            slots.append((where, backend.description))
            for field in backend.fields:
                slots.append((f"{where} field '{field.name}'", field.description))
        untranslated.extend(
            where for where, text in slots if isinstance(text, str) and text
        )
    assert not untranslated, "descriptions without a translation: " + ", ".join(
        untranslated
    )


def test_declared_text_carries_its_translations():
    """Every slot a reader sees is localizable, options included: a
    dropdown whose choices stay English while their field is translated
    reads worse than either language alone."""
    provider = CacheProvider(
        name="Pool",
        default_image="repo/pool:v1",
        versions={"v1.0": {}},
        fields=[
            {
                "name": "pool_mode",
                "type": "select",
                "label": {"default": "Mode", "zh-CN": "模式"},
                "options": [
                    {
                        "value": "embedded",
                        "label": {"default": "Embedded", "zh-CN": "内嵌"},
                        "description": {
                            "default": "The engines hold the pool.",
                            "zh-CN": "由引擎持有缓存池。",
                        },
                    },
                    # a bare string stays a string: an untranslated
                    # choice is not a violation, it is the same text in
                    # every locale
                    "standalone-store",
                ],
            }
        ],
    )
    assert validate_localized_text(provider) == []

    fields = {field.name: field for field in provider.fields}
    mode = fields["pool_mode"]
    assert set(localized_values(mode.label)) == {"Mode", "模式"}
    assert len(localized_values(mode.options[0].label)) == 2
    assert len(localized_values(mode.options[0].description)) == 2
    assert localized_default(mode.options[1]) == "standalone-store"


def test_template_filters_convert_on_the_way_out():
    params = {"cap": 3, "flag": True, "off": False}
    assert render_template("{{cap|gib_to_bytes}}", params) == str(3 * 1024**3)
    # a filtered placeholder is a converted value, never the raw one
    assert render_typed_template("{{cap|gib_to_bytes}}", params) == str(3 * 1024**3)
    assert render_typed_template("{{cap}}", params) == 3
    # one declared boolean serves JSON and gflags alike
    assert render_template("--enable={{flag}} --off={{off}}", params) == (
        "--enable=true --off=false"
    )


def test_a_gated_gate_closes_the_fields_behind_it():
    """A path that hangs off a switch that itself hangs off a mode: a
    switch left on for one mode must not keep the path alive under
    another. Gates resolve through the chain, not one level."""
    fields = [
        {
            "name": "pool_mode",
            "type": "select",
            "default": "embedded",
            "options": ["embedded", "standalone-store"],
        },
        {
            "name": "enable_offload",
            "type": "boolean",
            "default": False,
            "visible_by": "pool_mode",
            "visible_when": "standalone-store",
            "gated_default": False,
        },
        {
            "name": "offload_path",
            "visible_by": "enable_offload",
            "visible_when": True,
            "gated_default": "",
        },
        {
            "name": "capacity_gb",
            "type": "number",
            "visible_by": "enable_offload",
            "visible_when": True,
            "gated_default": "",
        },
    ]
    provider = CacheProvider(
        name="Pool",
        default_image="repo/pool:v1",
        versions={"v1.0": {}},
        fields=fields,
    )
    params = resolved_field_values(
        provider.fields,
        {
            "pool_mode": "embedded",
            "enable_offload": True,
            "offload_path": "/nvme/pool",
            "capacity_gb": 200,
        },
    )
    assert params["enable_offload"] is False
    assert params["offload_path"] == ""
    assert params["capacity_gb"] == ""


def test_one_asset_naming_a_provider_twice_yields_one_card(monkeypatch):
    """A name is a provider's identity across assets, and within one:
    a declaration repeated in the same file keeps its last form rather
    than putting two cards of the same name in the catalog."""
    asset = (
        "- name: Twice\n"
        '  default_image: "repo/one:{{version}}"\n'
        "  versions:\n"
        '    "v1.0": {}\n'
        "- name: Twice\n"
        '  default_image: "repo/two:{{version}}"\n'
        "  versions:\n"
        '    "v2.0": {}\n'
    )

    class _Asset:
        def is_file(self):
            return True

        def read_text(self, encoding=None):
            return asset

        def joinpath(self, _name):
            return self

    try:
        monkeypatch.setattr(cache_provider_catalog, "files", lambda _package: _Asset())
        providers = asset_providers()
        assert [provider.name for provider in providers] == ["Twice"]
        assert list(providers[0].versions) == ["v2.0"]
    finally:
        monkeypatch.undo()


def test_a_port_gated_on_an_undeclared_field_is_rejected():
    """A port whose gate names no declared field never opens, and every
    flag carrying it drops — at launch, silently, looking exactly like a
    feature left off. The declaration is refused instead."""
    from gpustack.schemas.cache_providers import CacheProviderComponent

    with pytest.raises(ValidationError):
        CacheProvider(
            name="Pool",
            default_image="repo/pool:{{version}}",
            versions={"v1.0": {}},
            components={
                "server": CacheProviderComponent(
                    attach_endpoint=True,
                    ports=[
                        "port",
                        {"name": "p2p", "enabled_by": "enable_pp2"},
                    ],
                )
            },
            fields=[{"name": "enable_p2p", "type": "boolean", "default": False}],
        )


def test_an_address_port_may_follow_its_own_component_s_gate():
    """A port gated exactly as its component is stays a valid address:
    the configurations that close the gate are the ones where the
    component has no instances to address."""
    from gpustack.schemas.cache_providers import CacheProviderComponent

    provider = CacheProvider(
        name="Pool",
        default_image="repo/pool:{{version}}",
        versions={"v1.0": {}},
        components={
            "master": CacheProviderComponent(attach_endpoint=True),
            "store": CacheProviderComponent(
                enabled_by="pool_mode",
                enabled_when="standalone",
                ports=[
                    {
                        "name": "rpc",
                        "enabled_by": "pool_mode",
                        "enabled_when": "standalone",
                    }
                ],
                address_port="rpc",
            ),
        },
        fields=[
            {
                "name": "pool_mode",
                "type": "select",
                "default": "embedded",
                "options": ["embedded", "standalone"],
            }
        ],
    )

    assert provider.address_port_name("store") == "rpc"
    assert provider.enabled_port_names("store", {"pool_mode": "standalone"}) == ["rpc"]
    assert provider.enabled_port_names("store", {"pool_mode": "embedded"}) == []


def test_a_component_may_declare_its_own_completion_hints():
    """Each role runs its own binary: the provider-level hints describe
    the one engines attach to, and a role that runs something else
    declares what its parser takes — offering the other's would suggest
    flags it rejects."""
    provider = _asset_provider("LMCache")
    coordinator = provider.components["coordinator"]
    server = provider.components["server"]

    assert "--chunk-size" in coordinator.common_parameters
    # the platform sets these; a hand-written copy would fight it
    assert not {"--host", "--port"} & set(coordinator.common_parameters)
    # the server is the attach component, so the provider-level list is
    # already its own
    assert server.common_parameters == []
    assert "--l1-size-gb" not in coordinator.common_parameters
