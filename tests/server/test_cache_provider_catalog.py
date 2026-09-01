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
    render_l2_adapter,
    validate_injection_templates,
    validate_localized_text,
)
from gpustack.schemas.cache_services import CacheServiceModeEnum
from gpustack.server import cache_provider_catalog
from gpustack.server.cache_provider_catalog import (
    get_cache_provider,
    load_cache_providers,
    render_injection,
)


def test_catalog_asset_loads():
    providers = load_cache_providers(reload=True)
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
        providers = load_cache_providers(reload=True)
        assert [provider.name for provider in providers] == ["Good"]
    finally:
        # The loader caches for the process lifetime; leave the bundled
        # catalog in place for the tests that read it.
        monkeypatch.undo()
        load_cache_providers(reload=True)


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


def test_managed_provider_without_versions_must_allow_the_custom_version():
    """A provider declaring no release line resolves no image of its own;
    without the custom version its managed services could never start."""
    with pytest.raises(ValidationError):
        CacheProvider(name="Versionless", supported_modes=["managed"])

    provider = CacheProvider(
        name="Versionless",
        supported_modes=["managed"],
        custom_version=True,
        default_run_args="--port {{port}}",
    )
    # The provider-level launch declaration is what the service's own
    # image runs on, reached through the same version-config contract.
    template = provider.custom_version_config()
    assert template.run_args == "--port {{port}}"
    assert template.run_command is None

    # An external-only provider runs no container at all, so declaring no
    # version says nothing about images.
    CacheProvider(name="Registered", supported_modes=["external"])


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


def test_lmcache_provider_declaration():
    provider = get_cache_provider("LMCache")
    assert provider is not None
    # Managed only: LMCache is the single-container engine GPUStack runs
    # itself; reference-only distributed caches are what external is for.
    assert provider.supported_modes == [CacheServiceModeEnum.MANAGED.value]
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

    # The declared version pins a verified image tag; a service may also
    # pin its own image via the reserved "custom" version.
    assert provider.default_version == "v0.5.4"
    assert set(provider.versions) == {"v0.5.2", "v0.5.3", "v0.5.4"}
    assert provider.custom_version is True

    version_config, version = provider.get_version_config()
    assert version_config is not None
    assert version == provider.default_version
    # Upstream's tag layout, asserted against the resolved version rather
    # than a copy of it: the bare tag is the CUDA 13 build and cu129
    # serves CUDA 12 nodes. The worker resolves per node, so a
    # heterogeneous per_node fleet mixes images; unknown runtimes and
    # accelerator-less workers get the plain image.
    assert version_config.image == f"lmcache/vllm-openai:{version}"
    assert (
        version_config.resolve_image("cuda", "13.0") == f"lmcache/vllm-openai:{version}"
    )
    assert (
        version_config.resolve_image("cuda", "12.8")
        == f"lmcache/vllm-openai:{version}-cu129"
    )
    assert version_config.resolve_image(None, None) == f"lmcache/vllm-openai:{version}"
    # The full CLI entry: the HTTP frontend on --http-port serves
    # /metrics (same registry as the standalone exposition) plus
    # /healthcheck and the admin APIs; --prometheus-port is ignored
    # there, so the frontend port doubles as the metrics port.
    assert version_config.run_command == (
        "lmcache server --host {{host}} "
        "--port {{port}} --l1-size-gb {{ram_size}} "
        "--chunk-size {{chunk_size}} "
        "--http-host {{host}} --http-port {{metrics_port}} "
        "--supported-transfer-mode auto --worker-reap-timeout-seconds 60 "
        "--eviction-policy {{eviction_policy}} "
        "--eviction-trigger-watermark {{eviction_trigger_watermark}} "
        "--eviction-ratio {{eviction_ratio}}"
    )
    # The server's argument groups are unchanged across the declared
    # release line, so every version inherits the one command template
    # and consumers read the effective command off the version config.
    for declared in provider.versions.values():
        assert declared.run_command == provider.default_run_command
    # The eviction knobs are declared fields: structured in the UI, wired
    # into the run command through their placeholders. The eviction policy
    # is required upstream, so its default renders even untouched.
    fields = {field.name: field for field in provider.managed_fields}
    assert set(fields) == {
        "eviction_policy",
        "eviction_trigger_watermark",
        "eviction_ratio",
    }
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
    # Every declared field is actually wired into a template, and none
    # shadows a reserved platform placeholder.
    for name in fields:
        assert f"{{{{{name}}}}}" in version_config.run_command
    assert not set(fields) & {"host", "port", "metrics_port", "ram_size", "chunk_size"}
    # Capacity flows through --l1-size-gb on the command line, not env.
    assert not version_config.env

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
    provider = get_cache_provider("LMCache")
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
    provider = get_cache_provider("LMCache")
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


def test_mooncake_provider_declaration():
    provider = get_cache_provider("Mooncake")
    assert provider is not None
    # Shipped and supported as part of the GPUStack catalog, like LMCache;
    # "partner" is reserved for vendor-branded providers.
    assert provider.source.value == "built_in"
    # A distributed pool is network-attachable from any worker — spanning
    # (multi-worker) instances attach by design.
    assert provider.attach_locality == "cluster"
    assert provider.icon == "/static/catalog_icons/mooncake.png"
    # Mooncake is a self-deployed distributed store; GPUStack only references
    # it, so it is external-only and runs no managed container.
    assert provider.supported_modes == [CacheServiceModeEnum.EXTERNAL.value]
    assert provider.versions == {}
    assert provider.health_check.scheme == "tcp"

    # Master-side metrics: the pool's allocated/capacity view on the
    # master's Prometheus endpoint (conventionally port 9003). Lookup-hit
    # accounting lives in the engine-side connector, so no hit_rate.
    metrics = provider.default_metrics
    assert metrics is not None
    assert metrics.path == "/metrics"
    assert metrics.default_port == 9003
    assert "hit_rate" not in metrics.mappings
    assert metrics.mappings["l1_usage_bytes"].gauge == "master_allocated_bytes"
    assert metrics.mappings["l1_usage_ratio"].gauge_ratio == {
        "numerator": "master_allocated_bytes",
        "denominator": "master_total_capacity_bytes",
    }
    assert provider.dashboard_uid == "gpustack-mooncake"

    fields = {field.name: field for field in provider.external_fields}
    assert set(fields) == {
        "metadata_server",
        "protocol",
        "device_name",
        "local_buffer_size",
    }
    assert fields["local_buffer_size"].default == "1GB"
    assert fields["metadata_server"].default == "P2PHANDSHAKE"
    assert fields["protocol"].default == "tcp"
    assert fields["protocol"].options == ["tcp", "rdma"]
    # None of the external fields are required: each has a usable default or
    # is only needed for RDMA.
    assert all(not field.required for field in provider.external_fields)

    compat = provider.integration_for("vLLM")
    assert compat is not None


def test_mooncake_injection_renders_store_connector_env():
    provider = get_cache_provider("Mooncake")
    rendered = render_injection(
        provider,
        "vLLM",
        {
            "master_server_address": "10.0.0.9:50051",
            "metadata_server": "P2PHANDSHAKE",
            "protocol": "tcp",
            "device_name": None,
            "local_hostname": "10.0.0.7",
        },
    )
    assert rendered is not None
    env, args, files = rendered
    # The connector reads its configuration solely from the JSON file
    # MOONCAKE_CONFIG_PATH points at; the injection materializes it.
    assert env == {
        "MOONCAKE_CONFIG_PATH": "/tmp/gpustack-mooncake.json",
        # TCP transport pools connections instead of opening one per
        # transfer slice, which exhausts ephemeral ports under prefill
        # bursts; the RDMA path ignores the switch.
        "MC_TCP_ENABLE_CONNECTION_POOL": "1",
    }
    config = json.loads(files["/tmp/gpustack-mooncake.json"])
    # A pure requester: the externally deployed cluster owns the pool.
    assert config["mode"] == "standalone-store"
    assert config["global_segment_size"] == 0
    assert config["master_server_address"] == "10.0.0.9:50051"
    assert config["metadata_server"] == "P2PHANDSHAKE"
    assert config["protocol"] == "tcp"
    # An unset optional field renders empty in the file; the config
    # schema treats empty device_name as "no RDMA device".
    assert config["device_name"] == ""
    assert config["local_buffer_size"] == "1GB"
    assert args[0] == "--kv-transfer-config"
    assert '"kv_connector":"MooncakeStoreConnector"' in args[1]


def test_meshfusion_provider_is_a_branded_lmcache_clone():
    meshfusion = get_cache_provider("XSKY MeshFusion")
    lmcache = get_cache_provider("LMCache")
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

    lmcache = get_cache_provider("LMCache")
    assert labels(lmcache) == {"Documentation", "GitHub"}
    assert all(link.url.startswith("https://") for link in lmcache.links)

    mooncake = get_cache_provider("Mooncake")
    assert labels(mooncake) == {"Documentation", "GitHub"}

    meshfusion = get_cache_provider("XSKY MeshFusion")
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
    assert get_cache_provider("lmcache") is not None
    assert get_cache_provider("no-such-provider") is None


def test_render_injection_substitutes_host_and_port():
    provider = get_cache_provider("LMCache")
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
    assert args[2] == "--disable-hybrid-kv-cache-manager"


def test_meshfusion_vllm_injection_includes_connector_module_path():
    provider = get_cache_provider("XSKY MeshFusion")
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
    provider = get_cache_provider("LMCache")
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
    # Free-form args follow the slot: the non-hybrid manager requirement
    # and the graceful-shutdown window (CUDA IPC teardown).
    assert args[2:] == [
        "--disable-hybrid-kv-cache-manager",
        "--shutdown-timeout",
        "20",
    ]


def test_secret_and_scrape_fields_never_enter_injection_templates():
    """Injection renders into the snapshot on the model instance row,
    outside the cache-service redaction's reach — so password-typed
    external fields must never be referenced by injection templates,
    and metrics_target fields (scrape addresses, not connector config)
    are excluded from the injection namespace by contract."""
    for provider in load_cache_providers():
        excluded = {
            field.name
            for field in provider.external_fields
            if field.type == "password" or field.metrics_target
        }
        if not excluded:
            continue
        for integration in provider.inference_backend_integrations:
            blob = integration.injection.model_dump_json()
            for name in excluded:
                assert f"{{{{{name}}}}}" not in blob, (
                    f"{provider.name}: injection references excluded " f"field {name}"
                )


def test_render_injection_drops_metrics_target_values():
    from gpustack.schemas.cache_providers import CacheProviderExternalField

    provider = CacheProvider(
        name="scrape-provider",
        external_fields=[
            CacheProviderExternalField(name="metadata_server"),
            CacheProviderExternalField(name="exporter_address", metrics_target=True),
        ],
        inference_backend_integrations=[
            {
                "backend": "vLLM",
                "injection": {"env": {"META": "{{metadata_server}}"}},
            }
        ],
    )
    rendered = render_injection(
        provider,
        "vLLM",
        {"metadata_server": "10.0.0.9:8000", "exporter_address": "10.0.0.9:9100"},
    )
    assert rendered is not None
    env, _, _ = rendered
    assert env == {"META": "10.0.0.9:8000"}


def test_render_injection_maps_node_local_locality_to_auto():
    """Engines attach node-local only (the resolver degrades instead of
    crossing nodes), so the declaration maps the sole placement fact to
    the auto-negotiated zero-copy path."""
    provider = get_cache_provider("LMCache")
    rendered = render_injection(
        provider,
        "vLLM",
        {"host": "10.0.0.5", "port": 9000, "locality": "node_local"},
    )
    assert rendered is not None
    _, args, _ = rendered
    assert '"lmcache.mp.mp_transfer_mode":"auto"' in args[1]


def test_render_injection_explicit_param_beats_locality_default():
    provider = get_cache_provider("LMCache")
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
    provider = get_cache_provider("LMCache")
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


def test_mooncake_injection_backfills_optional_fields_as_empty():
    """A declared field without a default (device_name is empty on TCP)
    must still backstop its placeholder — otherwise the literal
    "{{device_name}}" lands in the rendered config file."""
    provider = get_cache_provider("Mooncake")
    rendered = render_injection(
        provider,
        "vLLM",
        {
            "master_server_address": "10.0.0.9:50051",
            "metadata_server": "P2PHANDSHAKE",
            "protocol": "tcp",
        },
    )
    assert rendered is not None
    _, _, files = rendered
    config = json.loads(files["/tmp/gpustack-mooncake.json"])
    assert config["device_name"] == ""
    assert "{{" not in files["/tmp/gpustack-mooncake.json"]


def test_lmcache_sglang_injection_renders_config_file():
    """SGLang attaches through --enable-lmcache with a YAML config file
    carrying the MP server address; the adapter pulls the chunk size from
    the server, so host/port is the whole contract."""
    provider = get_cache_provider("LMCache")
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
    for provider in load_cache_providers():
        assert validate_injection_templates(provider) == []


def test_injection_contract_flags_violations():
    from gpustack.schemas.cache_providers import CacheProviderExternalField

    provider = CacheProvider(
        name="bad-provider",
        external_fields=[
            CacheProviderExternalField(name="token", type="password"),
            CacheProviderExternalField(name="exporter", metrics_target=True),
        ],
        inference_backend_integrations=[
            {
                "backend": "vLLM",
                "injection": {
                    "env": {"TOKEN": "{{token}}", "EXP": "{{exporter}}"},
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
    assert "references field 'token'" in joined
    assert "references field 'exporter'" in joined
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
        display_name="Localized",
        description={"default": "A cache", "zh-CN": "一个缓存"},
        links=[{"label": {"default": "Docs", "zh-CN": "文档"}, "url": "https://x"}],
        managed_fields=[
            {"name": "size", "label": {"default": "Size", "ja-JP": "サイズ"}}
        ],
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
        external_fields=[
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
        providers = load_cache_providers(reload=True)
        assert [provider.name for provider in providers] == ["Good"]
    finally:
        monkeypatch.undo()
        load_cache_providers(reload=True)


def test_every_form_field_declares_a_label():
    """The UI humanizes a missing label from the field name, which is an
    English identifier: a field without a label is a slot that stays
    English in every other locale."""
    missing = []
    for provider in load_cache_providers(reload=True):
        for field in provider.external_fields:
            if field.label is None:
                missing.append(f"{provider.name} external field '{field.name}'")
        for field in provider.managed_fields:
            if field.label is None:
                missing.append(f"{provider.name} managed field '{field.name}'")
        for key, backend in provider.l2_backends.items():
            for field in backend.fields:
                if field.label is None:
                    missing.append(f"{provider.name} l2 '{key}' field '{field.name}'")
    assert not missing, "fields missing a label: " + ", ".join(missing)
