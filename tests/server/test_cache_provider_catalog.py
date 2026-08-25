import json

import pytest
from pydantic import ValidationError

from gpustack.schemas.cache_providers import (
    CacheProvider,
    CacheProviderVersionConfig,
    render_l2_adapter,
    validate_injection_templates,
)
from gpustack.schemas.cache_services import CacheServiceModeEnum
from gpustack.server.cache_provider_catalog import (
    get_cache_provider,
    load_cache_providers,
    render_injection,
)


def test_catalog_asset_loads():
    providers = load_cache_providers(reload=True)
    assert providers, "bundled cache-providers.yaml should yield at least one provider"


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
    assert provider.default_version == "v0.5.3"
    assert set(provider.versions) == {"v0.5.2", "v0.5.3", "v0.5.4"}
    assert provider.custom_version is True

    version_config, version = provider.get_version_config()
    assert version_config is not None
    assert version == provider.default_version
    assert version_config.image == "lmcache/vllm-openai:v0.5.3"
    # The bare tag is the CUDA 13 build; cu129 serves CUDA 12 nodes. The
    # worker resolves per node, so a heterogeneous per_node fleet mixes
    # images; unknown runtimes and accelerator-less workers get the
    # plain image.
    assert version_config.resolve_image("cuda", "13.0") == "lmcache/vllm-openai:v0.5.3"
    assert (
        version_config.resolve_image("cuda", "12.8")
        == "lmcache/vllm-openai:v0.5.3-cu129"
    )
    assert version_config.resolve_image(None, None) == "lmcache/vllm-openai:v0.5.3"
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


def test_lmcache_metrics_declaration():
    provider = get_cache_provider("LMCache")
    assert provider is not None

    # The declaration only locates the exposition (scrape targets ride
    # on it); semantic metric mappings return with the native-UI
    # metrics integration.
    metrics = provider.metrics
    assert metrics is not None
    assert metrics.path == "/metrics"


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

    # Master-side exposition on the master's conventional metrics port;
    # default_port seeds the registration form's metrics-port field.
    metrics = provider.metrics
    assert metrics is not None
    assert metrics.path == "/metrics"
    assert metrics.default_port == 9003
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
    }
    diverging_fields = {
        "l2_backends",
        "versions",
        "default_version",
        # The staged Ascend build lives in the image templates; the CUDA
        # layout and the run command are asserted equal below.
        "default_runtime_images",
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

    # The versions diverge from LMCache's only by the staged Ascend
    # build (an assumed image for XSKY to correct) and by which version
    # each provider defaults to; at a given version the CUDA builds and
    # the run command stay LMCache's.
    mf_version = meshfusion.versions[meshfusion.default_version]
    lm_version = lmcache.versions[meshfusion.default_version]
    assert mf_version.run_command == lm_version.run_command
    assert mf_version.runtime_images["cuda"] == lm_version.runtime_images["cuda"]
    assert "cann" in mf_version.runtime_images
    assert "cann" not in lm_version.runtime_images

    # Every integration is framework-scoped — the catalog is the single
    # accelerator gate. MeshFusion diverges from LMCache only by the
    # extra cann-scoped vLLM entry (an assumed placeholder for XSKY;
    # vllm-ascend trails vLLM, so its attachable range is declared
    # separately). The cuda entries mirror LMCache's.
    vllm_entries = [
        c for c in meshfusion.inference_backend_integrations if c.backend == "vLLM"
    ]
    assert [(c.frameworks, c.versions) for c in vllm_entries] == [
        (["cuda"], ">=0.25.0"),
        (["cann"], ">=0.25.0"),
    ]
    lm_vllm = lmcache.integration_for("vLLM", "cuda")
    assert lm_vllm.frameworks == ["cuda"]
    for entry in vllm_entries:
        assert entry.injection == lm_vllm.injection
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

    # MeshFusion adds XSKY's store L2 backend (adapter type "xdfs",
    # branded with the XSKY icon) on top of the LMCache-inherited
    # backends; LMCache has none of it.
    assert "xdfs" not in lmcache.l2_backends
    # The inherited backends must stay byte-identical, not just share keys.
    for key, backend in lmcache.l2_backends.items():
        assert meshfusion.l2_backends[key] == backend
    xdfs = meshfusion.l2_backends["xdfs"]
    assert xdfs.icon == "/static/catalog_icons/xsky.png"
    xdfs_fields = {field.name for field in xdfs.fields}
    assert {"metadata_endpoint", "sdk_config_file", "max_write_inflight_bytes"} <= (
        xdfs_fields
    )
    assert next(f for f in xdfs.fields if f.name == "metadata_endpoint").required
    # No store-side metrics scrape this version: L2 observability rides on
    # the cache server's own lmcache_mp_* metrics.
    assert all(field.metrics_target is False for field in xdfs.fields)

    args, env = render_l2_adapter(
        meshfusion,
        "xdfs",
        {"metadata_endpoint": "10.0.0.20:8000"},
    )
    assert '"metadata_endpoint":"10.0.0.20:8000"' in args[1]
    assert env == {}


def test_provider_brand_links():
    lmcache = get_cache_provider("LMCache")
    assert {link.label for link in lmcache.links} == {"Documentation", "GitHub"}
    assert all(link.url.startswith("https://") for link in lmcache.links)

    mooncake = get_cache_provider("Mooncake")
    assert {link.label for link in mooncake.links} == {"Documentation", "GitHub"}

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
