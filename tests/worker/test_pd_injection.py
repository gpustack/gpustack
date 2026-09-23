"""The PD-mode catalog rendered into an engine launch.

Two properties are what these pin:

* a recipe reaches the process — `vllm-nixl`'s prefill role comes out as the
  three NIXL env variables plus one `--kv-transfer-config`, and its decode
  role differs from it in exactly one field (`kv_role`), because that is the
  whole of what the two sides disagree on;
* everything that cannot render correctly is visible. An unresolved
  placeholder survives verbatim with a warning, and a launch that cannot
  resolve one at all is refused rather than started wrong.

Never two `--kv-transfer-config` documents, whichever way the second one
arrives: vLLM reads the flag once, so a role that sets it itself takes it over
and the recipe stops contributing its own.
"""

import json
import logging
import types

import pytest

from gpustack.schemas.models import (
    DisaggregationSpec,
    ExtendedKVCacheConfig,
    KVCacheModeEnum,
    Model,
    ModelInstance,
    PDModeEnum,
    PortBand,
    RoleSpec,
    SourceEnum,
)
from gpustack.utils.template import deployment_variables
from gpustack.worker.pd_injection import (
    KV_TRANSFER_CONFIG_FLAG,
    PDInjectionError,
    render_pd_injection,
)


def _model(mode=PDModeEnum.VLLM_NIXL, roles=None, **kwargs) -> Model:
    if roles is None:
        roles = [
            RoleSpec(name="prefill", replicas=1),
            RoleSpec(name="decode", replicas=1),
            RoleSpec(name="router", replicas=1),
        ]
    return Model(
        id=1,
        name="llm",
        replicas=1,
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        owner_principal_id=1,
        roles=roles,
        disaggregation=(DisaggregationSpec(mode=mode) if mode is not None else None),
        **kwargs,
    )


def _instance(role="prefill", named_ports=None, **kwargs) -> ModelInstance:
    if named_ports is None:
        named_ports = {"kv_side_channel": PortBand(base=5600, count=1)}
    return ModelInstance(
        id=1,
        name="llm-0",
        model_id=1,
        model_name="llm",
        source=SourceEnum.HUGGING_FACE,
        huggingface_repo_id="org/repo",
        worker_id=1,
        port=40000,
        role=role,
        group_id="g1",
        named_ports=named_ports,
        **kwargs,
    )


def _variables(**overrides):
    """What `_template_variables()` hands the injector on the worker."""
    variables = deployment_variables(
        model_path="/models/llm",
        port=40000,
        worker_ip="192.168.50.10",
        model_name="llm",
        gpu_count=1,
        gpu_ids=[0],
        role=overrides.pop("role", "prefill"),
        group_id="g1",
    )
    variables["net_device"] = "eth0"
    variables["runner_image"] = "gpustack/runner:cuda12.8-vllm0.20.0"
    variables.update(overrides)
    return variables


def _connector(injection):
    """The rendered `--kv-transfer-config` document."""
    assert KV_TRANSFER_CONFIG_FLAG in injection.args
    return json.loads(injection.args[injection.args.index(KV_TRANSFER_CONFIG_FLAG) + 1])


def test_vllm_nixl_prefill_renders_the_whole_launch():
    injection = render_pd_injection(_model(), _instance(), _variables())

    assert injection.env == {
        "VLLM_NIXL_SIDE_CHANNEL_HOST": "192.168.50.10",
        "VLLM_NIXL_SIDE_CHANNEL_PORT": "5600",
        "UCX_NET_DEVICES": "eth0",
    }
    assert injection.args == [
        KV_TRANSFER_CONFIG_FLAG,
        '{"kv_connector":"NixlConnector","kv_role":"kv_producer",'
        '"kv_load_failure_policy":"fail",'
        '"kv_connector_extra_config":{"kv_lease_duration":60}}',
    ]
    assert injection.files == {}
    # The lease window is a number in the document, not the string the
    # template was: the engine parses this as JSON.
    assert _connector(injection)["kv_connector_extra_config"] == {
        "kv_lease_duration": 60
    }


def test_vllm_nixl_decode_differs_only_in_kv_role():
    prefill = render_pd_injection(_model(), _instance("prefill"), _variables())
    decode = render_pd_injection(
        _model(), _instance("decode"), _variables(role="decode")
    )

    assert _connector(decode)["kv_role"] == "kv_consumer"
    assert _connector(prefill)["kv_role"] == "kv_producer"
    # Both sides bind and advertise their own side channel.
    assert decode.env == prefill.env


def test_kv_load_failure_policy_comes_from_the_deployment():
    model = _model()
    model.disaggregation.kv_load_failure_policy = "recompute"

    injection = render_pd_injection(model, _instance(), _variables())

    assert _connector(injection)["kv_load_failure_policy"] == "recompute"


def test_not_a_pd_instance_returns_none():
    # No `disaggregation` at all: plain multi-role orchestration.
    assert render_pd_injection(_model(mode=None), _instance(), _variables()) is None
    # No role: a single-role deployment.
    assert render_pd_injection(_model(), _instance(role=None), _variables()) is None


def test_router_role_is_not_an_engine_injection():
    """The router's command is assembled by the router path, not merged into
    an engine's launch — `mode.roles` has no entry for it."""
    assert (
        render_pd_injection(
            _model(),
            _instance("router", named_ports={"prometheus": PortBand(base=29000)}),
            _variables(role="router"),
            peers={"prefill": [("192.168.50.10", 40000)]},
        )
        is None
    )


def test_custom_mode_injects_nothing():
    assert (
        render_pd_injection(_model(mode=PDModeEnum.CUSTOM), _instance(), _variables())
        is None
    )


def test_an_unallocated_port_band_stops_the_launch(caplog):
    """The renderer still leaves the placeholder rather than blanking it — a
    blank port is a plausible-looking wrong value — but the launch no longer
    proceeds with it. An argument carrying one is at least visible in the
    engine's echoed argv; an env var is not, and the port band lands in a
    variable."""
    with caplog.at_level(logging.WARNING):
        with pytest.raises(PDInjectionError, match="ports.kv_side_channel"):
            render_pd_injection(_model(), _instance(named_ports={}), _variables())

    assert "ports.kv_side_channel" in caplog.text


def test_an_underivable_net_device_stops_the_launch(caplog):
    """An unresolved `{{net_device}}` must not be rendered. A host with several
    candidate NICs makes `derive_net_device` refuse to guess, and a rendered
    literal `{{net_device}}` reaches HCCL as the name of an interface that does
    not exist: nothing fails loudly, the transport simply never connects.

    An Ascend recipe whose `{{net_device}}` is a control-plane socket takes
    `Worker.ifname` instead (`net_device_plane: control`) and so never reaches
    the refusal; the refusal governs the NIXL recipes, where an unresolvable NIC
    stops the launch rather than being rendered as `all`.

    The message has to name the escape hatch, because the operator's next
    question is where to put the answer."""
    variables = _variables()
    variables.pop("net_device")

    with caplog.at_level(logging.WARNING):
        with pytest.raises(PDInjectionError, match="kv_ifname") as e:
            render_pd_injection(_model(), _instance(), variables)

    assert "net_device" in str(e.value)


def test_spaced_placeholder_is_not_one():
    """`{{ worker_ip }}` is not a placeholder — the catalog loader rejects the
    spelling, and the renderer would leave it alone anyway."""
    model = _model()
    injection = render_pd_injection(model, _instance(), _variables())
    rendered = render_pd_injection(
        model, _instance(), {**_variables(), " worker_ip ": "10.0.0.1"}
    )
    assert rendered.env == injection.env


@pytest.mark.parametrize(
    "extended",
    [
        ExtendedKVCacheConfig(enabled=True, mode=KVCacheModeEnum.LOCAL),
        ExtendedKVCacheConfig(
            enabled=True, mode=KVCacheModeEnum.SHARED, cache_service_id=7
        ),
    ],
)
def test_pd_renders_its_own_connector_beside_an_extended_cache(extended):
    """The injector's job is this role's connector, whole. A cache that also
    contributes one is folded in later by `kv_transfer`, once the whole argv
    exists — which is where the per-role ordering can be applied. Refusing
    here would make the two mutually exclusive."""
    model = _model(extended_kv_cache=extended)

    injection = render_pd_injection(model, _instance(), _variables())

    assert KV_TRANSFER_CONFIG_FLAG in injection.args
    # Its own descriptor, not a composite: composing is not this seam's call.
    index = injection.args.index(KV_TRANSFER_CONFIG_FLAG)
    assert "NixlConnector" in injection.args[index + 1]


def test_a_per_role_cache_leaves_every_role_renderable():
    """Which sides take a cache is per role, and none of them is a reason to
    refuse the PD connector — the prefill side is exactly where a shared cache
    pays."""
    model = _model(
        roles=[
            RoleSpec(
                name="prefill",
                extended_kv_cache=ExtendedKVCacheConfig(
                    enabled=True, mode=KVCacheModeEnum.SHARED, cache_service_id=7
                ),
            ),
            RoleSpec(name="decode"),
        ]
    )

    assert render_pd_injection(model, _instance("prefill"), _variables()).args
    assert render_pd_injection(
        model, _instance("decode"), _variables(role="decode")
    ).args


def test_a_user_written_kv_transfer_config_takes_the_flag_over():
    """This asserted a refusal until the form started seeding the recipe's
    descriptor into the role's parameter list as an editable row.

    Setting the flag is the documented way to change it now, so it can no
    longer be read as going around us. What still must not happen is two
    descriptors — vLLM reads the flag once — so the injection drops its own
    and the user's is what reaches the engine."""
    model = _model(
        backend_parameters=[KV_TRANSFER_CONFIG_FLAG, '{"kv_connector":"Mine"}']
    )

    injection = render_pd_injection(model, _instance(), _variables())

    assert KV_TRANSFER_CONFIG_FLAG not in injection.args
    # The rest of the recipe still applies: only the one flag changed hands.
    assert injection.env


def test_sglang_pd_does_not_collide_with_a_shared_cache():
    """SGLang's disaggregation is configured through its own flags, so the
    exclusion is on the flag, not on "PD plus cache"."""
    model = _model(
        mode=PDModeEnum.SGLANG_MOONCAKE,
        extended_kv_cache=ExtendedKVCacheConfig(
            enabled=True, mode=KVCacheModeEnum.SHARED, cache_service_id=7
        ),
    )

    injection = render_pd_injection(
        model,
        _instance(named_ports={"bootstrap": PortBand(base=8998, count=1)}),
        _variables(),
    )

    assert injection.args == [
        "--disaggregation-mode",
        "prefill",
        "--disaggregation-bootstrap-port",
        "8998",
        "--disaggregation-transfer-backend",
        "mooncake",
    ]
    assert KV_TRANSFER_CONFIG_FLAG not in injection.args


def test_ascend_mooncake_carries_both_sides_parallelism_and_a_lease_env():
    model = _model(
        mode=PDModeEnum.VLLM_ASCEND_MOONCAKE,
        roles=[
            RoleSpec(
                name="prefill",
                backend_parameters=["--tensor-parallel-size", "8"],
            ),
            RoleSpec(
                name="decode",
                backend_parameters=["--tensor-parallel-size", "4", "--dp", "2"],
            ),
        ],
    )
    instance = _instance(named_ports={"kv_port": PortBand(base=41100, count=8)})

    injection = render_pd_injection(model, instance, _variables())
    descriptor = _connector(injection)

    # Cross-role: prefill's own config names decode's parallelism.
    assert descriptor["kv_connector_extra_config"]["prefill"]["tp_size"] == 8
    assert descriptor["kv_connector_extra_config"]["decode"]["tp_size"] == 4
    assert descriptor["kv_connector_extra_config"]["decode"]["dp_size"] == 2
    # The port lives inside the descriptor for this connector, as a number.
    assert descriptor["kv_port"] == 41100
    # A data parallelism nobody wrote resolves to 1 for a single-worker member,
    # because that is what the engine will run at — not a default chosen here.
    # Leaving it unresolved would force the user to write out a parameter that
    # only restates what the engine would do.
    assert descriptor["kv_connector_extra_config"]["prefill"]["dp_size"] == 1
    # Mooncake's abort window is an env var, and its engine default is 8
    # minutes with no Prometheus counter to notice an expiry.
    assert injection.env["VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT"] == "60"
    assert injection.env["HCCL_SOCKET_IFNAME"] == "eth0"


def test_nixl_lease_is_not_injected_as_an_env_var():
    """NIXL's window is a connector-config field; injecting it as an env var
    as well would be a second spelling of one fact."""
    injection = render_pd_injection(_model(), _instance(), _variables())
    assert "kv_lease_duration" not in injection.env


def _backend(model, instance):
    """A worker-side server object with only what the injection seams read.

    `_model_spec` is the unprojected model and `_model` the projection, which
    is the split the start path itself has: the injector must be handed the
    former, or a cross-role reference resolves against the running role's
    values.
    """
    from gpustack.schemas.models import role_effective_model
    from gpustack.worker.backends.custom import CustomServer

    backend = CustomServer.__new__(CustomServer)
    backend._model_spec = model
    backend._model = role_effective_model(model, instance.role)
    backend._model_instance = instance
    backend._worker = types.SimpleNamespace(id=1, ip="192.168.50.10", ifname="eth0")
    backend._config = types.SimpleNamespace(data_dir="/var/lib/gpustack")
    backend._model_path = "/models/llm"
    backend.inference_backend = None
    backend._get_selected_gpu_devices = lambda: []
    backend._resolve_image = lambda backend_name=None: (
        "gpustack/runner:cuda12.8-vllm0.20.0",
        None,
    )
    return backend


def test_start_path_carries_env_args_and_attributes_them_to_gpustack():
    model = _model(backend_parameters=["--max-model-len", "8192"])
    backend = _backend(model, _instance())

    env = backend._get_configured_env()
    assert env["VLLM_NIXL_SIDE_CHANNEL_HOST"] == "192.168.50.10"
    assert env["VLLM_NIXL_SIDE_CHANNEL_PORT"] == "5600"
    # Derived from the worker's own interface until the net-device module
    # lands; either way it is never left as "all".
    assert env["UCX_NET_DEVICES"] == "eth0"

    tokens = backend._flatten_backend_param()
    assert tokens[0] == KV_TRANSFER_CONFIG_FLAG
    assert json.loads(tokens[1])["kv_role"] == "kv_producer"
    assert tokens[2:] == ["--max-model-len", "8192"]

    arguments = ["vllm", "serve", "/models/llm"] + tokens
    injected = backend._get_injected_backend_parameters(arguments, tokens)
    assert injected[0] == KV_TRANSFER_CONFIG_FLAG
    assert "--max-model-len" not in injected


def test_a_users_parameter_is_rendered_like_an_env_value():
    """The two halves of a deployment's configuration follow one rule.

    `env` values have always been rendered; parameters were not, so the same
    `{{worker_ip}}` resolved in one and reached the engine verbatim in the
    other. The PD form is what made the split untenable — it seeds the recipe's
    own rows into the role's parameter list so they can be edited, and those
    rows are written in placeholders."""
    # Non-PD on purpose: the rule under test is the shared parameter path, and
    # a PD model would additionally need a derivable KV interface, which a
    # developer laptop with two candidate NICs does not have.
    model = _model(
        mode=None,
        backend_parameters=[
            "--served-model-name",
            "{{model_name}}",
            "--max-model-len=8192",
        ],
    )
    backend = _backend(model, _instance())

    tokens = backend._flatten_backend_param()

    assert "{{model_name}}" not in tokens
    assert tokens[tokens.index("--served-model-name") + 1] == "llm"
    # Everything without a placeholder is untouched.
    assert "--max-model-len=8192" in tokens


def test_an_unknown_placeholder_in_a_parameter_survives(caplog):
    """Same posture as `render` takes everywhere else: verbatim plus a
    warning. A user parameter may legitimately contain braces that are not
    ours, and blanking one would turn text they chose into a plausible-looking
    wrong value."""
    model = _model(mode=None, backend_parameters=["--chat-template", "{{not_ours}}"])
    backend = _backend(model, _instance())

    with caplog.at_level(logging.WARNING):
        tokens = backend._flatten_backend_param()

    assert "{{not_ours}}" in tokens
    assert "not_ours" in caplog.text


def test_a_json_parameter_keeps_its_quotes_through_rendering():
    """Rendering is per token, after `flatten_to_argv` has decided what a token
    is. Doing it over the joined string would give a value containing a space a
    second chance to be split — which for a JSON document means losing its
    quotes."""
    model = _model(
        mode=None,
        backend_parameters=[
            "--kv-transfer-config",
            '{"kv_connector":"Mine","host":"{{worker_ip}}"}',
        ],
    )
    backend = _backend(model, _instance())

    tokens = backend._flatten_backend_param()
    document = tokens[tokens.index("--kv-transfer-config") + 1]

    assert json.loads(document) == {
        "kv_connector": "Mine",
        "host": "192.168.50.10",
    }


def test_non_pd_deploy_takes_the_same_path_it_takes_today():
    model = _model(mode=None, backend_parameters=["--max-model-len", "8192"])
    backend = _backend(model, _instance())

    assert backend._pd_injection() is None
    assert backend._flatten_backend_param() == ["--max-model-len", "8192"]
    assert "VLLM_NIXL_SIDE_CHANNEL_HOST" not in backend._get_configured_env()
    assert backend._cache_injection_files() == {}


def test_a_refused_injection_stays_refused_at_every_seam():
    """The refusal must not be cached as "nothing to inject": the seams run in
    sequence, and a seam that swallowed the first one would then start the
    engine with neither connector.

    Uses the one clash that is still a refusal — a hand-written
    `--kv-transfer-config` under a mode that injects its own. The cache is no
    longer one of these, because it is composed rather than refused."""
    model = _model(
        backend_parameters=[KV_TRANSFER_CONFIG_FLAG, '{"kv_connector":"Mine"}']
    )
    backend = _backend(model, _instance())

    with pytest.raises(PDInjectionError):
        backend._get_configured_env()
    with pytest.raises(PDInjectionError):
        backend._flatten_backend_param()


def test_model_env_overrides_the_injection_but_says_so(caplog):
    model = _model(env={"UCX_NET_DEVICES": "mlx5_0:1"})
    backend = _backend(model, _instance())

    with caplog.at_level(logging.WARNING):
        env = backend._get_configured_env()

    assert env["UCX_NET_DEVICES"] == "mlx5_0:1"
    assert env["VLLM_NIXL_SIDE_CHANNEL_HOST"] == "192.168.50.10"
    assert "UCX_NET_DEVICES" in caplog.text


def test_injection_files_join_the_shared_cache_files(caplog):
    from gpustack.schemas.cache_services import CacheConfigSnapshot
    from gpustack.worker.pd_injection import PDInjection

    instance = _instance()
    instance.cache_config = CacheConfigSnapshot(
        cache_service_id=1,
        injected=True,
        files={"/tmp/cache.json": "cache", "/tmp/both.json": "cache"},
    )
    backend = _backend(_model(mode=None), instance)
    backend._pd_injection_resolved = True
    backend._pd_injection_cache = PDInjection(
        files={"/tmp/pd.json": "pd", "/tmp/both.json": "pd"}
    )

    with caplog.at_level(logging.WARNING):
        files = backend._cache_injection_files()

    assert files == {
        "/tmp/cache.json": "cache",
        "/tmp/pd.json": "pd",
        "/tmp/both.json": "pd",
    }
    assert "/tmp/both.json" in caplog.text
    # And the serving script writes every one of them before the engine runs.
    script = backend._get_serving_command_script({})
    assert "/tmp/pd.json" in script and "/tmp/cache.json" in script


def test_unknown_mode_returns_none_and_warns(caplog):
    model = _model()
    model.disaggregation = types.SimpleNamespace(
        mode="not-a-mode", kv_load_failure_policy="fail"
    )

    with caplog.at_level(logging.WARNING):
        assert render_pd_injection(model, _instance(), _variables()) is None

    assert "not-a-mode" in caplog.text


# --- the host IPC trade-off ------------------------------------------------ #


def test_a_disaggregated_member_with_a_cache_is_told_what_it_traded(caplog):
    """A shared cache wants the host IPC namespace, for the CUDA-IPC path that
    passes KV buffers instead of copying them. A KV connector wants a private
    /dev/shm, and joining the host namespace replaces it with the host's,
    dropping the shm_size the workload was given.

    Both configurations run, so this does not refuse. What it must not do is
    decide silently: the person who cares cannot otherwise see the question
    was asked, and both directions are one env away."""
    from gpustack.schemas.cache_services import CacheConfigSnapshot

    instance = _instance()
    instance.cache_config = CacheConfigSnapshot(cache_service_id=7, injected=True)
    backend = _backend(_model(), instance)

    with caplog.at_level(logging.WARNING):
        assert backend._host_ipc_enabled() is True

    assert "/dev/shm" in caplog.text
    assert "GPUSTACK_HOST_IPC" in caplog.text


def test_a_role_less_deployment_with_a_cache_is_not_warned(caplog):
    """No connector, no tension — the derivation is just the cache's own
    requirement and there is nothing being traded away."""
    from gpustack.schemas.cache_services import CacheConfigSnapshot

    instance = _instance()
    instance.role = None
    instance.cache_config = CacheConfigSnapshot(cache_service_id=7, injected=True)
    model = _model()
    model.disaggregation = None
    backend = _backend(model, instance)

    with caplog.at_level(logging.WARNING):
        assert backend._host_ipc_enabled() is True

    assert "/dev/shm" not in caplog.text


def test_the_warning_is_said_once(caplog):
    """It is derived on every workload build; repeating it per build would
    bury the things that happen once."""
    from gpustack.schemas.cache_services import CacheConfigSnapshot

    instance = _instance()
    instance.cache_config = CacheConfigSnapshot(cache_service_id=7, injected=True)
    backend = _backend(_model(), instance)

    with caplog.at_level(logging.WARNING):
        backend._host_ipc_enabled()
        backend._host_ipc_enabled()

    assert caplog.text.count("GPUSTACK_HOST_IPC") == 1


def test_the_escape_hatch_wins_and_says_nothing(caplog):
    """An explicit answer is not a trade-off being made for anyone."""
    from gpustack.schemas.cache_services import CacheConfigSnapshot

    instance = _instance()
    instance.cache_config = CacheConfigSnapshot(cache_service_id=7, injected=True)
    backend = _backend(_model(env={"GPUSTACK_HOST_IPC": "false"}), instance)

    with caplog.at_level(logging.WARNING):
        assert backend._host_ipc_enabled() is False

    assert "/dev/shm" not in caplog.text


# --- a placeholder must not reach the engine ------------------------------- #


def test_an_unrendered_env_var_stops_the_launch():
    """An argument that keeps its placeholder survives visibly — vLLM echoes
    its argv into the log and the scanner recognises the shape. An environment
    variable does not: `HCCL_SOCKET_IFNAME={{net_device}}` reaches HCCL as the
    literal name of an interface that does not exist, and what comes back is a
    transport that quietly never connects.

    A placeholder reaches this point when a host has several candidate NICs and
    `derive_net_device` refuses to guess between them. The refusal is
    defensible; its consequence would be invisible — hence the hard stop."""
    from gpustack.worker.pd_injection import PDInjection, _refuse_unrendered

    injection = PDInjection(
        env={"HCCL_SOCKET_IFNAME": "{{net_device}}", "HCCL_IF_IP": "10.0.0.1"},
        args=["--host", "10.0.0.1"],
    )

    with pytest.raises(
        PDInjectionError, match="HCCL_SOCKET_IFNAME=\\{\\{net_device\\}\\}"
    ):
        _refuse_unrendered(injection, "mode 'm' role 'prefill'")


def test_an_unrendered_argument_stops_the_launch_too():
    from gpustack.worker.pd_injection import PDInjection, _refuse_unrendered

    injection = PDInjection(args=["--port", "{{ports.kv_port}}"])

    with pytest.raises(PDInjectionError, match="ports.kv_port"):
        _refuse_unrendered(injection, "where")


def test_every_unrendered_placeholder_is_named_not_counted():
    """`{{net_device}}` and `{{ports.kv_port}}` are fixed in entirely different
    places, so a count tells the operator nothing about where to go."""
    from gpustack.worker.pd_injection import PDInjection, _refuse_unrendered

    injection = PDInjection(env={"A": "{{net_device}}"}, args=["{{ports.kv_port}}"])

    with pytest.raises(PDInjectionError) as e:
        _refuse_unrendered(injection, "where")
    assert "net_device" in str(e.value) and "ports.kv_port" in str(e.value)


def test_a_fully_rendered_injection_passes():
    from gpustack.worker.pd_injection import PDInjection, _refuse_unrendered

    _refuse_unrendered(
        PDInjection(env={"A": "eno1"}, args=["--port", "40001"]), "where"
    )


def test_a_multi_worker_member_leaves_its_parallelism_to_the_engine():
    """The other half of the implicit rule, and the reason it is not a plain
    default. For a member spanning workers the shape decides dp and dpl, that
    decision is made further down the vLLM path, and a number guessed here
    would go into a Mooncake descriptor the engine then contradicts — a pairing
    that fails at handshake, which is worse than the launch refusing."""
    from types import SimpleNamespace

    from gpustack.worker.pd_injection import _implicit_parallelism

    single = SimpleNamespace(gpu_indexes=[0, 1], distributed_servers=None)
    spanning = SimpleNamespace(
        gpu_indexes=[0, 1],
        distributed_servers=SimpleNamespace(subordinate_workers=[object()]),
    )

    assert _implicit_parallelism({}, single, "prefill") == {
        "tensor_parallel_size": 2,
        "data_parallel_size": 1,
    }
    assert _implicit_parallelism({}, spanning, "prefill") == {}


def test_a_declared_parallelism_is_never_overwritten():
    """The implicit rule fills gaps; it does not have opinions."""
    from types import SimpleNamespace

    from gpustack.worker.pd_injection import _implicit_parallelism

    instance = SimpleNamespace(gpu_indexes=[0, 1, 2, 3], distributed_servers=None)

    assert (
        _implicit_parallelism(
            {"tensor_parallel_size": 2, "data_parallel_size": 4}, instance, "prefill"
        )
        == {}
    )


# --- host mounts ----------------------------------------------------------- #


def test_the_ascend_recipe_asks_for_the_hccn_map_on_both_sides():
    """The mount is what makes cross-host transfers work at all.

    AscendDirectTransport resolves a peer's host address to per-card RoCE
    addresses through `/etc/hccn.conf`, and the Ascend container runtime
    injects the devices and the driver but not that file. Without it every
    rank's `batch_transfer_sync_read` returns -1 cross-host while same-host
    transfers succeed, and the connector logs the exception, sends its done
    signal and lets decode answer 200 with garbage.
    """
    # The recipe's connector names both roles' parallelism, so both roles
    # have to declare it or the render refuses before reaching the mounts.
    roles = [
        RoleSpec(
            name="prefill", replicas=1, backend_parameters=["--tensor-parallel-size=4"]
        ),
        RoleSpec(
            name="decode", replicas=1, backend_parameters=["--tensor-parallel-size=2"]
        ),
        RoleSpec(name="router", replicas=1),
    ]
    for role in ("prefill", "decode"):
        injection = render_pd_injection(
            _model(mode=PDModeEnum.VLLM_ASCEND_MOONCAKE, roles=roles),
            _instance(
                role=role, named_ports={"kv_port": PortBand(base=41100, count=8)}
            ),
            _variables(role=role),
        )
        assert injection.host_mounts == ["/etc/hccn.conf"], role


def test_a_recipe_that_needs_no_host_file_asks_for_none():
    """NIXL reaches its peers over the host stack, so there is nothing to
    mount — and a mount every recipe carried would be a host path bound into
    containers that have no use for it."""
    injection = render_pd_injection(_model(), _instance(), _variables())
    assert injection.host_mounts == []


def test_an_unrendered_mount_path_stops_the_launch():
    """Same class as an unrendered env: the bind either fails or creates an
    empty directory where the transport expects a file, and neither says
    why."""
    from gpustack.schemas.pd_modes import PDMode, PDModeRole
    from gpustack.worker import pd_injection as module

    mode = PDMode(
        name=PDModeEnum.VLLM_NIXL.value,
        backends=["vLLM"],
        roles={"prefill": PDModeRole(host_mounts=["/etc/{{nowhere}}.conf"])},
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(module, "get_pd_mode", lambda name: mode)
        with pytest.raises(PDInjectionError) as excinfo:
            render_pd_injection(_model(), _instance(), _variables())
    assert "{{nowhere}}" in str(excinfo.value)


def test_the_mount_reaches_the_container_read_only():
    """The channel only pays off if the container actually gets it, so the
    assertion is on the runtime's mount list rather than on the injection."""
    from gpustack_runtime.deployer import ContainerMountModeEnum

    from gpustack.worker.backends.base import InferenceServer
    from gpustack.worker.pd_injection import PDInjection

    class _Stub:
        _model_path = None

        def _pd_injection(self):
            return PDInjection(host_mounts=["/etc/hccn.conf"])

    mounts = InferenceServer._get_configured_mounts(_Stub())
    assert [(m.path, m.mode) for m in mounts] == [
        ("/etc/hccn.conf", ContainerMountModeEnum.ROX)
    ]


# --- whose cards are whose ------------------------------------------------- #


def test_a_peer_that_pins_its_own_cards_is_not_read_off_the_running_member():
    """`{{roles.decode.tensor_parallel_size}}` must not resolve off the running
    member: rendering the prefill's own card count there writes 4 into
    Mooncake's descriptor for a decode pinned to one card — a number decode's
    engine contradicts the moment it derives its own, which is the handshake
    failure the implicit rule exists to avoid.

    A peer that pinned its own cards can be answered from the spec, and is."""
    from gpustack.schemas.models import GPUSelector

    model = _model(
        mode=PDModeEnum.VLLM_ASCEND_MOONCAKE,
        roles=[
            RoleSpec(name="prefill", replicas=1),
            RoleSpec(
                name="decode",
                replicas=1,
                gpu_selector=GPUSelector(gpu_ids=["w1:npu:4"]),
            ),
        ],
    )
    instance = _instance(
        named_ports={"kv_port": PortBand(base=41100, count=4)},
        gpu_indexes=[0, 1, 2, 3],
    )

    descriptor = _connector(render_pd_injection(model, instance, _variables()))
    extra = descriptor["kv_connector_extra_config"]

    assert extra["prefill"]["tp_size"] == 4
    assert extra["decode"]["tp_size"] == 1


def test_two_silent_roles_still_render_from_the_member_that_is_starting():
    """The deliberate limit of the fix above. The peer's own member cannot be
    consulted here — prefill routinely starts before decode has been placed, so
    its `gpu_indexes` may not exist yet — and refusing to render would break
    the silent 1P1D the implicit rule was written for. With no per-role pin
    both roles are sized by the same auto-selection of the same model, so the
    running member's count is the honest answer rather than a guess."""
    model = _model(mode=PDModeEnum.VLLM_ASCEND_MOONCAKE)
    instance = _instance(
        named_ports={"kv_port": PortBand(base=41100, count=2)},
        gpu_indexes=[0, 1],
    )

    extra = _connector(render_pd_injection(model, instance, _variables()))[
        "kv_connector_extra_config"
    ]
    assert extra["prefill"]["tp_size"] == 2
    assert extra["decode"]["tp_size"] == 2


def test_a_peers_declared_parallelism_still_wins_over_its_pin():
    """The pin only answers for a role that wrote nothing. A decode that
    declares TP2 on four pinned cards is a decode at TP2, and the descriptor
    has to carry what the engine will run."""
    from gpustack.schemas.models import GPUSelector

    model = _model(
        mode=PDModeEnum.VLLM_ASCEND_MOONCAKE,
        roles=[
            RoleSpec(name="prefill", replicas=1),
            RoleSpec(
                name="decode",
                replicas=1,
                backend_parameters=["--tensor-parallel-size", "2"],
                gpu_selector=GPUSelector(
                    gpu_ids=["w1:npu:4", "w1:npu:5", "w1:npu:6", "w1:npu:7"]
                ),
            ),
        ],
    )
    instance = _instance(
        named_ports={"kv_port": PortBand(base=41100, count=4)},
        gpu_indexes=[0, 1, 2, 3],
    )

    extra = _connector(render_pd_injection(model, instance, _variables()))[
        "kv_connector_extra_config"
    ]
    assert extra["decode"]["tp_size"] == 2
