import shlex

import pytest

from gpustack.api.exceptions import BadRequestException
from gpustack.schemas.models import (
    DisaggregationSpec,
    Model,
    RoleSpec,
    role_effective_model,
)
from gpustack.server.pd_mode_catalog import get_pd_mode, load_pd_modes
from gpustack.utils.pd_router_membership import membership_update_policy
from gpustack.validation.pd.roles import validate_roles
from gpustack.worker.pd_router import (
    PeerAddress,
    apply_managed_router,
    is_managed_router,
)

load_pd_modes()


def model_for(mode, command):
    return Model(
        id=42,
        name="model",
        source="local_path",
        local_path="/models/model",
        backend="SGLang" if mode.startswith("sglang-") else "vLLM",
        roles=[
            RoleSpec(name="prefill"),
            RoleSpec(name="decode"),
            RoleSpec(
                name="router",
                backend="Custom",
                image_name="upstream:router",
                run_command=command,
            ),
        ],
        disaggregation=DisaggregationSpec(mode=mode),
        owner_principal_id=1,
    )


VARIABLES = {
    "worker_ip": "10.0.0.3",
    "port": 41000,
    "ports.prometheus": 41001,
    "runner_image": "engine:image",
    "model_name": "name with spaces;$(literal)",
}
PEERS = {
    "prefill": [
        PeerAddress("10.0.0.1", 42000, {"bootstrap": 42001}),
        PeerAddress("10.0.0.2", 43000, {"bootstrap": 43001}),
    ],
    "decode": [PeerAddress("10.0.0.4", 44000), PeerAddress("10.0.0.5", 45000)],
}
COMMAND = """my-router --listen {{ worker_ip }} --listen-port {{port}}
--label {{model_name}} --metrics-port {{ports.prometheus}}
{% for p in prefill %} --prompt-worker {{p.url}} {{p.ports.bootstrap}} {% endfor %}
{% for d in decode %} --generation-worker {{d.ip}}:{{d.port}} {% endfor %}
"""


def launch(model, peers=PEERS, variables=VARIABLES):
    return apply_managed_router(
        role_effective_model(model, "router"),
        "router",
        peers=peers,
        variables=variables,
    )


@pytest.mark.parametrize(
    "mode",
    [
        "vllm-nixl",
        "vllm-ascend-mooncake",
        "vllm-mooncake",
        "sglang-nixl",
        "sglang-mooncake",
        "custom",
    ],
)
def test_template_owns_flags_and_expands_every_ready_endpoint(mode):
    model = model_for(mode, COMMAND)
    validate_roles(model)
    assert is_managed_router(model, "router")
    rendered = launch(model)
    assert shlex.split(rendered.run_command) == [
        "my-router",
        "--listen",
        "10.0.0.3",
        "--listen-port",
        "41000",
        "--label",
        VARIABLES["model_name"],
        "--metrics-port",
        "41001",
        "--prompt-worker",
        "http://10.0.0.1:42000",
        "42001",
        "--prompt-worker",
        "http://10.0.0.2:43000",
        "43001",
        "--generation-worker",
        "10.0.0.4:44000",
        "--generation-worker",
        "10.0.0.5:45000",
    ]
    assert rendered.image_name == "upstream:router"
    assert model.roles[-1].run_command == COMMAND
    assert membership_update_policy(model, get_pd_mode(mode)).value == "restart"
    moved = {**PEERS, "decode": [PeerAddress("2001:db8::1", 46000)]}
    replacement = launch(model, moved)
    assert "2001:db8::1:46000" in replacement.run_command
    assert "10.0.0.5:45000" not in replacement.run_command


def test_join_and_quoting_preserve_argument_boundaries_without_shell_evaluation():
    model = model_for(
        "vllm-nixl",
        '''router --host 0.0.0.0 --port {{port}}
--prefill-endpoints "{{ prefill_urls | join(',') }}"
--decode-endpoints={{decode_urls | join(' ')}}
--label="prefix {{model_name}} suffix"''',
    )
    validate_roles(model)
    assert shlex.split(launch(model).run_command) == [
        "router",
        "--host",
        "0.0.0.0",
        "--port",
        "41000",
        "--prefill-endpoints",
        "http://10.0.0.1:42000,http://10.0.0.2:43000",
        "--decode-endpoints=http://10.0.0.4:44000 http://10.0.0.5:45000",
        "--label=prefix name with spaces;$(literal) suffix",
    ]


@pytest.mark.parametrize(
    "expression",
    [
        "{{missing}}",
        "{{router_args}}",
        "{{prefill_args}}",
        "{{decode_args}}",
        "{{worker_ip.__class__}}",
        "{{prefill[0].url}}",
        "{{prefill}}",
        "{{prefill_urls | safe}}",
        "{{prefill_urls | join(worker_ip)}}",
        "{{range(10)}}",
        "{% include 'file' %}",
        "{% set x = 1 %}",
        "{% if prefill %}x{% endif %}",
        "{% for p in range(100) %}x{% endfor %}",
        "{% for p in prefill %}{{p.missing}}{% endfor %}",
        "{% for p in prefill %}{% for d in decode %}{{p.url}}{% endfor %}{% endfor %}",
        "{% for port in prefill %}{{port.url}}{% endfor %}",
        "{% for p in prefill if p.port %}{{p.url}}{% endfor %}",
        "{% for p in prefill %}",
        '"{{worker_ip}}',
    ],
)
def test_unsupported_templates_are_rejected_before_scheduling(expression):
    with pytest.raises(BadRequestException):
        validate_roles(model_for("vllm-nixl", "router " + expression))


@pytest.mark.parametrize(
    "expression",
    ["{{ports.absent}}", "{% for p in decode %}{{p.ports.bootstrap}}{% endfor %}"],
)
def test_missing_runtime_data_fails_with_the_variable_name(expression):
    model = model_for("vllm-nixl", "router " + expression)
    validate_roles(model)
    with pytest.raises(ValueError, match="absent|bootstrap"):
        launch(model)


def test_empty_peer_role_is_not_silently_rendered_as_no_arguments():
    model = model_for("custom", COMMAND)
    with pytest.raises(ValueError, match="decode"):
        launch(model, {"prefill": PEERS["prefill"]})


def test_ipv6_urls_and_url_loops_use_routable_endpoint_format():
    model = model_for(
        "custom", "router {% for u in prefill_urls %}--worker {{u}} {% endfor %}"
    )
    assert shlex.split(
        launch(
            model,
            {"prefill": [PeerAddress("2001:db8::1", 80)], "decode": PEERS["decode"]},
        ).run_command
    ) == ["router", "--worker", "http://[2001:db8::1]:80"]


@pytest.mark.parametrize(
    "command,expected",
    [
        (None, "dynamic"),
        ("  ", "dynamic"),
        ("router --port {{ port }}", "manual"),
        ("router --workers static", "manual"),
    ],
)
def test_only_peer_dependent_commands_require_restart(command, expected):
    model = model_for("vllm-nixl", command)
    assert membership_update_policy(model, get_pd_mode("vllm-nixl")).value == expected
    if command and "{{" in command:
        assert shlex.split(launch(model).run_command) == ["router", "--port", "41000"]


def test_template_backend_parameters_may_use_user_defined_peer_flags():
    model = model_for("vllm-nixl", COMMAND)
    model.roles[-1].backend_parameters = ["--prefill http://other-service:80"]
    validate_roles(model)
    assert launch(model).backend_parameters == ["--prefill http://other-service:80"]


def test_peer_templates_are_router_only():
    model = model_for("vllm-nixl", None)
    model.roles[0].run_command = "engine {% for p in prefill %}{{p.url}}{% endfor %}"
    with pytest.raises(BadRequestException):
        validate_roles(model)


def test_blank_custom_router_command_means_automatic_generation():
    model = model_for("vllm-nixl", "   ")
    validate_roles(model)
    assert launch(model).run_command.startswith("vllm-router ")


@pytest.mark.parametrize('mode', ['vllm-nixl', 'custom'])
def test_template_uses_the_selected_backend_image_without_an_explicit_override(mode):
    model = model_for(mode, COMMAND)
    model.roles[-1].backend = 'vLLM'
    model.roles[-1].image_name = None
    model.roles[-1].backend_version = '0.33.0'
    validate_roles(model)
    assert launch(model).image_name == VARIABLES['runner_image']


@pytest.mark.parametrize(
    'command,image',
    [
        (None, None),
        ('router --port 41000', None),
        ('router --port {{port}}', 'upstream:router'),
        ("router --peers {{prefill_urls | join(',')}}", 'upstream:router'),
    ],
)
@pytest.mark.parametrize('override_env', [False, True])
def test_router_commands_keep_rendered_catalog_env_defaults(
    monkeypatch, command, image, override_env
):
    mode = get_pd_mode('vllm-nixl').model_copy(deep=True)
    mode.router.env = {
        'NEEDS_THIS_TO_BOOT': '0',
        'ADVERTISE': '{{worker_ip}}',
        'CHOICE': 'catalog',
    }
    monkeypatch.setattr('gpustack.server.pd_mode_catalog.get_pd_mode', lambda _: mode)
    model = model_for('vllm-nixl', command)
    model.roles[-1].backend = 'vLLM'
    model.roles[-1].image_name = image
    user_env = {'CHOICE': 'user', 'USER_ONLY': 'kept'} if override_env else None
    model.roles[-1].env = user_env
    rendered = launch(model)
    assert rendered.env['NEEDS_THIS_TO_BOOT'] == '0'
    assert rendered.env['ADVERTISE'] == VARIABLES['worker_ip']
    assert rendered.env['CHOICE'] == ('user' if override_env else 'catalog')
    if override_env:
        assert rendered.env['USER_ONLY'] == 'kept'
    assert model.roles[-1].env == user_env
    assert mode.router.env['ADVERTISE'] == '{{worker_ip}}'


@pytest.mark.parametrize('env_source', ['model', 'role', 'catalog'])
@pytest.mark.parametrize('with_peers', [False, True])
def test_router_templates_accept_declared_environment_values(
    monkeypatch, env_source, with_peers
):
    source = 'router --config {{ROUTER_CONFIG}} --port {{port}}'
    if with_peers:
        source += ' {% for p in prefill %}--peer {{p.url}} {% endfor %}'
    model = model_for('vllm-nixl', source)
    value = "/etc/router config 'quoted'.json;$(literal)"
    mode = get_pd_mode('vllm-nixl').model_copy(deep=True)
    monkeypatch.setattr('gpustack.server.pd_mode_catalog.get_pd_mode', lambda _: mode)
    monkeypatch.setattr('gpustack.validation.pd.roles.get_pd_mode', lambda _: mode)
    if env_source == 'catalog':
        mode.router.env = {'ROUTER_CONFIG': value}
    elif env_source == 'role':
        model.roles[-1].env = {'ROUTER_CONFIG': value}
    else:
        model.env = {'ROUTER_CONFIG': value}
    validate_roles(model)
    argv = shlex.split(launch(model).run_command)
    assert argv[:5] == ['router', '--config', value, '--port', '41000']
    assert argv.count('--peer') == (2 if with_peers else 0)
    assert membership_update_policy(model, mode).value == (
        'restart' if with_peers else 'manual'
    )


def test_a_role_env_override_does_not_expose_uninherited_parent_values():
    model = model_for('vllm-nixl', 'router --config {{ROUTER_CONFIG}}')
    model.env = {'ROUTER_CONFIG': 'parent.json'}
    model.roles[-1].env = {}
    with pytest.raises(BadRequestException):
        validate_roles(model)


def inherited_router_model(command):
    model = model_for('custom', None)
    model.backend = 'Custom'
    model.image_name = 'upstream:router'
    model.run_command = command
    model.env = {'ROUTER_CONFIG': '/etc/router config.json'}
    for role in model.roles[:2]:
        role.backend = 'vLLM'
    model.roles[-1].backend = None
    model.roles[-1].image_name = None
    return model


def test_custom_pd_never_uses_the_parent_router_template():
    model = inherited_router_model("router --peers {{decode_urls | join(',')}}")
    validate_roles(model)
    assert not is_managed_router(model, 'router')
    assert membership_update_policy(model, get_pd_mode('custom')).value == 'manual'


@pytest.mark.parametrize('command,backend', [('', None), (None, 'vLLM')])
def test_an_explicit_router_command_or_backend_stops_command_inheritance(
    command, backend
):
    model = inherited_router_model(
        'router {% for p in prefill %}{{p.url}} {% endfor %}'
    )
    model.roles[-1].backend = backend
    model.roles[-1].run_command = command
    assert not is_managed_router(model, 'router')
    assert membership_update_policy(model, get_pd_mode('custom')).value == 'manual'


def test_builtin_router_does_not_inherit_the_deployments_inference_command():
    model = inherited_router_model('engine --config {{ENGINE_CONFIG}}')
    model.disaggregation = DisaggregationSpec(mode='vllm-nixl')
    validate_roles(model)
    rendered = launch(model)
    assert shlex.split(rendered.run_command)[0] == 'vllm-router'
    assert membership_update_policy(model, get_pd_mode('vllm-nixl')).value == 'dynamic'


@pytest.mark.parametrize(
    'command',
    [
        'engine --peers {{decode_urls | join(",")}}',
        'engine --peers {{decode_urls}}',
        'engine --peers {{ (prefill_urls) }}',
        'engine {% for p in prefill %}--peer {{p.url}} {% endfor %}',
    ],
)
@pytest.mark.parametrize('sparse_update', [False, True])
def test_engine_inherited_peer_templates_are_rejected_before_scheduling(
    command, sparse_update
):
    from gpustack.schemas.models import ModelUpdate

    model = model_for('vllm-nixl', None)
    model.run_command = command
    request = (
        ModelUpdate(
            name=model.name,
            source=model.source,
            local_path=model.local_path,
            replicas=1,
        )
        if sparse_update
        else model
    )
    with pytest.raises(BadRequestException) as exc:
        validate_roles(request, stored=model if sparse_update else None)
    assert 'Peer command templates require a Router role' in exc.value.message


@pytest.mark.parametrize('override', ['backend', 'command'])
def test_unused_parent_peer_templates_do_not_apply_to_independent_engine_backends(
    override,
):
    model = model_for('vllm-nixl', None)
    model.run_command = 'engine {% for p in prefill %}{{p.url}} {% endfor %}'
    for role in model.roles[:2]:
        if override == 'backend':
            role.backend = 'vLLM'
        else:
            role.run_command = 'engine {{model_path}}'
    validate_roles(model)


def test_engine_scalar_commands_can_still_be_inherited():
    model = model_for('vllm-nixl', None)
    model.run_command = 'engine {{model_path}} --port {{port}}'
    validate_roles(model)


@pytest.mark.parametrize(
    'command', ['engine {{decode_urls', 'engine ' + 'x' * 65536 + ' {{decode_urls}}']
)
def test_malformed_or_oversized_engine_peer_references_are_rejected(command):
    model = model_for('vllm-nixl', None)
    model.run_command = command
    with pytest.raises(BadRequestException) as exc:
        validate_roles(model)
    assert 'Peer command templates require a Router role' in exc.value.message
