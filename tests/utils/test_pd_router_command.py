import shlex
from types import SimpleNamespace

import pytest

from gpustack.utils.pd_router_command import (
    render_router_command_template,
    validate_router_command_template,
)


def test_runtime_values_are_substituted_once_and_keep_quotes_and_metacharacters():
    value = "a 'quoted' path; $(literal) --fake {{port}}"
    argv = render_router_command_template(
        "router --tag {{model_name}} --port {{port}}",
        {"model_name": value, "port": 80},
        {},
    )
    assert argv == ["router", "--tag", value, "--port", "80"]
    assert shlex.split(shlex.join(argv)) == argv


@pytest.mark.parametrize(
    'source,variables',
    [
        ('router --tag {{model_name}}', {'model_name': 'x' * 1048577}),
        ('router --tag {{model_name}}', {'model_name': 'x\x00y'}),
        ('router \x00', {}),
        ('router --port {{port}}', {'port': None}),
    ],
)
def test_unusable_runtime_values_fail_before_container_start(source, variables):
    with pytest.raises(ValueError):
        render_router_command_template(source, variables, {})


def test_loop_expansion_has_a_finite_limit():
    peers = {'prefill': [SimpleNamespace(ip='127.0.0.1', port=80, ports={})] * 8193}
    with pytest.raises(ValueError, match='too many'):
        render_router_command_template(
            'router {% for p in prefill %}--peer {{p.url}} {% endfor %}', {}, peers
        )


@pytest.mark.parametrize(
    'source',
    [
        'router {{prefill_urls | join(attribute="url")}}',
        'router {{prefill_urls | join(",", ":")}}',
        'router {% macro command() %}evil{% endmacro %}',
        'router {{worker_ip.upper()}}',
        'router {% for p in prefill %}{{p.ports.__class__}}{% endfor %}',
    ],
)
def test_template_evaluation_is_limited_to_the_documented_language(source):
    with pytest.raises(ValueError):
        validate_router_command_template(source)


@pytest.mark.parametrize(
    'command,argument',
    [
        (r"router --value '\d+\s+' --port {{port}}", r'\d+\s+'),
        (
            r"router --value 'C:\model files\qwen' --port {{port}}",
            r'C:\model files\qwen',
        ),
        (
            r'router --value="C:\model files\qwen" --port {{port}}',
            r'C:\model files\qwen',
        ),
        (r'router --value a\ b --port {{port}}', 'a b'),
    ],
)
def test_static_arguments_follow_posix_quoting_without_losing_quoted_backslashes(
    command, argument
):
    argv = render_router_command_template(command, {'port': 80}, {})
    expected = ['router', '--value', argument, '--port', '80']
    if '--value=' in command:
        expected = ['router', '--value=' + argument, '--port', '80']
    assert argv == expected
    assert shlex.split(shlex.join(argv)) == expected


def test_declared_env_values_are_atomic_and_cannot_shadow_runtime_values():
    env = {
        'ROUTER_CONFIG': "path with 'quotes';$(literal) {{port}}",
        'port': 'wrong',
        'worker_ip': 'wrong',
    }
    command = 'router --config={{ROUTER_CONFIG}} --port {{port}} --host {{worker_ip}}'
    validate_router_command_template(command, env=env)
    assert render_router_command_template(
        command, {'port': 80, 'worker_ip': '10.0.0.1'}, {}, env=env
    ) == [
        'router',
        '--config=' + env['ROUTER_CONFIG'],
        '--port',
        '80',
        '--host',
        '10.0.0.1',
    ]


def test_declared_dotted_env_names_keep_flat_environment_lookup():
    command = 'router --config {{ROUTER.CONFIG}}'
    env = {'ROUTER.CONFIG': '/etc/my config.json'}
    validate_router_command_template(command, env=env)
    assert render_router_command_template(command, {}, {}, env=env) == [
        'router',
        '--config',
        env['ROUTER.CONFIG'],
    ]


@pytest.mark.parametrize(
    'expression',
    [
        '{{ROUTER_CONFIG.upper()}}',
        '{{ROUTER_CONFIG.__class__}}',
        '{{ROUTER_CONFIG | attr("__class__")}}',
    ],
)
def test_environment_support_does_not_enable_template_code_execution(expression):
    with pytest.raises(ValueError):
        validate_router_command_template(
            'router ' + expression, env={'ROUTER_CONFIG': '/etc/config.json'}
        )


@pytest.mark.parametrize(
    'expression,env,variables',
    [
        ('worker_ip', {'worker_ip': 'spoofed-host'}, {'worker_ip': '10.0.0.3'}),
        ('ports.prometheus', {'ports.prometheus': '9999'}, {'ports.prometheus': 9001}),
    ],
)
def test_reserved_runtime_values_never_fall_back_to_environment(
    expression, env, variables
):
    command = 'router --value {{ ' + expression + ' }}'
    for declared_env in (None, env):
        validate_router_command_template(command, env=declared_env)
        with pytest.raises(ValueError, match='unavailable'):
            render_router_command_template(command, {}, {}, env=declared_env)
        assert render_router_command_template(command, variables, {}, env=declared_env)[
            -1
        ] == str(next(iter(variables.values())))
