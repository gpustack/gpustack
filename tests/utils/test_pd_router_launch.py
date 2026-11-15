import pytest

from gpustack.schemas.inference_backend import InferenceBackend, VersionConfigDict
from gpustack.schemas.models import DisaggregationSpec, Model, RoleSpec
from gpustack.utils.pd_router_launch import resolve_router_launch


def model_for(mode='vllm-nixl', **router):
    return Model(
        name='model',
        source='local_path',
        local_path='/model',
        backend='vLLM',
        backend_version='0.29.0',
        image_name='parent:engine',
        run_command='vllm serve /model',
        roles=[
            RoleSpec(name='prefill'),
            RoleSpec(name='decode'),
            RoleSpec(name='router', **router),
        ],
        disaggregation=DisaggregationSpec(mode=mode),
    )


def backend_for(name='router-custom', built_in=False):
    return InferenceBackend(
        backend_name=name,
        is_built_in=built_in,
        default_version='1',
        default_run_command='--default-port {{port}}',
        default_entrypoint='router',
        default_env={'CONFIG': 'default'},
        version_configs=VersionConfigDict(
            root={
                '1': {
                    'image_name': 'router:1',
                    'run_command': '--version-port {{port}}',
                    'entrypoint': 'router-v1',
                    'env': {'CONFIG': 'version'},
                },
                '2': {'image_name': 'router:2'},
            }
        ),
    )


@pytest.mark.parametrize(
    'name,built_in', [('router-custom', False), ('vLLM', True), ('SGLang', True)]
)
def test_selected_backend_command_priority_is_independent_of_its_origin(name, built_in):
    backend = backend_for(name, built_in)
    model = model_for(backend=name, backend_version='1')
    plan = resolve_router_launch(model, backend)
    assert (plan.source, plan.command, plan.entrypoint) == (
        'backend_version',
        '--version-port {{port}}',
        ['router-v1'],
    )
    assert plan.version == '1'
    assert plan.env == {'CONFIG': 'version'}
    model.roles[-1].backend_version = '2'
    plan = resolve_router_launch(model, backend)
    assert (plan.source, plan.command, plan.entrypoint) == (
        'backend_default',
        '--default-port {{port}}',
        ['router'],
    )
    model.roles[-1].run_command = 'own-router --port {{port}}'
    plan = resolve_router_launch(model, backend)
    assert (plan.source, plan.command, plan.entrypoint) == (
        'role',
        'own-router --port {{port}}',
        None,
    )


@pytest.mark.parametrize('mode', ['vllm-nixl', 'custom'])
@pytest.mark.parametrize('image', [None, 'own:router'])
def test_inheritance_and_image_only_overrides_never_reuse_parent_command(mode, image):
    model = model_for(mode, image_name=image)
    plan = resolve_router_launch(model, backend_for('vLLM', True))
    assert plan.command is None
    assert plan.source == 'recipe'
    assert plan.entrypoint is None


def test_blank_commands_fall_back_without_reusing_parent_command():
    backend = backend_for()
    backend.version_configs.root['1'].run_command = '  '
    backend.default_run_command = '\n'
    plan = resolve_router_launch(
        model_for(backend=backend.backend_name, run_command=' '), backend
    )
    assert plan.source == 'recipe' and plan.command is None


def test_auto_version_and_explicit_image_use_the_same_backend_defaults():
    backend = backend_for()
    plan = resolve_router_launch(
        model_for(backend=backend.backend_name, image_name='own:router'), backend
    )
    assert plan.version == '1'
    assert plan.command == '--version-port {{port}}'


def test_unselected_backend_definition_cannot_supply_router_command():
    plan = resolve_router_launch(model_for(backend='other-custom'), backend_for())
    assert plan.command is None


@pytest.mark.parametrize('mode', ['vllm-nixl', 'custom'])
@pytest.mark.parametrize('static', [False, True])
def test_backend_command_is_rendered_without_recipe_flags_and_owns_membership(
    mode, static
):
    import shlex
    from gpustack.schemas.models import role_effective_model
    from gpustack.server.pd_mode_catalog import get_pd_mode, load_pd_modes
    from gpustack.utils.pd_router_membership import membership_update_policy
    from gpustack.validation.pd.roles import validate_roles
    from gpustack.worker.pd_router import apply_managed_router, PeerAddress

    load_pd_modes()
    backend = backend_for()
    command = (
        '--fixed-peer http://fixed:80'
        if static
        else "--listen {{worker_ip}}:{{port}} --config {{CONFIG}} --workers {{decode_urls | join(',')}}"
    )
    backend.version_configs.root['1'].run_command = command
    model = model_for(mode, backend=backend.backend_name)
    model.roles[-1].backend_parameters = ['--user-flag']
    validate_roles(model, router_backend=backend)
    rendered = apply_managed_router(
        role_effective_model(model, 'router'),
        'router',
        inference_backend=backend,
        peers={
            'decode': [PeerAddress('10.0.0.1', 8000), PeerAddress('10.0.0.2', 8001)]
        },
        variables={'worker_ip': '10.0.0.3', 'port': 9000, 'runner_image': 'router:1'},
    )
    assert rendered.backend == 'Custom'
    assert rendered.backend_version == '1'
    assert rendered.image_name == 'router:1'
    assert rendered.backend_parameters == ['--user-flag']
    assert rendered.env['CONFIG'] == 'version'
    assert shlex.split(rendered.run_command) == (
        ['--fixed-peer', 'http://fixed:80']
        if static
        else [
            '--listen',
            '10.0.0.3:9000',
            '--config',
            'version',
            '--workers',
            'http://10.0.0.1:8000,http://10.0.0.2:8001',
        ]
    )
    assert membership_update_policy(model, get_pd_mode(mode), backend).value == (
        'manual' if static else 'restart'
    )


@pytest.mark.asyncio
async def test_custom_mode_requires_its_own_command_instead_of_inheriting_parent(
    monkeypatch,
):
    from gpustack.api.exceptions import BadRequestException
    from gpustack.validation.pd import validate_pd_config

    with pytest.raises(BadRequestException) as exc:
        await validate_pd_config(None, model_for('custom'))
    assert 'Router requires a command' in exc.value.message


@pytest.mark.asyncio
async def test_backend_lookup_only_reads_the_deployment_owner_and_platform(monkeypatch):
    from unittest.mock import AsyncMock
    from gpustack.utils.pd_router_launch import router_backend_definition

    platform = backend_for()
    owner = InferenceBackend(
        backend_name='router-custom',
        owner_principal_id=7,
        default_run_command=None,
        version_configs=VersionConfigDict(root={'1': {'run_command': 'org-router'}}),
    )
    lookup = AsyncMock(side_effect=[platform, owner])
    monkeypatch.setattr(InferenceBackend, 'one_by_fields', lookup)
    model = model_for(backend='router-custom')
    model.owner_principal_id = 7
    backend = await router_backend_definition(None, model)
    assert [call.args[1]['owner_principal_id'] for call in lookup.call_args_list] == [
        None,
        7,
    ]
    assert backend.default_run_command == platform.default_run_command
    assert resolve_router_launch(model, backend).command == 'org-router'
    assert platform.version_configs.root['1'].run_command != 'org-router'


@pytest.mark.parametrize('own_command', [None, 'own-router --port {{port}}'])
def test_selected_router_backend_health_is_reused_only_for_router(own_command):
    from gpustack.worker.pd_router import managed_router_health_path

    model = model_for(backend='router-custom', run_command=own_command)
    backend = backend_for()
    backend.health_check_path = '/ready'
    assert managed_router_health_path(model, 'router', backend) == '/ready'
    assert managed_router_health_path(model, 'prefill', backend) is None


@pytest.mark.asyncio
async def test_cached_router_backend_reads_reuse_the_catalog_and_observe_invalidation(
    monkeypatch,
):
    from unittest.mock import AsyncMock
    from gpustack.utils.pd_router_launch import router_backend_definition

    platform = backend_for()
    own = backend_for()
    own.owner_principal_id = 7
    own.version_configs.root['1'].run_command = 'org-router'
    other = backend_for()
    other.owner_principal_id = 99
    other.version_configs.root['1'].run_command = 'other-org-router'
    query = AsyncMock(return_value=[platform, own, other])
    monkeypatch.setattr(InferenceBackend, '_do_cached_all_query', query)
    direct = AsyncMock(
        side_effect=AssertionError('reconcile must use the catalog cache')
    )
    monkeypatch.setattr(InferenceBackend, 'one_by_fields', direct)
    model = model_for(backend='router-custom')
    model.owner_principal_id = 7
    await InferenceBackend._invalidate_cached_all()
    try:
        for _ in range(3):
            result = await router_backend_definition(None, model, cached=True)
            assert resolve_router_launch(model, result).command == 'org-router'
        assert query.await_count == 1
        model.owner_principal_id = 8
        result = await router_backend_definition(None, model, cached=True)
        assert resolve_router_launch(model, result).command == '--version-port {{port}}'
        assert query.await_count == 1
        replacement = backend_for()
        replacement.version_configs.root['1'].run_command = 'updated-router'
        query.return_value = [replacement]
        await InferenceBackend._invalidate_cached_all()
        result = await router_backend_definition(None, model, cached=True)
        assert resolve_router_launch(model, result).command == 'updated-router'
        assert query.await_count == 2
    finally:
        await InferenceBackend._invalidate_cached_all()


@pytest.mark.parametrize(
    'command_source', ['backend_version', 'backend_default', 'role']
)
def test_router_backend_environment_is_rendered_before_the_startup_guard(
    command_source,
):
    import shlex
    from types import SimpleNamespace
    from gpustack.schemas.models import role_effective_model
    from gpustack.validation.pd.roles import validate_roles
    from gpustack.worker.backends.base import _refuse_unrendered_router
    from gpustack.worker.pd_router import apply_managed_router

    backend = backend_for()
    backend.default_env = {'ADVERTISE_HOST': '{{worker_ip}}', 'CONFIG': 'backend'}
    backend.version_configs.root['1'].env = {
        'SERVICE_PORT': '{{port}}',
        'METRICS_PORT': '{{ports.prometheus}}',
    }
    command = '--advertise {{ADVERTISE_HOST}} --port {{SERVICE_PORT}}'
    backend.version_configs.root['1'].run_command = (
        command if command_source == 'backend_version' else None
    )
    backend.default_run_command = (
        command if command_source == 'backend_default' else None
    )
    model = model_for(
        backend='router-custom',
        run_command=('own-router ' + command) if command_source == 'role' else None,
    )
    model.env = {'USER_BIND': '{{worker_ip}}'}
    validate_roles(model, router_backend=backend)
    rendered = apply_managed_router(
        role_effective_model(model, 'router'),
        'router',
        inference_backend=backend,
        peers={},
        variables={
            'worker_ip': '10.0.0.3',
            'port': 9000,
            'ports.prometheus': 9001,
            'runner_image': 'router:1',
        },
    )
    _refuse_unrendered_router(
        rendered, SimpleNamespace(name='router'), command_rendered=True
    )
    assert rendered.env['ADVERTISE_HOST'] == rendered.env['USER_BIND'] == '10.0.0.3'
    assert rendered.env['SERVICE_PORT'] == '9000'
    assert rendered.env['METRICS_PORT'] == '9001'
    argv = shlex.split(rendered.run_command)
    assert argv[argv.index('--advertise') + 1] == '10.0.0.3'
    assert argv[argv.index('--port') + 1] == '9000'
    assert (
        backend.default_env['ADVERTISE_HOST']
        == model.env['USER_BIND']
        == '{{worker_ip}}'
    )


@pytest.mark.parametrize('mode', ['vllm-nixl', 'sglang-nixl'])
def test_recipe_fallback_keeps_backend_environment_and_user_overrides(mode):
    from types import SimpleNamespace
    from gpustack.schemas.models import role_effective_model
    from gpustack.worker.backends.base import _refuse_unrendered_router
    from gpustack.worker.pd_router import apply_managed_router, PeerAddress

    backend = backend_for()
    backend.default_run_command = None
    backend.version_configs.root['1'].run_command = None
    backend.default_env = {
        'ROUTER_CONFIG': '/etc/router.json',
        'ADVERTISE_HOST': '{{worker_ip}}',
        'CONFIG': 'backend',
    }
    backend.version_configs.root['1'].env = {
        'VERSION_ONLY': 'kept',
        'CONFIG': 'version',
    }
    model = model_for(mode, backend='router-custom', env={'CONFIG': 'role'})
    rendered = apply_managed_router(
        role_effective_model(model, 'router'),
        'router',
        inference_backend=backend,
        peers={
            'prefill': [PeerAddress('10.0.0.1', 8000, {'bootstrap': 8001})],
            'decode': [PeerAddress('10.0.0.2', 8100)],
        },
        variables={
            'worker_ip': '10.0.0.3',
            'port': 9000,
            'ports.prometheus': 9001,
            'runner_image': 'router:1',
            'net_device': 'eth0',
        },
    )
    _refuse_unrendered_router(rendered, SimpleNamespace(name='router'))
    assert rendered.env['ROUTER_CONFIG'] == '/etc/router.json'
    assert rendered.env['ADVERTISE_HOST'] == '10.0.0.3'
    assert rendered.env['VERSION_ONLY'] == 'kept'
    assert rendered.env['CONFIG'] == 'role'


def test_a_missing_backend_environment_value_still_fails_the_startup_guard():
    from types import SimpleNamespace
    from gpustack.worker.backends.base import _refuse_unrendered_router
    from gpustack.worker.pd_injection import PDInjectionError
    from gpustack.worker.pd_router import apply_managed_router

    backend = backend_for()
    backend.default_env = {'ADVERTISE_HOST': '{{NOT_PROVIDED}}'}
    model = model_for(backend='router-custom')
    rendered = apply_managed_router(
        model,
        'router',
        inference_backend=backend,
        peers={},
        variables={'port': 9000, 'runner_image': 'router:1'},
    )
    with pytest.raises(PDInjectionError, match='ADVERTISE_HOST'):
        _refuse_unrendered_router(
            rendered, SimpleNamespace(name='router'), command_rendered=True
        )


@pytest.mark.parametrize(
    'command_source', ['role', 'backend_version', 'backend_default', 'recipe']
)
@pytest.mark.parametrize('health_path', [None, '/my-ready'])
def test_router_health_path_follows_the_command_owner(command_source, health_path):
    from gpustack.worker.pd_router import managed_router_health_path
    from gpustack.server.pd_mode_catalog import get_pd_mode, load_pd_modes

    load_pd_modes()
    backend = backend_for()
    backend.health_check_path = health_path
    backend.version_configs.root['1'].run_command = (
        '--port {{port}}' if command_source == 'backend_version' else None
    )
    backend.default_run_command = (
        '--port {{port}}' if command_source == 'backend_default' else None
    )
    model = model_for(
        backend='router-custom',
        run_command='my-router --port {{port}}' if command_source == 'role' else None,
    )
    expected = (
        get_pd_mode('vllm-nixl').router.health_path
        if command_source == 'recipe'
        else health_path
    )
    assert managed_router_health_path(model, 'router', backend) == expected
