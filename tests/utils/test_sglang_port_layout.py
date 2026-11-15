from dataclasses import FrozenInstanceError

import pytest

from gpustack.schemas.models import (
    DisaggregationSpec,
    DistributedServers,
    ModelInstanceSubordinateWorker,
    PortBand,
)
from gpustack.utils.command import find_parameter
from gpustack.utils.sglang_port_layout import (
    sglang_port_layout,
    sglang_port_layout_for_instance,
)
from tests.utils.model import new_model, new_model_instance


def _model(*, attention=True, rust=False, pd=False):
    return new_model(
        1,
        'sg',
        huggingface_repo_id='org/model',
        backend='SGLang',
        backend_version='0.5.22',
        backend_parameters=['--tp=2', '--dp=2']
        + (['--enable-dp-attention'] if attention else []),
        env={'SGLANG_RUST_SERVER': '1'} if rust else {},
        disaggregation=DisaggregationSpec(mode='sglang-mooncake') if pd else None,
    )


def _instance(*, managed=False, cross=False, role=None):
    mi = new_model_instance(1, 'sg', 1, worker_id=1, gpu_indexes=[0])
    mi.worker_ip, mi.role = '10.0.0.1', role
    mi.port, mi.ports = 40000, [40000, 40010]
    mi.named_ports = {'_sglang_connecting': PortBand(base=40010, count=14)}
    mi.distributed_servers = DistributedServers(
        workloads=[] if managed else None,
        subordinate_workers=(
            [
                ModelInstanceSubordinateWorker(
                    worker_id=2, worker_ip='10.0.0.2', gpu_indexes=[0]
                )
            ]
            if cross
            else []
        ),
    )
    if managed:
        mi.named_ports.update(
            {
                '_sglang_engine_api': PortBand(base=40000),
                '_sglang_smg_api': PortBand(base=40050),
                '_sglang_smg_metrics': PortBand(base=40051),
            }
        )
        mi.ports.extend([40050, 40051])
    return mi


@pytest.mark.parametrize('cross', [False, True])
@pytest.mark.parametrize('rust', [False, True])
def test_layout_returns_only_native_component_and_engine_requests(cross, rust):
    allocation = [1, 1] if cross else [2]
    for role in (None, 'prefill', 'decode'):
        model = _model(rust=rust, pd=role is not None)
        layout = sglang_port_layout(model, role, allocation)
        assert layout.engine_version == '0.5.22'
        assert layout.internal_count == 14
        assert layout.nccl_offset == (9 if rust else 7)
        requests = layout.port_requests()
        assert ('_sglang_connecting', 14, None) in requests
        assert not {'bootstrap', 'prometheus'} & {name for name, _, _ in requests}
        assert sum(count for _, count, _ in requests) == (17 if role is None else 15)
        with pytest.raises(FrozenInstanceError):
            layout.engine_version = 'other'


def test_router_does_not_inherit_engine_port_rules():
    layout = sglang_port_layout(_model(pd=True), 'router', [0])
    assert layout.internal_count == 0
    assert layout.runtime_ports == ()
    assert layout.port_requests() == [('_api', 1, None)]


def test_native_layout_does_not_accept_connector_declarations():
    layout = sglang_port_layout(_model(pd=True), 'prefill', [1, 1])
    with pytest.raises(TypeError):
        layout.port_requests([('bootstrap', 1)])


def test_router_does_not_emit_engine_initialization_for_inherited_topology():
    model, mi = _model(pd=True), _instance(role='router', cross=True)
    mi.named_ports = {'prometheus': PortBand(base=40001)}
    binding = sglang_port_layout_for_instance(model, mi).bind(mi, '10.0.0.1')
    assert binding.arguments == ()
    assert binding.runtime_ports == ()


@pytest.mark.parametrize('rust', [False, True])
def test_fixed_and_runtime_ports_are_distinct(rust):
    layout = sglang_port_layout(_model(pd=True, rust=rust), 'prefill', [1, 1])
    fixed = dict(layout.internal_offsets)
    assert fixed['rendezvous'] == 0
    assert fixed['tokenizer'] == 1
    assert fixed['rpc'] == 3
    runtime = {port.name: (port.owner, port.scope) for port in layout.runtime_ports}
    assert runtime['pd_kv_rank'] == ('sglang', 'transfer_rank')
    if rust:
        assert fixed['scheduler_input_0'] == 7
        assert fixed['scheduler_input_1'] == 8
        assert fixed['nccl'] == 9
        assert 'dp_scheduler_input' not in runtime
    else:
        assert fixed['controller_handshake'] == 13
        assert fixed['nccl'] == 7
        assert runtime['dp_scheduler_input'] == ('sglang', 'routing_rank')


@pytest.mark.parametrize('managed', [False, True])
@pytest.mark.parametrize('leader', [False, True])
def test_resolved_bindings_preserve_primary_address_and_declare_engine_ports(
    managed, leader
):
    model = _model(pd=not managed)
    mi = _instance(managed=managed, cross=True, role=None if managed else 'prefill')
    mi.named_ports['bootstrap'] = PortBand(base=40030)
    mi.ports.append(40030)
    layout = sglang_port_layout_for_instance(model, mi)
    bindings = layout.bind(mi, '10.0.0.1' if leader else '10.0.0.2', is_leader=leader)
    assert (
        find_parameter(list(bindings.arguments), ['dist-init-addr']) == '10.0.0.1:40010'
    )
    assert find_parameter(list(bindings.arguments), ['node-rank']) == (
        '0' if leader else '1'
    )
    assert set(range(40010, 40024)) <= set(bindings.declared_ports)
    if managed:
        assert not {40030, 40050, 40051} & set(bindings.declared_ports)
        assert (40000 in bindings.declared_ports) is leader
    else:
        assert {40000, 40030} <= set(bindings.declared_ports)


@pytest.mark.parametrize(
    'parameters,valid',
    [
        (['--nccl-init-addr=49999'], False),
        (['--nnodes=2'], False),
        (['--node-rank=1'], False),
        (['--dist-init-addr=49999', '--nccl-init-addr=127.0.0.1:40010'], True),
        (['--nccl-init-addr=127.0.0.1:40010', '--dist-init-addr=49999'], False),
        (['--nnodes=1', '--node-rank=0', '--nccl-port=40017'], True),
    ],
)
def test_resolved_binding_validates_defaults_and_last_alias(parameters, valid):
    model, mi = _model(pd=True), _instance(role='prefill')
    binding = sglang_port_layout_for_instance(model, mi).bind(mi, '10.0.0.1')
    if valid:
        binding.validate_parameters(parameters)
    else:
        with pytest.raises(ValueError, match='reserved allocation'):
            binding.validate_parameters(parameters)


def test_legacy_single_node_native_selection_is_explicit():
    model, mi = _model(), _instance()
    mi.named_ports = None
    layout = sglang_port_layout_for_instance(model, mi)
    binding = layout.bind(mi, '10.0.0.1')
    assert binding.arguments == ()
    assert not binding.enforce_parameters
    assert 'native_dpa_tcp' in {port.name for port in binding.runtime_ports}


def test_missing_published_pd_band_is_rejected():
    model, mi = _model(pd=True), _instance(role='prefill', cross=True)
    mi.named_ports = {'bootstrap': PortBand(base=40001)}
    with pytest.raises(ValueError, match='restart the model deployment'):
        sglang_port_layout_for_instance(model, mi).bind(mi, '10.0.0.1')


@pytest.mark.parametrize('leader', [False, True])
def test_multinode_bind_handles_missing_subordinate_metadata(leader):
    layout = sglang_port_layout(_model(pd=True), 'prefill', [1, 1])
    mi = _instance(role='prefill')
    mi.distributed_servers = None
    if leader:
        binding = layout.bind(mi, mi.worker_ip)
        assert find_parameter(binding.arguments, ['node-rank']) == '0'
    else:
        with pytest.raises(ValueError, match='is not a SGLang subordinate'):
            layout.bind(mi, '10.0.0.2', is_leader=False)


def test_legacy_non_dpa_multinode_uses_the_connecting_tail():
    model, mi = _model(attention=False, pd=True), _instance(role='prefill', cross=True)
    mi.named_ports = {'bootstrap': PortBand(base=40001)}
    mi.ports = [40000, 40001, 40010]
    binding = sglang_port_layout_for_instance(model, mi).bind(mi, '10.0.0.1')
    assert (
        find_parameter(list(binding.arguments), ['dist-init-addr']) == '10.0.0.1:40010'
    )


def test_external_rendezvous_override_is_recorded_as_external_ownership():
    model = _model(pd=True)
    model.env = {'SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE': 'env://'}
    layout = sglang_port_layout(model, 'prefill', [1, 1])
    assert layout.rendezvous_override == 'env://'
    assert any(
        port.name == 'rendezvous' and port.owner == 'external'
        for port in layout.runtime_ports
    )


@pytest.mark.parametrize(
    'worker_ip,host,valid',
    [
        ('10.0.0.1', '0.0.0.0', True),
        ('10.0.0.1', '::', False),
        ('fd00::1', '::', True),
        ('fd00::1', '0.0.0.0', False),
        ('10.0.0.1', '10.0.0.2', False),
    ],
)
def test_api_wildcard_host_matches_worker_address_family(worker_ip, host, valid):
    model, mi = _model(pd=True), _instance(role='prefill')
    mi.worker_ip = worker_ip
    bindings = sglang_port_layout_for_instance(model, mi).bind(mi, worker_ip)
    parameters = [f'--host={host}', f'--port={mi.port}']
    if valid:
        bindings.validate_parameters(parameters)
        with pytest.raises(ValueError, match='--port conflicts'):
            bindings.validate_parameters([f'--host={host}', '--port=49999'])
    else:
        with pytest.raises(ValueError, match='--host conflicts'):
            bindings.validate_parameters(parameters)


@pytest.mark.parametrize('rust,count,required', [(False, 14, 14), (True, 24, 24)])
def test_rendered_dpa_checks_native_port_capacity_without_mutating_allocation(
    rust, count, required
):
    model, mi = _model(attention=False, pd=True, rust=rust), _instance(
        role='prefill', cross=True
    )
    layout = sglang_port_layout_for_instance(model, mi)
    argv = ['--dp=16', '--enable-dp-attention']
    band = mi.named_ports['_sglang_connecting']
    band.count = count
    before = mi.model_dump()
    layout.validate_runtime_port_demand(mi, argv)
    assert mi.model_dump() == before
    band.count -= 1
    before = mi.model_dump()
    with pytest.raises(ValueError, match=f'{required} ports.*backend_parameters'):
        layout.validate_runtime_port_demand(mi, argv)
    assert mi.model_dump() == before


@pytest.mark.parametrize('role', [None, 'prefill'])
def test_rendered_dpa_preserves_native_single_node_port_selection(role):
    model, mi = _model(attention=False, pd=role is not None), _instance(role=role)
    mi.named_ports = None
    sglang_port_layout_for_instance(model, mi).validate_runtime_port_demand(
        mi, ['--dp=2', '--enable-dp-attention']
    )
