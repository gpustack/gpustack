"""Pure SGLang port contracts shared by admission, allocation and launch.

Fixed native listeners are reserved by GPUStack. Dynamic DP scheduler sockets
and PD KV sockets remain engine-owned and are not counted as reserved bands.
The selected version is retained for diagnostics; the known Python/Rust native
contracts are selected by runtime options, not guessed from custom image tags.
"""

from dataclasses import dataclass
from ipaddress import ip_address
from typing import List, Optional, Tuple

from gpustack.schemas.models import CategoryEnum, Model, ModelInstance, PortBand
from gpustack.utils.command import (
    find_bool_parameter,
    find_last_int_parameter,
    find_last_parameter,
)
from gpustack.utils.model_instance_workers import (
    get_model_instance_workloads,
    sglang_parallelism,
    sglang_smg_mode,
)

SGLANG_CONNECTING_BAND = '_sglang_connecting'
SGLANG_ENGINE_API_BAND = '_sglang_engine_api'
SGLANG_SMG_API_BAND = '_sglang_smg_api'
SGLANG_SMG_METRICS_BAND = '_sglang_smg_metrics'


@dataclass(frozen=True)
class SGLangRuntimePort:
    """A runtime-selected listener, outside the platform's fixed reservation."""

    name: str
    owner: str
    scope: str


@dataclass(frozen=True)
class SGLangPortBindings:
    """Allocated endpoints consumed by commands and container declarations."""

    arguments: Tuple[str, ...]
    declared_ports: Tuple[int, ...]
    expected_parameters: Tuple[Tuple[str, str], ...]
    runtime_ports: Tuple[SGLangRuntimePort, ...]
    enforce_parameters: bool

    def validate_parameters(self, parameters: List[str]) -> None:
        """Reject options redirecting platform-owned listeners or topology."""
        for key, expected in self.expected_parameters:
            aliases = [key, 'nccl-init-addr'] if key == 'dist-init-addr' else [key]
            supplied = find_last_parameter(parameters, aliases, tokenized=True)
            if key == 'host' and supplied in ('0.0.0.0', '::'):
                try:
                    if ip_address(supplied).version == ip_address(expected).version:
                        continue
                except ValueError:
                    pass
            if supplied is not None and supplied != expected:
                raise ValueError(
                    f'Managed SGLang --{key} conflicts with its reserved allocation'
                )


@dataclass(frozen=True)
class SGLangPortLayout:
    """Native listener rules before any concrete port has been allocated."""

    role: Optional[str]
    nnodes: int
    ranks: int
    attention: bool
    rust: bool
    managed: bool
    components: bool
    engine: bool
    engine_version: Optional[str]
    rendezvous_override: Optional[str]

    @property
    def api_band(self) -> str:
        return SGLANG_ENGINE_API_BAND if self.components else '_api'

    @property
    def nccl_offset(self) -> Optional[int]:
        if not self.attention:
            return None
        # Rust assigns scheduler inputs at +7..6+ranks. NCCL follows them.
        return 7 + self.ranks if self.rust else 7

    @property
    def internal_count(self) -> int:
        offsets = self.internal_offsets
        if not offsets:
            return 0
        # Keep the DPA protection window on one node too, independent of
        # startup order. Wider native layouts grow from their highest offset.
        return max(
            14 if self.attention else 1, max(offset for _, offset in offsets) + 1
        )

    @property
    def internal_offsets(self) -> Tuple[Tuple[str, int], ...]:
        if not self.engine or not (
            self.attention or self.components or self.nnodes > 1
        ):
            return ()
        ports = [('rendezvous', 0)]
        if self.attention:
            ports.extend(
                [
                    ('tokenizer', 1),
                    ('detokenizer', 2),
                    ('rpc', 3),
                    ('metrics', 4),
                    ('controller_input', 5),
                    ('load_collector', 6),
                    ('nccl', self.nccl_offset),
                ]
            )
            if self.rust:
                ports.extend(
                    (f'scheduler_input_{rank}', 7 + rank) for rank in range(self.ranks)
                )
            else:
                ports.append(('controller_handshake', 13))
        return tuple(sorted(ports, key=lambda item: item[1]))

    @property
    def runtime_ports(self) -> Tuple[SGLangRuntimePort, ...]:
        if not self.engine:
            return ()
        ports = []
        if not self.attention:
            ports.append(SGLangRuntimePort('native_nccl', 'sglang', 'tp_group'))
        elif not self.rust:
            ports.append(
                SGLangRuntimePort('dp_scheduler_input', 'sglang', 'routing_rank')
            )
        if self.role in ('prefill', 'decode'):
            ports.append(SGLangRuntimePort('pd_kv_rank', 'sglang', 'transfer_rank'))
        if self.rendezvous_override:
            ports.append(SGLangRuntimePort('rendezvous', 'external', 'instance'))
        return tuple(ports)

    def port_requests(self) -> List[Tuple[str, int, Optional[int]]]:
        """Return native Engine/SMG bands without PD connector declarations."""
        internal = (
            [(SGLANG_CONNECTING_BAND, self.internal_count, None)]
            if self.internal_count
            else []
        )
        if self.components:
            return (
                [(self.api_band, 1, None)]
                + internal
                + [(SGLANG_SMG_API_BAND, 1, None), (SGLANG_SMG_METRICS_BAND, 1, None)]
            )
        return internal + [(self.api_band, 1, None)]

    def validate_runtime_port_demand(
        self, instance: ModelInstance, parameters: List[str]
    ) -> None:
        """Check rendered DPA port demand without changing parallelism or ports."""
        band = (instance.named_ports or {}).get(SGLANG_CONNECTING_BAND)
        if not self.engine or (
            self.nnodes == 1
            and not self.components
            and (band is None or band.count <= 1)
        ):
            return
        dp = (
            find_last_int_parameter(
                parameters, ['dp', 'dp-size', 'data-parallel-size'], tokenized=True
            )
            or 1
        )
        attention_dp = (
            find_last_int_parameter(
                parameters,
                ['attn-dp-size', 'attention-data-parallel-size'],
                tokenized=True,
            )
            or 1
        )
        if not (
            attention_dp > 1
            or (
                dp > 1
                and find_bool_parameter(
                    parameters, ['enable-dp-attention'], tokenized=True
                )
            )
        ):
            return
        ranks = dp * attention_dp
        required = max(14, ranks + 8) if self.rust else 14
        if band is None or band.count < required:
            raise ValueError(
                f'SGLang DPA requires a complete reserved internal port band '
                f'of {required} ports; declare DPA options in backend_parameters '
                'and restart the model deployment after all Workers are upgraded'
            )

    def _uses_native_ports(self, band: Optional[PortBand]) -> bool:
        return (
            not self.components
            and self.nnodes == 1
            and self.role is None
            and (band is None or band.count <= 1)
        )

    def bind(
        self,
        instance: ModelInstance,
        worker_ip: str,
        *,
        is_leader: bool = True,
        api_port: Optional[int] = None,
    ) -> SGLangPortBindings:
        """Resolve published reservations without allocating or expanding them."""
        bands = instance.named_ports or {}
        band = bands.get(SGLANG_CONNECTING_BAND)
        native = self._uses_native_ports(band)
        arguments = self._internal_arguments(
            instance, worker_ip, is_leader, band, native
        )
        port = api_port if api_port is not None else instance.port
        expected = {'host': worker_ip}
        if port is not None:
            expected['port'] = str(port)
        if arguments:
            expected.update({'nnodes': '1', 'node-rank': '0'})
            expected.update(
                (arguments[index].removeprefix('--'), arguments[index + 1])
                for index in range(0, len(arguments), 2)
            )
        runtime = self.runtime_ports
        if native and self.attention:
            runtime += (SGLangRuntimePort('native_dpa_tcp', 'sglang', 'instance'),)
        return SGLangPortBindings(
            arguments=arguments,
            declared_ports=self._declared_ports(instance, is_leader, band),
            expected_parameters=tuple(expected.items()),
            runtime_ports=runtime,
            enforce_parameters=self.managed or band is not None,
        )

    def _internal_arguments(
        self, instance: ModelInstance, worker_ip: str, leader: bool, band, native: bool
    ) -> Tuple[str, ...]:
        if not self.engine or native or not (self.attention or self.nnodes > 1):
            return ()
        if self.attention and (band is None or band.count < self.internal_count):
            raise ValueError(
                'SGLang DPA requires a complete reserved internal port band; '
                'restart the model deployment after all Workers are upgraded '
                'to allocate it'
            )
        published = instance.ports or []
        if band is None and not published:
            raise ValueError(
                'SGLang distributed initialization port has not been reserved'
            )
        # Non-DPA legacy instances publish the connecting port at the tail.
        connecting = band.base if band is not None else published[-1]
        host = instance.worker_ip if self.nnodes > 1 else '127.0.0.1'
        arguments = ['--dist-init-addr', f'{host}:{connecting}']
        if self.nccl_offset is not None:
            arguments.extend(['--nccl-port', str(connecting + self.nccl_offset)])
        if self.nnodes > 1:
            subordinates = (
                instance.distributed_servers.subordinate_workers
                if instance.distributed_servers
                else []
            ) or []
            rank = (
                0
                if leader
                else next(
                    (
                        index + 1
                        for index, item in enumerate(subordinates)
                        if item.worker_ip == worker_ip
                    ),
                    None,
                )
            )
            if rank is None:
                raise ValueError(f'Worker {worker_ip} is not a SGLang subordinate')
            arguments.extend(['--nnodes', str(self.nnodes), '--node-rank', str(rank)])
        return tuple(arguments)

    def _declared_ports(
        self, instance: ModelInstance, leader: bool, band
    ) -> Tuple[int, ...]:
        if self.components:
            ports = {instance.port} if leader and instance.port is not None else set()
        else:
            ports = set(instance.ports or [])
            if instance.port is not None:
                ports.add(instance.port)
            for item in (instance.named_ports or {}).values():
                ports.update(range(item.base, item.base + item.count))
        if band is not None:
            ports.update(range(band.base, band.base + band.count))
        return tuple(sorted(ports))


def sglang_port_layout(
    model: Model, role: Optional[str], gpu_per_node: List[int], *, managed: bool = True
) -> SGLangPortLayout:
    """Describe native ports; PD connector ports are supplied by the catalog."""
    engine = role in (None, 'prefill', 'decode') and CategoryEnum.IMAGE not in (
        model.categories or []
    )
    ranks, _, attention = (
        sglang_parallelism(model.backend_parameters) if engine else (1, 1, False)
    )
    env = getattr(model, 'env', None) or {}
    return SGLangPortLayout(
        role=role,
        nnodes=max(1, len(gpu_per_node)),
        ranks=ranks,
        attention=attention,
        rust=str(env.get('SGLANG_RUST_SERVER', '')).lower()
        in ('1', 'true', 'yes', 'y'),
        managed=managed,
        components=engine and managed and role is None and sglang_smg_mode(model),
        engine=engine,
        engine_version=model.backend_version,
        rendezvous_override=env.get('SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE'),
    )


def sglang_port_layout_for_instance(
    model: Model, instance: ModelInstance
) -> SGLangPortLayout:
    """Build the same contract from a persisted instance allocation."""
    servers = instance.distributed_servers
    allocation = (
        [len(instance.gpu_indexes or [])]
        + [len(item.gpu_indexes or []) for item in (servers.subordinate_workers or [])]
        if servers
        else [len(instance.gpu_indexes or [])]
    )
    return sglang_port_layout(
        model,
        getattr(instance, 'role', None),
        allocation,
        managed=get_model_instance_workloads(instance) is not None,
    )
