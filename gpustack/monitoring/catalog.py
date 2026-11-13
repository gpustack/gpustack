"""Page metric definitions over normalized engine and Worker exporter families."""

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class MetricDefinition:
    key: str
    family: str
    kind: str
    unit: str
    measurement_scope: str
    denominator: Optional[str] = None
    scale: float = 1.0
    per_source: bool = False


def model_metric_definitions(backend: str) -> List[MetricDefinition]:
    latencies = {
        'ttft': 'time_to_first_token_seconds',
        'tpot': 'time_per_output_token_seconds',
        'itl': 'inter_token_latency_seconds',
        'e2e': 'e2e_request_latency_seconds',
        'queue': 'request_queue_time_seconds',
        'prefill': 'request_prefill_time_seconds',
        'decode': 'request_decode_time_seconds',
    }
    definitions = [
        MetricDefinition(k, f'gpustack:{v}', 'histogram', 'seconds', 'engine_local')
        for k, v in latencies.items()
    ]
    for key, family in (
        ('input_tokens', 'prompt_tokens_total'),
        ('output_tokens', 'generation_tokens_total'),
    ):
        definitions.append(
            MetricDefinition(
                key,
                f'gpustack:{family}',
                'counter',
                'tokens_per_second',
                'engine_processed_tokens',
            )
        )
    for key, family in (
        ('running_requests', 'num_requests_running'),
        ('waiting_requests', 'num_requests_waiting'),
    ):
        definitions.append(
            MetricDefinition(
                key, f'gpustack:{family}', 'gauge', 'requests', 'engine_local_requests'
            )
        )
    definitions.append(
        MetricDefinition(
            'kv_usage',
            'gpustack:kv_cache_usage_ratio',
            'gauge',
            'percent',
            'engine_kv_pool',
            scale=100,
            per_source=True,
        )
    )
    if backend in ('sglang', 'mindie'):
        definitions.append(
            MetricDefinition(
                'prefix_cache_hit_rate',
                'gpustack:prefix_cache_hit_rate',
                'gauge',
                'percent',
                'reported_source_cache_ratio',
                scale=100,
                per_source=True,
            )
        )
    else:
        definitions.append(
            MetricDefinition(
                'prefix_cache_hit_rate',
                'gpustack:prefix_cache_hits_total',
                'counter',
                'percent',
                'engine_cache_lookup_ratio',
                denominator='gpustack:prefix_cache_queries_total',
                scale=100,
            )
        )
    definitions.extend(
        [
            MetricDefinition(
                'external_cache_hit_rate',
                'gpustack:external_prefix_cache_hits_total',
                'counter',
                'percent',
                'engine_external_cache_lookup_ratio',
                denominator='gpustack:external_prefix_cache_queries_total',
                scale=100,
            ),
            MetricDefinition(
                'pd_kv_transfer',
                'gpustack:pd_kv_transfer_seconds',
                'histogram',
                'seconds',
                (
                    'bootstrap_handshake'
                    if backend == 'sglang'
                    else 'kv_transfer_duration'
                ),
            ),
            MetricDefinition(
                'router_health',
                'gpustack:pd_router_worker_health',
                'gauge',
                'state',
                'router_upstream_health',
            ),
        ]
    )
    return definitions


HARDWARE_DEFINITIONS = {
    'gpu_utilization': ('worker_gpu_utilization_percent', 'percent', 0, 100),
    'gpu_memory_used': ('worker_gpu_memory_used_bytes', 'bytes', 0, None),
    'gpu_memory_total': ('worker_gpu_memory_total_bytes', 'bytes', 0, None),
    'gpu_power': ('worker_gpu_power_watts', 'watts', 0, None),
    'gpu_temperature': ('worker_gpu_temperature_celsius', 'celsius', None, None),
    'host_cpu': ('worker_host_cpu_utilization_percent', 'percent', 0, 100),
    'host_memory_used': ('worker_host_memory_used_bytes', 'bytes', 0, None),
    'host_memory_total': ('worker_host_memory_total_bytes', 'bytes', 0, None),
}
HARDWARE_KEYS = tuple(HARDWARE_DEFINITIONS)
HARDWARE_DEFINITIONS['filesystem_usage'] = (
    'worker_filesystem_utilization_percent',
    'percent',
    0,
    100,
)
