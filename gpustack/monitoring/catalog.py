"""Page metric definitions over normalized engine and Worker exporter families."""

from dataclasses import dataclass
from typing import List, Optional

from . import config as query_config


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
    query_profile: str = 'gauge'
    section: str = 'llm'


def model_metric_definitions(backend: str) -> List[MetricDefinition]:
    return [
        MetricDefinition(
            key=key,
            **metric.for_backend(backend).model_dump(exclude={'backend_overrides'}),
        )
        for key, metric in query_config.get_query_config().metrics.items()
    ]


_HARDWARE = query_config.get_query_config().hardware_metrics
HARDWARE_DEFINITIONS = {
    key: (metric.family, metric.unit, metric.minimum, metric.maximum)
    for key, metric in _HARDWARE.items()
}
HARDWARE_KEYS = tuple(key for key, metric in _HARDWARE.items() if metric.model_resource)
HARDWARE_QUERY_PROFILES = {
    key: metric.query_profile for key, metric in _HARDWARE.items()
}
