"""Validated, packaged declarations for native monitoring queries."""

from functools import lru_cache
import math
from pathlib import Path
import re
from string import Template
from typing import Dict, Literal, Optional, Tuple

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    field_validator,
    model_validator,
)
import yaml

from gpustack.utils.compat_importlib import pkg_resources


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)


class WindowPolicy(_StrictModel):
    target_points: int = Field(ge=1, le=1000, strict=True)
    min_step_seconds: int = Field(ge=1, le=3600, strict=True)
    rate_window_extra_seconds: int = Field(default=0, ge=0, le=86400, strict=True)
    freshness_seconds: int = Field(default=120, ge=1, le=86400, strict=True)

    def step(self, seconds: int) -> int:
        if type(seconds) is not int or seconds <= 0:
            raise ValueError('Invalid query span')
        return max(math.ceil(seconds / self.target_points), self.min_step_seconds)

    def rate_window(self, step: int) -> int:
        return step + self.rate_window_extra_seconds


class TimeWindows(_StrictModel):
    llm: WindowPolicy
    hardware: WindowPolicy


class QueryProfile(_StrictModel):
    kind: Literal['gauge', 'counter', 'histogram']
    scope: Literal['engine', 'hardware'] = 'engine'
    expressions: Dict[str, str]


class RuntimeMetric(_StrictModel):
    family: str = Field(pattern=r'^gpustack:[A-Za-z_][A-Za-z0-9_]*$')
    kind: Literal['gauge', 'counter', 'histogram']
    unit: str
    measurement_scope: str
    query_profile: str
    section: Literal['llm', 'pd'] = 'llm'
    denominator: Optional[str] = Field(
        default=None, pattern=r'^gpustack:[A-Za-z_][A-Za-z0-9_]*$'
    )
    scale: FiniteFloat = Field(default=1.0, gt=0)
    per_source: bool = False
    backend_overrides: Dict[str, dict] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_ratio(self):
        if self.denominator is not None and self.kind != 'counter':
            raise ValueError('Only counter metrics may declare a denominator')
        return self

    def for_backend(self, backend: str) -> 'RuntimeMetric':
        override = self.backend_overrides.get(backend, {})
        if 'backend_overrides' in override:
            raise ValueError('Nested backend overrides are not supported')
        return RuntimeMetric.model_validate(
            {
                **self.model_dump(exclude={'backend_overrides'}),
                **override,
            }
        )


class HardwareMetric(_StrictModel):
    family: str = Field(pattern=r'^worker_[A-Za-z0-9_]+$')
    kind: Literal['gauge']
    unit: str
    query_profile: str = 'hardware_gauge'
    minimum: Optional[FiniteFloat] = None
    maximum: Optional[FiniteFloat] = None
    model_resource: bool = True

    @model_validator(mode='after')
    def validate_bounds(self):
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError('Invalid sensor bounds')
        return self


class ChartMapping(_StrictModel):
    description: str = Field(
        description='English developer metadata; localized chart titles belong to the UI'
    )
    metrics: Tuple[str, ...] = Field(min_length=1)
    apis: Tuple[
        Literal['model_runtime', 'model_resource', 'worker_resource', 'gpu_resource'],
        ...,
    ] = Field(min_length=1)
    derived_in_ui: Optional[str] = None


_COMMON_VARIABLES = {
    'sample_since': {'expression', 'selector', 'start'},
    'engine_success': {'selector'},
    'engine_inventory': {'selector'},
    'history_presence': {'expression', 'window_seconds'},
    'engine_guard': {
        'expression',
        'source_labels',
        'plumbing_labels',
        'success',
        'inventory',
    },
    'aggregate': {'operation', 'grouping', 'expression'},
    'divide': {'numerator', 'denominator'},
    'pair': {'left', 'right'},
    'compare_population': {'paired', 'population'},
    'match_group': {'expression', 'grouping', 'condition'},
}
_PROFILE_VARIABLES = {
    ('engine', 'gauge'): {'sample': {'family', 'selector'}},
    ('engine', 'counter'): {
        'rate': {'family', 'selector', 'window_seconds'},
        'increase': {'family', 'selector', 'window_seconds'},
        'window_rate': {'expression', 'window_seconds'},
    },
    ('engine', 'histogram'): {
        'rate': {'family', 'part', 'selector', 'window_seconds'},
        'increase': {'family', 'part', 'selector', 'window_seconds'},
        'avg': {'sum', 'count'},
        'p50': {'buckets'},
        'p95': {'buckets'},
        'p99': {'buckets'},
    },
    ('hardware', 'gauge'): {'sample': {'selector', 'freshness_seconds'}},
}


def _validate_expression(expression, variables):
    if (
        not isinstance(expression, str)
        or not expression.strip()
        or len(expression) > 20000
    ):
        raise ValueError('Invalid query expression')
    template = Template(expression)
    if not template.is_valid() or set(template.get_identifiers()) != variables:
        raise ValueError(
            f'Expression must use exactly these variables: {sorted(variables)}'
        )


def _render(expression: str, variables: dict) -> str:
    if set(Template(expression).get_identifiers()) != variables.keys():
        raise ValueError('Query bindings do not match the declared expression')
    seconds = variables.get('window_seconds')
    if isinstance(seconds, float) and seconds.is_integer():
        variables = {**variables, 'window_seconds': int(seconds)}
    if isinstance(seconds, float) and not seconds.is_integer():
        # PromQL duration literals use integral units. Runner windows retain
        # millisecond boundaries instead of extending into a neighboring phase.
        milliseconds = max(1, math.floor(seconds * 1000))
        expression = expression.replace('${window_seconds}s', f'{milliseconds}ms')
    return Template(expression).substitute(variables).strip()


class MonitoringQueries(_StrictModel):
    schema_version: Literal[1]
    time_windows: TimeWindows
    expressions: Dict[str, str]
    profiles: Dict[str, QueryProfile]
    metrics: Dict[str, RuntimeMetric]
    hardware_metrics: Dict[str, HardwareMetric]
    charts: Dict[str, ChartMapping]

    @field_validator('schema_version', mode='before')
    @classmethod
    def validate_schema_version(cls, value):
        if type(value) is not int:
            raise ValueError('Query schema_version must be an integer')
        return value

    @model_validator(mode='after')
    def validate_declarations(self):
        if self.expressions.keys() != _COMMON_VARIABLES.keys():
            raise ValueError('The common query expressions are incomplete or unknown')
        for name, variables in _COMMON_VARIABLES.items():
            _validate_expression(self.expressions[name], variables)
        self._validate_profiles()
        self._validate_metrics()
        self._validate_charts()
        return self

    def _validate_profiles(self):
        for name, profile in self.profiles.items():
            expected = _PROFILE_VARIABLES.get((profile.scope, profile.kind))
            if expected is None or profile.expressions.keys() != expected.keys():
                raise ValueError(f'Invalid query profile: {name}')
            for key, variables in expected.items():
                _validate_expression(profile.expressions[key], variables)
        for name, kind, scope in (
            ('gauge', 'gauge', 'engine'),
            ('counter', 'counter', 'engine'),
            ('histogram', 'histogram', 'engine'),
            ('hardware_gauge', 'gauge', 'hardware'),
        ):
            self._profile(name, kind, scope)

    def _profile(self, name, kind, scope):
        profile = self.profiles.get(name)
        if profile is None or (profile.kind, profile.scope) != (kind, scope):
            raise ValueError(f'Unknown or incompatible query profile: {name}')
        return profile

    def _validate_metrics(self):
        if self.metrics.keys() & self.hardware_metrics.keys():
            raise ValueError('Runtime and hardware metric keys must be distinct')
        for name, metric in self.metrics.items():
            if not re.fullmatch(r'[a-z][a-z0-9_]*', name):
                raise ValueError('Invalid metric key')
            for backend in ('', *metric.backend_overrides):
                resolved = metric.for_backend(backend)
                self._profile(resolved.query_profile, resolved.kind, 'engine')
        for name, metric in self.hardware_metrics.items():
            if not re.fullmatch(r'[a-z][a-z0-9_]*', name):
                raise ValueError('Invalid hardware metric key')
            self._profile(metric.query_profile, metric.kind, 'hardware')

    def _validate_charts(self):
        for chart in self.charts.values():
            for key in chart.metrics:
                if key not in self.metrics and key not in self.hardware_metrics:
                    raise ValueError(f'Chart references unknown metric: {key}')
                engine = key in self.metrics
                if (engine and set(chart.apis) != {'model_runtime'}) or (
                    not engine and 'model_runtime' in chart.apis
                ):
                    raise ValueError(f'Chart API does not match metric source: {key}')
                if (
                    not engine
                    and not self.hardware_metrics[key].model_resource
                    and (set(chart.apis) != {'worker_resource'})
                ):
                    raise ValueError(f'Metric is only exposed by the Worker API: {key}')

    def common(self, name: str, **variables) -> str:
        return _render(self.expressions[name], variables)

    def query(
        self, profile: str, kind: str, name: str, *, scope='engine', **variables
    ) -> str:
        declaration = self._profile(profile, kind, scope)
        return _render(declaration.expressions[name], variables)


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in keys:
                raise ValueError(f'Duplicate query configuration key: {key}')
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


def load_query_config(path: Optional[Path] = None) -> MonitoringQueries:
    source = path or pkg_resources.files('gpustack.assets').joinpath(
        'monitoring-queries.yaml'
    )
    return MonitoringQueries.model_validate(
        yaml.load(source.read_text(encoding='utf-8'), Loader=_UniqueKeyLoader)
    )


@lru_cache(maxsize=1)
def get_query_config() -> MonitoringQueries:
    return load_query_config()
