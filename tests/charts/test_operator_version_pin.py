"""The gpustack-operator dependency pin must track ``__operator_version__``.

The pin in ``Chart.yaml`` is the only thing that selects an operator version: the
operator chart keeps its ``version`` and ``appVersion`` equal and defaults its
image tag to ``v<appVersion>``, so nothing else in this chart names an operator
image. CI patches the pin from ``__operator_version__`` at release time, which
means a stale pin only shows up in a locally installed chart — exactly the case
that pairs an old operator with fresh templates. Asserting it here fails the
drift at review time instead.
"""

import pathlib

import pytest
import yaml

from gpustack import __operator_version__

CHART_YAML = (
    pathlib.Path(__file__).resolve().parents[2]
    / "charts"
    / "gpustack-chart"
    / "Chart.yaml"
)
DEPENDENCY_NAME = "gpustack-operator"


@pytest.fixture(scope="module")
def operator_dependency() -> dict:
    chart = yaml.safe_load(CHART_YAML.read_text())
    for dependency in chart.get("dependencies") or []:
        if dependency.get("name") == DEPENDENCY_NAME:
            return dependency
    pytest.fail(f"{DEPENDENCY_NAME} dependency missing from {CHART_YAML}")


def test_pinned_version_matches_operator_version(operator_dependency):
    # A Helm dependency version is SemVer, so the `v` prefix carried by
    # __operator_version__ is stripped rather than compared.
    assert operator_dependency["version"] == __operator_version__.lstrip("v"), (
        "Chart.yaml pins gpustack-operator "
        f"{operator_dependency['version']} but gpustack.__operator_version__ is "
        f"{__operator_version__}. Update the dependency version in "
        "charts/gpustack-chart/Chart.yaml (without the 'v' prefix)."
    )


def test_operator_is_gated_on_worker_enabled(operator_dependency):
    # The operator only exists to run worker workloads, so a server-only install
    # must not deploy it. Losing this condition would silently add an operator,
    # Kueue, NFD and the CSI drivers to every server-only release.
    assert operator_dependency.get("condition") == "worker.enabled"
