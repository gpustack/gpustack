"""Keep pinned Tempo and Alloy image versions aligned across chart, Dockerfile, and CLI."""

import importlib
import pathlib
import re

from gpustack.cmd import images

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_observability_image_pins_stay_in_sync():
    dockerfile = (REPO_ROOT / "pack" / "Dockerfile").read_text()
    alloy_dockerfile = re.search(r"^ARG ALLOY_VERSION=(\S+)$", dockerfile, re.MULTILINE)
    tempo_dockerfile = re.search(r"^ARG TEMPO_VERSION=(\S+)$", dockerfile, re.MULTILINE)
    assert alloy_dockerfile is not None
    assert tempo_dockerfile is not None
    assert alloy_dockerfile.group(1) == images.alloy_version
    assert tempo_dockerfile.group(1) == images.tempo_version

    alloy_template = (
        REPO_ROOT / "charts" / "gpustack-chart" / "templates" / "alloy.yaml"
    ).read_text()
    tempo_template = (
        REPO_ROOT / "charts" / "gpustack-chart" / "templates" / "tempo.yaml"
    ).read_text()
    assert f"grafana/alloy:{images.alloy_version}" in alloy_template
    assert f"grafana/tempo:{images.tempo_version}" in tempo_template


def test_exported_images_include_observability_workloads():
    from gpustack_runtime.cmds import list_images

    importlib.import_module("gpustack.cmd.images")
    exported = {image.name for image in list_images()}

    assert {
        f"grafana/alloy:{images.alloy_version}",
        f"grafana/tempo:{images.tempo_version}",
    } <= exported
