import argparse

from gpustack import __version__, __benchmark_runner_version__
from gpustack_higress_plugins import __version__ as __higress_plugins_version__

from gpustack_runtime.cmds import (
    CopyImagesSubCommand,
    ListImagesSubCommand,
    SaveImagesSubCommand,
    LoadImagesSubCommand,
    append_images,
)

# The higress version should be sync with HIGRESS_VERSION in pack/Dockerfile.
higress_version = "2.1.9"

# Append images used by GPUStack here.
append_images(
    f"gpustack/gpustack:{'dev' if __version__.removeprefix('v') == '0.0.0' else __version__}",
    f"gpustack/benchmark-runner:{__benchmark_runner_version__}",
    f"gpustack/higress-plugins:{__higress_plugins_version__}",
    f"gpustack/mirrored-higress-higress:{higress_version}",
    f"gpustack/mirrored-higress-pilot:{higress_version}",
    f"gpustack/mirrored-higress-gateway:{higress_version}",
)


def setup_images_cmd(subparsers: argparse._SubParsersAction):
    ListImagesSubCommand.register(subparsers)
    SaveImagesSubCommand.register(subparsers)
    CopyImagesSubCommand.register(subparsers)
    LoadImagesSubCommand.register(subparsers)
