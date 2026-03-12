import argparse

from gpustack import __version__, __benchmark_runner_version__

from gpustack_runtime.cmds import (
    CopyImagesSubCommand,
    ListImagesSubCommand,
    SaveImagesSubCommand,
    LoadImagesSubCommand,
    append_images,
)

# Append images used by GPUStack here.
append_images(
    f"gpustack/gpustack:{'dev' if __version__.removeprefix('v') == '0.0.0' else __version__}",
    f"gpustack/benchmark-runner:{__benchmark_runner_version__}",
)


def setup_images_cmd(subparsers: argparse._SubParsersAction):
    ListImagesSubCommand.register(subparsers)
    SaveImagesSubCommand.register(subparsers)
    CopyImagesSubCommand.register(subparsers)
    LoadImagesSubCommand.register(subparsers)
