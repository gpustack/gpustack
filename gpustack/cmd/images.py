import argparse

from gpustack import __version__

from gpustack_runtime.cmds import (
    CopyImagesSubCommand,
    ListImagesSubCommand,
    SaveImagesSubCommand,
    append_images,
)

# Append images used by GPUStack here.
append_images(
    f"gpustack/gpustack:{'main' if __version__.removeprefix('v') == '0.0.0' else __version__}",
)


def setup_images_cmd(subparsers: argparse._SubParsersAction):
    ListImagesSubCommand.register(subparsers)
    SaveImagesSubCommand.register(subparsers)
    CopyImagesSubCommand.register(subparsers)
