import argparse

from gpustack import __benchmark_runner_version__, __operator_version__
from gpustack_higress_plugins import __version__ as __higress_plugins_version__

from gpustack_runtime.cmds import (
    CopyImagesSubCommand,
    ListImagesSubCommand,
    SaveImagesSubCommand,
    LoadImagesSubCommand,
    append_images,
)

from gpustack.config.config import get_image_name
from gpustack.extension import iter_plugin_classes, Plugin
from gpustack.utils.envs import get_gpustack_env

# The higress version should be sync with HIGRESS_VERSION in pack/Dockerfile.
higress_version = "2.1.9"

ssh_server_version = "v1.3.0"
kueue_version = "v0.18.4"
node_feature_discovery_version = "v0.19.0-gpustack1"
csi_nfs_driver_version = "v4.13.4"
csi_s3_driver_version = "v0.43.7"
csi_provisioner_version = "v6.3.0"
csi_resizer_version = "v2.2.0"
csi_snapshotter_version = "v8.6.0"
csi_livenessprobe_version = "v2.19.0"
csi_node_driver_registrar_version = "v2.17.0"

# Append images used by GPUStack here. The GPUStack image itself is appended
# from _append_self_image() instead, since it cannot be resolved at import time.
append_images(
    f"gpustack/benchmark-runner:{__benchmark_runner_version__}",
    f"gpustack/higress-plugins:{__higress_plugins_version__}",
    f"gpustack/mirrored-higress-higress:{higress_version}",
    f"gpustack/mirrored-higress-pilot:{higress_version}",
    f"gpustack/mirrored-higress-gateway:{higress_version}",
    f"gpustack/gpustack-operator:{__operator_version__}",
    f"gpustack/ssh-server:{ssh_server_version}",
    f"gpustack/mirrored-kueue:{kueue_version}",
    f"gpustack/mirrored-node-feature-discovery:{node_feature_discovery_version}",
    f"gpustack/mirrored-csi-nfs-driver:{csi_nfs_driver_version}",
    f"gpustack/mirrored-csi-s3-driver:{csi_s3_driver_version}",
    f"gpustack/mirrored-csi-provisioner:{csi_provisioner_version}",
    f"gpustack/mirrored-csi-resizer:{csi_resizer_version}",
    f"gpustack/mirrored-csi-snapshotter:{csi_snapshotter_version}",
    f"gpustack/mirrored-csi-livenessprobe:{csi_livenessprobe_version}",
    f"gpustack/mirrored-csi-node-driver-registrar:{csi_node_driver_registrar_version}",
)


def _append_self_image():
    # Resolved through get_image_name(), the helper that also names the image
    # workers pull: repository from GPUSTACK_IMAGE_REPO -- where the
    # --image-repo default comes from, and the only seam this command has,
    # running as its own process with no server config to read -- and version
    # from resolve_version_info(). Deferred until the images subcommand is
    # wired because plugins are loaded by then: a repackaged distribution
    # reports its own version and ships its own repository, and the baked-in
    # __version__ names an image that was never published for it.
    #
    # An image name override stays out: it is a full reference that may carry
    # its own registry, while save-images / copy-images prefix every listed
    # name with --source, which would build docker.io/quay.io/gpustack/...
    append_images(
        get_image_name(
            image_name_override=None,
            image_repo=get_gpustack_env("IMAGE_REPO") or "gpustack/gpustack",
        )
    )


def _append_plugin_images():
    # Deferred until the images subcommand is wired so a misbehaving plugin
    # can't crash unrelated CLI entry points (start, --help, version).
    for name, plugin_cls in iter_plugin_classes():
        if not (isinstance(plugin_cls, type) and issubclass(plugin_cls, Plugin)):
            continue
        try:
            append_images(*plugin_cls.extra_image_list())
        except Exception as e:
            raise RuntimeError(
                f"Failed to append images from plugin '{name}': {e}"
            ) from e


def setup_images_cmd(subparsers: argparse._SubParsersAction):
    _append_self_image()
    _append_plugin_images()
    ListImagesSubCommand.register(subparsers)
    SaveImagesSubCommand.register(subparsers)
    CopyImagesSubCommand.register(subparsers)
    LoadImagesSubCommand.register(subparsers)
