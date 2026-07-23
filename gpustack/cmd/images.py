import argparse

from gpustack import __version__, __benchmark_runner_version__, __operator_version__
from gpustack_higress_plugins import __version__ as __higress_plugins_version__

from gpustack_runtime.cmds import (
    CopyImagesSubCommand,
    ListImagesSubCommand,
    SaveImagesSubCommand,
    LoadImagesSubCommand,
    append_images,
)

from gpustack.extension import iter_plugin_classes, Plugin

# The higress version should be sync with HIGRESS_VERSION in pack/Dockerfile.
higress_version = "2.1.9"

ssh_server_version = "v1.3.0"
kueue_version = "v0.18.2"
node_feature_discovery_version = "v0.18.3"
csi_nfs_driver_version = "v4.13.0"
csi_s3_driver_version = "v0.43.7"
csi_provisioner_version = "v6.1.0"
csi_resizer_version = "v2.0.0"
csi_snapshotter_version = "v8.4.0"
csi_livenessprobe_version = "v2.17.0"
csi_node_driver_registrar_version = "v2.15.0"

# Append images used by GPUStack here.
append_images(
    f"gpustack/gpustack:{'dev' if __version__.removeprefix('v') == '0.0.0' else __version__}",
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
    _append_plugin_images()
    ListImagesSubCommand.register(subparsers)
    SaveImagesSubCommand.register(subparsers)
    CopyImagesSubCommand.register(subparsers)
    LoadImagesSubCommand.register(subparsers)
