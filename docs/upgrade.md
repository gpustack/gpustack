# Upgrade

You can upgrade GPUStack using the installation script or by manually installing the desired version of the GPUStack Python package.

!!! note

    When upgrading, upgrade the GPUStack server first, then upgrade the workers.

## Upgrade GPUStack Using the Installation Script

To upgrade GPUStack from an older version, re-run the installation script using the same configuration options you originally used.

Running the installation script will:

1. Install the latest version of the GPUStack Python package.
2. Update the system service (systemd, launchd, or Windows) init script to reflect the arguments passed to the installation script.
3. Restart the GPUStack service.

For example, to upgrade GPUStack to the latest version on a Linux system:

```bash
curl -sfL https://get.gpustack.ai | <EXISTING_INSTALL_ENV> sh -s - <EXISTING_GPUSTACK_ARGS>
```

To upgrade to a specific version, specify the `INSTALL_PACKAGE_SPEC` environment variable similar to the `pip install` command:

```bash
curl -sfL https://get.gpustack.ai | INSTALL_PACKAGE_SPEC=gpustack==x.y.z <EXISTING_INSTALL_ENV> sh -s - <EXISTING_GPUSTACK_ARGS>
```

## Manual Upgrade

If you install GPUStack manually, upgrade using the common `pip` workflow.

For example, to upgrade GPUStack to the latest version:

```bash
pip install --upgrade gpustack
```

Then restart the GPUStack service according to your setup.
