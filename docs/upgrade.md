# Upgrade

You can upgrade GPUStack using the installation script or by manually installing the desired version of the GPUStack Python package.

!!! note

    1. When upgrading, upgrade the GPUStack server first, then upgrade the workers.
    2. Please **DO NOT** upgrade from/to the main(dev) version or a release candidate(rc) version, as they may contain breaking changes. Use a fresh installation if you want to try the main or rc versions.

## Upgrade GPUStack Using the Installation Script

To upgrade GPUStack from an older version, re-run the installation script using the same configuration options you originally used.

Running the installation script will:

1. Install the latest version of the GPUStack Python package.
2. Update the system service (systemd, launchd, or Windows) init script to reflect the arguments passed to the installation script.
3. Restart the GPUStack service.

### Linux or macOS

For example, to upgrade GPUStack to the latest version on a Linux system and macOS:

```bash
curl -sfL https://get.gpustack.ai | <EXISTING_INSTALL_ENV> sh -s - <EXISTING_GPUSTACK_ARGS>
```

!!! Note

    `<EXISTING_INSTALL_ENV>` are the environment variables you set during the initial installation, and `<EXISTING_GPUSTACK_ARGS>` are the startup parameters you configured back then.

    **Simply execute the same installation command again, and the system will automatically perform an upgrade.**

To upgrade to a specific version, specify the `INSTALL_PACKAGE_SPEC` environment variable similar to the `pip install` command:

```bash
curl -sfL https://get.gpustack.ai | INSTALL_PACKAGE_SPEC=gpustack==x.y.z <EXISTING_INSTALL_ENV> sh -s - <EXISTING_GPUSTACK_ARGS>
```

### Windows

To upgrade GPUStack to the latest version on a Windows system:

```powershell
$env:<EXISTING_INSTALL_ENV> = <EXISTING_INSTALL_ENV_VALUE>
Invoke-Expression (Invoke-WebRequest -Uri "https://get.gpustack.ai" -UseBasicParsing).Content
```

!!! Note

    `<EXISTING_INSTALL_ENV>` are the environment variables you set during the initial installation, and `<EXISTING_GPUSTACK_ARGS>` are the startup parameters you configured back then.

    **Simply execute the same installation command again, and the system will automatically perform an upgrade.**

To upgrade to a specific version:

```powershell
$env:INSTALL_PACKAGE_SPEC = gpustack==x.y.z
$env:<EXISTING_INSTALL_ENV> = <EXISTING_INSTALL_ENV_VALUE>
Invoke-Expression "& { $((Invoke-WebRequest -Uri 'https://get.gpustack.ai' -UseBasicParsing).Content) } <EXISTING_GPUSTACK_ARGS>"
```

## Docker Upgrade

If you installed GPUStack using Docker, upgrade to the a new version by pulling the Docker image with the desired version tag.

For example:

```bash
docker pull gpustack/gpustack:vX.Y.Z
```

Then restart the GPUStack service with the new image.

## pip Upgrade

If you install GPUStack manually using pip, upgrade using the common `pip` workflow.

For example, to upgrade GPUStack to the latest version:

```bash
pip install --upgrade gpustack
```

Then restart the GPUStack service according to your setup.
