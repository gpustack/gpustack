# Upgrade

You can upgrade GPUStack using the installation script or by manually installing the desired version of the GPUStack Python package.

!!! note

    1. When upgrading, upgrade the GPUStack server first, then upgrade the workers.
    2. Please **DO NOT** upgrade from/to the main(dev) version or a release candidate(rc) version, as they may contain breaking changes. Use a fresh installation if you want to try the main or rc versions.

!!! note

    Before proceeding with an upgrade, it’s strongly recommended to back up your database. For default installations, stop the GPUStack server and create a backup of the file located at `/var/lib/gpustack/database.db`.

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

## Upgrading GPUStack Installed via Script (Legacy Script Installation)

!!! note
      The installation script method is deprecated as of version 0.7. We recommend using Docker on Linux, and the [desktop installer](https://gpustack.ai/) on macOS or Windows. 


If you previously installed GPUStack using the legacy installation script, follow the steps below to upgrade:

### On Linux

#### Step 1: Locate Your Existing Database

Find the path to your existing database directory (used in the legacy installation). For example:

```bash
/path/to/your/legacy/gpustack/data
```

We'll refer to this as `${your-database-file-location}` in the next step.

#### Step 2: Reinstall GPUStack via Docker

Make sure your hardware platform is supported. Then run the following Docker command, replacing the volume mount path with your database location.

**Example: For NVIDIA GPUs**

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --gpus all \
    --network=host \
    --ipc=host \
    -v ${your-database-file-location}:/var/lib/gpustack \
    gpustack/gpustack
```

This will launch GPUStack using Docker, preserving your existing data.

For other hardware platforms, please refer to the commands in the installation documentation.

- [Apple Metal](installation/apple-metal-installation.md)
- [AMD ROCm](installation/amd-rocm/online-installation.md)
- [Ascend CANN](installation/ascend-cann/online-installation.md)
- [Hygon DTK](installation/hygon-dtk/online-installation.md)
- [Moore Threads MUSA](installation/moorethreads-musa/online-installation.md)
- [Iluvatar Corex](installation/iluvatar-corex/online-installation.md)

### On macOS or Windows

- Run the Desktop Installer.

- After installation, the status will be shown as To Upgrade.

- Click `Start` to automatically migrate data and configuration for the upgrade.
