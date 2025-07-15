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

##  Upgrade from Legacy Script

If you installed GPUStack using the installation script, follow the [migration guide](migration.md) to upgrade and preserve your existing data.
