# Upgrade

You can upgrade GPUStack by pulling and running a newer Docker image.

The following upgrade instructions apply only to GPUStack v2.0 and later.

For installations prior to v0.7, please refer to the [migration guide](migration.md).

!!! note

    1. When upgrading, upgrade the GPUStack server first, then upgrade the workers.

    2. Please **DO NOT** upgrade from/to the main(dev) version or a release candidate(rc) version, as they may contain breaking changes. Use a fresh installation if you want to try the main or rc versions.

!!! warning

    **Backup First:** Before proceeding with an upgrade, it’s strongly recommended to back up your database.

    For default installations, stop the GPUStack server and create a backup of the PostgreSQL database directory located inside the container at:

    ```
    /var/lib/gpustack/postgresql/data
    ```

You can upgrade by pulling a new image (either a specific version tag or the latest tag), removing the old container, and starting a new one with the updated image.

For example:

```bash
docker pull gpustack/gpustack:latest  # or: docker pull gpustack/gpustack:vx.y.z

docker stop gpustack
docker rm gpustack

docker run -d --name gpustack \
  ... \
  gpustack/gpustack:latest
  ...
```
