# Upgrade via Docker

You can upgrade a Docker-based GPUStack installation by pulling a newer image and recreating the containers.

The following upgrade instructions apply only to GPUStack v2.0 and later.

For installations prior to v0.7, please refer to the [migration guide](../migration.md).

!!! note

    1. When upgrading, upgrade the GPUStack server first, then upgrade the workers.

    2. Please **DO NOT** upgrade from/to the main(dev) version or a release candidate(rc) version, as they may contain breaking changes. Use a fresh installation if you want to try the main or rc versions.

!!! warning

    **Backup First:** Before proceeding with an upgrade, it’s strongly recommended to back up your database.

    For default installations that use the embedded PostgreSQL database, stop the GPUStack server and create a backup of the PostgreSQL database directory located inside the container at:

    ```
    /var/lib/gpustack/postgresql/data
    ```

    If you use an external database, follow your database provider's backup procedure instead.

Upgrade the **server** first, then upgrade the **workers**. The server is upgraded by pulling a new image (either a specific version tag or the `latest` tag), removing the old container, and starting a new one with the updated image using the **same arguments and volumes** as before.

## Upgrade the Server

```bash
docker pull gpustack/gpustack:latest  # or: docker pull gpustack/gpustack:vx.y.z

docker stop gpustack
docker rm gpustack

docker run -d --name gpustack \
  ... \
  gpustack/gpustack:latest
  ...
```

## Upgrade the Workers

After the server is up and running with the new version, upgrade the workers in each cluster. See [Upgrade a Cluster Deployment](cluster.md) for the Docker, Kubernetes, and cloud cluster procedures.

## Upgrade via Docker Compose

If you deployed with Docker Compose from a cloned repository, check out the new release tag and recreate the containers with the updated image:

```bash
cd gpustack

# Fetch the tags and check out the target stable release
git fetch --tags
git checkout <new-tag>   # e.g. vx.y.z

cd docker-compose

# Pull the new image and recreate the containers
sudo docker compose -f docker-compose.server.yaml pull
sudo docker compose -f docker-compose.server.yaml up -d
```
