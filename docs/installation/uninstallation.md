# Uninstallation

GPUStack is typically installed using containerization, 
so uninstallation mainly involves removing the container and any associated data volumes.

For example, if GPUStack is running in a Docker container named `gpustack`, run:

```bash
docker rm -f gpustack

```

To optionally remove associated data volumes, use:

```bash
docker volume rm <data_volume_name>

```
