# GPUStack Installation via Docker Compose

This guide explains how to deploy GPUStack and observability components (Prometheus, Grafana) using Docker Compose. NVIDIA and Ascend platforms are covered, with notes for other GPU types.

## Overview of Services

**Services:**

- **gpustack-server**: Central server for scheduling, management, and built-in inference.
- **gpustack-worker**: (Optional) Distributed inference worker, can run on separate nodes.
- **prometheus**: Metrics collection.
- **grafana**: Metrics visualization.

## Prerequisites

- Docker Compose installed ([guide](https://docs.docker.com/compose/install/)).
- Required ports available ([see requirements](../requirements.md#port-requirements)).

## NVIDIA

### Requirements
- NVIDIA GPU driver (CUDA 12.4+), verify with:
  ```bash
  nvidia-smi
  ```
- NVIDIA Container Toolkit, verify with:
  ```bash
  sudo docker info | grep nvidia
  ```

### Deployment
- **Server** ([compose file](https://github.com/gpustack/gpustack/blob/v2.0.0/docker-compose/docker-compose.server.nvidia.yaml)):
  ```bash
  sudo docker compose -f docker-compose.server.nvidia.yaml up -d
  ```
  Access UI: `http://your_host_ip`
  Get admin password:
  ```bash
  sudo docker exec -it gpustack-server cat /var/lib/gpustack/initial_admin_password
  ```
- **Worker** ([compose file](https://github.com/gpustack/gpustack/blob/v2.0.0/docker-compose/docker-compose.worker.nvidia.yaml)):
  - Edit file: set `server-url` and `token`.
  - Start:
  ```bash
  sudo docker compose -f docker-compose.worker.nvidia.yaml up -d
  ```

## Ascend

### Requirements
- [Ascend NPU Driver](https://www.hiascend.com/hardware/firmware-drivers/community) supporting Ascend CANN 8.2 or higher, verify with:
  ```bash
  sudo npu-smi info
  ```
- Ascend Container Toolkit, verify with:
  ```bash
  sudo docker info 2>/dev/null | grep -q "ascend" \
        && echo "Ascend Container Toolkit OK" \
        || (echo "Ascend Container Toolkit not configured"; exit 1)
  ```

### Deployment
- **Device detection** (before starting):
  ```bash
  export ASCEND_VISIBLE_DEVICES=$(ls /dev/davinci* 2>/dev/null | head -1 | grep -o '[0-9]\+' || echo "0")
  ```
- **Server** ([compose file](https://github.com/gpustack/gpustack/blob/v2.0.0/docker-compose/docker-compose.server.ascend.yaml)):
  ```bash
  sudo -E docker compose -f docker-compose.server.ascend.yaml up -d
  ```
  Access UI: `http://your_host_ip`
  Get admin password:
  ```bash
  sudo docker exec -it gpustack-server cat /var/lib/gpustack/initial_admin_password
  ```
- **Worker** ([compose file](https://github.com/gpustack/gpustack/blob/v2.0.0/docker-compose/docker-compose.worker.ascend.yaml)):
  - Edit file: set `server-url` and `token`.
  - Start:
  ```bash
  sudo -E docker compose -f docker-compose.worker.ascend.yaml up -d
  ```

## Other GPU Platforms
Refer to [requirements](../requirements) for platform-specific setup (AMD, MLU, etc.).

### Deployment

- Edit Compose files as needed:
  - Adjust/remove `runtime: nvidia`.
  - Set environment variables and volumes for your hardware.
  - For workers, set correct `server-url` and `token`.
- Start services using `docker compose -f <compose-file> up -d`.
