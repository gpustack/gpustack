# Air-Gapped Installation

You can install GPUStack in an air-gapped environment. An air-gapped environment refers to a setup where GPUStack will be installed offline.

The following methods are available for installing GPUStack in an air-gapped environment:

| OS    | Arch  | Supported methods                           |
| ----- | ----- | ------------------------------------------- |
| Linux | AMD64 | [Docker Installation](#docker-installation) |

## Supported Devices

| Devices                                             | Supported Backends |
| --------------------------------------------------- | ------------------ |
| gfx1100: AMD Radeon RX 7900 XTX/7900 XT/7900 GRE    | llama-box, vLLM    |
| gfx1101: AMD Radeon RX 7800 XT/7700 XT              | llama-box, vLLM    |
| gfx1102: AMD Radeon RX 7600 XT/7600                 | llama-box, vLLM    |
| gfx942: AMD Instinct MI300X/MI300A                  | llama-box, vLLM    |
| gfx90a: AMD Instinct MI250X/MI250/MI210             | llama-box, vLLM    |
| gfx1030: AMD Radeon RX 6950 XT/6900 XT/6800 XT/6800 | llama-box          |
| gfx1031: AMD Radeon RX 6750 XT/6700 XT/6700         | llama-box          |
| gfx1032: AMD Radeon RX 6650 XT/6600 XT/6600         | llama-box          |
| gfx908: AMD Instinct MI100                          | llama-box          |
| gfx906: AMD Instinct MI50                           | llama-box          |

## Docker Installation

### Prerequisites

- [Docker](https://docs.docker.com/engine/install/)
- [ROCm 6.2.4](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.2.4/install/install-overview.html)

Select the appropriate installation method for your system. Here, we provide steps for Linux (Ubuntu 22.04). For other systems, refer to the ROCm documentation.

1. Install ROCm:

```bash
sudo apt update
wget https://repo.radeon.com/amdgpu-install/6.2.4/ubuntu/jammy/amdgpu-install_6.2.60204-1_all.deb
sudo apt install ./amdgpu-install_6.2.60204-1_all.deb
amdgpu-install -y --usecase=graphics,rocm
sudo reboot
```

2. Set Groups permissions:

```bash
sudo usermod -a -G render,video $LOGNAME
sudo reboot
```

3. Verify Installation:

Verify that the current user is added to the render and video groups:

```bash
groups
```

Check if amdgpu kernel driver is installed:

```bash
dkms status
```

Check if the GPU is listed as an agent:

```bash
rocminfo
```

Check `rocm-smi`:

```bash
rocm-smi -i --showmeminfo vram --showpower --showserial --showuse --showtemp --showproductname --showuniqueid --json
```

### Run GPUStack

When running GPUStack with Docker, it works out of the box in an air-gapped environment as long as the Docker images are available. To do this, follow these steps:

1. Pull GPUStack docker image in an online environment:

```bash
docker pull gpustack/gpustack:latest-rocm
```

If your online environment differs from the air-gapped environment in terms of OS or arch, specify the OS and arch of the air-gapped environment when pulling the image:

```bash
docker pull --platform linux/amd64 gpustack/gpustack:latest-rocm
```

2. Publish docker image to a private registry or load it directly in the air-gapped environment.
3. Refer to the [Docker Installation](./online-installation.md#docker-installation) guide to run GPUStack using Docker.
