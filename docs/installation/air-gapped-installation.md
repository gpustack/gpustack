# Air-Gapped Installation

You can install GPUStack in an air-gapped environment. An air-gapped environment refers to a setup where GPUStack will be installed offline, behind a firewall, or behind a proxy.

The following methods are available for installing GPUStack in an air-gapped environment:

- [Docker Installation](#docker-installation)
- [Manual Installation](#manual-installation)

## Docker Installation

When running GPUStack with Docker, it works out of the box in an air-gapped environment as long as the Docker images are available. To do this, follow these steps:

1. Pull GPUStack Docker images in an online environment.
2. Publish Docker images to a private registry.
3. Refer to the [Docker Installation](docker-installation.md) guide to run GPUStack using Docker.

## Manual Installation

For manual installation, you need to prepare the required packages and tools in an online environment and then transfer them to the air-gapped environment.

### Prerequisites

Set up an online environment identical to the air-gapped environment, including **OS**, **architecture**, and **Python version**.

### Step 1: Download the Required Packages

Run the following commands in an online environment:

```bash
# Optional: To include vLLM dependencies or install a specific version
# PACKAGE_SPEC="gpustack[vllm]"
# PACKAGE_SPEC="gpustack==0.4.0"
PACKAGE_SPEC="gpustack"

# Download all required packages
pip download $PACKAGE_SPEC --only-binary=:all: -d gpustack_offline_packages

# Install GPUStack to access its CLI
pip install gpustack

# Download dependency tools and save them as an archive
gpustack download-tools --save-archive gpustack_offline_tools.tar.gz
```

!!!note

    This instruction assumes that the online environment uses the same GPU type as the air-gapped environment. If the GPU types differ, use the `--device` flag to specify the device type for the air-gapped environment. Refer to the [download-tools](../cli-reference/download-tools.md) command for more information.

### Step 2: Transfer the Packages

Transfer the following files from the online environment to the air-gapped environment.

- `gpustack_offline_packages` directory.
- `gpustack_offline_tools.tar.gz` file.

### Step 3: Install GPUStack

In the air-gapped environment, run the following commands:

```bash
# Install GPUStack from the downloaded packages
pip install --no-index --find-links=gpustack_offline_packages gpustack

# Load and apply the pre-downloaded tools archive
gpustack download-tools --load-archive gpustack_offline_tools.tar.gz
```

Now you can run GPUStack by following the instructions in the [Manual Installation](manual-installation.md#run-gpustack) guide.
