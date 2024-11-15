# Air-Gapped Installation

You can install GPUStack in an air-gapped environment. An air-gapped environment could be where GPUStack will be installed offline, behind a firewall, or behind a proxy.

The following ways are available to install GPUStack in an air-gapped environment:

- Docker Installation
- Manual Installation

## Docker Installation

When you run GPUStack using Docker, it works out of the box as long as the Docker images are available in the air-gapped environment. Please publish the GPUStack docker images to your private registry, then refer to the [Docker Installation](docker-installation.md) guide for how to run GPUStack using Docker.

## Manual Installation

When you install GPUStack manually, you need to download the required packages in an online environment and then transfer them to the air-gapped environment.

### Step 1: Download the Required Packages

Run the following commands in an online environment to download the required packages:

```bash
# To install extra vllm dependencies: PACKAGE_SPEC="gpustack[vllm]"
# To install a specific version: PACKAGE_SPEC="gpustack==0.4.0"
PACKAGE_SPEC="gpustack"
pip download $PACKAGE_SPEC --only-binary=:all: -d gpustack_offline_packages

pip install gpustack
gpustack download-tools --save-archive gpustack_offline_tools.tar
```

!!! note

    Here we assume that the online environment is the same as the air-gapped environment. If the online environment is different, you can specify the `--system`, `--arch`, and `--device` flags to download the tools for the air-gapped environment. Refer to the [download-tools](../cli-reference/download-tools.md) command for more information.

### Step 2: Transfer the Packages

Transfer the `gpustack_offline_packages` directory and the `gpustack_offline_tools.tar` file to the air-gapped environment.

### Step 3: Install GPUStack

Run the following commands in the air-gapped environment to install GPUStack:

```bash
pip install --no-index --find-links=gpustack_offline_packages gpustack
gpustack download-tools --load-archive gpustack_offline_tools.tar
```

Now you can run GPUStack as in the [Manual Installation](manual-installation.md#run-gpustack) guide.
