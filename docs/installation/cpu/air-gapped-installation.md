# Air-Gapped Installation

You can install GPUStack in an air-gapped environment. An air-gapped environment refers to a setup where GPUStack will be installed offline.

The following methods are available for installing GPUStack in an air-gapped environment:

| OS      | Arch           | Supported methods                                                                                  |
| ------- | -------------- | -------------------------------------------------------------------------------------------------- |
| Linux   | AMD64<br>ARM64 | [Docker Installation](#docker-installation) (Recommended)<br>[pip Installation](#pip-installation) |
| Windows | AMD64<br>ARM64 | [pip Installation](#pip-installation)                                                              |

## Prerequisites

- [Port Requirements](../installation-requirements.md#port-requirements)
- CPUs (AMD64 with AVX2 or ARM64 with NEON)

Check if the CPU is supported:

=== "Linux"

    === "AMD64"

        ```bash
        lscpu | grep avx2
        ```

    === "ARM64"

        ```bash
        grep -E -i "neon|asimd" /proc/cpuinfo
        ```

=== "Windows"

    Windows users need to manually verify support for the above instructions.

## Docker Installation

### Prerequisites

- [Docker](https://docs.docker.com/engine/install/)

### Run GPUStack

When running GPUStack with Docker, it works out of the box in an air-gapped environment as long as the Docker images are available. To do this, follow these steps:

1. Pull GPUStack docker image in an online environment:

```bash
docker pull gpustack/gpustack:latest-cpu
```

If your online environment differs from the air-gapped environment in terms of OS or arch, specify the OS and arch of the air-gapped environment when pulling the image:

```bash
docker pull --platform linux/amd64 gpustack/gpustack:latest-cpu
```

2. Publish docker image to a private registry or load it directly in the air-gapped environment.
3. Refer to the [Docker Installation](./online-installation.md#docker-installation) guide to run GPUStack using Docker.

## pip Installation

### Prerequisites

- Python 3.10 ~ 3.12

Check the Python version:

```bash
python -V
```

### Install GPUStack

For manually pip installation, you need to prepare the required packages and tools in an online environment and then transfer them to the air-gapped environment.

Set up an online environment identical to the air-gapped environment, including **OS**, **architecture**, and **Python version**.

#### Step 1: Download the Required Packages

Run the following commands in an online environment:

=== "Linux"

    ```bash
    PACKAGE_SPEC="gpustack[audio]"
    # To install a specific version
    # PACKAGE_SPEC="gpustack[audio]==0.6.0"
    ```

    If you don’t need support for audio models, just set:

    ```bash
    PACKAGE_SPEC="gpustack"
    ```

=== "Windows"

    ```powershell
    $PACKAGE_SPEC = "gpustack[audio]"
    # To install a specific version
    # $PACKAGE_SPEC = "gpustack[audio]==0.6.0"
    ```

    If you don’t need support for audio models, just set:

    ```powershell
    $PACKAGE_SPEC = "gpustack"
    ```

Download all required packages:

```bash
pip wheel $PACKAGE_SPEC -w gpustack_offline_packages
```

Install GPUStack to use its CLI:

```bash
pip install gpustack
```

Download dependency tools and save them as an archive:

```bash
gpustack download-tools --save-archive gpustack_offline_tools.tar.gz
```

If your online environment differs from the air-gapped environment, specify the **OS**, **architecture**, and **device** explicitly:

```bash
gpustack download-tools --save-archive gpustack_offline_tools.tar.gz --system linux --arch amd64 --device cpu
```

#### Step 2: Transfer the Packages

Transfer the following files from the online environment to the air-gapped environment.

- `gpustack_offline_packages` directory.
- `gpustack_offline_tools.tar.gz` file.

#### Step 3: Install GPUStack

In the air-gapped environment, run the following commands:

```bash
# Install GPUStack from the downloaded packages
pip install --no-index --find-links=gpustack_offline_packages gpustack

# Load and apply the pre-downloaded tools archive
gpustack download-tools --load-archive gpustack_offline_tools.tar.gz
```

Now you can run GPUStack by following the instructions in the [pip Installation](online-installation.md#pip-installation) guide.
