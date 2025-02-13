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
# On Windows (PowerShell):
# $PACKAGE_SPEC = "gpustack"

# Optional: To include extra dependencies (vllm, audio, all) or install a specific version
# PACKAGE_SPEC="gpustack[all]"
# PACKAGE_SPEC="gpustack==0.4.0"
PACKAGE_SPEC="gpustack"

# Download all required packages
pip wheel $PACKAGE_SPEC -w gpustack_offline_packages

# Install GPUStack to access its CLI
pip install gpustack

# Download dependency tools and save them as an archive
gpustack download-tools --save-archive gpustack_offline_tools.tar.gz
```

Optional: Additional Dependencies for macOS.

Download save and load dependencies scripts:

- [load_macos_dependencies.sh](../assets/installation/air-gapped-installation/load_macos_dependencies.sh)
- [save_macos_dependencies.sh](../assets/installation/air-gapped-installation/save_macos_dependencies.sh)

```bash
# Deploying the speech-to-text CosyVoice model on macOS requires additional dependencies.
./save_macos_dependencies.sh
```

After running the script, will generate two files in the current directory:

- `local-openfst-1.8.3.tar.gz`
- `c9d6de4616377b76b6c6c71920a0b24f1b19aeed734fe12dbd2a169d0893b541--openfst-1.8.3.tar.gz`

```bash
export AUDIO_DEPENDENCY_PACKAGE_SPEC="wetextprocessing==1.0.4.1"
export CPLUS_INCLUDE_PATH=$(brew --prefix openfst@1.8.3)/include
export LIBRARY_PATH=$(brew --prefix openfst@1.8.3)/lib

pip wheel $AUDIO_DEPENDENCY_PACKAGE_SPEC -w gpustack_audio_dependency_offline_packages
mv gpustack_audio_dependency_offline_packages/* gpustack_offline_packages/ && rm -rf gpustack_audio_dependency_offline_packages
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

Optional: Additional Dependencies for macOS.

Transfer the following files from the online environment to the air-gapped environment.

- `load_macos_dependencies.sh` file.
- `local-openfst-1.8.3.tar.gz` file.
- `c9d6de4616377b76b6c6c71920a0b24f1b19aeed734fe12dbd2a169d0893b541--openfst-1.8.3.tar.gz` file.

```bash
# Install the additional dependencies for speech-to-text CosyVoice model on macOS.
# load-dir is the directory include local-openfst-1.8.3.tar.gz and c9d6de4616377b76b6c6c71920a0b24f1b19aeed734fe12dbd2a169d0893b541--openfst-1.8.3.tar.gz
./load_macos_dependencies.sh --load-dir ./

pip install --no-index --find-links=gpustack_offline_packages wetextprocessing
```

Now you can run GPUStack by following the instructions in the [Manual Installation](manual-installation.md#run-gpustack) guide.
