# Pinned Backend Versions

Inference engines in the generative AI domain are evolving rapidly to enhance performance and unlock new capabilities. This constant evolution provides exciting opportunities but also presents challenges for maintaining model compatibility and deployment stability.

GPUStack allows you to pin [inference backend](./inference-backends.md) versions to specific releases, offering a balance between staying up-to-date with the latest advancements and ensuring a reliable runtime environment. This feature is particularly beneficial in the following scenarios:

- Leveraging the newest backend features without waiting for a GPUStack update.
- Locking in a specific backend version to maintain compatibility with existing models.
- Assigning different backend versions to models with varying requirements.

By pinning backend versions, you gain full control over your inference environment, enabling both flexibility and predictability in deployment.

## Automatic Installation of Pinned Backend Versions

To simplify deployment, GPUStack supports the automatic installation of pinned backend versions when feasible. The process depends on the type of backend:

1. **Prebuilt Binaries**  
   For backends like `llama-box`, GPUStack downloads the specified version using the same mechanism as in GPUStack bootstrapping.

!!! tip

    You can customize the download source using the `--tools-download-base-url` [configuration option](../cli-reference/start.md).

2. **Python-based Backends**  
   For backends like `vLLM` and `vox-box`, GPUStack uses `pipx` to install the specified version in an isolated Python environment.

!!! tip

    - Ensure that `pipx` is installed on the worker nodes.
    - If `pipx` is not in the system PATH, specify its location with the `--pipx-path` [configuration option](../cli-reference/start.md).

This automation reduces manual intervention, allowing you to focus on deploying and using your models.

## Manual Installation of Pinned Backend Versions

When automatic installation is not feasible or preferred, GPUStack provides a straightforward way to manually install specific versions of inference backends. Follow these steps:

1. **Prepare the Executable**  
   Install the backend executable or link it under the GPUStack bin directory. The default locations are:

   - **Linux/macOS:** `/var/lib/gpustack/bin`
   - **Windows:** `$env:AppData\gpustack\bin`

!!! tip

    You can customize the bin directory using the `--bin-dir` [configuration option](../cli-reference/start.md).

2. **Name the Executable**  
   Ensure the executable is named in the following format:

   - **Linux/macOS:** `<backend>_<version>`
   - **Windows:** `<backend>_<version>.exe`

For example, the vLLM executable for version v0.7.3 should be named `vllm_v0.7.3` on Linux.

By following these steps, you can maintain full control over the backend installation process, ensuring that the correct version is used for your deployment.
