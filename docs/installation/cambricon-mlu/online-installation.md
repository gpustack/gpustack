# Online Installation

## Supported Devices

- [x] Cambricon MLUs 

## Supported Platforms

| OS      | Arch           | Supported methods                                                                                                                                 |
| ------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| Linux   | AMD64 | [pip Installation](#pip-installation) |

## Supported backends

- [x] vLLM

## Prerequisites

- [Cambricon Driver]

Check if the Cambricon driver is installed:

```bash
cnmon
```
- [Cambricon Pytorch docker image]

Please contact Cambricon engineers to get the Cambricon Pytorch docker image.

## pip Installation

Use Cambricon Pytorch docker image and Activate the "pytorch_infer" virtual environment:

```bash
source /torch/venv3/pytorch_infer/bin/activate
```

### Install GPUStack

Run the following to install GPUStack.

    ```bash
    # Vllm has been installed in Cambricon Pytorch docker
    pip install "gpustack[audio]"
    ```

To verify, run:

```bash
gpustack version
```

### Run GPUStack

Run the following command to start the GPUStack server **and built-in worker**:

```bash
gpustack start
```

If the startup logs are normal, open `http://your_host_ip` in the browser to access the GPUStack UI. Log in to GPUStack with username `admin` and the default password. You can run the following command to get the password for the default setup:

=== "Linux"

    ```bash
    cat /var/lib/gpustack/initial_admin_password
    ```

By default, GPUStack uses `/var/lib/gpustack` as the data directory so you need `sudo` or proper permission for that. You can also set a custom data directory by running:

```bash
gpustack start --data-dir mypath
```

You can refer to the [CLI Reference](../../cli-reference/start.md) for available CLI Flags.
