# Online Installation

## Supported Devices

- [x] Cambricon MLUs

## Supported Platforms

| OS    | Arch  | Supported methods                     |
| ----- | ----- | ------------------------------------- |
| Linux | AMD64 | [pip Installation](#pip-installation) |

## Supported backends

- [x] vLLM

## Prerequisites

- Cambricon Driver

Check if the Cambricon driver is installed:

```bash
cnmon
```

- Cambricon Pytorch docker image

Please contact Cambricon engineers to get the Cambricon Pytorch docker image.

<a id="pip-installation"></a>

## pip Installation

Use Cambricon Pytorch docker image and Activate the `pytorch_infer` virtual environment:

```bash
source /torch/venv3/pytorch_infer/bin/activate
```

### Install GPUStack

Run the following to install GPUStack.

```bash
# vLLM has been installed in Cambricon Pytorch docker
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

```bash
cat /var/lib/gpustack/initial_admin_password
```

By default, GPUStack uses `/var/lib/gpustack` as the data directory so you need `sudo` or proper permission for that. You can also set a custom data directory by running:

```bash
gpustack start --data-dir mypath
```

You can refer to the [CLI Reference](../../cli-reference/start.md) for available CLI Flags.

### (Optional) Add Worker

To add a worker to the GPUStack cluster, you need to specify the server URL and the authentication token.

To get the token used for adding workers, run the following command on the GPUStack **server node**:

```bash
cat /var/lib/gpustack/token
```

To start GPUStack as a worker, and **register it with the GPUStack server**, run the following command on the **worker node**. Be sure to replace the URL, token and node IP with your specific values:

```bash
gpustack start --server-url http://your_gpustack_url --token your_gpustack_token --worker-ip your_worker_host_ip
```
