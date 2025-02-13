# FAQ

## Support Matrix

### Hybird Cluster Support

Supports a mix of Linux, Windows, and macOS nodes, as well as a mix of x86_64 and arm64 architectures. It also supports various GPUs/NPUs, including NVIDIA, Apple Metal, AMD, Ascend, Hygon and Moore Threads.

### OS Support

Please refer to: **https://docs.gpustack.ai/latest/installation/installation-requirements/#operating-system-requirements**

### NVIDIA GPU Support

GPUStack now supports all Nvidia GPUs with compute capability from 6.0 to 8.9. For more information, refer to Nvidia’s GPU compute capability guide: **https://developer.nvidia.com/cuda-gpus**.

### NVIDIA Driver and CUDA Requirements

The NVIDIA driver must be `550` or higher.

For non-container installations, `CUDA 12.4` or higher is required, for container installations, CUDA is not necessary.

### Apple Metal GPU Support

Supports the entire Mac M1 to M4 series.

### AMD GPU Support

Please refer to: **https://docs.gpustack.ai/latest/tutorials/running-inference-with-amd-gpus/**

### Ascend NPU Support

Currently, Ascend 910B series (910B1 to 910B4) and 310P are supported. Ascend 910 series (910A, 910B, etc.), 310, and 310B are not supported.

When deploying models, choose the llama-box backend (GGUF model), which only supports `FP16`, `Q8_0`, and `Q4_0` models, and it's recommended to deploy `FP16` models. The vLLM backend currently does not support Ascend NPU. Please follow the merge status of this PR: **https://github.com/vllm-project/vllm/pull/8054**.

### NPU Driver and CANN Requirements

The NPU driver must be `24.1.rc2` or higher.

For non-container installations, `CANN 8.0.RC2.beta1` is required, for container installations, CANN is not necessary.

### Hygon DCU Support

Please refer to: **https://docs.gpustack.ai/latest/tutorials/running-inference-with-hygon-dcus/**

### Moore Threads GPU Support

Please refer to: **https://docs.gpustack.ai/latest/tutorials/running-inference-with-moorethreads-gpus/**

### Audio Models Support

Please refer to: **https://github.com/gpustack/vox-box#supported-models**

### Multi-GPU Scheduling, Parallelism and Distributed Inference

Supports multi-GPU scheduling and parallelism, collectively referred to as distributed inference.

Currently, vLLM tensor parallelism (single machine, multiple GPUs) and llama-box pipeline parallelism (single machine, multi-GPU, multi-node multi-GPU, and heterogeneous multi-GPU) are supported.

However, vLLM pipeline parallelism (multi-node, multi-GPU) is not yet supported. For updates on vLLM multi-node multi-GPU support, please follow this issue: **https://github.com/gpustack/gpustack/issues/750**

Related documentation:

**vLLM**：**https://docs.vllm.ai/en/latest/serving/distributed_serving.html**

**llama-box**：**https://github.com/ggerganov/llama.cpp/tree/master/examples/rpc**

### Function Calling / Tool Calling

Both the vLLM backend and the llama-box backend support Function Calling / Tool Calling, but the model also needs to support it.

## Installation

### Port Requirements

Please refer to: **https://docs.gpustack.ai/latest/installation/installation-requirements/**

### Modify the Default GPUStack Port

By default, the GPUStack server uses port 80. You can modify it using the following method:

- Script Installation

**Linux**

```bash
sudo vim /etc/systemd/system/gpustack.service
```

Add the `--port` parameter:

```bash
ExecStart=/root/.local/bin/gpustack start --port 9090
```

Save and restart GPUStack:

```bash
sudo systemctl daemon-reload && sudo systemctl restart gpustack
```

**macOS**

```bash
sudo launchctl unload /Library/LaunchDaemons/ai.gpustack.plist
sudo vim /Library/LaunchDaemons/ai.gpustack.plist
```

```bash
  <array>
    <string>/Users/gpustack/.local/bin/gpustack</string>
    <string>start</string>
    <string>--port</string>
    <string>9090</string>
  </array>
```

Save and start GPUStack:

```bash
sudo launchctl load /Library/LaunchDaemons/ai.gpustack.plist
```

**Windows**

```powershell
nssm edit GPUStack
```

```
gpustack start --port 9090
```

Save and restart GPUStack:

```powershell
Restart-Service -Name "GPUStack"
```

- Docker Installation

Add the `--port` parameter at the end of the `docker run` command, as shown below:

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --gpus all \
    --network=host \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack \
    --port 9090
```

If the host network is not used, only the mapped host port needs to be modified:

```
docker run -d --name gpustack \
    --restart=unless-stopped \
    --gpus all \
    -p 9090:80 \
    -p 10150:10150 \
    -p 40000-41024:40000-41024 \
    -p 50000-51024:50000-51024 \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack --worker-ip your_host_ip
```

- pip Installation

```bash
gpustack start --port 9090
```

### Default Data Path

- Script & pip Installation

**Linux & macOS**

The default path is as follows, and you can set it to a custom path using the `--data-dir` parameter when running GPUStack:

```bash
/var/lib/gpustack
```

**Windows**

The default path is as follows, and you can set it to a custom path using the `--data-dir` parameter when running GPUStack:

```powershell
"$env:APPDATA\gpustack"
```

- Docker Installation

When running the GPUStack container, the Docker volume is mounted using `-v` parameter. The default data path is under the Docker data directory, specifically in the volumes subdirectory, and the default path is:

```bash
/var/lib/docker/volumes/gpustack-data/_data
```

You can check it by the following method:

```bash
docker volume ls
docker volume inspect gpustack-data
```

If you need to change it to a custom path, modify the mount configuration when running container. For example, to mount the host directory `/data/gpustack`:

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --gpus all \
    -p 80:80 \
    --ipc=host \
    -v /data/gpustack:/var/lib/gpustack  \
    gpustack/gpustack
```

### K8s Installation

Please refer to: **https://github.com/linyinli/gpustack/blob/feat/k8s-manifest/gpustack.yaml**

This example is based on NVIDIA CUDA. You need to configure `NVIDIA Container Runtime` and `nvidia-device-plugin`. Modify the Ingress host as needed, or you can change to NodePort. It mounts an NFS storage, which can be adjusted based on your actual setup. Other backend configurations can be modified in the YAML according to the `Docker Installation`: **https://docs.gpustack.ai/latest/installation/docker-installation/**.

!!! note

    K8s Installation is not officially supported yet, we are conducting tests.

### GPUStack startup parameters

Please refer to: **https://docs.gpustack.ai/latest/cli-reference/start/**

## Upgrade

### How to upgrade GPUStack?

Upgrade the server first, then upgrade workers.

- Script Installation

Re-run the installation command to upgrade GPUStack automatically.

- Docker Installation

Remove the old version of the GPUStack container and run the new version:

```bash
docker stop $GPUSTACK_CONTAINER_ID
docker rm $GPUSTACK_CONTAINER_ID
# Use the previous running command, replacing with the new image.
docker run ……
```

- pip Installation

```bash
pip list | grep gpustack
# Linux x86_64
pip install -U gpustack[all]
# Linux arm64
pip install -U gpustack[audio]
```

### How to Upgrade Workers in Bulk?

Batch upgrading is not supported yet. After upgrading the server, you need to upgrade each worker individually.

### How to upgrade the built-in vLLM?

GPUStack supports multiple versions of inference backends. When deploying a model, you can specify the backend version to use a newly released vLLM version (**https://github.com/vllm-project/vllm/releases**). GPUStack will automatically create a virtual environment using pipx to install it:

![faq-customized-backend-version-vllm](assets/faq/faq-customized-backend-version-vllm.png)

If you still need to upgrade the built-in vLLM, you can upgrade vLLM on all worker nodes using the following method:

- Script Installation

```bash
pipx runpip gpustack list | grep vllm
pipx runpip gpustack install -U vllm
```

- Docker Installation

```bash
docker exec -it $GPUSTACK_CONTAINER_ID bash
pip list | grep vllm
pip install -U vllm
```

- pip Installation

```bash
pip list | grep vllm
pip install -U vllm
```

**How to upgrade the built-in llama-box?**

GPUStack supports multiple versions of inference backends. When deploying a model, you can specify the backend version to use a newly released llama-box version (**https://github.com/gpustack/llama-box/releases**). GPUStack will automatically download and configure it:

![faq-customized-backend-version-llama-box](assets/faq/faq-customized-backend-version-llama-box.png)

If you are using distributed inference, you should upgrade llama-box on all worker nodes using the following method:

Download a newly released llama-box binary from **https://github.com/gpustack/llama-box/releases**.

And you need to stop the GPUStack first, then replace the binary, and finally restart the GPUStack. You can check the file location through some directories, for example:

- Script Installation

```bash
ps -ef | grep llama-box
```

- Docker Installation

```bash
docker exec -it $GPUSTACK_CONTAINER_ID bash
ps -ef | grep llama-box
```

- pip Installation

```bash
ps -ef | grep llama-box
```

## Troubleshooting

### View Logs

### View GPUStack Logs

The GPUStack logs provide information on the startup status, calculated model resource requirements, and more. The methods for viewing the GPUStack logs are as follows:

- Script Installation

**Linux & macOS**

```bash
tail -200f /var/log/gpustack.log
```

**Windows**

```powershell
Get-Content "$env:APPDATA\gpustack\log\gpustack.log" -Tail 200 -Wait
```

- Docker Installation

```bash
docker logs $GPUSTACK_CONTAINER_ID
```

- pip Installation

View the output of `gpustack start`.

### Enable GPUStack Debug Mode

GPUStack server can enable debug mode in the following way:

```bash
curl -X PUT http://localhost/debug/log_level -d "debug"
```

Alternatively, both server and worker can add the `--debug` parameter when running GPUStack:

- Script Installation

**Linux**

```bash
sudo vim /etc/systemd/system/gpustack.service
```

```bash
ExecStart=/root/.local/bin/gpustack start --debug
```

Save and restart GPUStack:

```bash
sudo systemctl daemon-reload && sudo systemctl restart gpustack
```

**macOS**

```bash
sudo launchctl unload /Library/LaunchDaemons/ai.gpustack.plist
sudo vim /Library/LaunchDaemons/ai.gpustack.plist
```

```bash
  <array>
    <string>/Users/gpustack/.local/bin/gpustack</string>
    <string>start</string>
    <string>--debug</string>
  </array>
```

```bash
sudo launchctl load /Library/LaunchDaemons/ai.gpustack.plist
```

**Windows**

```powershell
nssm edit GPUStack
```

```powershell
gpustack start --debug
```

Save and restart GPUStack:

```powershell
Restart-Service -Name "GPUStack"
```

- Docker Installation

Add the `--debug` parameter at the end of the `docker run` command, as shown below:

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --gpus all \
    --network=host \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack \
    --debug
```

- pip Installation

```bash
gpustack start --debug
```

### View RPC Server Logs

RPC Server is used for distributed inference of GGUF models. If the model starts abnormally or if there are issues with distributed inference, you can check the RPC Server logs on the corresponding node:

- Script Installation

**Linux & macOS**

The default path is as follows. If the `--data-dir` or `--log-dir` parameters are set, please modify it to the actual path you have configured:

```bash
tail -200f /var/lib/gpustack/log/rpc_server/gpu-0.log
```

Each GPU corresponds to an RPC Server. For other GPU indices, modify it to the actual index:

```bash
tail -200f /var/lib/gpustack/log/rpc_server/gpu-n.log
```

**Windows**

The default path is as follows. If the `--data-dir` or `--log-dir` parameters are set, please modify it to the actual path you have configured:

```powershell
Get-Content "$env:APPDATA\gpustack\log\rpc_server\gpu-0.log" -Tail 200 -Wait
```

Each GPU corresponds to an RPC Server. For other GPU indices, modify it to the actual index:

```bash
Get-Content "$env:APPDATA\gpustack\log\rpc_server\gpu-n.log" -Tail 200 -Wait
```

- Docker Installation

The default path is as follows. If the `--data-dir` or `--log-dir` parameters are set, please modify it to the actual path you have configured:

```bash
docker exec -it $GPUSTACK_CONTAINER_ID tail -200f /var/lib/gpustack/log/rpc_server/gpu-0.log
```

Each GPU corresponds to an RPC Server. For other GPU indices, modify it to the actual index:

```bash
docker exec -it $GPUSTACK_CONTAINER_ID tail -200f /var/lib/gpustack/log/rpc_server/gpu-n.log
```

### View Model Logs

Click the `View Logs` button in the model instance operations:

![faq-view-model-logs](assets/faq/faq-view-model-logs.png)

The model instance logs are stored in the `/var/lib/gpustack/log/serve/` directory of the corresponding worker node or worker container, with the log file named `id.log`, where id is the model instance ID. If the `--data-dir` or `--log-dir` parameter is set, the logs will be stored in the actual path specified by the parameter.

### Enable Model Debug Mode

- llama-box backend (GGUF models)

Add the `--verbose` parameter and recreate the model instance:

![faq-enable-mode-debug-mode-llama-box](assets/faq/faq-enable-mode-debug-mode-llama-box.png)

- vLLM backends (Safetensors models)

Create or edit `/etc/default/gpustack` , add the `VLLM_LOGGING_LEVEL=DEBUG` environment variable:

```bash
vim /etc/default/gpustack
```

```
VLLM_LOGGING_LEVEL=DEBUG
```

Save and restart GPUStack:

```bash
systemctl restart gpustack
```

### Detect GPUs

### Failed to detect Ascend NPU

- Script Installation

Check if `npu-smi` can be executed on the host:

```bash
npu-smi info
```

- Docker Installation

Check if `npu-smi` can be executed in the container:

```bash
docker exec -it $GPUSTACK_CONTAINER_ID bash
npu-smi info
```

When the following error occurs, it indicates that other containers are also mounting the NPU device, and sharing is not supported:

```
dcmi model initialized failed, because the device is used. ret is -8020
```

Check if any containers on the host have mounted NPU devices:

```bash
if [ $(docker ps | wc -l) -gt 1 ]; then docker ps | grep -v CONT | awk '{print $1}' | xargs docker inspect --format='{{printf "%.5s" .ID}} {{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' | sort -k2; fi; echo ok
```

Only mount NPUs that are not mounted by other containers, and specify them using the `ASCEND_VISIBLE_DEVICES` environment variable:

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    -e ASCEND_VISIBLE_DEVICES=1,3,5,7 \
    -p 80:80 \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-npu
```

## Managing Models

### Deploying Models

### Requirements for deploying models from Hugging Face

To deploy models from Hugging Face, the server node and the worker nodes where the model instances are scheduled must have access to Hugging Face, or you can use a mirror.

Configure the `hf-mirror.com` mirror:

- Script Installation

**Linux & macOS**

Create or edit `/etc/default/gpustack` on **all nodes** , add the `HF_ENDPOINT` environment variable to use `https://hf-mirror.com` as the Hugging Face mirror:

```shell
vim /etc/default/gpustack
```

```shell
HF_ENDPOINT=https://hf-mirror.com
```

Save and restart GPUStack:

```shell
systemctl restart gpustack
```

- Docker Installation

Add the `HF_ENDPOINT` environment variable when running container, as shown below:

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --gpus all \
    -e HF_ENDPOINT=https://hf-mirror.com \
    --network=host \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack
```

- pip

### Requirements for deploying models from Local Path

When deploying models from Local Path, it is recommended to upload the model files to each node and maintain the same absolute path. Alternatively, the model instance should be manually scheduled to nodes that have the model files via manual scheduling or label selection. Another option is to mount a shared storage across multiple nodes.

When deploying GGUF models from Local Path, the path must point to the absolute path of the `.gguf` file. For sharded model files, use the absolute path of the first `.gguf` file (00001). If using container installation, the model files must be mounted into the container, and the path should point to the container’s path, not the host’s path.

When deploying Safetensors models from Local Path, the path must point to the absolute path of the model directory which contain `*.safetensors`, `config.json`, and other files. If using container installation, the model files must be mounted into the container, and the path should point to the container’s path, not the host’s path.

### \*.so: cannot open shared object file: No such file or directory

If the error occurs during model startup indicating that any `.so` file cannot be opened, for example:

```bash
llama-box: error while loading shared libraries: libascend_dump.so: cannot open shared object file: No such file or directory
```

The cause is that GPUStack doesn’t recognize the `LD_LIBRARY_PATH` environment variable, which may be due to a missing environment variable or unconfigured toolkits (such as CUDA, CANN, etc.) during GPUStack installation.

To check if the environment variable is set:

```bash
echo $LD_LIBRARY_PATH
```

If not configured, here’s an example configuration for CUDA and CANN:

- CUDA

Ensure that the `nvidia-smi` is executable and the NVIDIA driver version is `550` or later:

```bash
nvidia-smi
```

Configure the CUDA environment variables. If not installed, install `CUDA 12.4` or later:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/targets/x86_64-linux/lib
export PATH=$PATH:/usr/local/cuda/bin
echo $LD_LIBRARY_PATH
echo $PATH
```

Create or edit `/etc/default/gpustack` , add the `PATH` and `LD_LIBRARY_PATH` environment variables:

```bash
vim /etc/default/gpustack
```

```bash
LD_LIBRARY_PATH=xxxxx
PATH=xxxxx
```

Save and restart GPUStack:

```bash
systemctl restart gpustack
```

- CANN

Ensure that the `npu-smi` is executable and the NPU driver version is `24.1.rc2` or later:

```bash
npu-smi info
```

Configure the CANN environment variables. If not installed, install `CANN 8.0.RC2.beta1` first:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
echo $LD_LIBRARY_PATH
echo $PATH
```

Create or edit `/etc/default/gpustack` , add the `PATH` and `LD_LIBRARY_PATH` environment variables:

```bash
vim /etc/default/gpustack
```

```bash
LD_LIBRARY_PATH=xxxxx
PATH=xxxxx
ASCEND_OPP_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp
```

Save and restart GPUStack:

```bash
systemctl restart gpustack
```

### Failed to load model when using Local Path

When deploying a model using Local Path and encountering a `failed to load model` error, you need to check whether the model files exist on the node that the model instance is scheduled to, and if the absolute path is correct.

For GGUF models, you need to specify the absolute path to the `.gguf` file. For sharded models, use the absolute path to the first `.gguf` file (typically 00001).

If using Docker installation, the model files must be mounted into the container. Make sure the path you provide is the one inside the container, not the host path.

![faq-deploy-from-local-path](assets/faq/faq-deploy-from-local-path.png)

### Backend Parameters

### Backend Parameter Description

**llama-box**：**https://github.com/gpustack/llama-box?tab=readme-ov-file#usage**

**vLLM**：**https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html#vllm-serve**

### Model Context Length

- llama-box backend (GGUF models)

GPUStack sets the default context length for models to 8K. You can customize the context length using the `--ctx-size` parameter, but it cannot exceed the model’s maximum context length:

![faq-model-context-length-llama-box.png](assets/faq/faq-model-context-length-llama-box.png)

If editing, save the change and then recreate the model instance to take effect.

- vLLM backend (Safetensors models)

GPUStack sets the default context length for models to 8K. You can customize the context length using the `--max-model-len` parameter, but it cannot exceed the model’s maximum context length:

![faq-model-context-length-vllm](assets/faq/faq-model-context-length-vllm.png)

If editing, save the change and then recreate the model instance to take effect.

## Using Models

### Embedding Models

### Normalization

The llama-box backend (GGUF models) applies normalization to the embedding model by default.

The vLLM backend (Safetensors models) enables normalization through the following backend parameters:

```bash
--override-pooler-config={"normalize": true}
```

### Image Generation Models

### Recommended Parameters for Image Generation Models

Please refer to: **https://docs.gpustack.ai/latest/tutorials/recommended-parameters-for-image-generation-models/**

### API

### API Docs

Access the API documentation through the GPUStack URL and the `/docs` path, for example: `http://localhost/docs`.

### OpenAI-Compatible API

Use the OpenAI-compatible API provided by GPUStack through its URL with the `/v1-openai` or `/v1` path, for example: `http://localhost/v1-openai` or `http://localhost/v1`.

## Integration

### Integration with Dify

![faq-dify-integration-add-model](assets/faq/faq-dify-integration-add-model.png)

![faq-view-model-logs](assets/faq/faq-dify-integration-system-model-settings.png)

### Integration with RAGFlow

![faq-ragflow-integration-add-model](assets/faq/faq-ragflow-integration-add-model.png)

![faq-ragflow-integration-added-models](assets/faq/faq-ragflow-integration-added-models.png)

![faq-ragflow-integration-system-model-settings](assets/faq/faq-ragflow-integration-system-model-settings.png)

## Architecture

### GPUStack Architecture

Please refer to: **https://docs.gpustack.ai/latest/architecture/**

## Scheduling

### GPUStack Scheduling Process

Please refer to: **https://docs.gpustack.ai/latest/scheduler/**

## Managing GPUStack

### Managing GPUStack Service

- Script Installation

**Linux**

Stop GPUStack:

```bash
sudo systemctl stop gpustack
```

Start GPUStack:

```bash
sudo systemctl start gpustack
```

Restart GPUStack:

```bash
sudo systemctl restart gpustack
```

**macOS**

Stop GPUStack:

```bash
sudo launchctl unload /Library/LaunchDaemons/ai.gpustack.plist
```

Start GPUStack:

```bash
sudo launchctl load /Library/LaunchDaemons/ai.gpustack.plist
```

Restart GPUStack:

```bash
sudo launchctl unload /Library/LaunchDaemons/ai.gpustack.plist
sudo launchctl load /Library/LaunchDaemons/ai.gpustack.plist
```

**Windows**

Stop GPUStack:

```powershell
Stop-Service -Name "GPUStack"
```

Start GPUStack:

```powershell
Start-Service -Name "GPUStack"
```

Restart GPUStack:

```powershell
Restart-Service -Name "GPUStack"
```

- Docker Installation

Restart GPUStack container:

```bash
docker restart $GPUSTACK_CONTAINER_ID
```

### Configure GPUStack to use proxy

- Script Installation

**Linux & macOS**

Create or edit `/etc/default/gpustack` and add the proxy configuration:

```bash
vim /etc/default/gpustack
```

```bash
http_proxy="http://username:password@proxy-server:port"
https_proxy="http://username:password@proxy-server:port"
all_proxy="socks5://username:password@proxy-server:port"
no_proxy="localhost,127.0.0.1,192.168.0.0/16,10.0.0.0/8"
```

Save and restart GPUStack:

```bash
systemctl restart gpustack
```

- Docker Installation

Pass environment variables when running GPUStack:

```bash
docker run -e http_proxy="http://username:password@proxy-server:port" \
           -e https_proxy="http://username:password@proxy-server:port" \
           -e all_proxy="socks5://username:password@proxy-server:port" \
           -e no_proxy="localhost,127.0.0.1,192.168.0.0/16,10.0.0.0/8" \
           ……
```

### View Admin Password

- Script Installation

View the default admin password on the server node:

**Linux & macOS**

```bash
cat /var/lib/gpustack/initial_admin_password
```

**Windows**

```powershell
Get-Content -Path (Join-Path -Path $env:APPDATA -ChildPath "gpustack\initial_admin_password") -Raw
```

- Docker Installation

View the default admin password inside the server container:

```bash
docker exec -it $GPUSTACK_CONTAINER_ID cat /var/lib/gpustack/initial_admin_password
```

### Reset Admin Password

Run the reset command on the server node or inside the server container:

```shell
gpustack reset-admin-password
```

If the default port has been changed, specify the GPUStack URL using the `--server-url` parameter. It must be run locally on the server and accessed via `localhost`.

```bash
gpustack reset-admin-password --server-url http://localhost:9090
```

## Uninstallation

### Uninstall GPUStack

- Script Installation

**Linux & macOS**

```bash
sudo /var/lib/gpustack/uninstall.sh
```

**Windows**

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; & "$env:APPDATA\gpustack\uninstall.ps1"
```

- Docker Installation

```bash
docker rm -f $GPUSTACK_CONTAINER_ID
docker volume rm gpustack-data
docker volume rm gpustack-server-data
docker volume rm gpustack-worker-data
```

- pip Installation

```bash
pip uninstall gpustack
rm -rf /var/lib/gpustack
```
