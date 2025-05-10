# FAQ

## Support Matrix

### Hybird Cluster Support

It supports a mix of Linux, Windows, and macOS nodes, as well as x86_64 and arm64 architectures. Additionally, It also supports various GPUs, including NVIDIA, Apple Metal, AMD, Ascend, Hygon and Moore Threads.

### Distributed Inference Support

**Single-Node Multi-GPU**

- [x] llama-box (Image Generation models are not supported)
- [x] vLLM
- [x] MindIE
- [ ] vox-box

**Multi-Node Multi-GPU**

- [x] llama-box
- [x] vLLM
- [ ] MindIE

**Heterogeneous-Node Multi-GPU**

- [x] llama-box

!!! tip

    Related documentations:

    **vLLM**：[Distributed Inference and Serving](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)

    **llama-box**：[Distributed LLM inference with llama.cpp](https://github.com/ggml-org/llama.cpp/tree/master/examples/rpc)

## Installation

### How can I change the default GPUStack port?

By default, the GPUStack server uses port 80. You can change it using the following method:

**Script Installation**

- Linux

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

- macOS

```bash
sudo launchctl bootout system /Library/LaunchDaemons/ai.gpustack.plist
sudo vim /Library/LaunchDaemons/ai.gpustack.plist
```

Add the `--port` parameter:

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
sudo launchctl bootstrap system /Library/LaunchDaemons/ai.gpustack.plist
```

- Windows

```powershell
nssm edit GPUStack
```

Add parameter after `start`:

```
start --port 9090
```

Save and restart GPUStack:

```powershell
Restart-Service -Name "GPUStack"
```

**Docker Installation**

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

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --gpus all \
    -p 9090:80 \
    -p 10150:10150 \
    -p 40064-40095:40064-40095 \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack \
    --worker-ip your_host_ip
```

**pip Installation**

Add the `--port` parameter at the end of the `gpustack start`:

```bash
gpustack start --port 9090
```

### How can I change the registered worker name?

You can set it to a custom name using the `--worker-name` parameter when running GPUStack:

**Script Installation**

- Linux

```bash
sudo vim /etc/systemd/system/gpustack.service
```

Add the `--worker-name` parameter:

```bash
ExecStart=/root/.local/bin/gpustack start --worker-name New-Name
```

Save and restart GPUStack:

```bash
sudo systemctl daemon-reload && sudo systemctl restart gpustack
```

- macOS

```bash
sudo launchctl bootout system /Library/LaunchDaemons/ai.gpustack.plist
sudo vim /Library/LaunchDaemons/ai.gpustack.plist
```

Add the `--worker-name` parameter:

```bash
  <array>
    <string>/Users/gpustack/.local/bin/gpustack</string>
    <string>start</string>
    <string>--worker-name</string>
    <string>New-Name</string>
  </array>
```

Save and start GPUStack:

```bash
sudo launchctl bootstrap system /Library/LaunchDaemons/ai.gpustack.plist
```

- Windows

```powershell
nssm edit GPUStack
```

Add parameter after `start`:

```
start --worker-name New-Name
```

Save and restart GPUStack:

```powershell
Restart-Service -Name "GPUStack"
```

**Docker Installation**

Add the `--worker-name` parameter at the end of the `docker run` command, as shown below:

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --gpus all \
    --network=host \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack \
    --worker-name New-Name
```

**pip Installation**

Add the `--worker-name` parameter at the end of the `gpustack start`:

```bash
gpustack start --worker-name New-Name
```

### How can I change the registered worker IP?

You can set it to a custom IP using the `--worker-ip` parameter when running GPUStack:

**Script Installation**

- Linux

```bash
sudo vim /etc/systemd/system/gpustack.service
```

Add the `--worker-ip` parameter:

```bash
ExecStart=/root/.local/bin/gpustack start --worker-ip xx.xx.xx.xx
```

Save and restart GPUStack:

```bash
sudo systemctl daemon-reload && sudo systemctl restart gpustack
```

- macOS

```bash
sudo launchctl bootout system /Library/LaunchDaemons/ai.gpustack.plist
sudo vim /Library/LaunchDaemons/ai.gpustack.plist
```

Add the `--worker-ip` parameter:

```bash
  <array>
    <string>/Users/gpustack/.local/bin/gpustack</string>
    <string>start</string>
    <string>--worker-ip</string>
    <string>xx.xx.xx.xx</string>
  </array>
```

Save and start GPUStack:

```bash
sudo launchctl bootstrap system /Library/LaunchDaemons/ai.gpustack.plist
```

- Windows

```powershell
nssm edit GPUStack
```

Add parameter after `start`:

```
start --worker-ip xx.xx.xx.xx
```

Save and restart GPUStack:

```powershell
Restart-Service -Name "GPUStack"
```

**Docker Installation**

Add the `--worker-ip` parameter at the end of the `docker run` command, as shown below:

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --gpus all \
    --network=host \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack \
    --worker-ip xx.xx.xx.xx
```

**pip Installation**

Add the `--worker-ip` parameter at the end of the `gpustack start`:

```bash
gpustack start --worker-ip xx.xx.xx.xx
```

### Where are GPUStack's data stored?

**Script Installation**

- Linux

The default path is as follows:

```bash
/var/lib/gpustack
```

You can set it to a custom path using the `--data-dir` parameter when running GPUStack:

```bash
sudo vim /etc/systemd/system/gpustack.service
```

Add the `--data-dir` parameter:

```bash
ExecStart=/root/.local/bin/gpustack start --data-dir /data/gpustack-data
```

Save and restart GPUStack:

```bash
sudo systemctl daemon-reload && sudo systemctl restart gpustack
```

- macOS

The default path is as follows:

```bash
/var/lib/gpustack
```

You can set it to a custom path using the `--data-dir` parameter when running GPUStack:

```bash
sudo launchctl bootout system /Library/LaunchDaemons/ai.gpustack.plist
sudo vim /Library/LaunchDaemons/ai.gpustack.plist
```

```bash
  <array>
    <string>/Users/gpustack/.local/bin/gpustack</string>
    <string>start</string>
    <string>--data-dir</string>
    <string>/Users/gpustack/data/gpustack-data</string>
  </array>
```

Save and start GPUStack:

```bash
sudo launchctl bootstrap system /Library/LaunchDaemons/ai.gpustack.plist
```

- Windows

The default path is as follows:

```powershell
"$env:APPDATA\gpustack"
```

You can set it to a custom path using the `--data-dir` parameter when running GPUStack:

```powershell
nssm edit GPUStack
```

Add parameter after `start`:

```
start --data-dir D:\gpustack-data
```

Save and restart GPUStack:

```powershell
Restart-Service -Name "GPUStack"
```

**Docker Installation**

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
    --network=host \
    --ipc=host \
    -v /data/gpustack:/var/lib/gpustack  \
    gpustack/gpustack
```

**pip Installation**

Add the `--data-dir` parameter at the end of the `gpustack start`:

```bash
gpustack start --data-dir /data/gpustack-data
```

### Where are model files stored?

**Script Installation**

- Linux

The default path is as follows:

```bash
/var/lib/gpustack/cache
```

You can set it to a custom path using the `--cache-dir` parameter when running GPUStack:

```bash
sudo vim /etc/systemd/system/gpustack.service
```

Add the `--cache-dir` parameter:

```bash
ExecStart=/root/.local/bin/gpustack start --cache-dir /data/model-cache
```

Save and restart GPUStack:

```bash
sudo systemctl daemon-reload && sudo systemctl restart gpustack
```

- macOS

The default path is as follows:

```bash
/var/lib/gpustack/cache
```

You can set it to a custom path using the `--cache-dir` parameter when running GPUStack:

```bash
sudo launchctl bootout system /Library/LaunchDaemons/ai.gpustack.plist
sudo vim /Library/LaunchDaemons/ai.gpustack.plist
```

```bash
  <array>
    <string>/Users/gpustack/.local/bin/gpustack</string>
    <string>start</string>
    <string>--cache-dir</string>
    <string>/Users/gpustack/data/model-cache</string>
  </array>
```

Save and start GPUStack:

```bash
sudo launchctl bootstrap system /Library/LaunchDaemons/ai.gpustack.plist
```

- Windows

The default path is as follows:

```powershell
"$env:APPDATA\gpustack\cache"
```

You can set it to a custom path using the `--cache-dir` parameter when running GPUStack:

```powershell
nssm edit GPUStack
```

Add parameter after `start`:

```
start --cache-dir D:\model-cache
```

Save and restart GPUStack:

```powershell
Restart-Service -Name "GPUStack"
```

**Docker Installation**

When running the GPUStack container, the Docker volume is mounted using `-v` parameter. The default cache path is under the Docker data directory, specifically in the volumes subdirectory, and the default path is:

```bash
/var/lib/docker/volumes/gpustack-data/_data/cache
```

You can check it by the following method:

```bash
docker volume ls
docker volume inspect gpustack-data
```

If you need to change it to a custom path, modify the mount configuration when running container.

> **Note**: If the data directory is already mounted, the cache directory should not be mounted inside the data directory. You need to specify a different path using the `--cache-dir` parameter.

For example, to mount the host directory `/data/model-cache`:

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --gpus all \
    --network=host \
    --ipc=host \
    -v /data/gpustack:/var/lib/gpustack  \
    -v /data/model-cache:/data/model-cache \
    gpustack/gpustack \
    --cache-dir /data/model-cache
```

**pip Installation**

Add the `--cache-dir` parameter at the end of the `gpustack start`:

```bash
gpustack start --cache-dir /data/model-cache
```

### What parameters can I set when starting GPUStack?

Please refer to: [gpustack start](cli-reference/start.md)

## Upgrade

### How can I upgrade the built-in vLLM?

GPUStack supports multiple versions of inference backends. When deploying a model, you can specify the backend version in `Edit Model` → `Advanced` → `Backend Version` to use a [newly released vLLM version](https://github.com/vllm-project/vllm/releases). GPUStack will automatically create a virtual environment using pipx to install it:

![pin-vllm-backend-version](assets/faq/pin-vllm-backend-version.png)

Or you can manually install the custom version, including using a custom PyPI mirror, and then link the executable to `/var/lib/gpustack/bin/`. After that, configure and use it as described above.

```bash
python3 -m venv $(pipx environment --value PIPX_LOCAL_VENVS)/vllm-v0-8-5-post1
$(pipx environment --value PIPX_LOCAL_VENVS)/vllm-v0-8-5-post1/bin/python -m pip install vllm==v0.8.5.post1 -i https://mirrors.aliyun.com/pypi/simple
sudo mkdir -p /var/lib/gpustack/bin
sudo ln -s $(pipx environment --value PIPX_LOCAL_VENVS)/vllm-v0-8-5-post1/bin/vllm /var/lib/gpustack/bin/vllm_v0.8.5.post1
```

If you still need to upgrade the built-in vLLM, you can upgrade vLLM on all worker nodes using the following method:

**Script Installation**

```bash
pipx runpip gpustack list | grep vllm
pipx runpip gpustack install -U vllm
```

**Docker Installation**

```bash
docker exec -it gpustack bash
pip list | grep vllm
pip install -U vllm
```

**pip Installation**

```bash
pip list | grep vllm
pip install -U vllm
```

### How can I upgrade the built-in Transformers?

**Script Installation**

```bash
pipx runpip gpustack list | grep transformers
pipx runpip gpustack install -U transformers
```

**Docker Installation**

```bash
docker exec -it gpustack bash
pip list | grep transformers
pip install -U transformers
```

**pip Installation**

```bash
pip list | grep transformers
pip install -U transformers
```

### How can I upgrade the built-in llama-box?

GPUStack supports multiple versions of inference backends. When deploying a model, you can specify the backend version in `Edit Model` → `Advanced` → `Backend Version` to use a [newly released llama-box version](https://github.com/gpustack/llama-box/releases). GPUStack will automatically download and configure it:

![pin-llama-box-backend-version](assets/faq/pin-llama-box-backend-version.png)

If you are using distributed inference, you should upgrade llama-box on all worker nodes using the following method:

Download a newly released llama-box binary from [llama-box releases](https://github.com/gpustack/llama-box/releases).

And you need to stop the GPUStack first, then replace the binary, and finally restart the GPUStack. You can check the file location through some directories, for example:

**Script & pip Installation**

```bash
ps -ef | grep llama-box
```

**Docker Installation**

```bash
docker exec -it gpustack bash
ps -ef | grep llama-box
```

## View Logs

### How can I view the GPUStack logs?

The GPUStack logs provide information on the startup status, calculated model resource requirements, and more. Refer to the [Troubleshooting](troubleshooting.md#view-gpustack-logs) for viewing the GPUStack logs.

### How can I enable debug mode in GPUStack?

You can temporarily enable debug mode without interrupting the GPUStack service. Refer to the [Troubleshooting](./troubleshooting.md#configure-log-level) for guidance.

If you want to enable debug mode persistently, both server and worker can add the `--debug` parameter when running GPUStack:

**Script Installation**

- Linux

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

- macOS

```bash
sudo launchctl bootout system /Library/LaunchDaemons/ai.gpustack.plist
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
sudo launchctl bootstrap system /Library/LaunchDaemons/ai.gpustack.plist
```

- Windows

```powershell
nssm edit GPUStack
```

Add parameter after `start`:

```
start --debug
```

Save and restart GPUStack:

```powershell
Restart-Service -Name "GPUStack"
```

**Docker Installation**

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

**pip Installation**

Add the `--debug` parameter at the end of the `gpustack start`:

```bash
gpustack start --debug
```

### How can I view the RPC server logs?

RPC Server is used for distributed inference of GGUF models. If the model starts abnormally or if there are issues with distributed inference, you can check the RPC Server logs on the corresponding node:

**Script Installation**

- Linux & macOS

The default path is as follows. If the `--data-dir` or `--log-dir` parameters are set, please modify it to the actual path you have configured:

```bash
tail -200f /var/lib/gpustack/log/rpc_server/gpu-0.log
```

Each GPU corresponds to an RPC Server. For other GPU indices, modify it to the actual index:

```bash
tail -200f /var/lib/gpustack/log/rpc_server/gpu-n.log
```

- Windows

The default path is as follows. If the `--data-dir` or `--log-dir` parameters are set, please modify it to the actual path you have configured:

```powershell
Get-Content "$env:APPDATA\gpustack\log\rpc_server\gpu-0.log" -Tail 200 -Wait
```

Each GPU corresponds to an RPC Server. For other GPU indices, modify it to the actual index:

```powershell
Get-Content "$env:APPDATA\gpustack\log\rpc_server\gpu-n.log" -Tail 200 -Wait
```

**Docker Installation**

The default path is as follows. If the `--data-dir` or `--log-dir` parameters are set, please modify it to the actual path you have configured:

```bash
docker exec -it gpustack tail -200f /var/lib/gpustack/log/rpc_server/gpu-0.log
```

Each GPU corresponds to an RPC Server. For other GPU indices, modify it to the actual index:

```bash
docker exec -it gpustack tail -200f /var/lib/gpustack/log/rpc_server/gpu-n.log
```

### Where are the model logs stored?

The model instance logs are stored in the `/var/lib/gpustack/log/serve/` directory of the corresponding worker node or worker container, with the log file named `id.log`, where id is the model instance ID. If the `--data-dir` or `--log-dir` parameter is set, the logs will be stored in the actual path specified by the parameter.

### How can I enable the backend debug mode?

**llama-box backend (GGUF models)**

Add the `--verbose` parameter in `Edit Model` → `Advanced` → `Backend Parameters` and recreate the model instance:

![enable-llama-box-debug-mode](assets/faq/enable-llama-box-debug-mode.png)

**vLLM backends (Safetensors models)**

Add the `VLLM_LOGGING_LEVEL=DEBUG` environment variable in `Edit Model` → `Advanced` → `Environment Variables` and recreate the model instance:

![enable-vllm-debug-mode](assets/faq/enable-vllm-debug-mode.png)

## Managing Workers

### What should I do if the worker is stuck in `Unreachable` state?

Try accessing the URL shown in the error from the server. If the server is running in container, you need to enter the server container to execute the command:

```
curl http://10.10.10.1:10150/healthz
```

### What should I do if the worker is stuck in `NotReady` state?

Check the GPUStack logs on the corresponding worker [here](troubleshooting.md#view-gpustack-logs). If there are no abnormalities in the logs, verify that the time zones and system clocks are consistent across all nodes.

## Detect GPUs

### Why did it fail to detect the Ascend NPUs?

Check if `npu-smi` can be executed in the container:

```bash
docker exec -it gpustack bash
npu-smi info
```

When the following error occurs, it indicates that other containers are also mounting the NPU device, and sharing is not supported:

```bash
dcmi model initialized failed, because the device is used. ret is -8020
```

Check if any containers on the host have mounted NPU devices:

```bash
if [ $(docker ps | wc -l) -gt 1 ]; then docker ps | grep -v CONT | awk '{print $1}' | xargs docker inspect --format='{{printf "%.5s" .ID}} {{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' | sort -k2; fi; echo ok
```

Only mount NPUs that are not mounted by other containers, specify them using the `--device`:

```bash
docker run -d --name gpustack \
    --restart=unless-stopped \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    --network=host \
    --ipc=host \
    -v gpustack-data:/var/lib/gpustack \
    gpustack/gpustack:latest-npu
```

## Managing Models

### How can I deploy the model?

#### How can I deploy the model from Hugging Face?

To deploy models from Hugging Face, the server node and the worker nodes where the model instances are scheduled must have access to Hugging Face, or you can use a mirror.

For example, configure the `hf-mirror.com` mirror:

**Script Installation**

- Linux

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

**Docker Installation**

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

**pip Installation**

```
HF_ENDPOINT=https://hf-mirror.com gpustack start
```

#### How can I deploy the model from Local Path?

When deploying models from Local Path, it is recommended to **upload the model files to each node** and **maintain the same absolute path**. Alternatively, the model instance should be manually scheduled to nodes that have the model files via manual scheduling or label selection. Another option is to mount a shared storage across multiple nodes.

When deploying GGUF models from Local Path, the path **must point to the absolute path of the `.gguf` file**. For sharded model files, use the absolute path of the first `.gguf` file (00001). If using container installation, the model files must be mounted into the container, and the path **should point to the container’s path**, not the host’s path.

When deploying Safetensors models from Local Path, the path **must point to the absolute path of the model directory which contain `*.safetensors`, `config.json`, and other files**. If using container installation, the model files must be mounted into the container, and the path **should point to the container’s path**, not the host’s path.

![deploy-model-from-local-path](./assets/faq/deploy-model-from-local-path.png)

#### How can I deploy a locally downloaded Ollama model?

Use the following command to find the full path of the model file and deploy it via the Local Path. The example below uses `deepseek-r1:14b-qwen-distill-q4_K_M`, be sure to replace it with your actual Ollama model name:

```bash
ollama show deepseek-r1:14b-qwen-distill-q4_K_M --modelfile | grep FROM | grep blobs | sed 's/^FROM[[:space:]]*//'
```

![deploy-downloaded-ollama-model](assets/faq/deploy-downloaded-ollama-model.png)

### What should I do if the model is stuck in `Pending` state?

`Pending` means that there are currently no workers meeting the model’s requirements, move the mouse over the `Pending` status to view the reason.

First, check the `Resources`-`Workers` section to ensure that the worker status is Ready.

Then, for different backends:

- llama-box

llama-box uses the [GGUF Parser](https://github.com/gpustack/gguf-parser-go) to calculate the model’s memory requirements. You need to ensure that the allocatable memory is greater than the calculated memory requirements of the model. Note that even if other models are in an `Error` or `Downloading` state, the GPU memory has already been allocated. If you are unsure how much GPU memory the model requires, you can use the [GGUF Parser](https://github.com/gpustack/gguf-parser-go) to calculate it.

The context size for the model also affects the required GPU memory. You can adjust the `--ctx-size` parameter to set a smaller context. In GPUStack, if this parameter is not set, its default value is `8192`. If it is specified in the backend parameters, the actual setting will take effect.

You can adjust it to a smaller context in `Edit Model` → `Advanced` → `Backend Parameters` as needed, for example, `--ctx-size=2048`. However, keep in mind that the max tokens for each inference request is influenced by both the `--ctx-size` and `--parallel` parameters:
`max tokens = context size / parallel`

The default value of `--parallel` is `4`, so in this case, the max tokens would be `512`. If the token count exceeds the max tokens, the inference output will be truncated.

On the other hand, the `--parallel` parameter represents the number of parallel sequences to decode, which can roughly be considered as a setting for the model’s concurrent request handling.

Therefore, it is important to appropriately set the `--ctx-size` and `--parallel` parameters, ensuring that the max tokens for a single request is within the limits and that the available GPU memory can support the specified context size.

If you need to align with Ollama’s configuration, you can refer to the following examples:

Set the following parameters in `Edit Model` → `Advanced` → `Backend Parameters`:

```
--ctx-size=8192
--parallel=4

```

If your GPU memory is insufficient, try launching with a lower configuration:

```
--ctx-size=2048
--parallel=1
```

- vLLM

vLLM requires that all GPUs have more than 90% of their memory available by default (controlled by the `--gpu-memory-utilization` parameter). Ensure that there is enough allocatable GPU memory exceeding 90%. Note that even if other models are in an `Error` or `Downloading` state, the GPU memory has already been allocated.

If all GPUs have more than 90% available memory but still show `Pending`, it indicates insufficient memory. For `safetensors` models in BF16 format, the required GPU memory (GB) can be estimated as:

```
GPU Memory (GB) = Number of Parameters (B) * 2 * 1.2 + 2
```

If the allocatable GPU memory is less than 90%, but you are sure the model can run with a lower allocation, you can adjust the `--gpu-memory-utilization` parameter. For example, add `--gpu-memory-utilization=0.5` in `Edit Model` → `Advanced` → `Backend Parameters` to allocate 50% of the GPU memory.

**Note**: If the model encounters an error after running and the logs show `CUDA: out of memory`, it means the allocated GPU memory is insufficient. You will need to further adjust `--gpu-memory-utilization`, add more resources, or deploy a smaller model.

The context size for the model also affects the required GPU memory. You can adjust the `--max-model-len` parameter to set a smaller context. In GPUStack, if this parameter is not set, its default value is 8192. If it is specified in the backend parameters, the actual setting will take effect.

You can adjust it to a smaller context as needed, for example, `--max-model-len=2048`. However, keep in mind that the max tokens for each inference request cannot exceed the value of `--max-model-len`. Therefore, setting a very small context may cause inference truncation.

The `--enforce-eager` parameter also helps reduce GPU memory usage. However, this parameter in vLLM forces the model to execute in eager execution mode, meaning that operations are executed immediately as they are called, rather than being deferred for optimization in graph-based execution (like in lazy execution). This can make the execution slower but easier to debug. However, it can also reduce performance due to the lack of optimizations provided by graph execution.

### What should I do if the model is stuck in `Scheduled` state?

Try [restarting the GPUStack service](faq.md#how-can-i-manage-the-gpustack-service) where the model is scheduled. If the issue persists, check the worker logs [here](troubleshooting.md#view-gpustack-logs) to analyze the cause.

### What should I do if the model is stuck in `Error` state?

Move the mouse over the `Error` status to view the reason. If there is a `View More` button, click it to check the error messages in the model logs and analyze the cause of the error.

### How can I resolve the error \*.so: cannot open shared object file: No such file or directory?

If the error occurs during model startup indicating that any `.so` file cannot be opened, for example:

```bash
llama-box: error while loading shared libraries: libcudart.so.12: cannot open shared object file: No such file or directory
```

The cause is that GPUStack doesn’t recognize the `LD_LIBRARY_PATH` environment variable, which may be due to a missing environment variable or unconfigured toolkits (such as CUDA, CANN, etc.) during GPUStack installation.

To check if the environment variable is set:

```bash
echo $LD_LIBRARY_PATH
```

If not configured, here’s an example configuration for CUDA.

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
LD_LIBRARY_PATH=......
PATH=......
```

Save and restart GPUStack:

```bash
systemctl restart gpustack
```

### Why did it fail to load the model when using the local path?

When deploying a model using Local Path and encountering a `failed to load model` error, you need to check whether the model files exist on the node that the model instance is scheduled to, and if the absolute path is correct.

For GGUF models, you need to specify the absolute path to the `.gguf` file. For sharded models, use the absolute path to the first `.gguf` file (typically 00001).

If using Docker installation, the model files must be mounted into the container. Make sure the path you provide is the one inside the container, not the host path.

![deploy-model-from-local-path](./assets/faq/deploy-model-from-local-path.png)

### Why doesn’t deleting a model free up disk space?

This is to avoid re-downloading the model when redeploying. You need to clean it up in `Resources` → `Model Files` manually.

### Why does each GPU have a llama-box process by default?

This process is the RPC server used for llama-box’s distributed inference. If you are sure that you do not need distributed inference with llama-box, you can disable the RPC server service by adding the `--disable-rpc-servers` parameter when running GPUStack.

### Backend Parameters

#### How can I know the purpose of the backend parameters?

- [llama-box](https://github.com/gpustack/llama-box?tab=readme-ov-file#usage)

- [vLLM](https://docs.vllm.ai/en/stable/serving/openai_compatible_server.html#named-arguments)

- [MindIE](https://github.com/gpustack/gpustack/blob/main/gpustack/worker/backends/ascend_mindie.py#L50-L211)

#### How can I set the model’s context length?

**llama-box backend (GGUF models)**

GPUStack sets the default context length for models to 8K. You can customize the context length using the `--ctx-size` parameter, but it cannot exceed the model’s maximum context length:

![set-the-model-context-length-for-llama-box](./assets/faq/set-the-model-context-length-for-llama-box.png)

If editing, save the change and then recreate the model instance to take effect.

**vLLM backend (Safetensors models)**

GPUStack sets the default context length for models to 8K. You can customize the context length using the `--max-model-len` parameter, but it cannot exceed the model’s maximum context length:

![set-the-model-context-length-for-vllm](./assets/faq/set-the-model-context-length-for-vllm.png)

**MindIE backend (Safetensors models)**

GPUStack sets the default context length for models to 8K. You can customize the context length using the `--max-seq-len` parameter, but it cannot exceed the model’s maximum context length:

![set-the-model-context-length-for-mindie](./assets/faq/set-the-model-context-length-for-mindie.png)

If editing, save the change and then recreate the model instance to take effect.

## Using Models

### Using Vision Language Models

#### How can I resolve the error At most 1 image(s) may be provided in one request?

This is a limitation of vLLM. You can adjust the `--limit-mm-per-prompt` parameter in `Edit Model` → `Advanced` → `Backend Parameters` as needed. For example, `--limit-mm-per-prompt=image=4` means that it supports up to 4 images per inference request, see the details [here](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#vllm-serve).

## Managing GPUStack

### How can I manage the GPUStack service?

**Script Installation**

- Linux

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

- macOS

Stop GPUStack:

```bash
sudo launchctl bootout system /Library/LaunchDaemons/ai.gpustack.plist
```

Start GPUStack:

```bash
sudo launchctl bootstrap system /Library/LaunchDaemons/ai.gpustack.plist
```

Restart GPUStack:

```bash
sudo launchctl bootout system /Library/LaunchDaemons/ai.gpustack.plist
sudo launchctl bootstrap system /Library/LaunchDaemons/ai.gpustack.plist
```

- Windows

Run PowerShell as administrator (**avoid** using PowerShell ISE).

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

**Docker Installation**

Restart GPUStack container:

```bash
docker restart gpustack
```

### How do I use GPUStack behind a proxy?

**Script Installation**

- Linux & macOS

Create or edit `/etc/default/gpustack` and add the proxy configuration:

```bash
vim /etc/default/gpustack
```

```bash
http_proxy="http://username:password@proxy-server:port"
https_proxy="http://username:password@proxy-server:port"
all_proxy="socks5://username:password@proxy-server:port"
no_proxy="localhost,127.0.0.1,192.168.0.0/24,172.16.0.0/16,10.0.0.0/8"
```

Save and restart GPUStack:

```bash
systemctl restart gpustack
```

**Docker Installation**

Pass environment variables when running GPUStack:

```bash
docker run -e http_proxy="http://username:password@proxy-server:port" \
           -e https_proxy="http://username:password@proxy-server:port" \
           -e all_proxy="socks5://username:password@proxy-server:port" \
           -e no_proxy="localhost,127.0.0.1,192.168.0.0/24,172.16.0.0/16,10.0.0.0/8" \
           ……
```
