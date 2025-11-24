<br>

<p align="center">
    <img alt="GPUStack" src="https://raw.githubusercontent.com/gpustack/gpustack/main/docs/assets/gpustack-logo.png" width="300px"/>
</p>
<br>

<p align="center">
    <a href="https://docs.gpustack.ai" target="_blank">
        <img alt="Documentation" src="https://img.shields.io/badge/Docs-GPUStack-blue?logo=readthedocs&logoColor=white"></a>
    <a href="./LICENSE" target="_blank">
        <img alt="License" src="https://img.shields.io/github/license/gpustack/gpustack?logo=github&logoColor=white&label=License&color=blue"></a>
    <a href="./docs/assets/wechat-group-qrcode.jpg" target="_blank">
        <img alt="WeChat" src="https://img.shields.io/badge/微信群-GPUStack-blue?logo=wechat&logoColor=white"></a>
    <a href="https://discord.gg/VXYJzuaqwD" target="_blank">
        <img alt="Discord" src="https://img.shields.io/badge/Discord-GPUStack-blue?logo=discord&logoColor=white"></a>
    <a href="https://twitter.com/intent/follow?screen_name=gpustack_ai" target="_blank">
        <img alt="Follow on X(Twitter)" src="https://img.shields.io/twitter/follow/gpustack_ai?logo=X"></a>
</p>
<br>

<p align="center">
  <a href="./README.md">English</a> |
  <a href="./README_CN.md">简体中文</a> |
  <a href="./README_JP.md">日本語</a>
</p>

<br>

GPUStack is an open-source GPU cluster manager for running AI models.

### Key Features

- **High Performance:** Optimized for high-throughput and low-latency inference.
- **GPU Cluster Management:** Efficiently manage multiple GPU clusters across different providers, including Docker-based, Kubernetes, and cloud platforms such as DigitalOcean.
- **Broad GPU Compatibility:** Seamless support for GPUs from various vendors.
- **Extensive Model Support:** Supports a wide range of models, including LLMs, VLMs, image models, audio models, embedding models, and rerank models.
- **Flexible Inference Backends:** Built-in support for fast inference engines such as vLLM and SGLang, with the ability to integrate custom backends.
- **Multi-Version Backend Support:** Run multiple versions of inference backends concurrently to meet diverse runtime requirements.
- **Distributed Inference:** Supports single-node and multi-node, multi-GPU inference, including heterogeneous GPUs across vendors and environments.
- **Scalable GPU Architecture:** Easily scale by adding more GPUs, nodes, or clusters to your infrastructure.
- **Robust Model Stability:** Ensures high availability through automatic failure recovery, multi-instance redundancy, and intelligent load balancing.
- **Intelligent Deployment Evaluation:** Automatically assesses model resource requirements, backend and architecture compatibility, OS compatibility, and other deployment factors.
- **Automated Scheduling:** Dynamically allocates models based on available resources.
- **OpenAI-Compatible APIs:** Fully compatible with OpenAI API specifications for seamless integration.
- **User & API Key Management:** Simplified management of users and API keys.
- **Real-Time GPU Monitoring:** Monitor GPU performance and utilization in real time.
- **Token and Rate Metrics:** Track token usage and API request rates.

## Installation

> GPUStack now supports Linux only. For Windows, use WSL2 and avoid Docker Desktop.

If you are using NVIDIA GPUs, ensure the NVIDIA driver, [Docker](https://docs.docker.com/engine/install/) and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) are installed. Then start the GPUStack with the following command:

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    --privileged \
    --network host \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume gpustack-data:/var/lib/gpustack \
    --runtime nvidia \
    gpustack/gpustack
```

If you cannot pull images from `Docker Hub` or the download is very slow, you can use our `Quay.io` mirror by pointing your registry to `quay.io`:

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    --privileged \
    --network host \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume gpustack-data:/var/lib/gpustack \
    --runtime nvidia \
    quay.io/gpustack/gpustack \
    --system-default-container-registry quay.io
```

For more details on the installation or other GPU hardware platforms, please refer to the [Installation Requirements](installation/requirements.md).

Check the GPUStack startup logs:

```bash
sudo docker logs -f gpustack
```

After GPUStack starts, run the following command to get the default admin password:

```bash
sudo docker exec gpustack cat /var/lib/gpustack/initial_admin_password
```

Open your browser and navigate to `http://your_host_ip` to access the GPUStack UI. Use the default username `admin` and the password you retrieved above to log in.

## Deploy a Model

1. Navigate to the `Catalog` page in the GPUStack UI.

2. Select the `Qwen3 0.6B` model from the list of available models.

3. After the deployment compatibility checks pass, click the `Save` button to deploy the model.

![deploy qwen3 from catalog](docs/assets/quick-start/quick-start-qwen3.png)

4. GPUStack will start downloading the model files and deploying the model. When the deployment status shows `Running`, the model has been deployed successfully.

![model is running](docs/assets/quick-start/model-running.png)

5. Click `Playground - Chat` in the navigation menu, check that the model `qwen3-0.6b` is selected from the top-right `Model` dropdown. Now you can chat with the model in the UI playground.

![quick chat](docs/assets/quick-start/quick-chat.png)

## Use the model via API

1. Hover over the user avatar and navigate to the `API Keys` page, then click the `New API Key` button.

2. Fill in the `Name` and click the `Save` button.

3. Copy the generated API key and save it somewhere safe. Please note that you can only see it once on creation.

4. You can now use the API key to access the OpenAI-compatible API endpoints provided by GPUStack. For example, use curl as the following:

```bash
# Replace `your_api_key` and `your_gpustack_server_url`
# with your actual API key and GPUStack server URL.
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
    "model": "qwen3-0.6b",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Tell me a joke."
      }
    ],
    "stream": true
  }'
```

## Supported Accelerators

GPUStack supports a variety of General-Purpose Accelerators, including:

- [x] NVIDIA GPU
- [x] AMD GPU
- [x] Ascend NPU
- [x] Hygon DCU (Experimental)
- [x] MThreads GPU (Experimental)
- [x] Iluvatar GPU (Experimental)
- [x] MetaX GPU (Experimental)
- [x] Cambricon MLU (Experimental)

## Supported Models

GPUStack uses [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [MindIE](https://www.hiascend.com/en/software/mindie) and [vox-box](https://github.com/gpustack/vox-box) as built-in inference backends, and it also supports any custom backend that can run in a container and expose a serving API. This allows GPUStack to work with a wide range of models.

Models can come from the following sources:

1. [Hugging Face](https://huggingface.co/)

2. [ModelScope](https://modelscope.cn/)

3. Local File Path

For information on which models are supported by each built-in inference backend, please refer to the supported models section in the [Built-in Inference Backends](docs/user-guide/built-in-inference-backends.md) documentation.

## OpenAI-Compatible APIs

GPUStack serves the following OpenAI compatible APIs under the `/v1` path:

- [x] [List Models](https://platform.openai.com/docs/api-reference/models/list)
- [x] [Create Completion](https://platform.openai.com/docs/api-reference/completions/create)
- [x] [Create Chat Completion](https://platform.openai.com/docs/api-reference/chat/create)
- [x] [Create Embeddings](https://platform.openai.com/docs/api-reference/embeddings/create)
- [x] [Create Image](https://platform.openai.com/docs/api-reference/images/create)
- [x] [Create Image Edit](https://platform.openai.com/docs/api-reference/images/createEdit)
- [x] [Create Speech](https://platform.openai.com/docs/api-reference/audio/createSpeech)
- [x] [Create Transcription](https://platform.openai.com/docs/api-reference/audio/createTranscription)

For example, you can use the official [OpenAI Python API library](https://github.com/openai/openai-python) to consume the APIs:

```python
from openai import OpenAI
client = OpenAI(base_url="http://your_gpustack_server_url/v1", api_key="your_api_key")

completion = client.chat.completions.create(
  model="llama3.2",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)
```

GPUStack users can generate their own API keys in the UI.

## Documentation

Please see the [official docs site](https://docs.gpustack.ai) for complete documentation.

## Build

1. Install Python (version 3.10 to 3.12).

2. Run `make build`.

You can find the built wheel package in `dist` directory.

## Contributing

Please read the [Contributing Guide](./docs/contributing.md) if you're interested in contributing to GPUStack.

## Join Community

Any issues or have suggestions, feel free to join our [Community](https://discord.gg/VXYJzuaqwD) for support.

## License

Copyright (c) 2024 The GPUStack authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at [LICENSE](./LICENSE) file for details.

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
