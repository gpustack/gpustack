# GPUStack

GPUStack aims to get you started with running LLMs and performing inference in a simple yet scalable manner.

- Supports a wide variety of hardware.
- Scales with your GPU inventory.
- A lightweight Python package with minimal dependencies and operational overhead.
- OpenAI-compatible APIs.
- User and API key management.
- GPU metrics monitoring.
- Token usage and rate metrics.

## Installation

### Linux or MacOS

GPUStack provides a script to install it as a service on systemd or launchd based systems. To install GPUStack using this method, just run:

```bash
curl -sfL https://get.gpustack.ai | sh -
```

You can add additional workers to form a GPUStack cluster by running the following command on worker nodes:

```bash
curl -sfL https://get.gpustack.ai | sh - --server-url http://myserver --token mytoken
```

The token here is a secret used for adding workers. In the default setup, you can run the following to get the token:

```bash
cat /var/lib/gpustack/token
```

### Windows

`// TODO`

### Manual Install

For manual installation or detail configurations, refer to the [installation](.docs/user-guide/installation.md) docs.

## Gettting Started

1. Run and chat with the llama3 model:

```bash
gpustack chat llama3 "tell me a joke."
```

2. Open `http://myserver` in the browser to access the GPUStack UI. Log in to GPUStack with username `admin` and the default password. You can run the following command to get the password for the default setup:

```bash
cat /var/lib/gpustack/initial_admin_password
```

3. Click `Playground` in the navigation menus. Now you can chat with the LLM in the UI playground.

// TODO add screenshot

4. Click `API Keys` in the navigation menus, then click the `New API Key` button.

5. Fill in the `Name` and click the `Save` button.

6. Copy the generated API key and save it somewhere safe. Please note that you can only see it once on creation.

7. Now you can use the API key to access the OpenAI-compatible API. For example, use curl as the following:

```bash
export GPUSTACK_API_KEY=myapikey
curl http://myserver/v1-openai/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
    "model": "llama3",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ],
    "stream": true
  }'
```

## Supported Platforms

- [x] MacOS
- [x] Linux
- [x] Windows

## Supported Accelerators

We haven't tested everything out in the wild, but we plan to support all of the following in the near future.

- [x] Apple Metal
- [x] NVIDIA CUDA
- [ ] AMD ROCm
- [ ] Intel oneAPI

## Supported Models

GPUStack uses [llama.cpp](https://github.com/ggerganov/llama.cpp) as the backend and supports large language models in [GGUF format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md). Models from the following sources are supported:

1. [Hugging Face](https://huggingface.co/)

2. [Ollama Library](https://ollama.com/library)

Here are some example models:

- [x] [LLaMA](https://huggingface.co/meta-llama)
- [x] [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1)
- [x] [Mixtral MoE](https://huggingface.co/models?search=mistral-ai/Mixtral)
- [x] [DBRX](https://huggingface.co/databricks/dbrx-instruct)
- [x] [Falcon](https://huggingface.co/models?search=tiiuae/falcon)
- [x] [Baichuan](https://huggingface.co/models?search=baichuan-inc/Baichuan)
- [x] [Aquila](https://huggingface.co/models?search=BAAI/Aquila)
- [x] [Yi](https://huggingface.co/models?search=01-ai/Yi)
- [x] [StableLM](https://huggingface.co/stabilityai)
- [x] [Deepseek](https://huggingface.co/models?search=deepseek-ai/deepseek)
- [x] [Qwen](https://huggingface.co/models?search=Qwen/Qwen)
- [x] [Phi](https://huggingface.co/models?search=microsoft/phi)
- [x] [Gemma](https://ai.google.dev/gemma)
- [x] [Mamba](https://github.com/state-spaces/mamba)
- [x] [Grok-1](https://huggingface.co/keyfan/grok-1-hf)

## OpenAI-Compatible APIs

GPUStack serves the following OpenAI compatible APIs under the `/v1-openai` path:

1. List models
2. Chat completions

For example, you can use the official [OpenAI Python API library](https://github.com/openai/openai-python) to consume the APIs:

```python
from openai import OpenAI
client = OpenAI(base_url="http://myserver/v1-openai",api_key="myapikey")

completion = client.chat.completions.create(
  model="llama3",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)
```

GPUStack users can generate their own API keys in the UI.

## Build

1. Install `python 3.10+`.

2. Run `make build`.

You can find the built wheel package in `dist` directory.

## Contributing

Please read the [Contributing Guide](./docs/contributing.md) if you're interested in contributing to GPUStack.

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
