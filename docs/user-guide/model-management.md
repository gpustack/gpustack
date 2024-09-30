# Model Management

You can manage large language models in GPUStack by navigating to the `Models` page. A model in GPUStack contains one or multiple replicas of model instances. On deployment, GPUStack automatically computes resource requirements for the model instances from model metadata and schedules them to available workers accordingly.

## Deploy Model

Currently, models from [Hugging Face](https://huggingface.co) and [Ollama](https://ollama.com/library) in [GGUF format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) are supported.

### Deploying a Hugging Face Model

1. Click the `Deploy Model` button, then select `Hugging Face` in the dropdown.

2. Search the model by name from Hugging Face using the search bar in the top left. For example, `microsoft/Phi-3-mini-4k-instruct-gguf`. If you only want to search for GGUF models, check the "GGUF" checkbox.

3. Select a file with the desired quantization format from `Available Files`.

4. Adjust the `Name` and `Replicas` as needed.

5. Expand the `Advanced` section for advanced configurations if needed. Please refer to the [Advanced Model Configuration](#advanced-model-configuration) section for more details.

6. Click the `Save` button.

### Deploying an Ollama Model

1. Click the `Deploy Model` button, then select `Ollama Library` in the dropdown.

2. Fill in the `Name` of the model.

3. Select an `Ollama Model` from the dropdown list, or input any Ollama model you need. For example, `llama3`, `llama3:70b` or `youraccount/llama3:70b`.

4. Adjust the `Replicas` as needed.

5. Expand the `Advanced` section for advanced configurations if needed. Please refer to the [Advanced Model Configuration](#advanced-model-configuration) section for more details.

6. Click the `Save` button.

### Deploying an ModelScope Model

1. Click the `Deploy Model` button, then select `ModelScope` in the dropdown.

2. Search the model by name from ModelScope using the search bar in the top left. For example, `Qwen/Qwen2-0.5B-Instruct`. If you only want to search for GGUF models, check the "GGUF" checkbox.

3. Select a file with the desired quantization format from `Available Files`.

4. Adjust the `Name` and `Replicas` as needed.

5. Expand the `Advanced` section for advanced configurations if needed. Please refer to the [Advanced Model Configuration](#advanced-model-configuration) section for more details.

6. Click the `Save` button.

## Edit Model

1. Find the model you want to edit on the model list page.
2. Click the `Edit` button in the `Operations` column.
3. Update the attributes as needed. For example, change the `Replicas` to scale up or down.
4. Click the `Save` button.

## Delete Model

1. Find the model you want to delete on the model list page.
2. Click the ellipsis button in the `Operations` column, then select `Delete`.
3. Confirm the deletion.

## View Model Instance

1. Find the model you want to check on the model list page.
2. Click the `>` symbol to view the instance list of the model.

## Delete Model Instance

1. Find the model you want to check on the model list page.
2. Click the `>` symbol to view the instance list of the model.
3. Find the model instance you want to delete.
4. Click the ellipsis button for the model instance in the `Operations` column, then select `Delete`.
5. Confirm the deletion.

!!! note

    After a model instance is deleted, GPUStack will recreate a new instance to satisfy the expected replicas of the model if necessary.

## View Model Instance Logs

1. Find the model you want to check on the model list page.
2. Click the `>` symbol to view the instance list of the model.
3. Find the model instance you want to check.
4. Click the `View Logs` button for the model instance in the `Operations` column.

## Use Self-hosted Model

You can deploy self-hosted Ollama models by configuring the `--ollama-library-base-url` option in the GPUStack server. The `Ollama Library` URL should point to the base URL of the Ollama model registry. For example, `https://registry.mycompany.com`.

Here is an example workflow to set up a registry, publish a model, and use it in GPUStack:

```bash
# Run a self-hosted OCI registry
docker run -d -p 5001:5000 --name registry registry:2

# Push a model to the registry using Ollama
ollama pull llama3
ollama cp llama3 localhost:5001/library/llama3
ollama push localhost:5001/library/llama3 --insecure

# Start GPUStack server with the custom Ollama library URL
curl -sfL https://get.gpustack.ai | sh -s - --ollama-library-base-url http://localhost:5001
```

That's it! You can now deploy the model `llama3` from `Ollama Library` source in GPUStack as usual, but the model will now be fetched from the self-hosted registry.

## Advanced Model Configuration

GPUStack supports tailored configurations for model deployment.

### Schedule Type

#### Auto

GPUStack automatically schedules model instances to appropriate GPUs/Workers based on current resource availability.

- Placement Strategy

    - Spread: Make the resources of the entire cluster relatively evenly distributed among all workers. It may produce more resource fragmentation on a single worker.

    - Binpack: Prioritize the overall utilization of cluster resources, reducing resource fragmentation on Workers/GPUs.

- Worker Selector

    When configured, the scheduler will deploy the model instance to the worker containing specified labels.

    1. Navigate to the `Resources` page and edit the desired worker. Assign custom labels to the worker by adding them in the labels section.

    2. Go to the `Models` page and click on the `Deploy Model` button. Expand the `Advanced` section and input the previously assigned worker labels in the `Worker Selector` configuration. During deployment, the Model Instance will be allocated to the corresponding worker based on these labels.

#### Manual

This schedule type allows users to specify which GPU to deploy the model instance on.

- GPU Selector

    Select a GPU from the list. The model instance will attempt to deploy to this GPU if resources permit.

### Backend

#### llama-box

[llama-box](https://github.com/gpustack/llama-box) is a LLM inference engine implementation based on llama.cpp. This backend supports deploying GGUF model files, CPU offloading and distributed inference across workers.

- Backend Parameters

    Input the paramters for llama-box you want to customize while running llama-box backend, for example `--ctx-size=8192`. You can see the list of supported parameters [here](https://github.com/gpustack/llama-box).

- Allow CPU Offloading

    After enabling CPU offloading, GPUStack prioritizes loading as many layers as possible onto the GPU to maximize performance. If GPU resources are limited, some layers will be offloaded to the CPU, with full CPU inference used only when no GPU is available.

- Allow Distributed Inference Across Workers

    Enable distributed inference across multiple workers. The primary Model Instance will communicate with backend instances on one or more others workers, offloading computation tasks to them.

#### vLLM

[vLLM](https://github.com/vllm-project/vllm) is a high-throughput and memory-efficient LLMs inference engine. vLLM seamlessly supports most popular open-source models, including: Transformer-like LLMs (e.g., Llama), Mixture-of-Expert LLMs (e.g., Mixtral), Embedding Models (e.g. E5-Mistral), Multi-modal LLMs (e.g., LLaVA)

- Backend Parameters

    Input the paramters for llama-box you want to customize while running vLLM backend, for example `--max-model-len=8192`. You can see the list of supported parameters [here](https://docs.vllm.ai/en/latest/models/engine_args.html).
