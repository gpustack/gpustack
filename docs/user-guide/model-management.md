# Model Management

You can manage large language models in GPUStack by navigating to the `Models` page. A model in GPUStack contains one or multiple replicas of model instances. On deployment, GPUStack automatically computes resource requirements for the model instances from model metadata and schedules them to available workers accordingly.

## Deploy Model

Currently, models from [Hugging Face](https://huggingface.co), [ModelScope](https://modelscope.cn), [Ollama](https://ollama.com/library) and local paths are supported.

### Deploying a Hugging Face Model

1. Click the `Deploy Model` button, then select `Hugging Face` in the dropdown.

2. Search the model by name from Hugging Face using the search bar in the top left. For example, `microsoft/Phi-3-mini-4k-instruct-gguf`. If you only want to search for GGUF models, check the "GGUF" checkbox.

3. Select a file with the desired quantization format from `Available Files`.

4. Adjust the `Name` and `Replicas` as needed.

5. Expand the `Advanced` section for advanced configurations if needed. Please refer to the [Advanced Model Configuration](#advanced-model-configuration) section for more details.

6. Click the `Save` button.

### Deploying a ModelScope Model

1. Click the `Deploy Model` button, then select `ModelScope` in the dropdown.

2. Search the model by name from ModelScope using the search bar in the top left. For example, `Qwen/Qwen2-0.5B-Instruct`. If you only want to search for GGUF models, check the "GGUF" checkbox.

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

### Deploying a Local Path Model

You can deploy a model from a local path. The model path can be a directory (e.g., a downloaded Hugging Face model directory) or a file (e.g., a GGUF model file) located on workers. This is useful when running in an air-gapped environment.

!!!note

    1. GPUStack does not check the validity of the model path for scheduling, which may lead to deployment failure if the model path is inaccessible. It is recommended to ensure the model path is accessible on all workers(e.g., using NFS, rsync, etc.). You can also use the worker selector configuration to deploy the model to specific workers.
    2. GPUStack cannot evaluate the model's resource requirements unless the server has access to the same model path. Consequently, you may observe empty VRAM/RAM allocations for a deployed model. To mitigate this, it is recommended to make the model files available on the same path on the server. Alternatively, you can customize backend parameters, such as `tensor-split`, to configure how the model is distributed across the GPUs.

To deploy a local path model:

1. Click the `Deploy Model` button, then select `Local Path` in the dropdown.

2. Fill in the `Name` of the model.

3. Fill in the `Model Path`.

4. Adjust the `Replicas` as needed.

5. Expand the `Advanced` section for advanced configurations if needed. Please refer to the [Advanced Model Configuration](#advanced-model-configuration) section for more details.

6. Click the `Save` button.

## Edit Model

1. Find the model you want to edit on the model list page.
2. Click the `Edit` button in the `Operations` column.
3. Update the attributes as needed. For example, change the `Replicas` to scale up or down.
4. Click the `Save` button.

!!! note

    After editing the model, the configuration will not be applied to existing model instances. You need to delete the existing model instances. GPUStack will recreate new instances based on the updated model configuration.

## Stop Model

Stopping a model will delete all model instances and release the resources. It is equivalent to scaling down the model to zero replicas.

1. Find the model you want to stop on the model list page.
2. Click the ellipsis button in the `Operations` column, then select `Stop`.
3. Confirm the operation.

## Start Model

Starting a model is equivalent to scaling up the model to one replica.

1. Find the model you want to start on the model list page.
2. Click the ellipsis button in the `Operations` column, then select `Start`.

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

## Use Self-hosted Ollama Models

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

### Model Category

The model category helps you organize and filter models. By default, GPUStack automatically detects the model category based on the model's metadata. You can also customize the category by selecting it from the dropdown list.

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

  Select one or more GPUs from the list. The model instance will attempt to deploy to the selected GPU if resources permit.

### Backend

The inference backend. Currently, GPUStack supports three backends: llama-box, vLLM and vox-box. GPUStack automatically selects the backend based on the model's configuration.

For more details, please refer to the [Inference Backends](./inference-backends.md) section.

### Backend Version

Specify a backend version, such as `v1.0.0`. The version format and availability depend on the selected backend. This option is useful for ensuring compatibility or taking advantage of features introduced in specific backend versions. Refer to the [Pinned Backend Versions](./pinned-backend-versions.md) section for more information.

### Backend Parameters

Input the parameters for the backend you want to customize when running the model. The parameter should be in the format `--parameter=value`, `--bool-parameter` or as separate fields for `--parameter` and `value`.
For example, use `--ctx-size=8192` for llama-box.

For full list of supported parameters, please refer to the [Inference Backends](./inference-backends.md) section.

### Allow CPU Offloading

!!! note

    Available for llama-box backend only.

After enabling CPU offloading, GPUStack prioritizes loading as many layers as possible onto the GPU to optimize performance. If GPU resources are limited, some layers will be offloaded to the CPU, with full CPU inference used only when no GPU is available.

### Allow Distributed Inference Across Workers

!!! note

    Available for llama-box backend only.

Enable distributed inference across multiple workers. The primary Model Instance will communicate with backend instances on one or more other workers, offloading computation tasks to them.
