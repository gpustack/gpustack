# Model Deployment Management

You can manage model deployments in GPUStack by navigating to the `Models - Deployments` page. A model deployment in GPUStack contains one or multiple replicas of model instances. On deployment, GPUStack automatically computes resource requirements for the model instances from model metadata and schedules them to available workers accordingly.

## Deploy Model

Currently, models from [Hugging Face](https://huggingface.co), [ModelScope](https://modelscope.cn), and local paths are supported.

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

### Deploying a Local Path Model

You can deploy a model from a local path. The model path can be a directory (e.g., a downloaded Hugging Face model directory) or a file (e.g., a GGUF model file) located on workers. This is useful when running in an air-gapped environment.

!!!note

    1. GPUStack does not check the validity of the model path for scheduling, which may lead to deployment failure if the model path is inaccessible. It is recommended to ensure the model path is accessible on all workers(e.g., using NFS, rsync, etc.). You can also use the worker selector configuration to deploy the model to specific workers.
    2. GPUStack cannot evaluate the model's resource requirements unless the server has access to the same model path. Consequently, you may observe empty VRAM/RAM allocations for a deployed model. To mitigate this, it is recommended to make the model files available on the same path on the server. Alternatively, you can customize backend parameters, such as `tensor-split`, to configure how the model is distributed across the GPUs.

To deploy a local path model:

1. Click the `Deploy Model` button, then select `Local Path` in the dropdown.

2. Fill in the `Name` of the deployment.

3. Fill in the `Model Path`.

4. Adjust the `Replicas` as needed.

5. Expand the `Advanced` section for advanced configurations if needed. Please refer to the [Advanced Model Configuration](#advanced-model-configuration) section for more details.

6. Click the `Save` button.

## Edit Model Deployment

1. Find the model deployment you want to edit on the deployment list page.
2. Click the `Edit` button in the `Operations` column.
3. Update the attributes as needed. For example, change the `Replicas` to scale up or down.
4. Click the `Save` button.

!!! note

    After editing the model deployment, the configuration will not be applied to existing model instances. You need to delete the existing model instances. GPUStack will recreate new instances based on the updated model configuration.

## Stop Model Deployment

Stopping a model deployment will delete all model instances and release the resources. It is equivalent to scaling down the model to zero replicas.

1. Find the model deployment you want to stop on the deployment list page.
2. Click the ellipsis button in the `Operations` column, then select `Stop`.
3. Confirm the operation.

## Start Model Deployment

Starting a model deployment is equivalent to scaling up the model to one replica.

1. Find the model deployment you want to start on the deployment list page.
2. Click the ellipsis button in the `Operations` column, then select `Start`.

## Delete Model Deployment

1. Find the model deployment you want to delete on the deployment list page.
2. Click the ellipsis button in the `Operations` column, then select `Delete`.
3. Confirm the deletion.

## View Model Instance

1. Find the model deployment you want to check on the deployment list page.
2. Click the `>` symbol to view the instance list of the deployment.

## Delete Model Instance

1. Find the model deployment you want to check on the deployment list page.
2. Click the `>` symbol to view the instance list of the deployment.
3. Find the model instance you want to delete.
4. Click the ellipsis button for the model instance in the `Operations` column, then select `Delete`.
5. Confirm the deletion.

!!! note

    After a model instance is deleted, GPUStack will recreate a new instance to satisfy the expected replicas of the deployment if necessary.

## View Model Instance Logs

1. Find the model deployment you want to check on the deployment list page.
2. Click the `>` symbol to view the instance list of the deployment.
3. Find the model instance you want to check.
4. Click the `View Logs` button for the model instance in the `Operations` column.

## Performance-Related Configuration

GPUStack provides the following configuration options to optimize model inference performance.

### Extended KV Cache

You can enable extended KV cache to offload the KV cache to CPU memory or remote storage. This feature is particularly useful for setups with limited GPU memory requiring long context lengths. Under the hood, GPUStack leverages [LMCache](https://github.com/LMCache/LMCache) to provide this functionality.

Available options:

- **Maximum CPU Cache Size**: The maximum size of the KV cache (in GiB) that can be offloaded to CPU memory.
- **Size of Cache Chunks**: Number of tokens per KV cache chunk.
- **Remote Storage URL**: The remote storage URL for offloading KV cache. Format: `protocol://host:port`. e.g., `redis://your-redis-server:6379`. For more details, please refer to the [LMCache documentation](https://docs.lmcache.ai/).

This feature works for certain backends and frameworks only.

#### Compatibility Matrix

| Backend    | Framework  |
|------------|------------|
| vLLM       | CUDA, ROCm |

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

  1. Navigate to the `Workers` page and edit the desired worker. Assign custom labels to the worker by adding them in the labels section.

  2. Go to the `Deployments` page and click on the `Deploy Model` button. Expand the `Advanced` section and input the previously assigned worker labels in the `Worker Selector` configuration. During deployment, the Model Instance will be allocated to the corresponding worker based on these labels.

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

### Environment Variables

Environment variables used when running the model. These variables are passed to the backend process at startup.

### Allow CPU Offloading

!!! note

    Available for llama-box backend only.

After enabling CPU offloading, GPUStack prioritizes loading as many layers as possible onto the GPU to optimize performance. If GPU resources are limited, some layers will be offloaded to the CPU, with full CPU inference used only when no GPU is available.

### Allow Distributed Inference Across Workers

!!! note

    Available for llama-box, vLLM and MindIE backends.

Enable distributed inference across multiple workers. The primary Model Instance will communicate with backend instances on one or more other workers, offloading computation tasks to them.

### Auto-Restrat on Error

Enable automatic restart of the model instance if it encounters an error. This feature ensures high availability and reliability of the model instance. If an error occurs, GPUStack will automatically attempt to restart the model instance using an exponential backoff strategy. The delay between restart attempts increases exponentially, up to a maximum interval of 5 minutes. This approach prevents the system from being overwhelmed by frequent restarts in the case of persistent errors.
