# Model Deployment Management

You can manage model deployments in GPUStack by navigating to the `Models - Deployments` page. A model deployment in GPUStack contains one or multiple replicas of model instances. On deployment, GPUStack automatically computes resource requirements for the model instances from model metadata and schedules them to available workers accordingly.

## Deploy Model

Currently, models from [Hugging Face](https://huggingface.co), [ModelScope](https://modelscope.cn), and local paths are supported.

### Deploying a Hugging Face Model

1. Click the `Deploy Model` button, then select `Hugging Face` in the dropdown.

2. Search the model by name from `Hugging Face` using the search bar in the top left. For example, `Qwen/Qwen3-0.6B`.

3. Adjust the `Name`, `Cluster`, `Backend`, `Backend Version`, and `Replicas` as needed.

4. Expand the `Performance` section for performance configurations if needed. Please refer to the [Performance-Related Configuration](#performance-related-configuration) section for more details.

5. Expand the `Scheduling` section for scheduling configurations if needed. Please refer to the [Scheduling Configuration](#scheduling-configuration) section for more details.

6. Expand the `Advanced` section for advanced configurations if needed. Please refer to the [Advanced Configuration](#advanced-configuration) section for more details.

7. Click the `Save` button.

### Deploying a ModelScope Model

1. Click the `Deploy Model` button, then select `ModelScope` in the dropdown.

2. Search the model by name from `ModelScope` using the search bar in the top left. For example, `Qwen/Qwen3-0.6B`.

3. Adjust the `Name`, `Cluster`, `Backend`, `Backend Version`, and `Replicas` as needed.

4. Expand the `Performance` section for performance configurations if needed. Please refer to the [Performance-Related Configuration](#performance-related-configuration) section for more details.

5. Expand the `Scheduling` section for scheduling configurations if needed. Please refer to the [Scheduling Configuration](#scheduling-configuration) section for more details.

6. Expand the `Advanced` section for advanced configurations if needed. Please refer to the [Advanced Configuration](#advanced-configuration) section for more details.

7. Click the `Save` button.

### Deploying a Local Path Model

You can deploy a model from a local path. The model path can be a directory (e.g., a downloaded Hugging Face model directory) or a file (e.g., a GGUF model file) located on workers. This is useful when running in an air-gapped environment.

!!! note

    1. GPUStack uses the model files to estimate resource requirements. If the model path is not accessible on the server, GPUStack will attempt to access it from the workers.
    2. GPUStack does not automatically synchronize model files. You must ensure the model path is accessible on the target workers (e.g., using NFS, rsync, etc.). You can also use the worker selector configuration to deploy the model to specific workers.

To deploy a local path model:

1. Click the `Deploy Model` button, then select `Local Path` in the dropdown.

2. Fill in the `Name` of the deployment.

3. Fill in the `Model Path`.

4. Adjust the `Cluster`, `Backend`, `Backend Version`, and `Replicas` as needed.

5. Expand the `Performance` section for performance configurations if needed. Please refer to the [Performance-Related Configuration](#performance-related-configuration) section for more details.

6. Expand the `Scheduling` section for scheduling configurations if needed. Please refer to the [Scheduling Configuration](#scheduling-configuration) section for more details.

7. Expand the `Advanced` section for advanced configurations if needed. Please refer to the [Advanced Configuration](#advanced-configuration) section for more details.

8. Click the `Save` button.

### Backend

Currently, GPUStack supports some built-in backends: vLLM, SGLang, MindIE and VoxBox.

For more details, please refer to the [Inference Backends](built-in-inference-backends.md) section.

### Backend Version

Select a backend version. The version availability depend on the selected backend. This option is useful for ensuring compatibility or taking advantage of features introduced in specific backend versions.

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

## Export and Import Deployments

Export deployments to a YAML file that holds their full configuration, and import that file to create them again. Typical uses: back up before an upgrade or migration, restore after reinstalling GPUStack, keep deployment configuration under version control, share a deployment template with your team, or reproduce the same deployments on another cluster.

### Export Deployments

1. Find the model deployment you want to export on the deployment list page.
2. Click the ellipsis button in the `Operations` column, then select `Export YAML`.
3. To export several deployments at once, select them first, then choose `Export YAML` from the batch actions dropdown above the list.

The browser downloads a YAML file named `<deployment-name>.yaml` for a single deployment, or `gpustack-deployments-<timestamp>.yaml` for several.

The file holds one deployment per YAML document, separated by `---`. Each one is the request body that created the deployment, minus the fields the server generates or that tie it to one environment:

```yaml
# Exported from GPUStack v2.3.0 at 2026-09-07T10:00:00Z
name: qwen3-8b
source: huggingface
huggingface_repo_id: Qwen/Qwen3-8B
replicas: 2
placement_strategy: binpack
worker_selector:
  zone: a
gpu_selector:
  gpu_ids:
  - worker-1:cuda:0
  gpus_per_replica: 1
backend: vLLM
backend_version: 0.11.0
backend_parameters:
- --max-model-len=32768
env:
  HF_TOKEN: hf_xxx
enable_model_route: true
---
name: bge-m3
source: huggingface
huggingface_repo_id: BAAI/bge-m3
replicas: 1
```

Because each deployment starts at the left margin, you can copy one out of the file, or write one by hand, without re-indenting it. Importing also accepts a file written as a single YAML list, which is how earlier GPUStack releases exported.

- **Kept**: everything you configured, including backend parameters, environment variables, speculative decoding, Extended KV Cache, the LoRA list, whether a model route is created, and the full scheduling configuration — replicas, placement strategy, CPU offloading, distributed inference, worker selector, GPU selector and scheduled scaling.
- **Dropped**: IDs, timestamps, runtime state such as ready replicas, metadata derived by the scheduler, the owning cluster and organization, the access policy, and LoRA runtime paths. The server regenerates these on import.

Where a deployment ran is never exported. A deployment scheduled automatically has no `gpu_selector` in the file at all, so importing it schedules it afresh against whatever the target cluster has available. GPUs you picked yourself are your intent rather than a scheduling result, so they are exported as written — see the note on importing into a different cluster below.

!!! warning

    Environment variables are exported exactly as you entered them and may contain credentials such as `HF_TOKEN`. Do not commit an exported file to a public repository; remove or replace sensitive values before sharing it.

### Import Deployments

1. Click the `Deploy Model` button, then select `YAML File` in the dropdown.
2. Select the target `Cluster`. The drawer reads that cluster and fills the editable side with the YAML that would reproduce what it runs today, so both sides of the diff start out identical and there is a working document to change rather than a blank page. The file carries no cluster information; every deployment is created in the cluster you pick.
3. Change that document, or import a file from the toolbar to replace it. Picking a different cluster re-reads it, unless you have already edited or imported something — that document is yours, and is re-checked against the new cluster instead.
4. Review the diff. The left-hand side is the deployment as the cluster holds it, the right-hand side is what would be imported. The list beside them starts with the whole document and then indexes each deployment in it, marked `Create`, `Update` or `Unchanged`, with a count of the fields an `Update` would change; pick an entry to edit that deployment on its own, or the first row to edit all of them at once.
5. Fix anything the list flags in red. The reason is shown above the diff, and the document is checked again as you edit it — shortly after you stop typing, or straight away when you leave the editor.
6. Click `Import`. If the plan replaces any existing deployment, confirm it once — the dialog names every deployment that would be replaced. Everything is written at once; if any one entry fails, nothing from the file is written.

`Import` stays disabled while the document would write nothing, which is how it starts: a freshly read cluster matches itself, and only what you change from it is an import.

The file itself is never uploaded: only the text in the editor travels with the import request.

The following rules apply when importing:

- **Overwriting is deliberate and narrow.** An entry whose name matches an existing deployment replaces it only if you confirm the replacement when you import, and only if that deployment is **stopped** and lives in the **target cluster**. Stopped means both scaled to zero replicas and no instances left running — scaling down returns immediately but the instances take a moment to shut down, so an import right after may ask you to wait. Edit a deployment in another cluster there instead: an import never migrates one between clusters.
- **An overwrite replaces the deployment, it does not merge into it.** The file is the desired state, so a field you delete from it goes back to its default. Removing the `gpu_selector` block is how you return a deployment to automatic scheduling. Entries the plan marks `Unchanged` are not written at all.
- **Model routes follow `enable_model_route`.** Setting it to `false` on an overwrite deletes the route that deployment created, along with its LoRA child routes. If that route also serves another deployment, the entry is rejected instead — detach the other targets first.
- **Replicas come from the file.** Edit `replicas` in the editor when the target environment is a different size from the one the file came from. With hand-picked GPUs, check that `gpu_selector` still holds enough of them for the new count, exactly as you would when editing the deployment. With scheduled scaling enabled the count sets `scaling_schedule.baseline_replicas`, since the schedule owns the replica count.
- A field GPUStack does not recognize is rejected rather than silently dropped. This usually means the file came from a newer GPUStack release; remove the field named in the error and retry.
- When importing into a different cluster, `gpu_ids` under `gpu_selector` and `worker_selector` still refer to the GPUs and workers of the original cluster. Change them to values from the target cluster, or remove them to let the scheduler place the deployment; otherwise the import fails because the GPUs cannot be found.

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

- **RAM-to-VRAM Ratio**: The ratio of system RAM to GPU VRAM used for KV cache. For example, 2.0 means the cache in RAM can be twice as large as the GPU VRAM.
- **Maximum RAM Size**: The maximum size of the KV cache stored in system memory (GiB). If set, this value overrides `RAM-to-VRAM Ratio`.
- **Size of Cache Chunks**: Number of tokens per KV cache chunk.

This feature works for certain backends and frameworks only.

#### Compatibility Matrix

| Backend | Framework  |
| ------- | ---------- |
| vLLM    | CUDA, ROCm |
| SGLang  | CUDA, ROCm |

## Scheduling Configuration

### Schedule Mode

#### Auto

GPUStack automatically schedules model instances to appropriate GPUs/Workers based on current resource availability.

- **Placement Strategy**

Spread: Make the resources of the entire cluster relatively evenly distributed among all workers. It may produce more resource fragmentation on a single worker.

Binpack: Prioritize the overall utilization of cluster resources, reducing resource fragmentation on Workers/GPUs.

- **Worker Selector**

When configured, the scheduler will deploy the model instance to the worker containing specified labels.

1. Navigate to the `Workers` page and edit the desired worker. Assign custom labels to the worker by adding them in the labels section.

2. Go to the `Deployments` page and click on the `Deploy Model` button. Expand the `Scheduling` section and input the previously assigned worker labels in the `Worker Selector` configuration. During deployment, the Model Instance will be allocated to the corresponding worker based on these labels.

#### Manual

This schedule type allows users to specify which GPU to deploy the model instance on.

- **GPU Selector**

  Select one or more GPUs from the list. The model instance will attempt to deploy to the selected GPU if resources permit.

- **GPUs per Replica**

Auto: The system automatically calculates the GPU count per replica, using powers of two by default and capped by the selected GPUs.

Manual: Select the number of GPUs each replica should use from the dropdown.

## Advanced Configuration

GPUStack supports tailored configurations for model deployment.

### Model Category

The model category helps you organize and filter models. By default, GPUStack automatically detects the model category based on the model's metadata. You can also customize the category by selecting it from the dropdown list.

### Backend Parameters

Input the parameters for the backend you want to customize when running the model. Supported parameter formats:

| Method           | Example                                            | Remarks                                                                     |
|------------------|----------------------------------------------------|-----------------------------------------------------------------------------|
| Equal Sign Split | `--hf-overrides={"architectures": ["NewModel"]}`   | -                                                                           |
| Space Split      | `--hf-overrides '{"architectures": ["NewModel"]}'` | Supports `shell-like` style splitting (e.g., for values containing spaces). |
| Separate Fields  | `--max-model-length`, `8192`                       | Input parameter name and value as two separate items.                       |

For full list of supported parameters, please refer to the [Inference Backends](built-in-inference-backends.md) section.

### Environment Variables

Environment variables used when running the model. These variables are passed to the backend process at startup.

### LoRA Adapters

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that adapts a base model to a specific domain by loading small adapter files instead of retraining the full weights. GPUStack lets you mount multiple LoRA adapters on a single deployed base LLM and automatically creates one Model Route per adapter.

!!! note

    LoRA is supported only on the `vLLM`, `SGLang`, and `Ascend MindIE` backends. The `LoRA Adapters` configuration is ignored on other backends.

Expand the `Advanced` section on the deployment form, locate `LoRA Adapters`, and add adapters one by one:

- Pick an adapter from the dropdown. The list supports search. If your adapter is not shown, first confirm on `Hugging Face` or `ModelScope` that the model actually provides one, then paste its repository ID into the search box to select it.
- `LoRA name`: Enter the bare adapter name, for example `alpaca`.

After deployment, GPUStack automatically creates a Model Route for each LoRA adapter, named `<base-model-name>:<adapter-name>`. To switch adapters, set the `model` field of the OpenAI-compatible API to the corresponding route name. All adapters share the same GPU instance. GPUStack injects backend startup arguments such as `--enable-lora` automatically, so you usually do not need to repeat them under `Backend Parameters`.

!!! note

    After you change the LoRA configuration of a deployed model, the corresponding instances must be restarted before the new configuration takes effect.

For the full end-to-end workflow, including invocation examples and per-backend compatibility details, see the [Serving Models with LoRA Adapters](../tutorials/serving-with-lora-adapters.md) tutorial.

### Allow CPU Offloading

!!! note

    Available for custom backends only.

When CPU offloading is enabled, GPUStack will allocate CPU memory if GPU resources are insufficient. You must correctly configure the inference backend to use hybrid CPU+GPU or full CPU inference.

### Allow Distributed Inference Across Workers

!!! note

    Available for vLLM, SGLang, and MindIE backends.

Enable distributed inference across multiple workers. The primary Model Instance will communicate with backend instances on one or more other workers, offloading computation tasks to them.

### Auto-Restart on Error

Enable automatic restart of the model instance if it encounters an error. This feature ensures high availability and reliability of the model instance. If an error occurs, GPUStack will automatically attempt to restart the model instance using an exponential backoff strategy. The delay between restart attempts increases exponentially, up to a maximum interval of 5 minutes. This approach prevents the system from being overwhelmed by frequent restarts in the case of persistent errors.

### Enable Generic Proxy

While it is common practice to integrate with the OpenAI compatible APIs, users may have different requirements for their use cases. GPUStack supports any inference APIs other than the OpenAI-compatible ones and make it more flexible for AI application development.

GPUStack offers two ways to address the target model when Generic Proxy is enabled. The **path-based form** is recommended; the **header-based form** is retained for backward compatibility and is deprecated.

#### Path-based form (recommended)

Append the numeric model route id to the proxy prefix — `/model/proxy/<model_route_id>/<upstream-path>` — and GPUStack dispatches the request to that route's targets. No extra header is needed.

```bash
# Assume the model route id is 42.
curl http://<server-url>/model/proxy/42/embed \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <GPUSTACK_API_KEY>" \
  -d '{"inputs":["What is Deep Learning?", "Deep Learning is not..."]}'
```

The gateway strips `/model/proxy/<id>` before forwarding, so the upstream inference server sees:

```bash
curl http://<inference-server-url>/embed \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"inputs":["What is Deep Learning?", "Deep Learning is not..."]}'
```

The model route id is stable across renames and can be found on the model detail page or retrieved from `GET /v2/model-routes`.

#### Header-based form (deprecated)

!!! warning "Deprecated"
    The `/model/proxy` + `X-GPUStack-Model` form is kept for backward compatibility and will be removed in a future release. Migrate to the path-based form above.

```bash
curl http://<server-url>/model/proxy/embed \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <GPUSTACK_API_KEY>" \
  -H "X-GPUStack-Model: bge-m3" \
  -d '{"inputs":["What is Deep Learning?", "Deep Learning is not..."]}'
```

The path prefix `/model/proxy` is stripped before forwarding. You must provide either the `X-GPUStack-Model` header or the `model` attribute in the JSON body so the gateway can resolve the target model. On the upstream inference server the request looks like:

```bash
curl http://<inference-server-url>/embed \
  -X POST \
  -H "Content-Type: application/json" \
  -H "X-GPUStack-Model: bge-m3" \
  -d '{"inputs":["What is Deep Learning?", "Deep Learning is not..."]}'
```
