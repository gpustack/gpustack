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

The file holds one deployment per YAML document, separated by `---`. Each one is the request body that created the deployment, plus the cluster it belongs to, minus the fields the server generates:

```yaml
# Exported from GPUStack v2.3.0 at 2026-09-07T10:00:00Z
name: qwen3-8b
cluster_name: production
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
cluster_name: staging
source: huggingface
huggingface_repo_id: BAAI/bge-m3
replicas: 1
```

Because each deployment starts at the left margin, you can copy one out of the file, or write one by hand, without re-indenting it. Importing also accepts a file written as a single YAML list, which is how earlier GPUStack releases exported.

#### Field Reference

Every field in the file is something you set on the deploy form. Below is what each one is called in either place, in the order an exported file writes them. What a field means is documented where that setting is, linked from the right-hand column.

| Field | On the deploy form |
| --- | --- |
| `name` | `Name` |
| `cluster_name` | `Cluster`, written as the cluster's name rather than its ID |
| `source` | `Source` |
| `huggingface_repo_id`, `huggingface_filename` | `Repo ID` and `File Name`, for a Hugging Face model |
| `model_scope_model_id`, `model_scope_file_path` | `Repo ID` and `File Name`, for a ModelScope model |
| `local_path` | `Model Path`, for a local path model |
| `replicas` | `Replicas` |
| `categories` | [`Model Category`](#model-category) |
| `placement_strategy` | `Scheduling` → [`Auto`](#auto), the placement strategy |
| `cpu_offloading` | [`Allow CPU Offloading`](#allow-cpu-offloading) |
| `distributed_inference_across_workers` | [`Allow Distributed Inference Across Workers`](#allow-distributed-inference-across-workers) |
| `worker_selector` | `Scheduling` → [`Manual`](#manual) → `Selector` |
| `gpu_selector` | `Scheduling` → `Specify GPU` |
| `gpu_type_selector` | `Scheduling` → `Specify GPU Type` |
| `backend` | [`Backend`](#backend) |
| `backend_version` | [`Backend Version`](#backend-version) |
| `backend_parameters` | [`Backend Parameters`](#backend-parameters) |
| `image_name`, `run_command` | The image and start command of a custom backend |
| `native_anthropic_api` | `Native Anthropic API` |
| `env` | [`Environment Variables`](#environment-variables) |
| `restart_on_error` | [`Auto-Restart On Error`](#auto-restart-on-error) |
| `extended_kv_cache` | [`Extended KV Cache`](#extended-kv-cache) |
| `speculative_config` | `Performance` → speculative decoding |
| `scaling_schedule` | `Scheduled Scaling` |
| `generic_proxy` | [`Enable Generic Proxy`](#enable-generic-proxy) |
| `lora_list` | [`LoRA Adapters`](#lora-adapters) |
| `enable_model_route` | `Enable Model Route`, which the form offers only while creating a deployment |

The last group of fields — `gpu_selector`, `gpu_type_selector`, `extended_kv_cache`, `speculative_config`, `scaling_schedule` and `lora_list` — hold a block rather than a single value, and the shape of that block is not written out here. Configure the setting on a deployment, export it, and copy the block from the file: that is both quicker than transcribing it and the only way to be sure of it.

`description` and `distributable` are carried through the file but have no control on the form.

What the file leaves out: IDs, timestamps, runtime state such as ready replicas, metadata derived by the scheduler, the owning organization, the access policy, and LoRA runtime paths. The server regenerates these on import.

Selecting deployments from several clusters at once is fine: each entry records its own `cluster_name`, so one file describes them all and imports back whole. The cluster is written by name rather than by ID, so a file still restores correctly into a GPUStack that numbers its clusters differently — as a reinstall does.

A file covers one organization, however many of its clusters. Deployment names are unique within an organization rather than across them, so a selection spanning organizations — which only a platform admin viewing `All` can make — is refused; export each organization on its own.

Where a deployment ran is never exported. A deployment scheduled automatically has no `gpu_selector` in the file at all, so importing it schedules it afresh against whatever the target cluster has available. GPUs you picked yourself are your intent rather than a scheduling result, so they are exported as written — see the note on importing into a different cluster below.

!!! warning

    Environment variables are exported exactly as you entered them and may contain credentials such as `HF_TOKEN`. Do not commit an exported file to a public repository; remove or replace sensitive values before sharing it.

### Import Deployments

1. Click the `Deploy Model` button, then select `YAML File` in the dropdown.
2. Choose a `.yaml` file. The file is the starting point — nothing is read from a cluster first — and the deployments it names are looked up to fill the read-only side of the diff. Choosing a second file replaces the first outright, edits and all.
3. Decide where the deployments go. `Follow the file`, which is how the drawer starts, lands each entry in the cluster its own `cluster_name` names — that is what restores a backup spanning several clusters in one import. An entry that names no cluster falls back to the one its deployment is already in. Pick a specific cluster instead and every entry goes there, whatever the file says.
4. Review the diff. The left-hand side is the deployment as GPUStack holds it, the right-hand side is what would be imported. The list beside them starts with the whole document and then indexes each deployment in it, marked `Create`, `Update` or `Unchanged`, with a count of the fields an `Update` would change; pick an entry to edit that deployment on its own, or the first row to edit all of them at once.
5. Fix anything the list flags in red. The reason is shown above the diff, and the document is checked again as you edit it — shortly after you stop typing, or straight away when you leave the editor.
6. Click `Import`. If the plan replaces any existing deployment, confirm it once — the dialog names every deployment that would be replaced. Everything is written at once; if any one entry fails, nothing from the file is written.

`Import` stays disabled while the document would write nothing. Re-importing an untouched export is exactly that: every entry reads as `Unchanged`, and only what you change from it is an import.

The file itself is never uploaded: only the text in the editor travels with the import request.

The following rules apply when importing:

- **Overwriting is deliberate and narrow.** An entry whose name matches an existing deployment replaces it only if you confirm the replacement when you import, and only if that deployment is **stopped**. Stopped means both scaled to zero replicas and no instances left running — scaling down returns immediately but the instances take a moment to shut down, so an import right after may ask you to wait. Deployment names are unique across your whole organization, so the name in an entry always identifies exactly one deployment, in whichever cluster it happens to run.
- **The cluster is an ordinary field of the entry.** Change `cluster_name` in the editor, or pick a different cluster for the import, and the overwrite moves the deployment there. Because an overwrite only reaches a **stopped** deployment, there are no instances to migrate: the next ones are placed in whichever cluster it names by then. Remember to update `gpu_selector` and `worker_selector` to match the new cluster, or remove them.
- **An overwrite replaces the deployment, it does not merge into it.** The file is the desired state, so a field you delete from it goes back to its default. Removing the `gpu_selector` block is how you return a deployment to automatic scheduling. Entries the plan marks `Unchanged` are not written at all.
- **Model routes follow `enable_model_route`.** Setting it to `false` on an overwrite deletes the route that deployment created, along with its LoRA child routes. Removing a LoRA adapter from the file deletes that adapter's route on its own, with the rest left alone. Either way, if a route being deleted also serves another deployment, the entry is rejected instead — detach the other targets first.
- **Replicas come from the file.** Edit `replicas` in the editor when the target environment is a different size from the one the file came from. With hand-picked GPUs, check that `gpu_selector` still holds enough of them for the new count, exactly as you would when editing the deployment. With scheduled scaling enabled the count sets `scaling_schedule.baseline_replicas`, since the schedule owns the replica count.
- A field GPUStack does not recognize is rejected rather than silently dropped. This usually means the file came from a newer GPUStack release; remove the field named in the error and retry.
- **A cluster this GPUStack does not have is named, not guessed at.** Importing a file from another installation, where the clusters are called something else, flags each entry whose `cluster_name` is unknown. Either rename it to a cluster that exists here, or pick a specific cluster to put everything in. An entry that names no cluster at all, for a deployment that does not exist yet, is flagged the same way — there is nowhere to put it.
- **The file says which organization it lands in.** A platform admin viewing `All` has no organization selected, so `Follow the file` reads one off the clusters the file names. A file naming clusters from two organizations, or none this GPUStack has, is refused rather than defaulting anywhere — pick a specific cluster for the import, or switch to the organization first.
- When an entry lands in a different cluster from the one it was exported from, `gpu_ids` under `gpu_selector` and `worker_selector` still refer to the GPUs and workers of the original cluster. Change them to values from the new cluster, or remove them to let the scheduler place the deployment; otherwise the import fails because the GPUs cannot be found.

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
