# Compatibility Check

GPUStack performs a compatibility check prior to model deployment. This check provides detailed information about the model’s compatibility with the current GPUStack environment. The following compatibility checks are performed:

## Inference Backend Compatibility

Checks whether the selected inference backend is compatible with the current environment, including operating system, GPU, and architecture.

## Model Compatibility

Determines whether the model is supported by the selected inference backend. This includes checking for supported model formats and architectures (e.g., `LlamaForCausalLM`, `Qwen3ForCausalLM`, etc.). This check is based on built-in inference backends and their supported models. If a custom backend or backend version is used, this check will be skipped.

## Schedulability Check

Evaluates whether the model can be scheduled in the current environment. This includes verifying available resources such as RAM and VRAM, as well as configured scheduling rules.

### Scheduling Rules

Scheduling rules (including worker selectors, GPU selectors, and scheduling policies) are used to determine whether a model can be scheduled in the current environment.

### Resource Check

The resource check ensures that sufficient system resources are available to deploy the model. GPUStack estimates the required resources and compares them with available resources in the environment. Estimations are performed using the following methods:

1. **For GGUF models**: GPUStack uses the [GGUF parser](https://github.com/gpustack/gguf-parser-go) to estimate the model's resource requirements.
2. **For other models**: GPUStack estimates VRAM usage using the following formula:

$$
\text{VRAM} = \text{WEIGHT\_SIZE} \times 1.2 + \text{FRAMEWORK\_FOOTPRINT}
$$

- `WEIGHT_SIZE` refers to the size of the model weights in bytes.
- `FRAMEWORK_FOOTPRINT` is a constant representing the framework’s memory overhead. For example, vLLM may use several gigabytes of VRAM for CUDA graphs.
- The 1.2x multiplier is an empirical estimate. For more details, refer to [this explanation](https://blog.eleuther.ai/transformer-math/#total-inference-memory).

This formula provides a rough estimate and may not be accurate for all models. Typically, it reflects a lower-bound estimate of the required VRAM. If the estimation is insufficient, users can perform fine-grained scheduling by manually selecting workers and GPUs, or by adjusting advanced backend parameters. For instance, with vLLM, users can specify `--tensor-parallel-size` and `--pipeline-parallel-size` to control GPU allocation for the model.

#### Offloaded lookup tables

!!! note

    Automatic lookup-table estimation currently covers only supported vLLM and SGLang architectures, formats, and offload settings. Native MindIE, custom backends, and vLLM Ascend paths that have not been adapted are not automatically covered. If automatic estimates do not meet your deployment expectations, you can adjust them through [custom resource requirements](#custom-resource-requirements).

For supported models with CPU-offloaded lookup tables, such as Engram and PLE, GPUStack estimates the table sizes from model configuration and precision metadata. It subtracts the offloaded bytes from `WEIGHT_SIZE` before applying the VRAM formula above, and reserves RAM for the tables on each participating Worker in addition to any local extended KV cache RAM.

The RAM estimate assumes that local data parallel replicas share table storage and accounts for independent copies required by the parallel layout, without extra headroom. Actual memory usage can be higher. If the table size or offload behavior cannot be determined, GPUStack retains the full-weight VRAM estimate.

#### Custom resource requirements

If automatic estimates do not match your deployment's requirements, use the model's [resource claim environment variables](../environment-variables.md#model-deployment-configuration) to declare RAM and VRAM requirements in bytes. The RAM declaration replaces the total per-Worker estimate, including table and local cache memory. These declarations guide scheduling; they do not configure the backend's actual memory usage. RAM and VRAM can be declared independently. Any undeclared resource retains its backend-specific estimate.

For example, to declare 64 GiB of RAM per Worker and 32 GiB of total model VRAM, add the following model environment variables (`1 GiB = 1024^3 bytes`). The VRAM value is the model's estimated requirement across its GPUs, not a per-GPU reservation.

```text
GPUSTACK_MODEL_RAM_CLAIM=68719476736
GPUSTACK_MODEL_VRAM_CLAIM=34359738368
```
