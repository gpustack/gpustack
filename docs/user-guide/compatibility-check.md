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
