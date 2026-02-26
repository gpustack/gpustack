# Scheduler

## Summary

The scheduler's primary responsibility is to calculate the resources required by model instances and to evaluate and select the optimal workers/GPUs for model instances through a series of strategies. This ensures that model instances can run efficiently. This document provides a detailed overview of the policies and processes used by the scheduler.

## Scheduling Process

### Filtering Phase

The filtering phase aims to narrow down the available workers or GPUs to those that meet specific criteria. The main policies involved are:

- Cluster Matching Policy
- GPU Matching Policy
- Label Matching Policy
- Status Policy
- Backend Framework Matching Policy
- Resource Fit Policy

#### Cluster Matching Policy

This policy filters workers based on the cluster configuration of the model. Only those workers that belong to the specified cluster are retained for further evaluation.

#### GPU Matching Policy

This policy filters workers based on the user selected GPUs. Only workers that included the selected GPUs are retained for further evaluation.

#### Label Matching Policy

This policy filters workers based on the label selectors configured for the model. If no label selectors are defined for the model, all workers are considered. Otherwise, the system checks whether the labels of each worker node match the model's label selectors, retaining only those workers that match.

#### Status Policy

This policy filters workers based on their status, retaining only those that are in a READY state.

#### Backend Framework Matching Policy

This policy filters workers based on the backend framework required by the model (e.g., vLLM, SGLang). Only those workers with GPUs that support the specified backend framework are retained for further evaluation.

#### Resource Fit Policy

The Resource Fit Policy is a critical strategy in the scheduling system, used to filter workers or GPUs based on resource compatibility. The goal of this policy is to ensure that model instances can run on the selected workers. The Resource Fit Policy prioritizes candidates in the following order:

Resource requirements are determined based on:

- For GGUF models: Uses the [GGUF parser](https://github.com/gpustack/gguf-parser-go) to estimate the model's resource requirements.

- For other model types: Estimated by the backend (e.g., vLLM, SGLang, MindIE, VoxBox).

Backends have different capabilities:

- vLLM, SGLang, MindIE: GPU-only, no CPU or partial offload.

- Custom backends, VoxBox: Support GPU offload or CPU execution.

Candidates are evaluated in the following order, and the process stops once the first valid placement is found:

1. Single Worker, Single GPU (Full Fit)
A single GPU fully satisfies the model’s requirements.

2. Single Worker, Multiple GPUs (Full Fit)
Multiple GPUs on the same worker jointly satisfy the requirements.

3. Distributed Inference (Across Workers)
GPUs across multiple workers can be used when the backend supports distributed execution.

4. Single Worker, CPU Execution
CPU-only execution, supported only for Custom and VoxBox backends.

### Scoring Phase

The scoring phase evaluates the filtered candidates, scoring them to select the optimal deployment location. The primary strategy involved is:

- Placement Strategy Policy

#### Placement Strategy Policy

- Binpack

  This strategy aims to "pack" as many model instances as possible into the fewest number of "bins" (e.g., Workers/GPUs) to optimize resource utilization. The goal is to minimize the number of bins used while maximizing resource efficiency, ensuring each bin is filled as efficiently as possible without exceeding its capacity. Model instances are placed in the bin with the least remaining space to minimize leftover capacity in each bin.

- Spread

  This strategy seeks to distribute multiple model instances across different workers as evenly as possible, improving system fault tolerance and load balancing.
