# Scheduler

## Summary

The scheduler's primary responsibility is to calculate the resources required by models instance and to evaluate and select the optimal workers/GPUs for model instances through a series of strategies. This ensures that model instances can run efficiently. This document provides a detailed overview of the policies and processes used by the scheduler.

## Scheduling Process

### Filtering Phase

The filtering phase aims to narrow down the available workers or GPUs to those that meet specific criteria. The main policies involved are:

- Label Matching Policy
- Status Policy
- Resource Fit Policy

#### Label Matching Policy

This policy filters workers based on the label selectors configured for the model. If no label selectors are defined for the model, all workers are considered. Otherwise, the system checks whether the labels of each worker node match the model's label selectors, retaining only those workers that match.

#### Status Policy

This policy filters workers based on their status, retaining only those that are in a READY state.

#### Resource Fit Policy

The Resource Fit Policy is a critical strategy in the scheduling system, used to filter workers or GPUs based on resource compatibility. The goal of this policy is to ensure that model instances can run on the selected nodes without exceeding resource limits. The Resource Fit Policy prioritizes candidates in the following order:

- Single Worker Node, Single GPU Full Offload: Identifies candidates where a single GPU on a single worker can fully offload the model, which usually offers the best performance.
- Single Worker Node, Multiple GPU Full Offload: Identifies candidates where multiple GPUs on a single worker can fully the offload the model.
- Distributed Inference Across Multiple Workers: Identifies candidates where a combination of GPUs across multiple workers can handle full or partial offloading, used only when distributed inference across nodes is permitted.
- Single Worker Node Partial Offload: Identifies candidates on a single worker that can handle a partial offload, used only when partial offloading is allowed.
- Single Worker Node, CPU: When no GPUs are available, the system will use the CPU for inference, identifying candidates where memory resources on a single worker are sufficient.

### Scoring Phase

The scoring phase evaluates the filtered candidates, scoring them to select the optimal deployment location. The primary strategy involved is:

- Placement Strategy Policy

#### Placement Strategy Policy

- Binpack

  This strategy aims to "pack" as many model instances as possible into the fewest number of "bins" (e.g., Workers/GPUs) to optimize resource utilization. The goal is to minimize the number of bins used while maximizing resource efficiency, ensuring each bin is filled as efficiently as possible without exceeding its capacity. Model instances are placed in the bin with the least remaining space to minimize leftover capacity in each bin.

- Spread

  This strategy seeks to distribute multiple model instances across different worker nodes as evenly as possible, improving system fault tolerance and load balancing.
