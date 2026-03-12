# Scheduler

## Summary

The scheduler is responsible for two related tasks:

- **Scale up:** select the best worker/GPU placement for new model instances.
- **Scale down:** rank existing model instances and remove the least preferred ones when replicas are reduced.

In both directions, the scheduler works in two simple stages:

1. Filter: find the candidates that can actually run the model. You can think of this as building the candidate list.
2. Score: give each candidate a score, then pick the best one for the current goal.

## Scale-Up Scheduling

### 1. Basic Worker Filtering

The scheduler first does a basic round of filtering on the worker list. This step is mainly about non-resource constraints: cluster, labels, backend compatibility, selected GPUs, and local-path availability.

The filter chain runs in this order:

1. **Cluster Filter**
   Keeps only workers in the model's target cluster.
2. **GPU Matching Filter**
   Narrows the worker set when the model explicitly selects GPUs.
3. **Label Matching Filter**
   Keeps only workers whose labels satisfy the model's label selectors.
4. **Status Filter**
   Keeps only workers in the `READY` state.
5. **Backend Framework Filter**
   Removes workers whose accelerators or runtime capabilities do not match the selected backend.
6. **Local Path Filter**
   Applies to `LOCAL_PATH` models when GPUs are explicitly selected. It removes workers where the configured model path does not exist.

Only workers that pass this basic filtering step move on to resource-based filtering.

### 2. Resource-Based Candidate Filtering

This step is also a filter. Instead of filtering by metadata or worker state, it filters by resources: can this worker or placement actually provide enough RAM/VRAM to run the model?

Resource requirements are determined differently depending on the model type:

- **GGUF models:** resource requirements are estimated with the [GGUF parser](https://github.com/gpustack/gguf-parser-go).
- **Other model types:** resource requirements are estimated by the corresponding backend, such as vLLM, SGLang, MindIE, or VoxBox.

Backend capabilities are different, so the available fallback paths are also different:

- **vLLM, SGLang, MindIE:** mainly use GPU-based placements and do not use CPU-only or partial-offload fallback paths here.
- **GGUF, custom backends, VoxBox:** Uses the [GGUF parser](https://github.com/gpustack/gguf-parser-go) to estimate the model's resource requirements.Support GPU offload, partial offload or CPU execution.
- **Custom backends, VoxBox:** Support GPU offload or CPU execution.

Candidates are then evaluated in order, and the process stops as soon as one strategy returns runnable candidates. In general, the scheduler tries:

1. **Single worker, single GPU**
   One GPU on one worker is enough to satisfy the model requirements.
2. **Single worker, multiple GPUs**
   Multiple GPUs on the same worker are used together when one GPU is not enough.
3. **Distributed inference across workers**
   GPUs across multiple workers can be used when the backend supports distributed execution.
4. **Partial offload or CPU execution**
   Used only when that model type and backend support those fallback modes.

A few details matter here:

- Resource-fit filtering stops at the first strategy that yields candidates.
- Distributed candidates can include subordinate workers in addition to the primary worker.
- Explicit GPU selections are honored both during basic worker filtering and during the resource-fit filtering step.

### 3. Candidate Scoring

Once runnable candidates are found, the scheduler scores them with a scorer chain and picks the candidate with the highest total score.

Current scale-up scoring is:

1. **Placement Scorer**
2. **Model File Locality Scorer**

The total candidate score is the sum of all enabled scorers.

#### Placement Scorer

The placement scorer is always enabled for scale-up.

- **Binpack**
  
  This strategy aims to "pack" as many model instances as possible into the fewest number of "bins" (e.g., Workers/GPUs) to optimize resource utilization. The goal is to minimize the number of bins used while maximizing resource efficiency, ensuring each bin is filled as efficiently as possible without exceeding its capacity. Model instances are placed in the bin with the least remaining space to minimize leftover capacity in each bin.
- **Spread**
  
  This strategy seeks to distribute multiple model instances across different workers as evenly as possible, improving system fault tolerance and load balancing.

Additional behavior:

- For GPU placements, VRAM pressure is weighted more heavily than RAM pressure.
- For CPU-only placements, only RAM utilization is considered.

#### Model File Locality Scorer

The model-file locality scorer was added to bias placement toward workers that already have the required model files in the `READY` state.

What it does:

- Looks up workers that already have the main model files ready.
- If speculative decoding is configured, it also looks up ready files for the draft model.
- Scores each candidate by the fraction of participating workers that already have those files.
- Gives the main model locality more weight than the draft model locality.

### 4. Final Selection

After scoring, the scheduler picks the candidate with the highest total score and assigns the model instance to that placement.

## Scale-Down Scheduling

When the desired replica count is lower than the number of existing model instances, the controller ranks current instances and deletes the lowest-ranked ones first.

Scale-down uses a separate scorer chain over existing model instances:

1. **Status Scorer**
2. **Offload Layer Scorer**
3. **Placement Scorer**

The resulting instances are sorted by score in ascending order, and the lowest-scoring instances are removed first.

### Status Scorer

The status scorer prefers healthy replicas. This makes unhealthy or not-yet-ready replicas the first candidates for removal.

### Offload Layer Scorer

For GGUF models that report `total_layers` and `offload_layers`, this scorer prefers instances with more layers already offloaded:

- Full offload gets the maximum score.
- Partial offload gets a proportional score.
- Instances without offload-layer metadata get `0` from this scorer.

### Placement Scorer in Scale-Down Mode

The same placement scorer is reused during scale-down, but it evaluates existing placements from a removal perspective instead of a placement perspective. Placement still reflects the model's `binpack` or `spread` policy, but the score is interpreted as a keep-preference, so lower placement scores are more likely to be removed first.
