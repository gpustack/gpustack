# Model Catalog

The Model Catalog is an index of GPUStack-tuned models.

## Browse Models

You can browse the Model Catalog by navigating to the `Catalog` page. You can filter models by name and category. The following screenshot shows the Model Catalog page:

![Model Catalog](../assets/model-catalog.png)

## Deploy a Model from the Catalog

You can deploy a model from the Model Catalog by clicking the model card. A model deployment configuration page will appear. You can review and customize the deployment configuration and click the `Save` button to deploy the model.

## Customize Model Catalog

You can customize the Model Catalog by providing a YAML file via GPUStack server configuration using the `--model-catalog-file` flag. It accepts either a local file path or a URL. You can refer to the built-in model catalog file [here](https://github.com/gpustack/gpustack/blob/main/gpustack/assets/model-catalog.yaml) for the schema.

The following is an example of a custom model catalog YAML file:

```yaml
draft_models:
- name: Qwen3-8B-EAGLE3
  algorithm: eagle3
  source: huggingface
  huggingface_repo_id: Tengyunw/qwen3_8b_eagle3
model_sets:
- name: Deepseek R1 0528
  description: DeepSeek-R1-0528 is a minor version of the DeepSeek R1 model that features enhanced reasoning depth and inference capabilities. These improvements are achieved through increased computational resources and algorithmic optimizations applied during post-training. The model delivers strong performance across a range of benchmark evaluations, including mathematics, programming, and general logic, with overall capabilities approaching those of leading models such as O3 and Gemini 2.5 Pro.
  home: https://www.deepseek.com
  icon: /static/catalog_icons/deepseek.png
  categories:
    - llm
  capabilities:
    - context/128K
  size: 671
  licenses:
    - mit
  release_date: "2025-05-28"
  specs:
    - mode: throughput
      quantization: FP8
      gpu_filters:
        vendor: nvidia
        compute_capability: ">=9.0" # Hopper or later
      source: huggingface
      huggingface_repo_id: deepseek-ai/DeepSeek-R1-0528
      backend: SGLang
      backend_parameters:
        - --enable-dp-attention
        - --context-length=32768
    - mode: standard
      quantization: FP8
      source: huggingface
      huggingface_repo_id: deepseek-ai/DeepSeek-R1-0528
      backend: vLLM
      backend_parameters:
        - --max-model-len=32768
```

### Using Model Catalog in Air-Gapped Environments

The built-in model catalog sources models from either Hugging Face or ModelScope. If you are using GPUStack in an air-gapped environment without internet access, you can customize the model catalog to use a local-path model source. Here is an example:

```yaml
model_sets:
- name: Deepseek R1 0528
  description: DeepSeek-R1-0528 is a minor version of the DeepSeek R1 model that features enhanced reasoning depth and inference capabilities. These improvements are achieved through increased computational resources and algorithmic optimizations applied during post-training. The model delivers strong performance across a range of benchmark evaluations, including mathematics, programming, and general logic, with overall capabilities approaching those of leading models such as O3 and Gemini 2.5 Pro.
  home: https://www.deepseek.com
  icon: /static/catalog_icons/deepseek.png
  categories:
    - llm
  capabilities:
    - context/128K
  size: 671
  licenses:
    - mit
  release_date: "2025-05-28"
  specs:
    - mode: throughput
      quantization: FP8
      gpu_filters:
        vendor: nvidia
        compute_capability: ">=9.0" # Hopper or later
      source: local_path
      local_path: /path/to/DeepSeek-R1-0528
      backend: SGLang
      backend_parameters:
        - --enable-dp-attention
        - --context-length=32768
    - mode: standard
      quantization: FP8
      source: local_path
      # assuming you have /path/to/DeepSeek-R1-0528 directory
      local_path: /path/to/DeepSeek-R1-0528
      backend: vLLM
      backend_parameters:
        - --max-model-len=32768
```

### Model Catalog Schema

The Model Catalog YAML file contains two main sections: `draft_models` and `model_sets`.

- `draft_models`: A list of draft models for speculative decoding.
- `model_sets`: A list of model sets that are tested and optimized.

Each draft model has the following fields:

| Field               | Type          | Description                                                                                     |
|---------------------|---------------|-------------------------------------------------------------------------------------------------|
| name                | string        | The name of the draft model.                                                                    |
| algorithm           | string        | The speculative decoding algorithm of the model. Currently, only eagle3 is supported.                                                    |
| source              | string        | The source of the model (e.g., huggingface, model_scope).                                               |
| huggingface_repo_id | string        | The Hugging Face repository ID of the model (if source is huggingface).                            |
| model_scope_model_id  | string        | The ModelScope repository ID of the model (if source is model_scope).                              | 

Each model set has the following fields:

| Field               | Type          | Description                                                                                     |
|---------------------|---------------|-------------------------------------------------------------------------------------------------|
| name                | string        | The name of the model.                                                                      |
| description         | string        | A brief description of the model.                                                           |
| home                | string        | The homepage URL of the model.                                                              |
| icon                | string        | The icon URL of the model.                                                                  |
| categories          | list of str   | A list of categories that the model belongs to.                                                  |
| capabilities        | list of str   | A list of capabilities of the model.                                                       |
| size                | int           | The size of the model in billions of parameters.                                                                 |
| licenses            | list of str   | A list of licenses of the model.                                                              |
| release_date        | string        | The release date of the model in YYYY-MM-DD format.                                         |
| specs               | list of spec  | A list of deployment specifications for the model. |


Each deployment spec has the following fields:

| Field               | Type          | Description                                                                                     |
|---------------------|---------------|-------------------------------------------------------------------------------------------------|
| mode                | string        | GPUStack provides both conventional and optimized modes for different use cases, including throughput, latency, and standard scenarios. Users can also define custom modes as needed. |
| quantization        | string        | The quantization type (e.g., FP16, FP8, INT8).                                               |
| gpu_filters         | dict          | GPU filters to specify compatible GPUs.                                                  |

Other fields in a deployment spec are similar to the models API fields. For more details, see the [API Reference](../api-reference.md) documentation.
