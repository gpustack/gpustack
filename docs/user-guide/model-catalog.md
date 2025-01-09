# Model Catalog

The Model Catalog is an index of popular models to help you quickly find and deploy models.

## Browse Models

You can browse the Model Catalog by navigating to the `Catalog` page. You can filter models by name and category. The following screenshot shows the Model Catalog page:

![Model Catalog](../assets/model-catalog.png)

## Deploy a Model from the Catalog

You can deploy a model from the Model Catalog by clicking the model card. A model deployment configuration page will appear. You can review and customize the deployment configuration and click the `Save` button to deploy the model.

## Customize Model Catalog

You can customize the Model Catalog by providing a YAML file via GPUStack server configuration using the `--model-catalog-file` flag. It accepts either a local file path or a URL. You can refer to the built-in model catalog file [here](https://github.com/gpustack/gpustack/blob/main/gpustack/assets/model-catalog.yaml) for the schema. It contains a list of model sets, each with model metadata and templates for deployment configurations.

The following is an example model set in the model catalog file:

```yaml
- name: Qwen2.5
  description: Qwen2.5 is the latest series of Qwen large language models developed by Alibaba. Qwen2.5-instruct includes a number of instruction-tuned language models ranging from 0.5 to 72 billion parameters.
  home: https://qwenlm.github.io
  icon: /static/catalog_icons/qwen.png
  categories:
    - llm
  capabilities:
    - context/128k
    - tools
  sizes: [0.5, 1.5, 3, 7, 14, 32, 72]
  licenses:
    - apache-2.0
    - qwen-research
  release_date: "2024-09-19"
  order: 1
  templates:
    - quantizations: *default_f16_quantizations
      source: huggingface
      huggingface_repo_id: bartowski/Qwen2.5-{size}B-Instruct-GGUF
      huggingface_filename: "*-{quantization}*.gguf"
      replicas: 1
      backend: llama-box
      cpu_offloading: true
      distributed_inference_across_workers: true
    - quantizations: ["BF16"]
      source: huggingface
      huggingface_repo_id: Qwen/Qwen2.5-{size}B-Instruct
      replicas: 1
      backend: vllm
      backend_parameters:
        - --enable-auto-tool-choice
        - --tool-call-parser=hermes
```

### Template Variables

The following template variables are available for the deployment configuration:

- `{size}`: Model size in billion parameters.
- `{quantization}`: Quantization method of the model.
- `{data_dir}`: GPUStack data directory path.
