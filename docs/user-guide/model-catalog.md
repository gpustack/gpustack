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
- name: Llama3.2
  description: The Llama 3.2 collection of multilingual large language models (LLMs) is a collection of pretrained and instruction-tuned generative models in 1B and 3B sizes (text in/text out). The Llama 3.2 instruction-tuned text only models are optimized for multilingual dialogue use cases, including agentic retrieval and summarization tasks. They outperform many of the available open source and closed chat models on common industry benchmarks.
  home: https://www.llama.com/
  icon: /static/catalog_icons/meta.png
  categories:
    - llm
  capabilities:
    - context/128k
    - tools
  sizes:
    - 1
    - 3
  licenses:
    - llama3.2
  release_date: "2024-09-25"
  order: 2
  templates:
    - quantizations:
        - Q3_K_L
        - Q4_K_M
        - Q5_K_M
        - Q6_K_L
        - Q8_0
        - f16
      source: huggingface
      huggingface_repo_id: bartowski/Llama-3.2-{size}B-Instruct-GGUF
      huggingface_filename: "*-{quantization}*.gguf"
      replicas: 1
      backend: llama-box
      cpu_offloading: true
      distributed_inference_across_workers: true
    - quantizations: ["BF16"]
      source: huggingface
      huggingface_repo_id: unsloth/Llama-3.2-{size}B-Instruct
      replicas: 1
      backend: vllm
      backend_parameters:
        - --enable-auto-tool-choice
        - --tool-call-parser=llama3_json
        - --chat-template={data_dir}/chat_templates/tool_chat_template_llama3.2_json.jinja
```

### Using Model Catalog in Air-Gapped Environments

The built-in model catalog sources models from either Hugging Face or ModelScope. If you are using GPUStack in an air-gapped environment without internet access, you can customize the model catalog to use a local-path model source. Here is an example:

```yaml
- name: Llama3.2
  description: The Llama 3.2 collection of multilingual large language models (LLMs) is a collection of pretrained and instruction-tuned generative models in 1B and 3B sizes (text in/text out). The Llama 3.2 instruction-tuned text only models are optimized for multilingual dialogue use cases, including agentic retrieval and summarization tasks. They outperform many of the available open source and closed chat models on common industry benchmarks.
  home: https://www.llama.com/
  icon: /static/catalog_icons/meta.png
  categories:
    - llm
  capabilities:
    - context/128k
    - tools
  sizes:
    - 1
    - 3
  licenses:
    - llama3.2
  release_date: "2024-09-25"
  order: 2
  templates:
    - quantizations:
        - Q3_K_L
        - Q4_K_M
        - Q5_K_M
        - Q6_K_L
        - Q8_0
        - f16
      source: local_path
      # assuming you have all the GGUF model files in /path/to/the/model/directory
      local_path: /path/to/the/model/directory/Llama-3.2-{size}B-Instruct-{quantization}.gguf
      replicas: 1
      backend: llama-box
      cpu_offloading: true
      distributed_inference_across_workers: true
    - quantizations: ["BF16"]
      source: local_path
      # assuming you have both /path/to/Llama-3.2-1B-Instruct and /path/to/Llama-3.2-3B-Instruct directories
      local_path: /path/to/Llama-3.2-{size}B-Instruct
      replicas: 1
      backend: vllm
      backend_parameters:
        - --enable-auto-tool-choice
        - --tool-call-parser=llama3_json
        - --chat-template={data_dir}/chat_templates/tool_chat_template_llama3.2_json.jinja
```

### Template Variables

The following template variables are available for the deployment configuration:

- `{size}`: Model size in billion parameters.
- `{quantization}`: Quantization method of the model.
- `{data_dir}`: GPUStack data directory path.
