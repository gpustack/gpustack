# Integrate with LiteLLM

LiteLLM can integrate with GPUStack to aggregate locally deployed LLMs, embeddings, reranking, Speech-to-Text, and Text-to-Speech capabilities into a unified OpenAI-compatible API gateway for enterprise employees.

## Deploying Models in GPUStack

1. In GPUStack UI, navigate to the `Deployments` page and click on `Deploy Model` to deploy the models you need. Here are some example models:

- qwen3-8b
- qwen2.5-vl-3b-instruct
- bge-m3
- bge-reranker-v2-m3

![gpustack-models](../assets/integrations/integration-gpustack-models.png)

2. In the model’s Operations, open `API Access Info` to see how to integrate with this model.

![gpustack-api-access-info](../assets/integrations/integration-gpustack-api-access-info.png)

## Create an API Key in GPUStack

1. Navigate to the `Access Control` > `API Keys` page in GPUStack, then click on `New API Key`.

2. Fill in the name, then click `Save`.

3. Copy the API key and save it for later use.

## Integrating GPUStack into LiteLLM Proxy

LiteLLM Proxy uses a `config.yaml` file to map model names to upstream OpenAI-compatible endpoints provided by GPUStack.

1. Open or create your `config.yaml` for LiteLLM.

2. Add your GPUStack models to the `model_list` section using the `openai/` provider prefix:

```yaml
model_list:
  - model_name: qwen3-8b
    litellm_params:
      model: openai/qwen3-8b
      api_base: "http://your-gpustack-url/v1"  # Replace with your GPUStack access URL
      api_key: "gpustack-xxxxxxxx"            # Replace with your GPUStack API Key

  - model_name: bge-m3
    litellm_params:
      model: openai/bge-m3
      api_base: "http://your-gpustack-url/v1"
      api_key: "gpustack-xxxxxxxx"

  - model_name: bge-reranker-v2-m3
    litellm_params:
      model: openai/bge-reranker-v2-m3
      api_base: "http://your-gpustack-url/v1"
      api_key: "gpustack-xxxxxxxx"
