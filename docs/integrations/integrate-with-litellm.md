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

## Integrating GPUStack into LiteLLM 

1. Open LiteLLM manage ui

http://$litellmip:4000/ui

![litellm-dashboard](../assets/integrations/litellm-1.png)

2. open models+endpoints menu -> add model

![litellm-add-model1](../assets/integrations/litellm-2.png)
![litellm-add-model2](../assets/integrations/litellm-3.png)


3. test in litellm  ->playgroud
![litellm-chat1](../assets/integrations/litellm-4.png)



