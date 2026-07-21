# Integrate with NewAPI

NewAPI can integrate with GPUStack to aggregate locally deployed LLMs, embeddings, reranking, Speech-to-Text, and Text-to-Speech capabilities into a unified OpenAI-compatible API gateway for enterprise employee.

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

## Integrating GPUStack into NewAPI

1. Access the NewAPI Management Console, navigate to `Channels`  on the left sidebar, and click `Add Channel` 

2. Configure the channel parameters as follows:

- **Type**: Select `OpenAI` or `Custom` (GPUStack provides standard OpenAI-compatible and anthropic endpoints).

- **Name**: Input a Channels name,often same with model-id e.g., `qwen3-8b`.

- **Base URL (Base URL)**: `http://your-gpustack-url`, the URL should point to GPUStack's access url or domain name both with OpenAI-compatible or anthropic(remove v1 because newapi will add it auto). Do not use `localhost` if NewAPI is running in a Docker container, as it refers to the container’s internal network. also can use https if gpustack server start with tls

- **Key (密钥)**: Input the GPUStack API Key you copied in the previous step.

- **Models (模型)**: Enter or select the exact model names deployed in GPUStack (e.g., `qwen3-8b`, `bge-m3`, `bge-reranker-v2-m3`). Ensure the model names match GPUStack's deployment names.

![newapi-add-channel](../assets/integrations/newapi1.png)
![newapi-add-channel](../assets/integrations/newapi2.png)
![newapi-add-channel](../assets/integrations/newapi3.png)
![newapi-add-channel](../assets/integrations/newapi4.png)

3. config model price

![newapi-add-channel](../assets/integrations/newapi5.png)
![newapi-add-channel](../assets/integrations/newapi6.png)
![newapi-add-channel](../assets/integrations/newapi7.png)

4. You can click `Test` on the channel list to verify the connection between NewAPI and GPUStack.

![newapi-add-channel](../assets/integrations/newapi8.png)

---

## Verifying and Using the Integrated Models in NewAPI

1. Go to the `api keys` page in NewAPI, click `create api key`

* note: api key must select group same with channels config group , or you can not access the model

2. Copy the newly generated NewAPI key (starts with `sk-`).

3. You can now access all GPUStack-backed models via NewAPI's unified endpoint:

- **Unified Base URL**: `http://your-newapi-url/v1`
- **Unified API Key**: `sk-xxxxxx` (Your NewAPI key)

Example request using `curl`:

```bash
curl http://your-newapi-url/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-newapi-token" \
  -d '{
    "model": "qwen3-8b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
