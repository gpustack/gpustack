# Using Reranker Models

**Reranker Models** are specialized models designed to improve the ranking of a list of items based on relevance to a given query. They are commonly used in information retrieval and search systems to refine initial search results, prioritizing items that are more likely to meet the user’s intent. Reranker models take the initial document list and reorder items to enhance precision in applications such as search engines, recommendation systems, and question-answering tasks.

In this guide, we will demonstrate how to deploy and use reranker models in GPUStack.

## Prerequisites

Before you begin, ensure that you have the following:

- GPUStack is installed and running. If not, refer to the [Quickstart Guide](../quickstart.md).
- Access to Hugging Face for downloading the model files.

## Step 1: Deploy the Model

Follow these steps to deploy the model from Catalog:

1. Navigate to the `Catalog` page in the GPUStack UI.
2. In the model list page, use dropdown to filter with `Reranker`.
3. Review the model description, maximum context length and supported sizes.

![Deploy Model](../assets/using-models/using-reranker-models/deploy-model.png)

After deployment, you can monitor the model deployment's status on the `Deployments` page.

![Model List](../assets/using-models/using-reranker-models/model-list.png)

## Step 2: Generate an API Key

We will use the GPUStack API to interact with the model. To do this, you need to generate an API key:

1. Hover over the user avatar and navigate to the `API Keys` page.
2. Click the `New API Key` button.
3. Enter a name for the API key and click the `Save` button.
4. Copy the generated API key. You can only view the API key once, so make sure to save it securely.

## Step 3: Reranking

With the model deployed and an API key, you can rerank a list of documents via the GPUStack API. Here is an example script using `curl`:

```bash
export SERVER_URL=<your-server-url>
export GPUSTACK_API_KEY=<your-api-key>
curl $SERVER_URL/v1/rerank \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $GPUSTACK_API_KEY" \
    -d '{
        "model": "qwen3-reranker-4b",
        "query": "What is a panda?",
        "top_n": 3,
        "documents": [
            "hi",
            "it is a bear",
            "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."
        ]
    }' | jq
```

Replace `<your-server-url>` with the URL of your GPUStack server and `<your-api-key>` with the API key you generated in the previous step.

Example response:

```json
{
  "model": "qwen3-reranker-4b",
  "object": "list",
  "results": [
    {
      "index": 0,
      "document": {
        "text": "hi",
        "multi_modal": null
      },
      "relevance_score": 0.9996911287307739
    },
    {
      "index": 2,
      "document": {
        "text": "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.",
        "multi_modal": null
      },
      "relevance_score": 0.8206241726875305
    },
    {
      "index": 1,
      "document": {
        "text": "it is a bear",
        "multi_modal": null
      },
      "relevance_score": 0.7244728803634644
    }
  ],
  "usage": {
    "total_tokens": 51
  }
}
```
