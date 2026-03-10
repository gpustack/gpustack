# Integrate with MaxKB

MaxKB can integrate with GPUStack to leverage locally deployed **LLMs, embedding models, and reranking models** for building knowledge-based AI assistants.

## Deploying Models

1. In GPUStack UI, navigate to the `Deployments` page and click on `Deploy Model` to deploy the models you need. Here are some example models:

* `qwen3.5-35b-a3b`

  ![](../assets/integrations/maxkb-01.png)
  ![](../assets/integrations/maxkb-02.png)

* `qwen3-embedding-4b`

  ![](../assets/integrations/maxkb-03.png)
  ![](../assets/integrations/maxkb-04.png)

* `qwen3-reranker-4b`

  ![](../assets/integrations/maxkb-05.png)
  ![](../assets/integrations/maxkb-06.png)

2. After deployment, you can test the model in **Playground**.

![](../assets/integrations/maxkb-07.png)
![](../assets/integrations/maxkb-09.png)
![](../assets/integrations/maxkb-08.png)

## Obtain Model Access Information

1. In the GPUStack sidebar, open the **Routes** page.

2. Click the **More actions menu** next to the route and select **API Access Info**.

![](../assets/integrations/maxkb-10.png)
![](../assets/integrations/maxkb-11.png)

Record the following information:

```
Base URL
Model Name
API Key
```

Example:

```
Base URL: http://your-gpustack-url/v1

Model Name:
qwen3.5-35b-a3b
qwen3-embedding-4b
qwen3-reranker-4b

API Key:
gpustack_xxxxxxxxxxxxx
```

!!! note

    You can create an API Key following the instructions in the UI.

## Deploy MaxKB

MaxKB can be deployed using Docker:

```bash
docker run -d \
  --name maxkb \
  --restart always \
  -p 8080:8080 \
  -v ~/.maxkb:/opt/maxkb \
  1panel/maxkb
```

Default credentials:

```
admin / MaxKB@123..
```

![](../assets/integrations/maxkb-50.png)

After logging in for the first time, follow the prompt to change the password.

## Integrating GPUStack into MaxKB

1. In the MaxKB UI, navigate to **Model** in the top navigation bar.

![](../assets/integrations/maxkb-51.png)

2. Click **Add Model** and configure the model.

![](../assets/integrations/maxkb-52.png)
![](../assets/integrations/maxkb-53.png)
![](../assets/integrations/maxkb-54.png)

When configuring the model:

* **Base Model**: Must match the model name deployed in GPUStack.
* **API URL**: `http://your-gpustack-url/v1`
* **API Key**: The API key created in GPUStack.

!!! note

    `API URL` and `API Key` fields will appear **after entering the Base Model and pressing Enter**.

3. Add the embedding and reranking models using the same method:

* `qwen3-embedding-4b`

  ![](../assets/integrations/maxkb-56.png)

* `qwen3-reranker-4b`

  For **qwen3-reranker-4b**, enable **Generic Proxy**.

  ![](../assets/integrations/maxkb-55.png)

  This is required because MaxKB uses the following endpoint:

  ```
  /v2/rerank
  ```

  ![](../assets/integrations/maxkb-57.png)

After configuration, the models should appear in the model list.

![](../assets/integrations/maxkb-58.png)

## Create a Knowledge Base

1. Navigate to the **Knowledge** page.

2. Click **Create** and select **Web Knowledge**.

![](../assets/integrations/maxkb-59.png)

3. Enter a documentation URL or other data source. MaxKB will automatically crawl and parse the content.

![](../assets/integrations/maxkb-60.png)

After the crawl is completed:

![](../assets/integrations/maxkb-61.png)

## Create an AI Agent

1. Go to the **Agent** page.

2. Click **Create** to create a new agent.

![](../assets/integrations/maxkb-62.png)
![](../assets/integrations/maxkb-63.png)

3. Configure the agent with:

* Chat model
* Knowledge base
* Retrieval settings

4. Click **Publish** to activate the agent.

![](../assets/integrations/maxkb-64.png)

## Chat with the Knowledge Base

Open the chat interface to start interacting with the assistant.

![](../assets/integrations/maxkb-65.png)
![](../assets/integrations/maxkb-66.png)

The assistant can now answer questions based on the connected knowledge base and models deployed on GPUStack.
