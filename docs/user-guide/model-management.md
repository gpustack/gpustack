# Model Management

You can manage large language models in GPUStack by navigating to the `Models` page. A model in GPUStack contains one or multiple replicas of model instances. On deployment, GPUStack automatically computes resource requirements for the model instances from model metadata and schedules them to available workers accordingly.

## Deploy Model

1. To deploy a model, click the `Deploy Model` button.

2. Fill in the `Name` of the model.

3. Select the `Source` of the model. Currently, models from `Hugging Face` and the `Ollama Library` in [GGUF format](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) are supported.

4. For `Hugging Face` models, search and fill in the Hugging Face repo ID, e.g., `microsoft/Phi-3-mini-4k-instruct-gguf`, then select the `File Name`, e.g., `phi-3-mini-4k-instruct-q4.gguf`. For `Ollama Library` models, select an `Ollama Model` from the dropdown list, or input any Ollama model you need, e.g., `llama3:70b`.

5. Adjust the `Replicas` as needed.

6. Click the `Save` button.

## Edit Model

1. Find the model you want to edit on the model list page.
2. Click the `Edit` button in the `Operations` column.
3. Update the attributes as needed. For example, change the `Replicas` to scale up or down.
4. Click the `Save` button.

## Delete Model

1. Find the model you want to delete on the model list page.
2. Click the ellipsis button in the `Operations` column, then select `Delete`.
3. Confirm the deletion.

## View Model Instance

1. Find the model you want to check on the model list page.
2. Click the `>` symbol to view the instance list of the model.

## Delete Model Instance

1. Find the model you want to check on the model list page.
2. Click the `>` symbol to view the instance list of the model.
3. Find the model instance you want to delete.
4. Click the ellipsis button for the model instance in the `Operations` column, then select `Delete`.
5. Confirm the deletion.

!!! note

    After a model instance is deleted, GPUStack will recreate a new instance to satisfy the expected replicas of the model if necessary.

## View Model Instance Logs

1. Find the model you want to check on the model list page.
2. Click the `>` symbol to view the instance list of the model.
3. Find the model instance you want to check.
4. Click the `View Logs` button for the model instance in the `Operations` column.

## Use Self-hosted Model

You can deploy self-hosted Ollama models by configuring the `--ollama-library-base-url` option in the GPUStack server. The `Ollama Library` URL should point to the base URL of the Ollama model registry. For example, `https://registry.mycompany.com`.

Here is an example workflow to set up a registry, publish a model, and use it in GPUStack:

```bash
# Run a self-hosted OCI registry
docker run -d -p 5001:5000 --name registry registry:2

# Push a model to the registry using Ollama
ollama pull llama3
ollama cp llama3 localhost:5001/library/llama3
ollama push localhost:5001/library/llama3 --insecure

# Start GPUStack server with the custom Ollama library URL
curl -sfL https://get.gpustack.ai | sh -s - --ollama-library-base-url http://localhost:5001
```

That's it! You can now deploy the model `llama3` from `Ollama Library` source in GPUStack as usual, but the model will now be fetched from the self-hosted registry.
