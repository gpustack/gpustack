# Model File Management

GPUStack allows admins to download and manage model files.

## Add Model File

GPUStack currently supports models from [Hugging Face](https://huggingface.co), [ModelScope](https://modelscope.cn), [Ollama](https://ollama.com/library), and local paths. To add model files, navigate to the `Resources` page and click the `Model Files` tab.

### Add a Hugging Face Model

1. Click the `Add Model File` button and select `Hugging Face` from the dropdown.
2. Use the search bar in the top left to find a model by name, e.g., `Qwen/Qwen2.5-0.5B-Instruct`. To search only for GGUF models, check the `GGUF` checkbox.

3. _(Optional)_ For GGUF models, select the desired quantization format from `Available Files`.
4. Select the target worker to download the model file.
5. _(Optional)_ Specify a `Local Directory` to download the model to a custom path instead of the GPUStack cache directory.
6. Click the `Save` button.

### Add a ModelScope Model

1. Click the `Add Model File` button and select `ModelScope` from the dropdown.
2. Use the search bar in the top left to find a model by name, e.g., `Qwen/Qwen2.5-0.5B-Instruct`. To search only for GGUF models, check the `GGUF` checkbox.
3. _(Optional)_ For GGUF models, select the desired quantization format from `Available Files`.
4. Select the target worker to download the model file.
5. _(Optional)_ Specify a `Local Directory` to download the model to a custom path instead of the GPUStack cache directory.
6. Click the `Save` button.

### Add an Ollama Model

1. Click the `Add Model File` button and select `Ollama Library` from the dropdown.
2. Select a model from the dropdown list or input a custom Ollama model, e.g., `llama3`, `llama3:70b`, or `youraccount/llama3:70b`.
3. Select the target worker to download the model file.
4. _(Optional)_ Specify a `Local Directory` to download the model to a custom path instead of the GPUStack cache directory.
5. Click the `Save` button.

### Add a Local Path Model

You can add models from a local path. The path can be a directory (e.g., a Hugging Face model folder) or a file (e.g., a GGUF model) located on the worker.

1. Click the `Add Model File` button and select `Local Path` from the dropdown.
2. Enter the `Model Path`.
3. Select the target worker.
4. Click the `Save` button.

## Retry Download

If a model file download fails, you can retry it:

1. Navigate to the `Resources` page and click the `Model Files` tab.
2. Locate the model file with an error status.
3. Click the ellipsis button in the `Operations` column and select `Retry Download`.
4. GPUStack will attempt to download the model file again from the specified source.

## Deploy Model

Models can be deployed from model files. Since the model is stored on a specific worker, GPUStack will add a worker selector using the `worker-name` key to ensure proper scheduling.

1. Navigate to the `Resources` page and click the `Model Files` tab.
2. Find the model file you want to deploy.
3. Click the `Deploy` button in the `Operations` column.
4. Review or adjust the `Name`, `Replicas`, and other deployment parameters.
5. Click the `Save` button.

## Delete Model File

1. Navigate to the `Resources` page and click the `Model Files` tab.
2. Find the model file you want to delete.
3. Click the ellipsis button in the `Operations` column and select `Delete`.
4. _(Optional)_ Check the `Also delete the file from disk` option.
5. Click the `Delete` button to confirm.
