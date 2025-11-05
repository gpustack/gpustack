# Custom Inference Backend

This article demonstrates how to add a custom inference backend that is not yet supported by GPUStack, using TensorRT-LLM as an example, at the end of the article, configuration methods for common inference backends are also listed.

## Core Steps
1. Prepare the Docker image for the required inference backend
2. Understand the image's ENTRYPOINT or CMD to determine the inference backend startup command
3. Add configuration in the Inference Backend page
4. Deploy models in the Deployment page and select the newly added inference backend

## Example
The following uses TensorRT-LLM as an example to illustrate how to add and use an inference backend.
> The following examples do not represent the optimal performance configuration and are for functional demonstration only. For better inference performance, please refer to the official documentation of each inference backend for tuning, or use GPUStack's Catalog.

1. Find the required image from the [release page](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release) mentioned in TensorRT-LLM documentation.
2. TensorRT-LLM images must launch the inference service using `trtllm-serve`; otherwise, it will start a regular bash session. The run_command supports {{model_path}} and {{port}} templates, which are automatically replaced with the actual model path and port number after scheduling.
3. Add configuration in Inference Backend page, supporting YAML import. Example:
```yaml
backend_name: TensorRT-LLM
default_version: 1.2.0rc0
version_configs:
  1.2.0rc0:
    image_name: nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc0
    run_command: 'trtllm-serve {{model_path}} --host 0.0.0.0 --port {{port}}'
    custom_framework: cuda
```
!!! note

    You may see some inference backends labeled as Built-in (such as vLLM, MindIE) in the Inference Backend page. These are GPUStack's built-in inference backends. When using built-in backends, appropriate container images matching the worker environment are automatically obtained based on the runtime.
    You can also add custom versions to these Built-in inference backends and specify the image names you need.

4. In Deployment page, select the newly added inference backend and deploy the model.
![image.png](../assets/tutorials/custom-inference-backend/deploy-by-custom-backend.png)

**Result**

After the inference backend service starts, you can see the model_instance status becomes RUNNING.
![image.png](../assets/tutorials/custom-inference-backend/custom-backend-running.png)
You can engage in conversations in the Playground.
![image.png](../assets/tutorials/custom-inference-backend/use-custom-backend-in-playground.png)

## Parameter Description
When adding a Backend, besides the YAML import method shown in the example, you can also use the form. Detailed parameter descriptions are as follows:

| Parameter Name             | Description                                                                                                                                                                                                                                                                                                                                                                                              | Required |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| Name                       | Inference backend name                                                                                                                                                                                                                                                                                                                                                                                   | Yes      |
| Health Check Path          | Health check path used to verify if the inference backend has started and is running properly. Default value is /v1/models (OpenAI API specification)                                                                                                                                                                                                                                                    | No       |
| Default Execution Command  | Execution command passed as `args` when the inference backend container starts. For example, for vllm backend, this would be `vllm serve {{model_path}} --port {{port}} --served-model-name {{model_name}}`. Note that this command supports {{model_path}}, {{model_name}}, {{port}} templates, allowing automatic substitution of model path and port number and model name(optional) after scheduling | No       |
| Default Backend Parameters | Default backend parameters used to pre-fill Advanced-Backend Parameters during deployment for convenient deployment and adjustment                                                                                                                                                                                                                                                                       | No       |
| Description                | Description                                                                                                                                                                                                                                                                                                                                                                                              | No       |
| Version Configs            | Inference backend version configurations, used to add inference backend versions                                                                                                                                                                                                                                                                                                                         | Yes      |
| Default Version            | Dropdown option used to pre-fill during deployment. If no version is selected during deployment, the image corresponding to Default Version will be used                                                                                                                                                                                                                                                 | No       |

Version Configs parameter description:

| Parameter Name                  | Description                                         | Required |
|---------------------------------|-----------------------------------------------------|----------|
| Version                         | Version name, displayed in BackendVersion options during deployment | Yes      |
| Image Name                      | Inference backend image name                         | Yes      |
| Framework<br/>(custom_framework) | Inference backend framework. Deployment and scheduling will filter based on supported Frameworks | Yes      |
| Execution Command               | Execution command for this version. If not set, uses Default Execution Command | No       |

## Flexible Testing Deployment
For quick and flexible verification or adjustment of the image and startup command, select `Custom` in the backend options on the deployment page. This will reveal two new input fields for specifying the custom `image_name` and `run_command` respectively.
![img.png](../assets/tutorials/custom-inference-backend/flexible-testing-deployment.png)

## Typical Examples

### Deploy GGUF Models with llama.cpp
1. Find the image name in the [documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md): `ghcr.io/ggml-org/llama.cpp:server` (ensure you select the variant that matches your worker platform).
2. Add the following backend configuration on the Inference Backend page:
    ```yaml
    backend_name: llama.cpp-custom
    default_run_command: '-m {{model_path}} --host 0.0.0.0 --port {{port}}'
    version_configs:
      v1-custom:
        image_name: ghcr.io/ggml-org/llama.cpp:server
        custom_framework: cpu
    default_version: v1-custom
    ```
3. On the Deployment page, locate a GGUF-format model, select `llama.cpp`, and deploy.

For more information, refer to the llama.cpp [GitHub repository](https://github.com/ggml-org/llama.cpp).

Screenshots:
![gguf-deploy.png](../assets/tutorials/custom-inference-backend/gguf-deploy.png)
![gguf-resp.png](../assets/tutorials/custom-inference-backend/gguf-resp.png)

### Use Kokoro-FastAPI
1. Find the image name in the [documentation](https://github.com/remsky/Kokoro-FastAPI?tab=readme-ov-file#get-started), and choose the variant that matches your worker platform:
   - `ghcr.io/remsky/kokoro-fastapi-cpu:latest`
   - `ghcr.io/remsky/kokoro-fastapi-gpu:latest`

!!! warning

    This image includes a built-in model, so the model you select on the Deployment page may be ignored. To avoid unexpected errors, choose a model consistent with the one bundled in the image.
    The kokoro-fastapi image uses the [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) model.

2. Add the following backend configuration on the Inference Backend page:
   ```yaml
   backend_name: kokoro-custom
   version_configs:
     v1:
       image_name: ghcr.io/remsky/kokoro-fastapi-gpu:latest
       custom_framework: cuda
   default_run_command: python -m uvicorn api.src.main:app --host 0.0.0.0 --port {{port}} --log-level debug
   ```
3. On the Deployment page, select the Kokoro-82M model, choose `kokoro` as the backend, and set `Name` to `kokoro`.


!!! warning "Known Limitations for Name"

    In kokoro-fastapi, the `model_name` is restricted to the keys below; other values will result in an "unsupported" error.

    ```
    "models": {
        "tts-1": "kokoro-v1_0",
        "tts-1-hd": "kokoro-v1_0",
        "kokoro": "kokoro-v1_0"
    }
    ```

    Therefore, restrict the `Name` during deployment to one of these supported keys.


Screenshots:

![kokoro-run2.png](../assets/tutorials/custom-inference-backend/kokoro-run2.png)
![kokoro-run.png](../assets/tutorials/custom-inference-backend/kokoro-run-1.png)
