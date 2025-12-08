# Inference Backend Management
GPUStack allows admins to configure inference backends and backend versions.

This article serves as an operational guide for the Inference Backend page. For supported built-in backends and their capabilities, see [Built-in Inference Backends](built-in-inference-backends.md).

For guidelines for configuring custom backends and examples of custom backends that have been verified to work, see [Custom Inference Backends](../tutorials/using-custom-backends.md).

## Parameter Description

| Parameter Name             | Description                                                                                                                                                                                                                                                                                                                                                                                              | Required |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|
| Name                       | Inference backend name                                                                                                                                                                                                                                                                                                                                                                                   | Yes      |
| Health Check Path          | Health check path used to verify the backend is up and responding. Default: /v1/models (OpenAI-compatible).                                                                                                                                                                                                                                                   | No       |
| Default Execution Command  | Container startup command/args. For example (vLLM): `vllm serve {{model_path}} --port {{port}} --served-model-name {{model_name}} --host {{worker_ip}}`. The placeholders `{{model_path}}`, `{{model_name}}`, `{{port}}`, and `{{worker_ip}}` are automatically substituted when the deployment is scheduled to a worker. | No       |
| Default Backend Parameters | Pre-populate the Advanced Backend Parameters section during deployment; you can adjust them before launching                                                                                                                                                                                                                                                                                             | No       |
| Description                | Description                                                                                                                                                                                                                                                                                                                                                                                              | No       |
| Version Configs            | Configure available versions of this backend                                                                                                                                                                                                                                                                                                                                                             | Yes      |
| Default Version            | Preselected during deployment. If you don’t choose a version, its image is used                                                                                                                                                                                                                                                                                                                          | No       |

Version Configs parameter description:

| Parameter Name                  | Description                                         | Required |
|---------------------------------|-----------------------------------------------------|----------|
| Version                         | Version name shown in the Backend Version dropdown during deployment | Yes      |
| Image Name                      | Container image name for the backend (e.g., `ghcr.io/org/image:tag`) | Yes      |
| Framework<br/>(custom_framework) | Backend framework (internal identifier: `custom_framework`). Deployment and scheduling are filtered by supported frameworks | Yes      |
| Execution Command               | Version-specific startup command. If omitted, the Default Execution Command is used | No       |


## Add Custom Inference Backend
1. Click the "Add Backend" button in the top-right corner.
2. You can add a custom inference backend by completing the form or by pasting a YAML definition. Refer to the parameter descriptions above for field meanings.
3. The backend name cannot be modified after creation. Custom backend names must end with "-custom" (pre-filled in the form).
4. Click "Save" to submit.

## Edit Inference Backend or Add Custom Version
1. On the Inference Backend page, locate the target backend. From the card's top-right dropdown menu, choose "Edit".
2. Modify backend properties (the name cannot be changed), or add a new version.
3. For built-in backends, custom versions must end with "-custom" (pre-filled in the form).
4. Click "Save" to submit.

### Example: Add a Custom Version to the Built-in vLLM Inference Backend
1. On the Inference Backend page, locate the vLLM inference backend. From the card's top-right dropdown menu, choose "Edit".
2. In the Version Configs section, click "Add Version".
3. Fill in the fields as follows:

    - Version: `0.12.0`
    - Image Name: `vllm/vllm-openai:v0.12.0`
    - Framework: `cuda`
    - Execution Command: `{{model_path}} --host {{worker_ip}} --port {{port}} --served-model-name {{model_name}}`

4. Click "Save" to submit.

!!! note

    vLLM has changed the entrypoint of its Docker image since v0.11.1. Therefore, when adding a custom version for vLLM v0.11.1 or later, you must specify the `Execution Command` field; otherwise, the model will fail to start.

## Delete Custom Inference Backend
1. On the Inference Backend page, locate the target backend and select "Delete" from the card's top-right dropdown menu.
2. Built-in backends cannot be deleted.
3. Click "Delete" in the confirmation dialog.

## List Versions of Inference Backend
On the Inference Backend page, click anywhere on the backend card (except the action buttons) to open a modal where you can browse all built-in and custom-added versions.

## Flexible Testing Deployment
Use this mode to quickly verify or tweak the image and startup command without editing the backend definition.

1. Navigate to the Deployments page, click the "Deploy Model" button, and choose any model source.
2. In the Basic tab, open the "Backend" dropdown and select "Custom" under the "Built-in" section.
3. Two fields appear: `image_name` and `run_command`. These override the backend configuration for this deployment only.
4. Review the remaining required settings and submit the deployment.
