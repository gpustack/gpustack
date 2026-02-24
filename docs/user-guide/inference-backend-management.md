# Inference Backend Management

GPUStack allows admins to configure inference backends and backend versions.

This article serves as an operational guide for the Inference Backend page. For supported built-in backends and their capabilities, see [Built-in Inference Backends](built-in-inference-backends.md).

For guidelines for configuring custom backends and examples of custom backends that have been verified to work, see [Custom Inference Backends](../tutorials/using-custom-backends.md).

## Backend Sources

GPUStack supports three types of inference backends:

- **Built-in**: Pre-configured backends maintained by GPUStack (e.g., vLLM, MindIE, VoxBox). These cannot be deleted.
- **Community**: Backends shared by the Community Backend Marketplace. You can enable them as needed.
- **Custom**: Backends you create with your own configurations. These can be freely added, edited, and deleted.

## Parameter Description

| Parameter Name                | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | Required |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- |
| Name                          | Inference backend name                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Yes      |
| Health Check Path             | Health check path used to verify the backend is up and responding. Default: /v1/models (OpenAI-compatible).                                                                                                                                                                                                                                                                                                                                                                                      | No       |
| Default Execution Command     | Container startup command/args. For example (vLLM): `vllm serve {{model_path}} --port {{port}} --served-model-name {{model_name}} --host {{worker_ip}}`. The placeholders `{{model_path}}`, `{{model_name}}`, `{{port}}`, `{{worker_ip}}`, and `{{VAR_NAME}}` (for environment variables) are automatically substituted when the deployment is scheduled to a worker; after placeholder substitution, arguments are split using POSIX-style. Quote values with spaces and avoid shell operators. | No       |
| Default Entrypoint            | Container entrypoint override. If set, it replaces the image entrypoint for this backend. Arguments are split using POSIX-style.                                                                                                                                                                                                                                                                                                                                                                 | No       |
| Default Environment Variables | Environment variables to set for all versions of this backend. Can be referenced in commands using `{{VAR_NAME}}` syntax. Version-specific environment variables take precedence.                                                                                                                                                                                                                                                                                                                | No       |
| Default Backend Parameters    | Pre-populate the Advanced Backend Parameters section during deployment; you can adjust them before launching                                                                                                                                                                                                                                                                                                                                                                                     | No       |
| Description                   | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | No       |
| Version Configs               | Configure available versions of this backend                                                                                                                                                                                                                                                                                                                                                                                                                                                     | Yes      |
| Default Version               | Preselected during deployment. If you don't choose a version, its image is used                                                                                                                                                                                                                                                                                                                                                                                                                  | No       |

## Environment Variables

You can define environment variables at two levels:

- **Backend level** (`Default Environment Variables`): Applied to all versions of the backend
- **Version level** (`Environment Variables`): Specific to a version, overrides backend-level variables

Environment variables can be referenced in commands using `{{VAR_NAME}}` syntax.

**Example:**

```yaml
backend_name: my-backend-custom
default_env:
  MODEL_CACHE_DIR: /cache
  LOG_LEVEL: info
version_configs:
  v1:
    image_name: my-image:v1
    custom_framework: cuda
    run_command: "serve {{model_path}} --cache {{MODEL_CACHE_DIR}} --log-level {{LOG_LEVEL}} --port {{port}}"
    env:
      MODEL_CACHE_DIR: /custom-cache # Overrides backend-level value
```

In this example:

- `MODEL_CACHE_DIR` is set to `/cache` at the backend level
- Version `v1` overrides it to `/custom-cache`
- `LOG_LEVEL` remains `info` for all versions
- Both variables are referenced in the command using `{{VAR_NAME}}` syntax

Version Configs parameter description:

| Parameter Name                   | Description                                                                                                                                                                    | Required |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------- |
| Version                          | Version name shown in the Backend Version dropdown during deployment                                                                                                           | Yes      |
| Image Name                       | Container image name for the backend (e.g., `ghcr.io/org/image:tag`)                                                                                                           | Yes      |
| Framework<br/>(custom_framework) | Backend framework (internal identifier: `custom_framework`). Deployment and scheduling are filtered by supported frameworks                                                    | Yes      |
| Environment Variables            | Environment variables specific to this version. Overrides backend-level `Default Environment Variables`. Can be referenced in `Execution Command` using `{{VAR_NAME}}` syntax. | No       |
| Entrypoint                       | Version-specific container entrypoint override. If omitted, `Default Entrypoint` is used. Arguments are split using POSIX-style.                                               | No       |
| Execution Command                | Version-specific startup command. If omitted, the Default Execution Command is used. Parsing and splitting rules are identical to `Default Execution Command`.                 | No       |

## Add Custom Inference Backend

1. Click the "Add Backend" button in the top-right corner.
2. You can add a custom inference backend by completing the form or by pasting a YAML definition. Refer to the parameter descriptions above for field meanings.
3. The backend name cannot be modified after creation. Custom backend names must end with "-custom" (pre-filled in the form).
4. Click "Save" to submit.

There are two ways to add a custom inference backend:

- Through the UI form: Navigate to the **Resources > Inference Backends** page and click the **Add Custom Inference Backend** button.
- Through YAML configuration: Import a YAML file containing the backend configuration.

### Enable Community Inference Backend

These are essentially custom backends with a "community" source label, allowing you to quickly create custom backends without manual configuration.

1. On the Inference Backend page, click the "Add Backend" button in the top-right corner.
2. Select the "Community" option to browse available community backends from the marketplace.
3. Locate the backend you want to use and click "Enable" from the card's action menu.
4. Once enabled, the backend becomes available for model deployments. To disable, delete the backend from the Inference Backend page.

## Edit Inference Backend or Add Custom Version

1. On the Inference Backend page, locate the target backend. From the card's top-right dropdown menu, choose "Edit".
2. Modify backend properties (the name cannot be changed), or add a new version.
3. For built-in backends, custom versions must end with "-custom" (pre-filled in the form).
4. Click "Save" to submit.

### Example: Add a Custom Version to the Built-in vLLM Inference Backend

1. On the Inference Backend page, locate the vLLM inference backend. From the card's top-right dropdown menu, choose "Edit".
2. In the Version Configs section, click "Add Version".
3. Fill in the fields as follows:
   - Version: `0.16.0`
   - Image Name: `vllm/vllm-openai:v0.16.0`
   - Framework: `cuda`
   - Override Image Entrypoint: `vllm serve`
   - Execution Command: `{{model_path}} --host {{worker_ip}} --port {{port}} --served-model-name {{model_name}}`

4. Click "Save" to submit.

!!! note

    vLLM has changed the entrypoint of its Docker image since v0.11.1. Therefore, when adding a custom version for vLLM v0.11.1 or later, you must specify the `Execution Command` field; otherwise, the model will fail to start. If you use newer versions of `gpustack/runner` images, you don't need to set the `Execution Command` field.

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

## Limitations of Custom Inference Backends

Custom inference backends do not support distributed inference across multiple workers.
