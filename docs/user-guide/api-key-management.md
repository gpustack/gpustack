# API Key Management

GPUStack supports authentication using API keys. Each GPUStack user can generate and manage their own API keys.

## Create API Key

1. Hover over the user avatar and navigate to the `API Keys` page.
2. Click the `Add API Key` button.
3. Fill in the `Name`, `Description`, and select the `Expiration` of the API key.
4. Select the `Type` of the API key:
   - **Auto-generated**: GPUStack generates the API key for you.
   - **Custom**: Provide your own key value in the `Key` field. This field is required when choosing **Custom**.
5. In the `Access Permissions` section, select the permissions for the API key:
   - **Platform Management**: Grants access to platform management endpoints (users, models, workers, etc.).
   - **Model Access**: Grants access to inference APIs. When selected, choose either **All models** or **Allowed models**, and if choosing **Allowed models**, select which models this API key can access from the list.

   If no permission is selected, the API key has no access permissions.
6. Click the `Save` button.
7. Copy and store the key somewhere safe, then click the `Done` button.

!!! note

    The full API key value is shown only once upon creation; afterwards only a masked value is displayed. Custom keys are the exception — you already know their value.

## Edit Access Permissions

1. Hover over the user avatar and navigate to the `API Keys` page.
2. Find the API key you want to edit.
3. Click the `Edit` button in the `Operations` column.
4. Update the `Description` if needed.
5. In the `Access Permissions` section, adjust the selected permissions:
   - **Platform Management**: Grants access to platform management endpoints (users, models, workers, etc.).
   - **Model Access**: Grants access to inference APIs. When selected, choose either **All models** or **Allowed models**, and if choosing **Allowed models**, select which models this API key can access from the list.
6. Click the `Save` button.

!!! note

    Changes will take effect within one minute.

## Delete API Key

1. Hover over the user avatar and navigate to the `API Keys` page.
2. Find the API key you want to delete.
3. Click the `Delete` button in the `Operations` column.
4. Confirm the deletion.

## Use API Key

GPUStack supports using the API key as a bearer token. The following is an example using curl:

```bash
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
    "model": "qwen3",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ],
    "stream": true
  }'
```
