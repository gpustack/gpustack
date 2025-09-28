# API Key Management

GPUStack supports authentication using API keys. Each GPUStack user can generate and manage their own API keys.

## Create API Key

1. Hover over the user avatar and navigate to the `API Keys` page.
2. Click the `New API Key` button.
3. Fill in the `Name`, `Description`, and select the `Expiration` of the API key.
4. Click the `Save` button.
5. Copy and store the key somewhere safe, then click the `Done` button.

!!! note

    Please note that you can only see the generated API key once upon creation.

## Delete API Key

1. Hover over the user avatar and navigate to the `API Keys` page.
2. Find the API key you want to delete.
3. Click the `Delete` button in the `Operations` column.
4. Confirm the deletion.

## Use API Key

GPUStack supports using the API key as a bearer token. The following is an example using curl:

```bash
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1-openai/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
    "model": "llama3",
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
