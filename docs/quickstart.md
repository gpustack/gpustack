# Guickstart

## Installation

### Linux or MacOS

GPUStack provides a script to install it as a service on systemd or launchd based systems. To install GPUStack using this method, just run:

```bash
curl -sfL https://get.gpustack.ai | sh -
```

You can add additional workers to form a GPUStack cluster by running the following command on worker nodes:

```bash
curl -sfL https://get.gpustack.ai | sh - --server-url http://myserver --token mytoken
```

The token here is a secret used for adding workers. In the default setup, you can run the following to get the token:

```bash
cat /var/lib/gpustack/token
```

### Windows

`// TODO`

### Manual Install

For manual installation or detail configurations, refer to the [installation](./user-guide/installation.md) docs.

## Gettting Started

1. Run and chat with the llama3 model:

```bash
gpustack chat llama3 "tell me a joke."
```

2. Open `http://myserver` in the browser to access the GPUStack UI. Log in to GPUStack with username `admin` and the default password. You can run the following command to get the password for the default setup:

```bash
cat /var/lib/gpustack/initial_admin_password
```

3. Click `Playground` in the navigation menus. Now you can chat with the LLM in the UI playground.

// TODO add screenshot

4. Click `API Keys` in the navigation menus, then click the `New API Key` button.

5. Fill in the `Name` and click the `Save` button.

6. Copy the generated API key and save it somewhere safe. Please note that you can only see it once on creation.

7. Now you can use the API key to access the OpenAI-compatible API. For example, use curl as the following:

```bash
export GPUSTACK_API_KEY=myapikey
curl http://myserver/v1-openai/chat/completions \
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
