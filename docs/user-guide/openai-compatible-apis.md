# OpenAI Compatible APIs

GPUStack serves [OpenAI-compatible APIs](https://platform.openai.com/docs/api-reference) using the `/v1` path.

!!! note

    For other APIs, GPUStack allows you to enable the Generic Proxy when deploying a model.

    With the Generic Proxy enabled, GPUStack determines which model to forward the request to by checking either of the following:

    - the "model" field in the JSON body

    - the `X-GPUStack-Model` header

    For more details, see [Enable Generic Proxy](model-deployment-management.md#enable-generic-proxy).

## Supported Endpoints

The following API endpoints are supported:

- [x] [List Models](https://platform.openai.com/docs/api-reference/models/list)
- [x] [Create Completion](https://platform.openai.com/docs/api-reference/completions/create)
- [x] [Create Chat Completion](https://platform.openai.com/docs/api-reference/chat/create)
- [x] [Create Embeddings](https://platform.openai.com/docs/api-reference/embeddings/create)
- [x] [Create Image](https://platform.openai.com/docs/api-reference/images/create)
- [x] [Create Image Edit](https://platform.openai.com/docs/api-reference/images/createEdit)
- [x] [Create Speech](https://platform.openai.com/docs/api-reference/audio/createSpeech)
- [x] [Create Transcription](https://platform.openai.com/docs/api-reference/audio/createTranscription)

## Usage

The following are examples using the APIs in different languages:

### cURL

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

### OpenAI Python API library

```python
from openai import OpenAI

client = OpenAI(base_url="http://your_gpustack_server_url/v1", api_key="your_api_key")

completion = client.chat.completions.create(
  model="qwen3",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

print(completion.choices[0].message)
```

### OpenAI Node API library

```javascript
const OpenAI = require("openai");

const openai = new OpenAI({
  apiKey: "your_api_key",
  baseURL: "http://your_gpustack_server_url/v1",
});

async function main() {
  const params = {
    model: "qwen3",
    messages: [
      {
        role: "system",
        content: "You are a helpful assistant.",
      },
      {
        role: "user",
        content: "Hello!",
      },
    ],
  };
  const chatCompletion = await openai.chat.completions.create(params);
  console.log(chatCompletion.choices[0].message);
}
main();
```
