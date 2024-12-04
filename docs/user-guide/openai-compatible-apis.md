# OpenAI Compatible APIs

GPUStack serves [OpenAI-compatible APIs](https://platform.openai.com/docs/api-reference) using the `/v1-openai` path. Most of the APIs also work under the `/v1` path as an alias, except for the `models` endpoint, which is reserved for GPUStack management APIs.

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

### curl

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

### OpenAI Python API library

```python
from openai import OpenAI

client = OpenAI(base_url="http://myserver/v1-openai", api_key="myapikey")

completion = client.chat.completions.create(
  model="llama3",
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
  apiKey: "myapikey",
  baseURL: "http://myserver/v1-openai",
});

async function main() {
  const params = {
    model: "llama3",
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
