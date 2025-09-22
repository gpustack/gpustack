# OpenAI 兼容 API

GPUStack 通过 `/v1-openai` 路径提供与 [OpenAI 兼容的 API](https://platform.openai.com/docs/api-reference)。大多数 API 也可以通过 `/v1` 路径作为别名使用，但 `models` 端点除外，该端点保留用于 GPUStack 管理 API。

## 支持的端点

支持以下 API 端点：

- [x] [列出模型](https://platform.openai.com/docs/api-reference/models/list)
- [x] [创建补全](https://platform.openai.com/docs/api-reference/completions/create)
- [x] [创建聊天补全](https://platform.openai.com/docs/api-reference/chat/create)
- [x] [创建嵌入](https://platform.openai.com/docs/api-reference/embeddings/create)
- [x] [创建图像](https://platform.openai.com/docs/api-reference/images/create)
- [x] [创建图像编辑](https://platform.openai.com/docs/api-reference/images/createEdit)
- [x] [创建语音](https://platform.openai.com/docs/api-reference/audio/createSpeech)
- [x] [创建转写](https://platform.openai.com/docs/api-reference/audio/createTranscription)

## 用法

以下是在不同语言中使用这些 API 的示例：

### curl

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
        "content": "你是一个乐于助人的助手。"
      },
      {
        "role": "user",
        "content": "你好！"
      }
    ],
    "stream": true
  }'
```

### OpenAI Python API 库

```python
from openai import OpenAI

client = OpenAI(base_url="http://your_gpustack_server_url/v1", api_key="your_api_key")

completion = client.chat.completions.create(
  model="llama3",
  messages=[
    {"role": "system", "content": "你是一个乐于助人的助手。"},
    {"role": "user", "content": "你好！"}
  ]
)

print(completion.choices[0].message)
```

### OpenAI Node.js API 库

```javascript
const OpenAI = require("openai");

const openai = new OpenAI({
  apiKey: "your_api_key",
  baseURL: "http://your_gpustack_server_url/v1",
});

async function main() {
  const params = {
    model: "llama3",
    messages: [
      {
        role: "system",
        content: "你是一个乐于助人的助手。",
      },
      {
        role: "user",
        content: "你好！",
      },
    ],
  };
  const chatCompletion = await openai.chat.completions.create(params);
  console.log(chatCompletion.choices[0].message);
}
main();
```