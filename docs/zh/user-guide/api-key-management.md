# API 密钥管理

GPUStack 支持使用 API 密钥进行身份验证。每位 GPUStack 用户都可以生成并管理自己的 API 密钥。

## 创建 API 密钥

1. 将鼠标悬停在用户头像上，导航至 `API Keys` 页面。
2. 点击 `New API Key` 按钮。
3. 填写 `Name`、`Description`，并选择该 API 密钥的 `Expiration`。
4. 点击 `Save` 按钮。
5. 复制并妥善保存该密钥，然后点击 `Done` 按钮。

!!! note

    请注意，生成的 API 密钥仅在创建时可见一次。

## 删除 API 密钥

1. 将鼠标悬停在用户头像上，导航至 `API Keys` 页面。
2. 找到要删除的 API 密钥。
3. 点击 `Operations` 列中的 `Delete` 按钮。
4. 确认删除。

## 使用 API 密钥

GPUStack 支持将 API 密钥作为 Bearer Token 使用。以下是使用 curl 的示例：

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