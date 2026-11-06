---
translation_status: translated
---

# 使用决策模型 { #using-decision-models }

**决策模型（Decision Models）**回答结构化的类型化问题，而非生成文本。给定一个 *state*（做出决策所依据的情境）和一组 *questions*（问题），决策模型对每个问题返回类型化的判定结果：选中的选项、级别评分或是否判断，通常附带置信度。它适用于完整生成代价过高的场景，例如模型选择与路由、分类以及轻量评判任务。

在 GPUStack 中，决策模型通过 System One（Jev）决策契约在 `/v1/systemone` 端点上提供服务。本指南介绍如何在 GPUStack 中部署和使用决策模型。

## 前提条件 { #prerequisites }

开始前，请确认：

- GPUStack 已安装并运行；如未完成，请参考[快速入门](../quickstart.md)。
- 可以访问 Hugging Face，以下载模型文件。

## 步骤 1：部署模型 { #step-1-deploy-the-model }

GPUStack 当前支持两类提供决策 API 的部署：

### vLLM：改造文本生成模型 { #vllm-repurpose-a-text-generation-model }

vLLM `0.32.0` 及以上版本在其常规服务端口上提供 System One 决策契约。像往常一样用 `vLLM` 后端部署任意文本生成模型，然后将模型类别改为 `Decision`——类别正是告诉 GPUStack 该部署是决策服务的关键。

1. 使用 `vLLM` 后端（`0.32.0` 或更高版本）部署一个文本生成模型。参见[模型部署管理](../user-guide/model-deployment-management.md)。
2. 编辑模型，在模型设置中将 `Category` 设为 `Decision`。

![模型类别](../assets/using-models/using-decision-models/model-category.png)

### llama.cpp：提供 Laya 决策 GGUF { #llamacpp-serve-a-laya-decision-gguf }

Laya 决策 GGUF 文件在通用骨干（modern-bert、qwen、llama）之上携带决策元数据键。像部署其他 GGUF 模型一样部署它，由 llama.cpp 提供服务：GPUStack 会在部署时检测决策元数据并自动将模型归入 `decision` 类别——否则同一文件会被误判为重排序或嵌入模型。

![模型列表](../assets/using-models/using-decision-models/model-list.png)

无论哪种方式，部署启动后 `Models` 页面的 `Category` 列都会显示 `Decision`。类别也可以手动修正；类别值有校验，误写的类别会被拒绝而不是被静默保存。

## 步骤 2：发起决策 { #step-2-ask-a-decision }

可以在 UI 上发起决策，也可以通过 API 发起。

### 通过 UI { #via-the-ui }

1. 进入 `Playground` > `Decision`，或在 `Models` 页面中点击模型行上的 `Open in Playground`。
2. 在模型选择器中选择决策模型——列表中只显示 `decision` 类别的模型，且模型是决策唯一的参数。
3. 在 `State` 编辑器中填写做出决策所依据的情境（预置了 chat-messages 示例，接受纯文本或结构化 JSON）；在 `Questions` 编辑器中填写要提出的问题（预置了一个 `choice` 问题，可通过 `+ choice`、`+ score`、`+ noul` 按钮扩展）。
4. 点击 `Run`（或按 `Ctrl`/`Cmd` + `Enter`）并查看类型化的答案。

编辑器与答案卡的详细说明参见 [Decision Playground](../user-guide/playground/decision.md)。

### 通过 API { #via-the-api }

通过 `POST /v1/systemone` 向模型发送 state 和 questions（使用在 `API Keys` 页面创建的 API 密钥认证）。`model` 字段必填，取部署模型的名称（或模型路由名称），与 OpenAI 兼容端点一致。

以下是使用 `curl` 的示例脚本：

```bash
export SERVER_URL=<your-server-url>
export GPUSTACK_API_KEY=<your-api-key>
curl $SERVER_URL/v1/systemone \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $GPUSTACK_API_KEY" \
    -d '{
        "model": "qwen3-decision-8b",
        "state": {
            "messages": [
                { "role": "user", "content": "Write a 200-word poem in Hungarian." }
            ]
        },
        "questions": {
            "model_selection": {
                "type": "choice",
                "instructions": "Pick the model best suited.",
                "criteria": {
                    "qwen3-8b": "Strong multilingual generation.",
                    "llama3-8b": "Faster, weaker multilingual."
                }
            }
        }
    }' | jq
```

将 `<your-server-url>` 替换为 GPUStack Server 的 URL，将 `<your-api-key>` 替换为在 `API Keys` 页面创建的 API 密钥。

响应示例：

```json
{
  "model": "qwen3-decision-8b",
  "usage": {
    "input_tokens": 189,
    "output_tokens": 12
  },
  "answers": {
    "model_selection": {
      "type": "choice",
      "choice": "qwen3-8b",
      "confidence": 0.93,
      "probabilities": {
        "qwen3-8b": 0.93,
        "llama3-8b": 0.07
      }
    }
  }
}
```

`questions` 下的每个条目将问题 id 映射到三种常规类型之一，问题的答案会在 `answers` 中以相同 id 返回：

| 类型 | 含义 | `criteria` | 答案字段 |
| --- | --- | --- | --- |
| `choice` | 选择一个选项 | 选项名 → 描述 | `choice`——选中的选项名 |
| `score` | 按有序级别对 state 评分 | 有序的级别描述列表 | `score`——级别索引 |
| `noul` | 二元（是/否）问题 | `true` / `false` 两侧的可选描述 | `noul`——布尔值 |

GPUStack 解析 `model` 字段，请求体其余部分和整个响应原样透传给决策引擎——契约中的问题集是开放的，引擎可以接受更多字段。

## 后续步骤 { #next-steps }

- 已部署的决策模型可以作为路由 [Decision Service Routing](../user-guide/model-route-management.md#decision-service-routing) 策略的决策服务，按请求为路由目标评分。
- 完整端点参考参见[推理 API](../integrations/inference-apis.md)。
