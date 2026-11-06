---
translation_status: translated
---

# 使用负载均衡策略进行路由 { #routing-with-lb-policies }

本教程演示 GPUStack 的路由级负载均衡策略：在分散负载的同时保持会话粘性，以及通过决策服务为每个请求选择小模型或大模型。配置参考见[模型路由管理](../user-guide/model-route-management.md#load-balancing)。

## 前提条件 { #prerequisites }

- GPUStack 已安装并运行。应用这些策略的 AI 网关默认随 GPUStack 部署，标准安装无需额外配置。
- 已部署两个模型作为路由目标，例如下文使用的 `qwen3-0.6b` 和 `qwen3-32b`。
- 对于决策服务部分：需要一个决策服务——可以是已在 `Model` - `Provider` 页面注册为 `TypeSafe Decision Service (Jev)` 提供商的 Jev 决策服务（参见[模型提供商管理](../user-guide/model-provider-management.md)），也可以是本地部署的决策模型（参见[使用决策模型](../using-models/using-decision-models.md)）。

## 步骤 1：创建路由 { #step-1-create-the-route }

1. 打开 `Routes` 页面。
2. 点击 `Add Route`，输入名称，例如 `qwen3`。
3. 选择 `qwen3-0.6b` 和 `qwen3-32b` 作为 `Route Targets`，不设置任何权重。
4. 点击 `Save`。

没有权重和策略时，路由的 `Route By` 为 round-robin，即在两个模型之间轮询。

## 步骤 2：理解会话亲和性与最少在途请求策略的交互 { #step-2-understand-how-session-affinity-and-least-inflight-request-interact }

启用策略插件后，每个插件为所有候选目标打分，最终选择器根据加权总分选择目标：

```
total(candidate) = Σ over plugins ( plugin weight × normalized score )
```

会话亲和性为会话当前归属目标打最高分；最少在途请求策略为最空闲的目标打最高分。因此，只有后者的分差超过前者的分差时，会话才会离开当前归属目标。

两个插件都使用默认权重 1 时，下表展示当前归属目标积累了 `x` 个在途请求、其他目标全部空闲时的行为；这是最容易发生切换的情况：

| 目标数（N） | x=0 | x=1 | x=2 | x=3 | x=4 | 是否会切换？ |
| ----------- | --- | --- | --- | --- | --- | ----------- |
| 2 | 保持 | 平局 ⚠️ | **切换** | **切换** | **切换** | 是 |
| 3 | 保持 | 保持 | 平局 ⚠️ | **切换** | **切换** | 是 |
| 4 | 保持 | 保持 | 保持 | 保持 | 保持 | 仅在 x ≈ 5.3 时 |
| ≥5 | 保持 | 保持 | 保持 | 保持 | 保持 | **否** |

- **切换**：会话迁移到下一个候选目标。
- **平局 ⚠️**：两个分差完全相同，最终选择器随机打破平局；同一请求可能迁移，也可能不迁移。
- **N ≥ 5**：最少在途请求策略的分差上限为 `1/(N−1) ≤ 0.25`，低于亲和性的分差，因此仅靠负载不均衡无法打破粘性。

要让较大路由中的负载发挥作用，需要重新平衡权重。[权重如何工作](#how-weights-work)给出了调整权重比例后的切换阈值。

## 步骤 3：启用会话亲和性和最少在途请求策略 { #step-3-enable-session-affinity-and-least-inflight-request }

1. 编辑步骤 1 中的路由。
2. 启用 `Session Affinity`，接受默认会话键链，或自行指定。默认依次使用 `session-id` 请求头、`x-client-request-id` 请求头，再使用请求体中的 `prompt_cache_key`。
3. 启用 `Least Inflight Request`，使用默认权重。
4. 保存，确认路由的 `Route By` 为 `Policy`。

此时会话具备粘性：携带相同会话键的请求会保留在同一目标上，使提示词缓存持续命中；当前目标负载较高时，按上表规则将会话迁移给空闲目标。

不同 API 路径支持的键来源不同：请求头键，例如 `session-id`，适用于所有路径；`prompt_cache_key` 等请求体键仅对 `/responses` 和 `/messages` 请求生效。`/chat/completions` 没有标准的请求体会话字段，因此只使用键链中的请求头键。对话流量应优先使用请求头键。

## 步骤 4：添加决策服务路由 { #step-4-add-decision-service-routing }

接下来让路由感知模型能力：不再始终将会话固定到最初选择的模型，而是让决策引擎判断每个请求适合哪个模型。

1. 编辑路由，启用 `Decision Service Routing`：
    - `Decision Service`：选择已注册的提供商及其某个引擎（例如 `jev-latest`），也可以改为选择本地部署的决策模型。
    - `Instructions`：“选择能够妥善处理请求的最小模型。简单、简短或事实类请求交给快速模型；复杂推理、数学或代码请求交给大模型。”
    - `Model Criteria`：为每个目标添加一项，也可以点击 `Generate from targets` 后再调整：
        - `qwen3-0.6b`：“非常快速、低成本的 0.6B 模型。适合问候、简单事实问答、格式整理和简短闲聊，不擅长多步推理、数学和代码。”
        - `qwen3-32b`：“能力较强的 32B 推理模型。适合复杂推理、数学、代码生成和长上下文分析，每次请求较慢且成本更高。”

2. 保存后，以路由名 `qwen3` 发送两个经过认证的请求。将 `$GPUSTACK_API_KEY` 替换为 `API Keys` 页面创建的密钥：

```bash
curl http://my-gpustack/v1/chat/completions \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -H "Content-Type: application/json" \
  -H "session-id: demo-session" \
  -d '{"model": "qwen3", "messages": [{"role": "user", "content": "Write a one-line greeting card."}]}'
```

```bash
curl http://my-gpustack/v1/chat/completions \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -H "Content-Type: application/json" \
  -H "session-id: demo-session" \
  -d '{"model": "qwen3", "messages": [{"role": "user", "content": "Prove that there are infinitely many primes, then implement the proof sketch in Python."}]}'
```

第一个请求会转发到 `qwen3-0.6b`，第二个会转发到 `qwen3-32b`。

## 权重如何工作 { #how-weights-work }

权重出现在两个位置，含义不同。

### 目标权重（流量份额） { #target-weight-traffic-share }

在 `Target Weight` 模式中，目标权重代表该目标的**总流量份额**，与实例数量无关。GPUStack 会调整实例级权重，确保分流结果始终符合目标权重：

- 目标 A 的权重为 100、运行 1 个实例，目标 B 的权重为 100、运行 2 个实例，两个目标各获得 50% 的流量，并非每个实例各获得 1/3。
- 将一个目标从 1 个实例扩容到 2 个实例，不会改变目标的总份额；各实例会各获得该份额的一半。

例如，`qwen3-0.6b` 权重为 100，`qwen3-32b` 权重为 300，则 25% 的请求发往小模型，75% 发往大模型，无论其副本数是多少。

### 策略权重（在总分中的影响力） { #policy-weight-influence-in-the-sum }

各策略插件的 `Weight` 控制其在 `Policy` 模式加权总分中的影响力。权重为正数，小数也有意义。不设置 `Weight` 时，使用插件内置默认值：Session Affinity 和 Least Inflight Request 默认为 1，Decision Service Routing 默认为 10。

当当前归属目标有 `x` 个在途请求、其他目标空闲时，会话亲和性与最少在途请求策略的平衡计算如下：

- 会话亲和性分差：`affDiff = 0.5 / (2·(1 − 2⁻ᴺ))`
- 最少在途请求分差：`loadDiff = x / ((1+x)·(N−1) + 1)`
- 当 `w_leastInflight × loadDiff > w_affinity × affDiff` 时，会话切换。

调整权重会改变切换阈值。下表按权重比例（Least Inflight Request : Session Affinity）和目标数 N，列出引起切换的最小在途请求数 `x`：

| N | 1:1（默认） | 1.5:1 | 2:1 |
| - | ------------- | ----- | --- |
| 2 | 2 | 1 | 1 |
| 3 | 3 | 1 | 1 |
| 4 | 6 | 2 | 1 |
| 5 | 从不 | 3 | 2 |
| 6 | 从不 | 7 | 3 |
| 7 | 从不 | 从不 | 4 |
| 8 | 从不 | 从不 | 9 |

表格解读：

- 默认 1:1 比例下，包含 5 个或更多目标的路由永远不会切换，因为最少在途请求策略的分差上限低于亲和性分差。
- 将 Session Affinity 的 `Weight` 减半，即采用 2:1 比例，会让最多 8 个目标的路由在 `x` = 2–4 时受负载影响；将 Least Inflight Request 的 `Weight` 加倍效果相同，因为仅比例影响结果。
- 完全相同的分差，例如 1:1 且 N=2/x=1，由最终选择器随机打破平局。

## 各策略如何分工 { #how-the-policies-divide-the-work }

所有已启用策略都参与同一加权总分计算，各司其职：

- **Decision Service Routing** 判断哪个模型适合请求，默认 `Weight` 为 10，通常具有主导作用。
- **Session Affinity** 将会话保留在当前模型上，维持提示词缓存的局部性。
- **Least Inflight Request** 在胜出模型的各实例之间分散请求。

模型部署了多个实例时，决策服务先将选择范围缩小到该模型，再由 Least Inflight Request 在模型内部选择实例。
