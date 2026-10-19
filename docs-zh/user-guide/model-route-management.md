---
translation_status: translated
---

# 模型路由管理 { #model-route-management }

GPUStack 提供模型路由管理能力。通过模型路由，可以实现模型别名、流量分配、容灾，以及公共模型和私有模型的统一入口。

## 创建路由 { #create-route }

1. 进入 `Routes` 页面。
2. 单击 `Add Route`。
3. 填写路由 `Name`，作为对外服务的模型名称。
4. 选择一个或多个 `Route Targets`。
5. 单击 `Save` 按钮。

## 管理路由目标 { #manage-route-targets }

1. 进入 `Routes` 页面。
2. 展开要管理目标的模型路由。
3. 对不再保留的目标，单击其 `Operations` 列中的 `Delete` 按钮。
4. 单击 `Operations` 列中的 `Edit` 按钮。在编辑路由页面中，为模型路由添加或移除目标，或调整目标的流量权重。
5. 在编辑路由页面中，为此路由选择或清除回退路由目标。

## 负载均衡 { #load-balancing }

模型路由包含多个目标时，GPUStack 网关侧的 LB 插件会决定每个请求由哪个目标处理。路由行为由路由配置推导：目标权重与已启用的策略插件（Session Affinity、Least Inflight Request、Decision Service Routing）共同决定路由上显示的 `Route By` 模式。

### Route By { #route-by }

路由列表中的 `Route By` 列显示请求在路由目标间的分配方式。它由 GPUStack 在每次路由变更时，根据配置中的目标权重和已启用策略插件计算得出，绝不是客户端输入。

| Route By | 配置形式 | 路由行为 |
| --- | --- | --- |
| `Target Weight` | 每个目标的权重均大于 0 | 按权重在目标间分配流量；每个目标的权重必须大于 0 |
| `Policy` | 无权重，且路由至少启用了一个策略插件（Session Affinity、Least Inflight Request、Decision Service Routing） | 由启用的策略插件选择目标 |
| *（轮询）* | 所有目标都没有权重，且未启用策略 | 在目标间轮询 |

说明：

- 权重与策略互斥：要么每个目标都有权重，要么所有目标都没有权重。混合配置会使路由不可用，请将每个目标权重都设置为大于 0，或全部设置为 0。
- 回退目标永远不作为候选目标，其权重在推导模式时会被忽略。
- 此模式描述配置形式，而非实时实例健康状态；即使实例停机，全部目标均配置权重的路由仍显示 `Target Weight`。

### 策略插件 { #policy-plugins }

策略插件参与 `Policy` 路由：每个插件为各目标返回一个分数，网关选择加权总分最高的目标。每个插件的 `Weight`（正数，允许小数）决定其在总分中的占比；未设置时使用插件的内置默认值。

#### 会话亲和性 { #session-affinity }

将同一会话的请求路由到同一目标，保持提示词缓存热度。会话通过有序键链识别。

- `Session Keys (Ordered, First Match Wins)`——按顺序排列的键来源链，首个产生值的来源生效。每个条目可以是请求 `Header`（例如 `session-id`）或 `Body Key`（例如 `prompt_cache_key`）。启用会话亲和性时，至少需要一个会话键。

#### 最少在途请求 { #least-inflight-request }

根据各目标正在处理的请求数评分，优先将请求路由到在途请求最少的目标。

同时启用 Session Affinity 和 Least Inflight Request 时，会话亲和性奖励会话当前绑定的目标，而最少在途请求策略奖励空闲目标。只有最少在途请求插件的投票差距超过亲和性插件时，会话才会迁离原目标；默认权重下，这在包含 2–3 个目标的路由中很容易发生，而在包含 5 个及以上目标的路由中不会发生。可以调整插件的 `Weight` 改变这一平衡；具体数值和示例参见[使用 LB 策略进行路由](../tutorials/routing-with-lb-policies.md)。

#### 决策服务路由 { #decision-service-routing }

按任务难度对目标评分：每个请求会被截断，并附带模型选择问题发送到兼容 Jev 的决策服务，即 `Model` - `Provider` 页面中的 `TypeSafe Decision Service (Jev)` 提供商，其回答会转换为加权投票。决策失败、超时或收到非 2xx 响应时，会静默回退到其他插件。

设置项：

- `Decision Service`——决策服务提供商（必填）。
- `Decision Model`——路由级决策模型，启用此策略时必填。选项来自所选提供商中缓存的引擎；未设置或值无效时，跳过决策路由，请求回退到其他插件。
- `Instructions`——发送给决策服务的模型选择问题的附加指导。
- `Model Criteria`——模型名称 → 能力描述：决策服务选择模型的依据。至少需要一条标准，可从路由目标生成条目。
- `Weight`——与其他策略组合时，在加权总分中的贡献权重；未设置时使用内置默认值（10）。

小模型与大模型之间路由的完整示例，参见[使用 LB 策略进行路由](../tutorials/routing-with-lb-policies.md)。

回退目标（`fallback_status_codes`）由独立的回退插件处理，不参与 LB 候选目标选择。路由可以将 LB 候选目标的流量分配与回退目标结合，用于容灾。

## 授权路由访问 { #authorize-route-access }

1. 进入 `Routes` 页面。
2. 找到要修改授权设置的路由。
3. 单击 `Operations` 列中的 `Access Settings` 按钮。
4. 根据需要更改 `Access Scope`。
5. 对于 `Allowed Users` 范围，选择要授权访问此路由的用户，单击 `>` 确认。
6. 单击 `Save` 按钮。

## 编辑路由 { #edit-route }

1. 进入 `Routes` 页面。
2. 找到要编辑的路由。
3. 单击 `Operations` 列中的 `Edit` 按钮。
4. 根据需要修改名称、模型类别、描述和路由目标。
5. 单击 `Save` 按钮。

## 删除路由 { #delete-route }

1. 进入 `Routes` 页面。
2. 找到要删除的路由。
3. 单击 `Operations` 列中的 `Delete` 按钮。
4. 确认删除。
