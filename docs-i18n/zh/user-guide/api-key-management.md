---
translation_status: translated
---

# API 密钥管理 { #api-key-management }

GPUStack 支持使用 API 密钥进行身份验证。每位 GPUStack 用户都可以生成和管理自己的 API 密钥。

## 创建 API 密钥 { #create-api-key }

1. 进入 `Access Control` > `API Keys` 页面。
2. 单击 `Add API Key` 按钮。
3. 填写 API 密钥的 `Name`、`Description`，并选择 `Expiration`。
4. 选择 API 密钥的 `Type`：

   - **Auto-generated**：由 GPUStack 为你生成 API 密钥。
   - **Custom**：在 `Key` 字段中提供自己的密钥值。选择 **Custom** 时，此字段为必填项。请使用足够强的值，达到无需人工记忆的密码应有的强度，详见下方说明。

5. 在 `Access Permissions` 部分选择 API 密钥的权限：

   - **Platform Management**：允许访问平台管理端点（用户、模型、Worker 等）。
   - **Model Access**：允许访问推理 API。选中后，选择 `All models` 或 `Allowed models`；若选择 `Allowed models`，请从列表中指定该 API 密钥可以访问的模型。

   如果未选择任何权限，该 API 密钥将没有访问权限。
6. 单击 `Save` 按钮。
7. 复制密钥并妥善保存，然后单击 `Done` 按钮。

!!! note

    完整的 API 密钥值仅在创建时显示一次，此后只显示掩码值。自定义密钥除外，因为其值由你自行提供。

!!! warning "自定义密钥的强度取决于你提供的值"

    GPUStack 接受任意字符串作为自定义密钥，无法区分随机值和 `123456`。自动生成的密钥来自密码学随机源，具有 128 位强度，不会意外生成更弱的密钥。

    这种差异不仅关系到密钥是否容易被猜中：为了在 API 网关而非 GPUStack Server 上验证密钥，GPUStack 会将各个密钥的快速哈希值发布到网关配置中。任何能够读取该配置的人都能看到这些哈希值，它们也会出现在诊断包中。自定义密钥的哈希不加盐，且仅由密钥本身决定，因此在所有部署中的值都相同，一份预计算表就能用于所有部署。随机密钥不受影响，因为无论哪种方式，搜索出其原值都不可行；容易被猜中的密钥则可能以较低成本被离线恢复。

    请尽可能使用随机源生成自定义密钥。如果密钥由人工选定，请设置 `GPUSTACK_GATEWAY_AUTH_ALLOW_CUSTOM_KEYS=false`。这样，自定义密钥会像此前一样在每次请求时交给 GPUStack Server 验证，且不会发布其哈希值。网关只会获知指定 ID 的密钥存在及其过期时间，这使得 GPUStack Server 停机期间仍可撤销密钥。

    此设置也适用于现有密钥，而不仅是新密钥：关闭后，下一次协调同步会撤回此前在开启状态下创建的自定义密钥的哈希值。

## 编辑访问权限 { #edit-access-permissions }

1. 进入 `Access Control` > `API Keys` 页面。
2. 找到要编辑的 API 密钥。
3. 单击 `Operations` 列中的 `Edit` 按钮。
4. 根据需要更新 `Description`。
5. 在 `Access Permissions` 部分调整所选权限：

   - **Platform Management**：允许访问平台管理端点（用户、模型、Worker 等）。
   - **Model Access**：允许访问推理 API。选中后，选择 `All models` 或 `Allowed models`；若选择 `Allowed models`，请从列表中指定该 API 密钥可以访问的模型。

6. 单击 `Save` 按钮。

!!! note

    更改将在一分钟内生效。

## 删除 API 密钥 { #delete-api-key }

1. 进入 `Access Control` > `API Keys` 页面。
2. 找到要删除的 API 密钥。
3. 单击 `Operations` 列中的 `Delete` 按钮。
4. 确认删除。

## 在 GPUStack Server 不可用时撤销密钥 { #revoke-a-key-while-the-server-is-unavailable }

通常，删除密钥就已足够：GPUStack Server 会通知 API 网关，密钥会在几秒内停止被接受。本节仅适用于必须**立即**使密钥失效，而 GPUStack Server 正处于停机状态（例如正在升级）的情况。

网关保留一份可接受密钥的副本，因此 GPUStack Server 不可用时仍能提供推理服务。这份副本存储在 Kubernetes 中，而非 GPUStack Server 中，所以无需 GPUStack Server 也可以编辑：

```bash
kubectl -n higress-system edit wasmplugin gpustack-llm-ext-auth
```

删除 `spec.defaultConfig.local_auth` 下该密钥的条目。条目几乎总是位于 `keys` 中，并以密钥的 access key 为索引；对于自动生成的密钥，access key 是 `gpustack_<access key>_<secret>` 中间的部分。自定义密钥内部没有 access key，因此请在 `keys` 中搜索其所属用户的数值型 `user_id`，或根据密钥的过期时间匹配 `exp` 值。

当 `GPUSTACK_GATEWAY_AUTH_ALLOW_CUSTOM_KEYS` 设为 `false` 时，自定义密钥位于 `refs` 中，索引是 API 所显示的密钥数值 ID。

如果密钥既不在 keys 中，也不在 refs 中，说明它本就会在每次请求时交给 GPUStack Server 验证，因此没有需要移除的条目；GPUStack Server 恢复并处理删除操作后，该密钥就会停止工作。

保存更改后，网关会在几秒内收到更新，其传播方式与此资源的其他更改相同。正在处理的会话也会受到影响：网关在无需询问 GPUStack Server 即可接受请求之前，都会先根据这些条目重新检查密钥，因此移除条目会立即使其在所有位置失效。

!!! warning

    这只是临时措施，不是真正的撤销。GPUStack Server 启动后会在几秒内根据数据库重建此列表，因此**仅在此处移除的密钥会随 GPUStack Server 恢复而重新出现**。GPUStack Server 可用后，还应通过正常方式删除该密钥。

!!! note

    编辑此资源需要具备网关所在集群的访问权限。上文的 `higress-system` 是网关的默认命名空间，如果已更改，请使用实际值替换。不支持直接修改 GPUStack 数据库作为替代方案，这可能导致部署状态不一致。

!!! warning "谁能读取此命名空间"

    同一资源还存储了网关用于签署内部身份令牌的密钥。因此，任何能够读取网关命名空间中资源的人，都可以签发令牌并以任意用户的身份通过验证。请将此处的读取权限视为平台级权限，并据此授权。

## 使用 API 密钥 { #use-api-key }

GPUStack 支持将 API 密钥用作 Bearer 令牌。

兼容 OpenAI 的 `GET /v1/models` 端点只返回在当前用户和组织上下文中可见且已就绪的模型。如果 API 密钥被限制为仅可访问指定模型，列表还会进一步按这些模型 ID 过滤，其中包括组织前缀（例如 `team-a/qwen3`）。这一行为也适用于旧版 `GET /v1-openai/models` 端点，以及管理员创建的密钥。如果 `allowed_model_names` 列表未设置或为空，则不会施加额外的模型访问限制。

以下是使用 curl 的示例：

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
