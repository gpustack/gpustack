---
translation_status: translated
---

# 环境变量 { #environment-variables }

GPUStack 支持通过多种环境变量进行配置。

大多数命令行参数也可通过带有 `GPUSTACK_` 前缀的全大写环境变量设置（例如，可通过 `GPUSTACK_DATA_DIR` 设置 `--data-dir`）。

可设置为环境变量的命令行参数完整列表，请参阅 [CLI 参考](cli-reference/start.md#available-environment-variables)，本文不再重复说明。

### 优先级顺序 { #priority-order }

配置值按以下优先级应用（从高到低）：

1. 命令行参数
2. 环境变量
3. 配置文件
4. 默认值

这意味着命令行参数始终覆盖环境变量，环境变量则覆盖配置文件中的值。

## GPUStack 核心环境变量 { #gpustack-core-environment-variables }

这些环境变量通常用于集成第三方服务。

**适用对象**列表示应在何处设置环境变量：

- **Server** - 适用于 GPUStack Server。
- **Worker** - 适用于 GPUStack Worker。
- **模型** - 适用于模型部署配置。

### 代理配置 { #proxy-configuration }

| <div style="width:100px">变量</div> | 说明                                                                                                                 | 默认值 | 适用对象      |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------- | --------------- |
| `HTTP_PROXY`                            | HTTP 代理 URL，例如 `http://proxy-server:port`。                                                                            | (empty) | Server & Worker |
| `HTTPS_PROXY`                           | HTTPS 代理 URL，例如 `https://proxy-server:port`。                                                                          | (empty) | Server & Worker |
| `NO_PROXY`                              | 不经过代理的主机列表，以逗号分隔。例如 `127.0.0.1,10.0.0.0/8,192.168.0.0/16,172.16.0.0/16,localhost,cluster.local`。 | (empty) | Server & Worker |

### Hugging Face Hub { #hugging-face-hub }

| <div style="width:100px">变量</div> | 说明                                              | 默认值 | 适用对象      |
| --------------------------------------- | -------------------------------------------------------- | ------- | --------------- |
| `HF_ENDPOINT`                           | Hugging Face Hub 端点，例如 `https://hf-mirror.com`。 | (empty) | Server & Worker |
| `HF_TOKEN`                              | Hugging Face Hub 访问 Token。                           | (empty) | Server & Worker |

### 调试 { #debugging }

| 变量 | 说明 | 默认值 | 适用范围 |
| --- | --- | --- | --- |
| `GPUSTACK_DEBUG_ENABLE_TRACEMALLOC` | 启动时启用 Python 内存分配追踪。独立于 `--debug`；追踪会增加 CPU 和内存开销。修改后需要重启 GPUStack。 | `false` | Server 和 Worker |

### 数据库配置 { #database-configuration }

| 变量                   | 说明                                  | 默认值 | 适用对象 |
| -------------------------- | -------------------------------------------- | ------- | ---------- |
| `GPUSTACK_DB_ECHO`         | 启用数据库查询日志。               | `false` | Server     |
| `GPUSTACK_DB_POOL_SIZE`    | 数据库连接池大小。               | `30`    | Server     |
| `GPUSTACK_DB_MAX_OVERFLOW` | 数据库连接池最大溢出连接数。       | `20`    | Server     |
| `GPUSTACK_DB_POOL_TIMEOUT` | 数据库连接池超时时间，单位为秒。 | `30`    | Server     |
| `GPUSTACK_DB_POOL_RECYCLE` | 数据库连接回收时间，单位为秒。 | `1800`  | Server     |

### 网络配置 { #network-configuration }

| 变量                                       | 说明                                                                                                                                                  | 默认值 | 适用对象      |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------- | --------------- |
| `GPUSTACK_PROXY_TIMEOUT_SECONDS`               | 代理推理请求的绝对最长时限，单位为秒。                                                                                                 | `1800`  | Server & Worker |
| `GPUSTACK_PROXY_TTFT_TIMEOUT_SECONDS`          | 等待流式响应首个数据块的兜底超时时间，单位为秒。由于包含排队和预填充阶段，应设置得足够宽裕。                             | `300`   | Server & Worker |
| `GPUSTACK_PROXY_STREAM_IDLE_TIMEOUT_SECONDS`   | 流式响应开始后，数据块之间允许的最长间隔，单位为秒。不适用于非流式响应，因为其响应体仅在结束时返回。 | `60`    | Server & Worker |
| `GPUSTACK_PROXY_UPSTREAM_IDLE_TIMEOUT_SECONDS` | Higress 上游连接池的空闲超时时间，单位为秒。与请求持续时间无关：它回收空闲连接，不限制单个请求的时长。        | `3`     | Server          |
| `GPUSTACK_TCP_CONNECTOR_LIMIT`                 | HTTP 客户端 TCP 连接器的连接数限制。                                                                                                                             | `1000`  | Server & Worker |

### Server 缓存配置 { #server-cache-configuration }

| 变量                            | 说明                  | 默认值 | 适用对象 |
| ----------------------------------- | ---------------------------- | ------- | ---------- |
| `GPUSTACK_SERVER_CACHE_TTL_SECONDS` | Server 缓存 TTL，单位为秒。 | `600`   | Server     |

### 身份认证与安全 { #authentication-security }

| 变量                            | 说明                                                                                                                                                                              | 默认值 | 适用对象      |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | --------------- |
| `GPUSTACK_JWT_TOKEN_EXPIRE_MINUTES` | JWT token 过期时间，单位为分钟。                                                                                                                                                    | `120`   | Server          |
| `GPUSTACK_PROVIDER_TEST_EGRESS_ALLOWLIST` | 提供方端点（`/get-models`、`/test-model`、`/test-decision-model`）连接调用方指定的基础 URL 时，允许访问的 CIDR、IP 地址和 DNS 名称列表。仅当目标主机名等于某个允许名称或以该名称结尾（以点为边界），或所有解析出的地址均位于某个允许 CIDR 内时，才允许访问。其他目标均被拒绝，包括非 http(s) 基础 URL。为空表示不限制。例如 `api.openai.com,10.0.1.0/24,vllm.internal.corp`。 | (empty) | Server          |

### 网关配置 { #gateway-configuration }

| 变量                                           | 说明                                                                                                                                                                                                                | 默认值   | 适用对象 |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ---------- |
| `GPUSTACK_GATEWAY_PORT_CHECK_INTERVAL`             | GPUStack Server 检查内置网关监听端口的间隔，单位为秒。                                                                                                                                        | `2`       | Server     |
| `GPUSTACK_GATEWAY_PORT_CHECK_RETRY_COUNT`          | GPUStack Server 检查内置网关监听端口的重试次数。                                                                                                                                                | `300`     | Server     |
| `GPUSTACK_GATEWAY_AUTH_RECONCILE_INTERVAL_SECONDS` | Server 根据数据库重新计算网关本地认证 API key 的频率。绕过 ORM 的删除不会发出事件，因此在公共路由上，此值表示这类已删除 key 最长可能继续生效的时间。 | `30`      | Server     |
| `GPUSTACK_GATEWAY_AUTH_ALLOW_CUSTOM_KEYS`          | 是否允许网关认证自定义 API key（用户自行提供密钥值的 key）。关闭时，key 仍可使用，但每次请求都由 Server 认证。开启时，key 会发布到网关配置中，并以密钥本身的无盐快速哈希作为索引；该哈希在不同部署间一致，因此弱密钥可被预计算表破解。`custom` 不要求密钥熵，因此允许用户自行选择密钥的环境应关闭此选项。每次协调时都会重新读取此值，所以关闭后会撤回开启期间已发布的自定义 key，而不只是影响新 key。 | `true`    | Server     |
| `GPUSTACK_GATEWAY_AUTH_MAX_CR_BYTES`               | Server 写入网关认证插件的 key 表和公共路由规则的字节预算。应低于 etcd 约 1.5 MiB 的对象上限；超出预算的 key 会在每次请求时通过 Server 认证。                | `1100000` | Server     |
| `GPUSTACK_GATEWAY_TLS_MIN_PROTOCOL_VERSION`        | 网关 HTTPS 监听器接受的最低 TLS 协议版本。可选值为 `TLSv1.0`、`TLSv1.1`、`TLSv1.2`、`TLSv1.3`，其他值会在启动时被拒绝。为空时，Higress 使用默认值，接受 TLS 1.0 和 1.1。在 `incluster` 模式下忽略此设置，详见下文。 | (empty)   | Server     |
| `GPUSTACK_GATEWAY_TLS_MAX_PROTOCOL_VERSION`        | 上述监听器接受的最高 TLS 协议版本。可选值和适用范围相同，通常无需设置。                                                                                                                                         | (empty)   | Server     |

TLS 协议版本配置会写入 `GPUSTACK_GATEWAY_MIRROR_INGRESS_NAME` 指定的 Ingress，GPUStack 为 LLM 路由生成的每个 Ingress 都从中复制版本上下限。
在 `incluster` 模式下，该 Ingress 由 Helm chart 管理，应在 chart 中设置 `server.ingress.tls.minProtocolVersion` / `maxProtocolVersion`。
上述环境变量会被忽略，设置时会输出警告。变更会立即应用到该 Ingress，并在下次协调时应用到已有 LLM 路由；重启 Server 可立即传播。

最终进入网关插件**内部**配置的设置，应放在配置文件的 `gateway_plugin` 下，而不是通过此处的环境变量设置，从而让两种机制各自具有明确的生效时间：配置文件中的值在下次协调时生效，此处的环境变量在重启时生效。
以下两个设置已移至配置文件，对应环境变量已弃用。但它们仍作为可由配置文件覆盖的默认值生效，避免升级时静默改变已有部署。设置其中任一变量都会输出警告。

| 已弃用变量                                   | 说明                                              | 替代设置                                                | 默认值                              | 适用对象 |
| ----------------------------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------ | ---------- |
| `GPUSTACK_HIGRESS_EXT_AUTH_TIMEOUT_MS`                | 网关身份认证调用超时时间，单位为毫秒。    | `gateway_plugin.gpustack-ext-auth.config.authz.timeout`    | `30000`                              | Server     |
| `GPUSTACK_GATEWAY_AI_STATISTICS_PLUGIN_CONTENT_TYPES` | 纳入 token 用量计量的内容类型，以逗号分隔。   | `gateway_plugin.ai-statistics.config.enable_content_types` | `application/json,text/event-stream` | Server     |

### 用量跟踪配置 { #usage-tracking-configuration }

| 变量                                           | 说明                                                                                                                                                                                                                                        | 默认值      | 适用对象 |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ---------- |
| `GPUSTACK_TIMEZONE`                                | 平台统一的 IANA 时区（例如 `Asia/Shanghai`），用于所有用量视图的时间分桶，以及最近活跃时间和事件时间的显示（参见下方说明）。为空时使用操作系统本地时区（`TZ` / `/etc/localtime`）。替代已弃用的 `GPUSTACK_USAGE_ROLLUP_TIMEZONE`（仍作为别名生效）。 | (empty)      | Server     |
| `GPUSTACK_USAGE_ESTIMATED_BYTES_PER_INPUT_TOKEN`   | 当网关报告为 `completed=false` 且 `input_token` 为空时，用于估算提示词 tokens 的每 token 字节数除数。默认值针对以英文为主的 GPT 风格分词器；对于以中日韩文字为主的工作负载，应降低此值（例如 `2`）。         | `4`          | Server     |
| `GPUSTACK_USAGE_ESTIMATED_TOKENS_PER_OUTPUT_CHUNK` | 对于 `output_token` 为空的不完整流式响应，用于估算生成 tokens 的每数据块 token 数乘数。                                                                                                                                | `1`          | Server     |
| `GPUSTACK_USAGE_DETAILS_RETENTION_MONTHS`          | `model_usage_details`（逐请求审计表）的保留期限。仅由 leader 运行的归档器会将早于此期限的记录（以 `COALESCE(completed_at, created_at)` 为基准）移至 `model_usage_details_archive`。                          | `13`         | Server     |
| `GPUSTACK_USAGE_DETAILS_ARCHIVE_CRON`              | 用量明细归档器定期扫描使用的 Cron 表达式（UTC）。无论此计划如何设置，归档器也会在 Server 启动时执行一次。                                                                                                 | `0 3 * * *`  | Server     |
| `GPUSTACK_USAGE_DETAILS_ARCHIVE_BATCH_SIZE`        | 每批归档迁移的行数。较小批次有助于在对复制延迟敏感的环境中缩短事务，较大批次可减少往返开销。                                                                           | `1000`       | Server     |
| `GPUSTACK_USAGE_DETAILS_BUFFER_MAX_SIZE`           | 两次刷写之间，内存中的逐请求明细缓冲区的硬上限。在持续刷写失败（数据库不可用、表结构不一致）时限制内存增长。溢出时丢弃最早的记录，并输出 `WARNING` 日志。                           | `100000`     | Server     |
| `GPUSTACK_RESOURCE_USAGE_TICK_SECONDS`             | 实例用量采集器的定时执行间隔（秒）。对于自上次执行以来没有生命周期事件的长时间运行 GPU/CPU 实例，此任务会刷写累积的运行时长。                                                    | `300`        | Server     |
| `GPUSTACK_STORAGE_USAGE_TICK_SECONDS`              | 持久卷（存储）用量采集器的定时执行间隔（秒）。                                                                                                                                                            | `300`        | Server     |
| `GPUSTACK_METERED_USAGE_SEAL_GRACE_SECONDS`        | 小时桶结束后、封存（最终确定）前的宽限窗口（秒）。当 `now >= bucket_end + grace` 时封存，用于容纳延迟事件或时钟偏差；此值应大于采集间隔。已封存的桶不可修改。 | `900`        | Server     |
| `GPUSTACK_METERED_USAGE_RETENTION_MONTHS`          | `metered_usage`（每小时资源用量汇总）的保留期限。仅由 leader 运行的归档器会将早于此期限的已封存记录移至 `metered_usage_archive`。该值必须远大于采集器的结算时间范围（小时）。            | `13`         | Server     |
| `GPUSTACK_METERED_USAGE_ARCHIVE_CRON`              | 计量用量归档器定期扫描使用的 Cron 表达式（UTC）。无论此计划如何设置，归档器也会在 Server 启动时执行一次。                                                                                                 | `0 4 * * *`  | Server     |
| `GPUSTACK_METERED_USAGE_ARCHIVE_BATCH_SIZE`        | 计量用量每批归档迁移的行数。较小批次可缩短事务，较大批次可减少往返开销。                                                                                                          | `5000`       | Server     |
| `GPUSTACK_USAGE_EVENTS_RETENTION_MONTHS`           | `resource_events`（资源生命周期与审计日志）的保留期限。仅由 leader 运行的归档器会将早于此期限的记录移至 `resource_events_archive`。                                                                              | `13`         | Server     |
| `GPUSTACK_USAGE_EVENTS_ARCHIVE_CRON`               | 资源事件归档器定期扫描使用的 Cron 表达式（UTC）。无论此计划如何设置，归档器也会在 Server 启动时执行一次。                                                                                               | `30 3 * * *` | Server     |
| `GPUSTACK_USAGE_EVENTS_ARCHIVE_BATCH_SIZE`         | 资源事件每批归档迁移的行数。                                                                                                                                                                                            | `5000`       | Server     |
| `GPUSTACK_USAGE_BREAKDOWN_MAX_NO_PAGINATION_ROWS`  | 不分页（`page=-1`）的明细分组查询最多可返回的桶数；趋势图和导出会一次获取完整序列。超出限制的请求会被拒绝（HTTP 400），不会静默截断。调高可支持更宽范围的仪表盘，调低可限制内存使用。       | `50000`      | Server     |

!!! note "`GPUSTACK_TIMEZONE` 的适用范围与夏令时"

    此统一时区控制所有用量视图：每日 `model_usages` token 汇总、GPU/存储时间桶，以及显示的最近活跃时间和资源事件时间。对于采用**夏令时（DST）**的时区，需要注意两点：

    - 夏令时切换附近，事件显示的时间可能与其计入的时间桶不在同一日历日。
    - 夏令时切换后重新运行相同的历史查询，旧记录可能被重新分配到相差一小时的桶中，因此结果会随时间变化。

    不采用夏令时的时区（UTC、`Asia/Shanghai` 等）计算准确，不受影响。

### 集群配置 { #cluster-configuration }

| 变量                              | 说明                                                                                                     | 默认值 | 适用对象 |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------- | ---------- |
| `GPUSTACK_DEFAULT_CLUSTER_KUBERNETES` | 自动创建默认集群时，如果设置此变量，将使用 `Kubernetes` 提供方。 | `false` | Server     |

### 内容来源配置 { #content-source-configuration }

| 变量                                   | 说明                                                                                                                                                                                                                                                                                                                 | 默认值 | 适用对象 |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ---------- |
| `GPUSTACK_BOOTSTRAP_WITH_EMBEDDED_SOURCES` | 使用内置内容启动所有内容类型（模型目录、社区后端、内置后端版本、缓存提供方），而不是使用已发布文档，因此不会从 OTA Server 获取任何内容。此设置仅决定初始状态：之后投入使用的内容类型仍保留其当前来源。适用于无法访问 OTA Server 的环境，或需要将内容固定为已安装版本的情况。 | `false` (`true` on a development build; an empty value counts as unset) | Server     |

### 调度器配置 { #scheduler-configuration }

| 变量                                            | 说明                                                                 | 默认值 | 适用对象 |
| --------------------------------------------------- | --------------------------------------------------------------------------- | ------- | ---------- |
| `GPUSTACK_SCHEDULER_SCALE_UP_PLACEMENT_MAX_SCORE`   | 调度器放置评分器使用的最高放置分数。                 | `100`   | Server     |
| `GPUSTACK_SCHEDULER_SCALE_UP_LOCALITY_MAX_SCORE`    | 模型文件已存在时，叠加到放置分数上的最高本地性分数。 | `5`     | Server     |
| `GPUSTACK_SCHEDULER_SCALE_DOWN_STATUS_MAX_SCORE`    | 缩容时，状态评分器的最大贡献分数（归一化后）。                 | `100`   | Server     |
| `GPUSTACK_SCHEDULER_SCALE_DOWN_OFFLOAD_MAX_SCORE`   | 缩容时，卸载评分器的最大贡献分数（归一化后）。                | `10`    | Server     |
| `GPUSTACK_SCHEDULER_SCALE_DOWN_PLACEMENT_MAX_SCORE` | 缩容时，放置评分器的最大贡献分数（归一化后）。              | `1`     | Server     |
| `GPUSTACK_SCALING_SCHEDULER_INTERVAL`               | 定时扩缩容根据各时间窗口重新计算模型副本数的间隔，单位为秒。协调采用状态驱动方式，因此此值只限制跨过窗口边界后最长多久会被检测到，不影响正确性。最小有效值为 `1` 秒。 | `30`    | Server     |

#### 多角色组调度 { #multi-role-group-scheduling }

这些权重和时间窗口适用于声明了角色的部署，即预填充与解码分离组及其路由器，不影响普通单角色部署。以下每项分数都是对调度器评分链的最大贡献；设为 `0` 会完全移除对应因素的影响。

| 变量                                             | 说明                                                                 | 默认值 | 适用对象 |
| ---------------------------------------------------- | --------------------------------------------------------------------------- | ------- | ---------- |
| `GPUSTACK_SCHEDULER_TOPOLOGY_PROXIMITY_MAX_SCORE`    | 为角色扩容或放置路由器时，候选 Worker 与组中已放置成员共享的每一级拓扑可获得的最高分。应高于放置评分器 100 分的分差，使空闲容量无法抵消声明拓扑中整整一级的亲近性。设置为 `0` 时，后续成员仅按资源匹配情况放置。 | `150`   | Server     |
| `GPUSTACK_SCHEDULER_PAIRING_AFFINITY_MAX_SCORE`      | 为分离式部署组中的某个角色扩容时，候选 Worker 上每个相反角色成员可贡献的最高分，用于使 KV 传输保持在同一宿主机内。此值是**下限**，并非最终使用的权重：调度器取此值与评分链中其他评分器分数上限之和加一的较大者，确保无论启用哪些评分器，都遵循“相反角色成员越多越优先，容量用于打破平局”。`0` 作为关闭开关，不会被提高到下限。 | `200`   | Server     |
| `GPUSTACK_SCHEDULER_SCALE_DOWN_PAIRING_MAX_SCORE`    | 缩减组内某个角色时，同一 Worker 上相反角色成员对该成员保留分数的最大贡献。在所有多余成员都健康、且状态和卸载分数相同的常见情形下，它决定移除对象。应保持**低于 50**，即状态评分器在“启动中”与“运行中”之间的分差；达到或超过该值时，周围有其他成员的启动中成员可能排在独处的健康成员之前，调度器将无法优先删除已异常的成员。`0` 表示仅根据状态和放置分数选择移除对象。 | `20`    | Server     |
| `GPUSTACK_SCHEDULER_DRAIN_WINDOW_SECONDS`            | 缩容的组成员从路由器注册表移除后继续运行的时长。此值不是进程退出宽限时间，而是供仍从该成员获取 KV cache 的解码任务完成的时间；引擎自身无法等待这些任务。`0` 表示立即删除，无角色部署始终采用立即删除。 | `60`    | Server     |
| `GPUSTACK_SCHEDULER_GATHER_BLOCKED_DWELL_SECONDS`    | `MustGather` 组成员等待放置多久后，部署才报告 `gather_blocked_scale_out`。超过此时间不会使任何内容过期；它仅用于区分“调度器尚未执行”和“调度器已执行，但聚集域内没有可放置的位置”。每次扩容都会经历前一阶段，若不设置等待时间，所有扩容都会显示此标记。`0` 表示只要存在尚未放置的成员，就立即报告拒绝。 | `120`   | Server     |

### GPU 实例配置 { #gpu-instance-configuration }

| 变量                                               | 说明                                                                                                                                                                                               | 默认值 | 适用对象 |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ---------- |
| `GPUSTACK_GPU_INSTANCE_TRANSITIONING_REQUEUE_INTERVAL`      | 控制器通过内存重新入队，重新观察仍在状态转换中（尚未稳定）的 GPU 实例的间隔，单位为秒；PV / PVT 完成清理控制器也复用此值，重新探测仍在清理的实例。已就绪记录的状态偏移由下游监听检测。最小有效值为 `1` 秒。 | `15`    | Server     |
| `GPUSTACK_GPU_INSTANCE_READY_SWEEP_INTERVAL`                | 可选低频扫描的间隔，单位为秒，用于重新观察已稳定就绪的 GPU 实例，补充下游监听在重连间隙可能遗漏的 Worker 端状态变化。`0`（默认）表示禁用；仅在实际发现此类监听覆盖缺口时，才应启用低频扫描。                            | `0`     | Server     |
| `GPUSTACK_GPU_INSTANCE_PV_FINALIZING_UNREACHABLE_TOLERANCE` | 持久卷（PV / PVT）终结器在 Kubernetes 集群的所有 Worker 都不可用时等待该集群的秒数；超过后放弃该集群并释放该行。完全没有 Worker 的集群会立即跳过——它从未持有过该卷。持续不可达的集群会先被等待，然后被释放，其卷可能仍留在该集群中。`0` 表示立即放弃——与 `GPUSTACK_GPU_INSTANCE_UNREADABLE_CR_TOLERANCE` 相反，后者的 `0` 表示不设上限。 | `1800` | Server |
| `GPUSTACK_GPU_INSTANCE_FINALIZING_UNREACHABLE_TOLERANCE` | 删除 GPU 实例时，在放弃并释放该行之前，等待其无法访问的集群的秒数——集群的所有 Worker 都不可用、网络不通，或集群已被删除。期间该行会等待，并写入一条点名该集群的消息，随后被释放；该实例的自定义资源可能仍留在该集群中。`0` 表示立即放弃——与 `GPUSTACK_GPU_INSTANCE_PV_FINALIZING_UNREACHABLE_TOLERANCE` 一致，与 `GPUSTACK_GPU_INSTANCE_UNREADABLE_CR_TOLERANCE` 相反，后者的 `0` 表示不设上限。 | `1800` | Server |

### Worker 与模型配置 { #worker-and-model-configuration }

| 变量                                                         | 说明                                                                                                                     | 默认值 | 适用对象     |
|------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|---------|----------------|
| `GPUSTACK_WORKER_HEARTBEAT_INTERVAL`                             | Worker 心跳间隔，单位为秒。                                                                                           | `30`    | Worker         |
| `GPUSTACK_WORKER_STATUS_SYNC_INTERVAL`                           | Worker 状态同步间隔，单位为秒。                                                                              | `30`    | Worker         |
| `GPUSTACK_WORKER_UNREACHABLE_CHECK_MODE`                         | Worker 不可达检测模式。可选值：`auto`、`enabled`、`disabled`。`auto` 在 Worker 数量 > 50 时禁用检测。            | `auto`  | Server         |
| `GPUSTACK_WORKER_HEARTBEAT_GRACE_PERIOD`                         | Worker 心跳宽限期，单位为秒。                                                                                       | `150`   | Server         |
| `GPUSTACK_MODEL_INSTANCE_RESCHEDULE_GRACE_PERIOD`                | 模型实例重新调度宽限期，单位为秒。                                                                              | `300`   | Server         |
| `GPUSTACK_MODEL_EVALUATION_CACHE_MAX_SIZE`                       | 模型评估缓存的最大容量。                                                                                         | `1000`  | Server         |
| `GPUSTACK_MODEL_EVALUATION_CACHE_TTL`                            | 模型评估缓存 TTL，单位为秒。                                                                                       | `3600`  | Server         |
| `GPUSTACK_WORKER_ORPHAN_WORKLOAD_CLEANUP_GRACE_PERIOD`           | Worker 孤立工作负载清理宽限期，单位为秒。                                                                         | `300`   | Worker         |
| `GPUSTACK_WORKER_ORPHAN_BENCHMARK_WORKLOAD_CLEANUP_GRACE_PERIOD` | Worker 孤立基准测试工作负载清理宽限期，单位为秒。                                                               | `300`   | Worker         |
| `GPUSTACK_WORKER_STATUS_COLLECTION_LOG_SLOW_SECONDS`             | Worker 状态采集耗时超过此秒数时，增加一条慢采集调试日志。                                             | `180`   | Worker         |
| `GPUSTACK_MODEL_INSTANCE_HEALTH_CHECK_INTERVAL`                  | 模型实例健康检查间隔，单位为秒。                                                                                | `3`     | Worker         |
| `GPUSTACK_DISABLE_OS_FILELOCK`                                   | 禁用操作系统文件锁。                                                                                                           | `false` | Worker         |
| `GPUSTACK_ENABLE_CUDA_MINOR_VERSION_COMPATIBILITY`               | 允许较低 CUDA 次版本的设备运行较高次版本的镜像。可在 Worker 上全局设置，或针对单个模型设置；模型级设置优先。 | `false` | Worker & 模型 |
| `GPUSTACK_RESTART_IN_FLIGHT_LAPSE_SECONDS`                       | 首次重启仍在重建期间，`POST /models/{id}/restart` 持续以 HTTP 409 拒绝第二次请求的时长。没有此保护时，第二次拆除会删除首次重启刚创建的替代实例。保护会自动失效，使始终无法达到 `running` 状态的部署仍可再次重启；此值应长于需要拉取镜像并从磁盘读取权重的冷启动时间。 | `900`   | Server         |
| `GPUSTACK_SERVE_LOG_MAX_BYTES`                                   | 单个推理服务日志在一次重启期间的字节预算。超出后保留头部，轮转尾部，并在两者之间插入标记。`0` 表示取消限制。 | `67108864` | Worker |
| `GPUSTACK_SERVE_LOG_HEAD_BYTES`                                  | 上述预算中为日志头部保留的字节数，头部只写入一次，不再改写。最大为总预算的一半；`0` 表示仅保留第一行。 | `8388608` | Worker |

### 基准测试配置 { #benchmark-configuration }

| 变量                                   | 说明                                                                                                                                      | 默认值                                                                               | 适用对象 |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- | ---------- |
| `GPUSTACK_BENCHMARK_DATASET_SHAREGPT_PATH` | 当 `dataset_name` 设置为 `ShareGPT` 时，基准测试容器使用的 ShareGPT 数据集路径。默认镜像已包含该数据集。 | `/workspace/benchmark-runner/sharegpt_data/ShareGPT_V3_unfiltered_cleaned_split.json` | Worker     |
| `GPUSTACK_BENCHMARK_REQUEST_TIMEOUT`       | 每个基准测试请求的超时时间，单位为秒。                                                                                                   | `3600`                                                                                | Worker     |

### 模型部署配置 { #model-deployment-configuration }

!!! note

    这些环境变量**不应**在启动 GPUStack 时设置，而应在部署模型时，在**高级选项 > 环境变量**中配置，用于自定义模型推理服务行为。

| <div style="width:180px">变量</div>                   | 说明                                                                                                                                                                                                                                                                                                                 | 默认值 | 适用对象     |
|-----------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|----------------|
| `GPUSTACK_MODEL_SERVING_COMMAND_SCRIPT_DISABLED`          | 禁用自动推理服务命令脚本。设为 `1` 或 `true` 时，不执行负责软件包安装及其他准备任务的脚本。                                                                                                                                                         | `0`     | 模型          |
| `PYPI_PACKAGES_INSTALL`                                   | 在模型推理环境中安装的额外 PyPI 包。多个包以空格分隔。可用时脚本使用 `uv pip install`，否则使用 `pip install`。                                                                                                                              | (empty) | 模型          |
| `GPUSTACK_MODEL_RAM_CLAIM`                                | 用户声明的模型 RAM 需求（单位为 Byte），供调度器进行容量规划。                                                                                                                                                                                                                         | (empty) | 模型          |
| `GPUSTACK_MODEL_VRAM_CLAIM`                               | 用户声明的模型 VRAM 需求（单位为 Byte），供调度器进行容量规划。                                                                                                                                                                                                                        | (empty) | 模型          |
| `GPUSTACK_APPLY_QWEN3_RERANKER_TEMPLATES`                 | 向请求体应用 Qwen3 reranker 模板。使用方法请参阅 https://huggingface.co/Qwen/Qwen3-Reranker-0.6B。                                                                                                                                                                                                    | (empty) | 模型          |
| `GPUSTACK_SKIP_MODEL_EVALUATION`                          | 在部署时跳过模型评估或验证步骤。                                                                                                                                                                                                                                                            | (empty) | 模型          |
| `GPUSTACK_SKIP_GPU_COUNT_CHECK`                           | 跳过“手动选择的 GPU 数量必须等于并行 world size（tp×pp×dp）”的验证。仅在手动选择 GPU 时生效，不影响自动调度。适用于并行参数有意不同于所选 GPU 数量的手动单节点部署。 | (empty) | 模型          |
| `GPUSTACK_DISABLE_METRICS`                                | 禁用模型指标的暴露和采集。                                                                                                                                                                                                                                                                        | (empty) | 模型          |
| `GPUSTACK_MODEL_HEALTH_CHECK_PATH`                        | 指定模型暴露的 HTTP 健康检查路径。                                                                                                                                                                                                                                                                  | (empty) | 模型          |
| `GPUSTACK_MODEL_RUNTIME_UID`                              | 控制容器内运行进程的用户权限。                                                                                                                                                                                                                                                     | (empty) | 模型          |
| `GPUSTACK_MODEL_RUNTIME_GID`                              | 控制容器内运行进程的用户组权限。                                                                                                                                                                                                                                                    | (empty) | 模型          |
| `GPUSTACK_MODEL_RUNTIME_SHM_SIZE_GIB`                     | 容器共享内存大小，单位为 GiB。                                                                                                                                                                                                                                                                                | `10.0`  | 模型          |
| `GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_ENABLED`           | 为此模型启用推理健康检查。启用后，GPUStack 定期发送最小推理请求，验证模型是否正确响应。                                                                                                                                                             | `false` | 模型          |
| `GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_INTERVAL`          | 推理健康检查间隔，单位为秒（最小值 `60`）。如果在此间隔内观察到近期成功的推理流量，则跳过主动检查。                                                                                                                                                            | `300`   | 模型          |
| `GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_TIMEOUT`           | 每次定期推理健康检查请求的超时时间，单位为秒。SGLang PD 引擎角色默认为 25，其他情况默认为 15；显式设置的值优先。                                                                                                                                                        | `15/25` | 模型          |
| `GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD` | 将实例标记为不健康前，允许的连续推理健康检查失败次数。                                                                                                                                                                                                                             | `3`     | 模型          |
| `GPUSTACK_ENABLE_CUDA_MINOR_VERSION_COMPATIBILITY`        | 允许较低 CUDA 次版本的设备运行较高次版本的镜像。可在 Worker 上全局设置，或针对单个模型设置；模型级设置优先。                                                                                                                                                                                             | `false` | Worker & 模型 |

#### 使用示例 { #usage-example }

部署模型时，进入**高级选项 > 环境变量**并添加：

```bash
# Install additional packages before model serving starts
PYPI_PACKAGES_INSTALL=torch-audio==2.0.0 transformers==4.30.0

# Disable the serving command script entirely
GPUSTACK_MODEL_SERVING_COMMAND_SCRIPT_DISABLED=1
```

推理服务命令脚本会自动处理：

- 安装 `PYPI_PACKAGES_INSTALL` 指定的额外 PyPI 包
- 同时支持使用 `uv pip` 和 `pip` 安装软件包
- 通过 `PIP_INDEX_URL` 和 `PIP_EXTRA_INDEX_URL` 处理自定义 PyPI 索引

## GPUStack Runtime 环境变量 { #gpustack-runtime-environment-variables }

这些环境变量由 GPUStack Runtime 使用，通常用于调整运行在 Docker/Kubernetes 中的推理后端行为。

它们仅在 Worker 中有效。请在 Worker 容器内设置环境变量，以确保正确生效。

### 全局变量 { #global-variables }

| 变量                         | 说明                | 默认值 |
| -------------------------------- | -------------------------- | ------- |
| `GPUSTACK_RUNTIME_LOG_LEVEL`     | 日志级别。                 | `INFO`  |
| `GPUSTACK_RUNTIME_LOG_WARNING`   | 启用警告日志。   | `0`     |
| `GPUSTACK_RUNTIME_LOG_EXCEPTION` | 启用异常日志。 | `0`     |

### 检测器变量 { #detector-variables }

| 变量                                           | 说明                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | 默认值                                 |
| -------------------------------------------------- |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| --------------------------------------- |
| `GPUSTACK_RUNTIME_DETECT`                          | 使用的检测器。可选值：`Auto`、`AMD`、`ASCEND`、`CAMBRICON`、`HYGON`、`ILUVATAR`、`METAX`、`MTHREADS`、`NVIDIA`。                                                                                                                                                                                                                                                                                                                                                                                                                                          | `Auto`                                  |
| `GPUSTACK_RUNTIME_DETECT_NO_PCI_CHECK`             | 检测时跳过 PCI 检查。适用于 WSL 环境。                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | (empty)                                 |
| `GPUSTACK_RUNTIME_DETECT_PCI_CLASS_PREFIXES`       | 检测时检查的 PCI 类别前缀。`GPUSTACK_RUNTIME_DETECT_NO_PCI_CHECK` 为 false 时，检测器检查设备的 PCI 类别是否以任一指定前缀开头。                                                                                                                                                                                                                                                                                                                                      | 02,03,0b,12                             |
| `GPUSTACK_RUNTIME_DETECT_BACKEND_MAP_RESOURCE_KEY` | 检测到的后端与资源键的映射，例如 `{"cuda": "nvidia.com/devices", "rocm": "amd.com/devices"}`。用于将 gpustack-runner 的后端名称映射到对应资源键。                                                                                                                                                                                                                                                                                                                                          | The default values named by each vendor |
| `GPUSTACK_RUNTIME_DETECT_PHYSICAL_INDEX_PRIORITY`  | 检测设备时优先使用物理索引。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | `1`                                     |

#### ROCm 检测器专用变量 { #rocm-detector-specific-variables }

!!! note

    同样适用于基于 ROCm 的后端，例如海光。

| 变量             | 说明             | 默认值     |
| -------------------- | ----------------------- | ----------- |
| `ROCM_SMI_LIB_PATH`  | ROCm SMI 库路径。  | (empty)     |
| `ROCM_HOME`          | ROCm 主目录。    | (empty)     |
| `ROCM_PATH`          | ROCm 路径。              | `/opt/rocm` |
| `ROCM_CORE_LIB_PATH` | ROCm 核心库路径。 | (empty)     |

### 部署器变量 { #deployer-variables }

| 变量                                                           | 说明                                                                                                                                                                                                                                                                                                                                               | 默认值                                         |
| ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `GPUSTACK_RUNTIME_DEPLOY`                                          | 使用的部署器。可选值：Auto、Docker、Kubernetes、Podman（实验性）。                                                                                                                                                                                                                                                                                 | `Auto`                                          |
| `GPUSTACK_RUNTIME_DEPLOY_API_CALL_ERROR_DETAIL`                    | 在部署时输出 API 调用错误详情。                                                                                                                                                                                                                                                                                                    | `1`                                             |
| `GPUSTACK_RUNTIME_DEPLOY_PRINT_CONVERSION`                         | 在部署时输出转换结果。GPUStack Runtime 提供统一的 Workload 定义 API，并将其转换为具体容器运行时 API 调用（例如 Docker SDK、Kubernetes API、Podman SDK）。启用此选项后，在 INFO 日志中输出最终转换的 API 调用，便于调试。                       | `0`                                             |
| `GPUSTACK_RUNTIME_DEPLOY_ASYNC`                                    | 启用异步部署。                                                                                                                                                                                                                                                                                                                           | `1`                                             |
| `GPUSTACK_RUNTIME_DEPLOY_ASYNC_THREADS`                            | 线程池中的线程数。                                                                                                                                                                                                                                                                                                                  | (empty)                                         |
| `GPUSTACK_RUNTIME_DEPLOY_MIRRORED_NAME`                            | 部署器名称。                                                                                                                                                                                                                                                                                                                                 | (empty)                                         |
| `GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_ENVIRONMENTS`  | 镜像部署时忽略的环境变量名称。                                                                                                                                                                                                                                                                                          | (empty)                                         |
| `GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_VOLUMES`       | 镜像部署时忽略的卷挂载目标。                                                                                                                                                                                                                                                                                           | (empty)                                         |
| `GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY_USERNAME`      | 默认容器镜像仓库的用户名。如果 GPUStack runner 镜像托管在需要身份认证的仓库中，请在每个 GPUStack Worker（或包含 Worker 的 GPUStack Server）启动时，将此变量设为访问用户名。实际使用中，建议直接在容器运行时中配置访问用户名。 | (empty) |
| `GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY_PASSWORD`      | 默认容器镜像仓库的密码。如果 GPUStack runner 镜像托管在需要身份认证的仓库中，请在每个 GPUStack Worker（或包含 Worker 的 GPUStack Server）启动时，将此变量设为访问密码。实际使用中，建议直接在容器运行时中配置访问密码。 | (empty) |
| `GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_NAMESPACE`              | 默认 runner 容器镜像的命名空间。如果 GPUStack runner 镜像位于 `gpustack` 命名空间之外，例如 `gpustack-ai`，请将此变量设置为 `gpustack-ai`。对于多级命名空间，例如 `awesome.com/amazing-group/amazing-subgroup/gpustack-ai`，请通过 `--system-default-container-registry`（或 Worker 的 `system_default_container_registry` 配置键，或 `GPUSTACK_SYSTEM_DEFAULT_CONTAINER_REGISTRY` 环境变量）声明包含父命名空间的镜像仓库，例如 `--system-default-container-registry=awesome.com/amazing-group/amazing-subgroup`。 | `gpustack` |
| `GPUSTACK_RUNTIME_DEPLOY_IMAGE_PULL_POLICY`                        | 部署器的镜像拉取策略（例如 Always、IfNotPresent、Never）。                                                                                                                                                                                                                                                                                   | `IfNotPresent`                                  |
| `GPUSTACK_RUNTIME_DEPLOY_CDI_SPECS_DIRECTORY`                      | 部署时包含 Container Device Interface（CDI）规范的目录路径，或生成 CDI 规范的目标目录。                                                                                                                                                                                                    | `/var/run/cdi`                                  |
| `GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_CDI`                     | 手动配置容器设备接口映射，告诉容器运行时应将哪些设备注入容器。例如 `nvidia.com/devices=nvidia.com/gpu;amd.com/devices=amd.com/gpu`。键为资源键，值为 Container Device Interface（CDI）键。                                                      | The default values named by each vendor         |
| `GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_RUNTIME_VISIBLE_DEVICES` | 手动配置运行时可见设备环境变量的映射，告诉容器运行时应将哪些设备注入容器。例如 `nvidia.com/devices=NVIDIA_VISIBLE_DEVICES;amd.com/devices=AMD_VISIBLE_DEVICES`。键为资源键，值为环境变量名称。                              | The default values named by each vendor         |
| `GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_BACKEND_VISIBLE_DEVICES` | 手动配置后端可见设备环境变量的映射，告诉设备运行时（例如 ROCm、CUDA、OneAPI）在容器内使用哪些设备。例如 `nvidia.com/devices=CUDA_VISIBLE_DEVICES;amd.com/devices=ROCR_VISIBLE_DEVICES`。键为资源键，值为环境变量名称列表。 | The default values named by each vendor         |
| `GPUSTACK_RUNTIME_DEPLOY_RUNTIME_VISIBLE_DEVICES_VALUE_UUID`       | 对指定的运行时可见设备环境变量使用 UUID。                                                                                                                                                                                                                                                                                    | (empty)                                         |
| `GPUSTACK_RUNTIME_DEPLOY_BACKEND_VISIBLE_DEVICES_VALUE_ALIGNMENT`  | 对指定的后端可见设备环境变量启用值对齐。                                                                                                                                                                                                                                                                       | `ASCEND_RT_VISIBLE_DEVICES,NPU_VISIBLE_DEVICES` |
| `GPUSTACK_RUNTIME_DEPLOY_CPU_AFFINITY`                             | 为部署的工作负载启用 CPU 亲和性。                                                                                                                                                                                                                                                                                                               | `0`                                             |
| `GPUSTACK_RUNTIME_DEPLOY_NUMA_AFFINITY`                            | 为部署的工作负载启用 NUMA 亲和性。启用后也会隐式启用 `GPUSTACK_RUNTIME_DEPLOY_CPU_AFFINITY`。                                                                                                                                                                                                                                        | `0`                                             |

#### Docker 部署器专用变量 { #docker-deployer-specific-variables }

| 变量                                            | 说明                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | 默认值                            |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `GPUSTACK_RUNTIME_DOCKER_HOST`                      | Docker 连接主机，用于覆盖默认 Docker 主机。                                                                                                                                                                                                                                                                                                                                                                                                                         | `http+unix:///var/run/docker.sock` |
| `GPUSTACK_RUNTIME_DOCKER_PAUSE_IMAGE`               | Docker 中 pause 容器使用的镜像。                                                                                                                                                                                                                                                                                                                                                                                                                                       | `gpustack/runtime:pause`           |
| `GPUSTACK_RUNTIME_DOCKER_UNHEALTHY_RESTART_IMAGE`   | Docker 中用于不健康重启的容器镜像。                                                                                                                                                                                                                                                                                                                                                                                                                               | `gpustack/runtime:health`          |
| `GPUSTACK_RUNTIME_DOCKER_RESOURCE_INJECTION_POLICY` | Docker 部署器的资源注入策略（例如 Env、CDI）。<br/>`Env`：通过标准环境变量注入资源，基于 `GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_RUNTIME_VISIBLE_DEVICES`。<br/>`CDI`：通过 CDI 注入资源，基于 `GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_CDI`。                                                                                                                                                                         | `Env`                              |
| `GPUSTACK_RUNTIME_DOCKER_CDI_SPECS_GENERATE`        | 使用 CDI 资源注入策略时，在部署期间生成 CDI 规范，要求 `GPUSTACK_RUNTIME_DEPLOY_CDI_SPECS_DIRECTORY` 存在。仅当 `GPUSTACK_RUNTIME_DOCKER_RESOURCE_INJECTION_POLICY` 设置为 `CDI` 时生效。使用内置规则为部署器生成 CDI 规范；如果输出文件与其他工具（例如 NVIDIA Container Toolkit）生成的 CDI 规范冲突，请禁用此选项，并手动移除输出文件。 | `1`                                |

#### Kubernetes 部署器专用变量 { #kubernetes-deployer-specific-variables }

| 变量                                                     | 说明                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 | 默认值         |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| `GPUSTACK_RUNTIME_KUBERNETES_NODE_NAME`                      | 部署工作负载的目标 Kubernetes Node 名称。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | (empty)         |
| `GPUSTACK_RUNTIME_KUBERNETES_NAMESPACE`                      | 部署工作负载的目标 Kubernetes 命名空间。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | `default`       |
| `GPUSTACK_RUNTIME_KUBERNETES_DOMAIN_SUFFIX`                  | Kubernetes 服务的域名后缀。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | `cluster.local` |
| `GPUSTACK_RUNTIME_KUBERNETES_SERVICE_TYPE`                   | Kubernetes 服务类型。可选值：ClusterIP、NodePort、LoadBalancer。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | `ClusterIP`     |
| `GPUSTACK_RUNTIME_KUBERNETES_QUORUM_READ`                    | 是否为 Kubernetes 服务使用 quorum read。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | `0`             |
| `GPUSTACK_RUNTIME_KUBERNETES_DELETE_PROPAGATION_POLICY`      | Kubernetes 资源的删除传播策略。可选值：Foreground、Background、Orphan。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | `Foreground`    |
| `GPUSTACK_RUNTIME_KUBERNETES_RESOURCE_INJECTION_POLICY`      | Kubernetes 部署器的资源注入策略。可选值：Auto、Env、KDP。<br/>`Auto`：根据环境自动选择资源注入策略。<br/>`Env`：通过标准环境变量注入资源，依赖底层 Container Toolkit，并基于 `GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_RUNTIME_VISIBLE_DEVICES`。<br/>`KDP`：通过 Kubernetes Device Plugin 注入资源。                                                                                                                                                                                                                                                                                                            | `Auto`          |
| `GPUSTACK_RUNTIME_KUBERNETES_KDP_PER_DEVICE_MAX_ALLOCATIONS` | Kubernetes Device Plugin 中，单个设备的最大分配次数。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | `10`            |
| `GPUSTACK_RUNTIME_KUBERNETES_KDP_DEVICE_ALLOCATION_POLICY`   | Kubernetes Device Plugin 的设备分配策略。可选值：Auto、CDI、Env、Opaque。<br/>`Auto`：根据环境自动选择设备分配策略。<br/>`Env`：通过运行时可见设备环境变量分配设备，基于 `GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_RUNTIME_VISIBLE_DEVICES`，要求 Container Toolkit 支持。<br/>`CDI`：通过生成的 CDI 规范分配设备，基于 `GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_CDI`，便于调试和排障；要求 `GPUSTACK_RUNTIME_DEPLOY_CDI_SPECS_DIRECTORY` 存在。<br/>`Opaque`：使用内置逻辑分配，便于部署，但较难排障。 | `Auto`          |
| `GPUSTACK_RUNTIME_KUBERNETES_KDP_CDI_SPECS_GENERATE`         | 在部署期间生成 CDI 规范，要求 `GPUSTACK_RUNTIME_DEPLOY_CDI_SPECS_DIRECTORY` 存在。仅当 `GPUSTACK_RUNTIME_KUBERNETES_KDP_DEVICE_ALLOCATION_POLICY` 设置为 `CDI` 时生效。使用内置规则为部署器生成 CDI 规范；如果输出文件与其他工具（例如 NVIDIA Container Toolkit）生成的 CDI 规范冲突，请禁用此选项，并手动移除输出文件。                                                                                                                                                                                                                                                                                 | `1`             |

#### Podman 部署器专用变量 { #podman-deployer-specific-variables }

!!! note

    Podman 部署器为实验性功能，需要 4.9 或更高版本。

| 变量                                          | 说明                                                                                                                                                                                                                                                                                                                                                                                                                      | 默认值                               |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| `GPUSTACK_RUNTIME_PODMAN_HOST`                    | Podman 连接主机，用于覆盖默认 Podman 主机。                                                                                                                                                                                                                                                                                                                                                            | `http+unix:///run/podman/podman.sock` |
| `GPUSTACK_RUNTIME_PODMAN_PAUSE_IMAGE`             | Podman 中 pause 容器使用的镜像。默认与 `GPUSTACK_RUNTIME_DOCKER_PAUSE_IMAGE` 相同。                                                                                                                                                                                                                                                                                                                | `gpustack/runtime:pause`              |
| `GPUSTACK_RUNTIME_PODMAN_UNHEALTHY_RESTART_IMAGE` | Podman 中用于不健康重启的容器镜像。默认与 `GPUSTACK_RUNTIME_DOCKER_UNHEALTHY_RESTART_IMAGE` 相同。                                                                                                                                                                                                                                                                                            | `gpustack/runtime:health`             |
| `GPUSTACK_RUNTIME_PODMAN_CDI_SPECS_GENERATE`      | 在部署期间生成 CDI 规范，要求 `GPUSTACK_RUNTIME_DEPLOY_CDI_SPECS_DIRECTORY` 存在。使用内置规则为部署器生成 CDI 规范；如果输出文件与其他工具（例如 NVIDIA Container Toolkit）生成的 CDI 规范冲突，请禁用此选项，并手动移除输出文件。默认与 `GPUSTACK_RUNTIME_DOCKER_CDI_SPECS_GENERATE` 相同。 | `1`                                   |
