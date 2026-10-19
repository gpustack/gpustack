---
translation_status: translated
---

# PD 分离 { #pd-disaggregation }

预填充/解码（PD）分离将部署拆为独立运行的角色：**prefill** 计算提示词的 KV 缓存，**decode** 根据缓存生成 token，前方的 **router** 将每个请求串联到这两个阶段。KV 缓存通过网络在两者之间传输，而无需重新计算。

两个阶段的资源需求相反。Prefill 受计算能力限制，希望采用较大的张量并行；decode 受内存带宽限制，希望处理大量并发序列。单体部署必须为两者选用同一组参数。分离后，可以根据各自实际任务独立确定规模和调优，并按工作负载调整二者比例：提示词占比较高的工作负载需要更多 prefill，生成较长的工作负载需要更多 decode。

它也带来实际成本：请求在处理过程中需要跨网络传输，需要维持的角色从一个增加为三个，组的可用性取决于其中可用性最低的角色。只有两个阶段的需求确实冲突时，分离才值得采用。

## 何时有效，何时无效 { #when-it-helps-and-when-it-does-not }

以下三个问题决定是否适用：

1. **Prefill 和 decode 是否需要不同参数？** 如果一组引擎参数就能让两阶段达到可接受的效果，分离只会增加复杂性，不会增加吞吐量。
2. **提示词与生成内容的比例是否远离 1:1？** 收益来自两侧独立调整规模，只有负载明显不同时才值得。
3. **KV 传输是否足够快？** 每次请求都需要通过网络传输提示词的 KV 缓存。在低速或共享链路上，传输时间会抵消节省的 prefill 时间。

单 GPU，或提示词和生成内容都较短的工作负载，通常在不分离时更快。

## 部署分离模型 { #deploying-a-disaggregated-model }

1. 按常规方式部署模型，然后展开 `Disaggregation` 部分并启用。
2. 选择 **PD 模式**。一个模式就是一套完整方案，包括引擎、KV 连接器和 router，因此连接器、握手变量和端口都由这一选择确定，不会单独出现在表单中。可用模式取决于后端；针对其他引擎的模式会显示为禁用，并说明原因。
3. 设置每个角色的**副本数**。它决定比例：2 个 prefill 和 1 个 decode 构成 2P1D 组。
4. 在两侧需求不同时，覆盖各角色设置。保持继承的字段使用部署级值，因此只需填写不同之处。
5. 保存。

Router 由平台管理：镜像、命令、对端地址和健康检查路径均来自模式方案，表单无需填写这些信息。需要填写的是方案无法预知的内容，即 router 自身的参数和环境；平台推导的连接参数也会预填到同一列表，便于查看实际生成结果。未修改的行会在保存时再次移除，由 GPUStack Server 继续生成。

Router 在结构上只有一个副本，因此无需设置数量；第二个副本会拆分前缀缓存，并使组具有两个地址。

如果需要自己的 router，请选择 `custom` 模式，并在**部署**而不是 router 角色上设置镜像和运行命令，角色会继承这两项。因此，一个镜像同时包含 router 和引擎，并按角色选择执行内容；工作量与在两处指定它相同，但不再需要维护第二处同步。

## 组的启动方式 { #how-a-group-starts }

启动顺序经过设计，了解它有助于理解组启动时显示的状态：

1. Prefill 和 decode 实例在同一事务中**同时**创建。支持 gang admission 时，它们也会一同准入，因为只启动一半的组无法提供服务。
2. 两个角色各自都有运行中的成员后，才创建 **router**。它最后创建，是因为其命令行包含对端地址，而对端启动前这些地址尚不存在。
3. Router 启动后注册组端点。只注册 router；prefill 和 decode 成员虽在各自端口提供 OpenAI 形式的 API，但都无法独立完成整个请求。

因此，组在开始服务之前会短暂报告 `Partially ready`，这是正常流程，而非故障。

## 理解组状态 { #reading-a-groups-state }

| 显示内容 | 含义 |
| --- | --- |
| `Pending` | 尚无运行中的成员。 |
| `Partially ready` | 已有成员启动，但组仍无法服务：某个角色没有就绪成员，或 router 的端点尚未注册。 |
| `Running` | 每个角色至少有一个就绪成员，且端点已注册。 |
| `Ratio 2:1 (currently 1:1)` | 组正在服务，但尚未达到请求的比例。显示在状态旁边，不替代状态。 |
| 时钟图标 / `Config changed` | 运行中的成员早于当前显示的配置；重启之前，角色无法扩容。参见[应用配置更改](#applying-a-configuration-change)。 |

未达到请求比例属于降级，而不是独立状态：请求 3P1D、实际运行 2P1D 的组仍可服务，只是吞吐量达不到规划值。

## 应用配置更改 { #applying-a-configuration-change }

编辑分离部署**不会**使其重启。更改会被保存，组标记为正在运行旧配置；重启前不会发生变动。

这是有意为之。逐个重启成员会在一段时间内将使用新配置的 prefill 与使用旧配置的 decode 配对，而引擎会接受这种配对。例如，上下文长度不匹配时，握手能完成，KV 能成功传输，短提示词也能正确回答；只有提示词超过 decode 侧窗口时才会失败，而此时 prefill 计算成本已经付出。错误答案比重启更糟，因此重新启动任何成员之前，会先停止整个组。

在部署上使用 `Restart` 应用更改。它会停止整个组，并让其按当前配置重新组成。不提供仅重启部分角色的操作，因为这正会产生上述新旧混配。

如果没有待应用的配置更改，伸缩会修改各角色的实例数，而无需重启整个组。Router 行为取决于 `pd-modes.yaml` 中的 `router.membership_update`：

| 策略 | 内置模式 | 伸缩或成员恢复时的行为 |
| --- | --- | --- |
| `dynamic` | vLLM NIXL、SGLang Mooncake/NIXL、vLLM Ascend Mooncake | GPUStack 通过成员管理 API 更新 router。 |
| `restart` | vLLM Mooncake | 新增 GPU 成员就绪后，GPUStack 替换 router，刷新启动时的对端和 bootstrap 元数据。移除或重启成员时，只要两侧均有可用对端，就立即触发刷新。保留的 GPU 实例继续运行，但服务会短暂不可用，正在处理的请求可能失败。 |
| `manual` | Custom | 操作人员更新 router 对端和 bootstrap 元数据，并在需要时重启 router。 |

用户自定义的 router 运行命令采用手动成员管理，即使所选方案支持自动更新。部署表单和副本编辑器显示来自目录的策略。删除实例时，如果恢复需要替换 router 或由操作人员处理，也会给出警告。

静态 router 快照包含实例身份、重启代次、HTTP 地址和具名端口。因此，即使替换或重启发生在相同地址，仍会刷新 Mooncake bootstrap 元数据。没有快照的现有 router，会在对端就绪时替换一次。

**存在已保存但尚未重启应用的更改时，角色不能扩容。** 新成员将根据已编辑配置创建，与未采用该配置的旧成员并行运行，形成上述新旧混配。伸缩设置会在重启时与配置更改一同生效。

**缩容**仍立即生效，无论是否存在待应用更改，因为减少成员不会产生不匹配的配对。

## 结合共享 KV 缓存使用 { #using-a-shared-kv-cache-with-it }

分离部署与共享 KV 缓存互补，而非相互替代。GPUStack 会自动组合两者：按常规方式接入缓存服务，两个连接器就会合并为引擎接受的一份配置。

每个接入缓存的 vLLM 角色都会生成 `MultiConnector[PD, Store]`。包装连接器在 prefill 上使用 `kv_producer`，在 decode 上使用 `kv_consumer`。各子连接器保留其传输或存储所需角色。

| 模式 | 无缓存 | 进程内缓存 | 缓存服务 |
| --- | --- | --- | --- |
| `vllm-nixl`（NVIDIA 默认） | `NixlConnector` | `NixlConnector` + `LMCacheConnectorV1` | 仅 prefill：`NixlConnector` + LMCache；仅支持 LMCache |
| `vllm-mooncake`（NVIDIA） | `MooncakeConnector` | `MooncakeConnector` + `LMCacheConnectorV1` | `MooncakeConnector` + `MooncakeStoreConnector`；仅支持 Mooncake |
| `vllm-ascend-mooncake` | 根据模型选择的 Mooncake PD 连接器 | PD 连接器 + `SimpleCPUOffloadConnector` | PD 连接器 + `AscendStoreConnector`；仅支持 Mooncake |

进程内 LMCache 子连接器在两个角色上都使用 `kv_both`。对于 NIXL，共享 LMCache 仅支持 prefill。即使 MultiConnector 将请求的加载分配给 NIXL，其 MP 连接器仍可能加载缓存前缀；第二次异步完成可能使 decode 崩溃（[vLLM #53049](https://github.com/vllm-project/vllm/issues/53049)）。NVIDIA Mooncake Store 在 prefill 上使用 `kv_both`，在 decode 上使用 `kv_consumer`。昇腾 Store 在 prefill 上使用 `kv_producer`，在 decode 上使用 `kv_consumer`。这些配置遵循 [vLLM 0.29 Mooncake Store](https://docs.vllm.ai/en/v0.29.0/features/mooncake_store_connector_usage/#disaggregated-prefill-decode-xpyd) 和 [vLLM-Ascend 0.23 KV pool](https://docs.vllm.ai/projects/ascend/en/v0.23.0/user_guide/feature_guide/kv_pool.html)。

昇腾为标准 KV 布局选择 `MooncakeConnectorV1`，包括 Qwen3、GLM-5 和 DeepSeek V3.1；为 DeepSeek V4、V4 Flash 和 V4 Pro 等混合布局选择 `MooncakeHybridConnector`。选择依据是预训练配置，包括压缩注意力和循环层，而非部署名称。两个角色使用同一份保存的选择。如果布局元数据不可用，混合连接器负责回退处理标准布局。GPUStack 会针对部署的模型输入保留此选择，避免某成员启动后元数据才变得可用，导致其他成员使用不同连接器。

NIXL 缓存组合使用 vLLM 0.26 起提供的真实块契约。PD 方案声明支持的缓存提供商和连接器设置，不定义或检查缓存服务版本或客户端包要求。缓存提供商负责自身服务与客户端的兼容性。Runner 依赖仍为 0.5.4。两种 NVIDIA 模式都以 vLLM 0.27.1 或更高版本为内置 runner 基线：这些 CUDA runner 提供从引擎镜像运行的 `vllm-router` 0.1.15。昇腾模式基于相同原因采用 CANN 0.23 基线。这些最低版本描述 GPUStack 内置镜像，而非每个连接器首次出现在上游的版本。

NVIDIA Mooncake 配合共享 Store 时，还有额外的 0.29 基线。GPUStack 0.29 runner 包含被拒绝加载的修复：Store 子连接器必须跳过已分配给 PD 子连接器的加载。MultiConnector 向未选中的 Store 传递零个外部 token，修复会在发出待处理加载前检查 `can_load`。这也适用于 decode 侧缓存命中。原版 vLLM 0.29 需要回移此修复，上游 0.30 已包含它。参见 [runner 修复](https://github.com/gpustack/runner/commit/f0264ed)。

请为每个需要缓存的角色单独接入服务。下拉菜单会按模式、提供商和角色筛选服务，API 也会强制执行这些兼容规则。PD 引擎版本范围属于建议：选择超出范围的内置引擎版本时，会针对模式或已启用缓存显示警告，但仍允许部署和注入缓存。自定义版本和显式指定的镜像不会显示此类警告。
启动失败通过实例状态和日志报告。

Auto 当前根据后端、加速器运行时和平台选择 runner，不会按 PD 模式或缓存组合的版本范围筛选。SGLang 通过参数和配置文件接入 LMCache 或 MeshFusion，而不使用 vLLM MultiConnector。

NVIDIA Mooncake 要求推理镜像中包含其 transfer-engine 运行时和 RDMA 支持。GPUStack runner 镜像还暴露 Mooncake 传输/过期指标，传输计数器在 prefill 上读取。GPUStack 分配唯一的 prefill bootstrap 端口，并传递给 router。使用 vllm-router 0.1.15 时，成员变化会重启 router 以刷新 bootstrap 元数据，可能短暂中断流量。

## 网络要求 { #networking-requirements }

### Kubernetes：Pod 需要集群提供的能力 { #on-kubernetes-what-the-pods-ask-the-cluster-for }

组成员同时需要主机网络、hostPort、主机 IPC 和设备挂载，超出了 `baseline` Pod Security Admission 级别允许的范围。GPUStack 会据此为自己管理的命名空间设置标签。其对自备集群的含义，以及**不**涵盖的内容，参见[命名空间与 Pod Security Admission](cluster-management.md#namespaces-and-pod-security-admission)。

### KV 传输接口 { #the-kv-transfer-interface }

每个请求都要通过网络传输 KV 缓存，所用接口至关重要。未干预时，传输层可能选择容器网桥（`docker0`、`br-*`、`flannel.1`），并公布对端无法访问的地址，最终在握手时表现为连接器错误，而非配置错误。

GPUStack 按 Worker 推导接口：

- **只有一个可路由接口**——直接使用，无需配置。
- **有多个可路由接口**——GPUStack 不会猜测。请在该 Worker 上设置 `--kv-transfer-ifname`（或 Worker 配置键 `kv_transfer_ifname`，或 `GPUSTACK_KV_TRANSFER_IFNAME`），指定承载 KV 流量的接口。Worker 日志会列出发现的候选接口。

对于具有多个网络互连的机器，请显式设置。常见情况是一块管理网卡加一块 RDMA 网卡，管理网卡几乎从来都不是正确选择。

### 端口 { #ports }

每个角色需要自己的 HTTP 端口，以及连接器要求的端口。GPUStack 从 Worker 的服务端口范围中按连续端口段分配，确保同一主机上的两个成员不会冲突。

默认范围包含 **64 个端口**，对大多数部署足够，但并非所有部署都适用：使用每 rank 一个端口的连接器时，张量并行大小为 8 的成员会占用九个端口。如果每台主机运行多个此类成员，请扩大 `--service-port-range`。

!!! warning "GPUStack 无法管理的三类端口"

    某些连接器会绑定由引擎在运行时选择的端口，并未在 GPUStack 能读取的任何位置声明：

    - SGLang 的 rank 端口，来自主机临时端口范围；
    - Mooncake 每 rank 的 transfer-engine 握手端口；
    - Mooncake 的 segment 端口。

    对于这些端口，配置文件中写入的值**不是**实际绑定端口，因此 GPUStack 无法预留它们，也无法将其声明为主机端口。观测到的范围约为 15000–17000 和 20000–20800，因此默认服务范围（40000–40063）不会与它们相遇；避免冲突靠的是范围分离，而非预留。

    **因此，不要将 `--service-port-range` 移入这些范围。** 实测中，当连接器端口段从 20001 开始时，成员自身的第二个 rank 尝试绑定 20003，却发现该端口已被引擎刚刚自行选择的 transfer-engine 端口占用。同样的部署将端口段放在默认范围后，第一次就成功启动。

    如果成员反复重启且始终无法达到 `Running`，请检查引擎日志中是否有 `Address already in use`。

    端口去重也仅覆盖一个 Worker 进程。同一主机上的第二个 GPUStack Worker，或手动启动的容器，都不在其管理范围内。

## 约束 { #constraints }

可能导致组返回错误答案的配置，会在保存时被拒绝，而非等到运行时：

- **Prefill 和 decode 必须声明相同的上下文长度。** 没有引擎会检查这一点，不匹配在长提示词到来之前不可见。
- **Decode 的张量并行大小必须至少等于 prefill。** 引擎确实会对此进行断言，但报告的是 decode 内部错误，而非配置问题。
- 两侧的**数据类型、KV 缓存数据类型、块大小和 KV 缓存布局必须一致**。
- **角色引擎必须是所选模式能够配置的引擎。** 跨角色混用引擎需要 `custom` 模式，并由你提供连接参数。
- **使用 `custom` 模式的分离部署不能同时使用扩展 KV 缓存。** 其他模式都会自动组合两者（参见[结合共享 KV 缓存使用](#using-a-shared-kv-cache-with-it)）；`custom` 完全不注入连接器配置，因此没有可以组合缓存的配置，需要自行编写组合配置。
- **定时伸缩与分离部署不能组合使用。**

对于 `vllm-nixl`，兼容性检查包括角色专用环境变量。`VLLM_KV_CACHE_LAYOUT` 接受等价别名（`NHD`/`LBNHC` 和 `HND`/`LBHNC`）。对于 MLA 模型，`VLLM_MLA_DISABLE` 必须产生兼容的潜在 KV 表示。在支持的运行时中，NIXL 循环状态传输要求 `VLLM_SSM_CONV_STATE_LAYOUT=DS`。对于分块局部注意力，会结合 HMA 参数检查 `VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE`。显式选择的注意力后端也会比较。

角色未设置环境时，继承部署环境；显式字典会替换它，空字典也不例外。依赖模型的检查使用源配置和角色覆盖值。未知配置、运行时版本和传输布局通过 `pairing_unverified` 报告。角色之间的网络地址、端口、设备名称和性能设置可以不同。实验性的布局置换仍受引擎传输方向和 HMA 限制约束。

## 限制 { #limitations }

- Router 以单副本运行。
- **`custom` router 完全按所写配置部署。** 内置模式会将上游 router 的熔断器设置为默认阈值，即十次失败请求；自定义 router 保留自身附带的默认值。
- **vLLM 模式下，返回错误的 prefill 成员不会从轮转中移除。** 上游 router 只将传输错误视为 prefill 失败，因此 prefill 返回 HTTP 500 时只会记录日志，请求仍继续发往 decode 并返回 200。实测使用模拟 prefill，让十二次请求全部失败，均未触发熔断。SGLang 模式则会把此类错误计入熔断器。无论哪种模式，组面板中的 **PD Effectiveness** 指标才是发现某成员已停止贡献的依据；router 自身的请求计数器无法反映。
- **组空闲期间死亡的成员，要到下一次请求时才会被发现。** 检测沿请求路径执行，没有流量就没有检测机会；router 后台扫描每分钟运行一次。空闲期后的首批请求会重试，然后绕过死亡成员路由。
- 组内所有成员必须使用相同 GPU 类型；混用卡型的组无法原子准入。
- 角色仅限 prefill、decode 和 router。
- **感知前缀的路由用于选择 prefill，不用于选择 decode。** Prefill 才是共享前缀能够带来收益的位置，因此内置模式按近似前缀匹配路由；decode 在整个生成过程中持有请求，因此采用轮询保持负载均衡。`vllm-ascend-mooncake` 模式两侧都保留 router 自身的默认值。
