---
translation_status: translated
---

# 基准测试 { #benchmarking }

GPUStack 可以对运行中的模型实例执行基准测试。基准测试由 Worker 在专用基准测试容器镜像中执行，结果和日志存储在 Worker 上。

## 前提条件 { #prerequisites }

- 模型实例正在运行且状态正常。

已有运行实例无需重新部署即可执行基准测试。报告仅使用与实例当前启动匹配的配置快照；缺少快照或无法确认快照属于本次启动时，使用创建基准测试时的模型配置（包含角色覆盖），并在报告中提示对应实例的配置仅供参考。这些参考配置可能与实际运行配置不同。基准测试不会将参考配置回填到实例的启动快照中。

### 路由基准测试的网关地址 { #route-benchmark-gateway-address }

路由基准测试使用基准测试 Worker 可访问的 Higress 基础 URL（不含 `/v1`），按以下顺序取值：

1. 执行 Worker 所属集群的 `server_url`。
2. Server 的 `server_external_url`。
3. 创建请求的源地址，且必须由 Server 的 `trusted_hosts` 显式允许。

使用请求源地址回退时不接受通配符。仅填写主机名只允许标准 HTTP/HTTPS 端口；自定义端口必须显式填写，例如 `gateway.example:8443`。如果 URL 包含路径前缀，前置代理必须在转发到 Higress 前剥离该前缀。

目标地址在创建基准测试时固定。修改 URL 后需重新创建基准测试。进度上报继续使用 Worker 的管理地址。

## 创建基准测试 { #create-benchmark }

1. 进入 `Benchmarks` 页面。
2. 单击 `Add Benchmark`。
3. 选择实例并填写配置。
4. 单击 `Save` 按钮。

### ShareGPT 工作负载长度 { #sharegpt-workload-lengths }

对于内置 ShareGPT 数据集，可以选择设置输入 token 长度的最小值、最大值，或同时设置两者。GPUStack 会保留经分词器计算后 token 数满足指定限制的首轮提示词。输出 token 长度为可选项：设置后，所有保留的提示词都会请求相同的最大输出长度；未设置时，每条提示词使用数据集中参考答案的长度。如果没有提示词落在指定范围内，本次运行会在发送基准测试请求前失败。

## 查看基准测试结果 { #view-benchmark-results }

1. 进入 `Benchmarks` 页面。
2. 找到要查看的基准测试。
3. 单击基准测试名称，查看结果和配置快照。

## 已保存的监控 { #saved-monitoring }

GPUStack 将新建基准测试的监控数据存入 Server 数据库。默认使用内置 PostgreSQL，也可通过 `--database-url` 配置数据库。即使部署资源已移除，仍可在「阶段与监控」回看整场和各测点已有的曲线。源指标不全时，归档可能只覆盖一部分。拥有归档硬件详情查看权限的用户可在保存状态旁看到压缩后的大小。基准测试结果可独立查看。

路由基准测试会在创建时记录所选部署中正在运行的副本。如果 GPUStack 检测到已记录的副本重启、重新分配资源或被移除，本次基准测试的实时监控将不可用。归档保留变更前符合条件的数据，并标记为部分覆盖。

归档采用压缩存储，压缩前后的数据都受单次运行上限约束，默认为 256 MiB。容量测量使用了两场各约 6 分钟的基准测试数据，普通部署与 PD 部署的数据压缩后分别约为 177 KiB 和 528 KiB。更大规模的运行可能产生更大的归档。请将归档空间纳入数据库容量规划和备份。

平台管理员可在「基准测试」页的「归档存储」设置全局容量上限。默认值为 `0`，表示全局容量不限。设置上限后，各组织共同使用按压缩后字节数计算的额度。单个组织用尽可用额度后，所有组织会暂停新增或增大归档。最后一次成功保存的内容仍可查看。

每次保存都通过一条数据库语句写入压缩归档。使用 MySQL 兼容数据库时，请检查实际生效的 `max_allowed_packet`，将 `--benchmark-archive-max-run-bytes` 设在数据包限制以下，并为 SQL 编码和元数据留出空间。[MySQL 数据包限制](https://dev.mysql.com/doc/refman/8.0/en/packet-too-large.html) · [OceanBase 数据包限制](https://en.oceanbase.com/docs/common-oceanbase-database-10000000003681766)

已封存的归档在上次修复结束后，至少要等一个归档写入间隔（默认 60 秒）才可再次修复。

## 导出基准测试结果 { #export-benchmark-results }

1. 进入 `Benchmarks` 页面。
2. 找到要导出的基准测试。
3. 选择一个或多个要导出的基准测试运行记录。
4. 单击 `Export` 按钮下载结果。

## 编辑基准测试 { #edit-benchmark }

1. 进入 `Benchmarks` 页面。
2. 找到要编辑的基准测试。
3. 单击 `Operations` 列中的 `Edit` 按钮。
4. 根据需要修改名称和描述。
5. 单击 `Save` 按钮。

## 删除基准测试 { #delete-benchmark }

1. 进入 `Benchmarks` 页面。
2. 找到要删除的基准测试。
3. 单击 `Operations` 列中的 `Delete` 按钮。
4. 确认删除。

## 使用私有 CA 的 HTTPS GPUStack Server 上的基准测试 { #benchmarks-on-an-https-server-with-a-private-ca }

进度上报和路由推理遵循 Worker 的 TLS 策略。默认启用证书验证，除非 Worker 启用了 `GPUSTACK_INSECURE_TLS` 或 `insecure_tls` 配置选项（`--insecure-tls`）。基准测试没有单独的 TLS 配置。

任一入口使用 HTTPS 且启用验证时，Worker 会将自身信任的 CA 包以文件内容的形式注入基准测试容器，包括 Worker 已导入的私有 CA（参见[额外受信任的 CA](../installation/installation.md#additional-trusted-cas)）。进度客户端使用 `--progress-ca-cert`；HTTPS 推理通过 `SSL_CERT_FILE` 使用注入的 CA 包，并启用后端验证。管理地址为 HTTP 不会阻止为 HTTPS 网关注入 CA。

如果 Worker 显式跳过验证，基准测试的两个通道都会跳过验证，且不注入 CA 包。如果需要验证 HTTPS 推理证书但无法读取 CA 包，基准测试会拒绝启动，而不是使用其他信任库。仅进度上报仍采用尽力而为的方式：上报失败会记录日志，不会丢弃基准测试测量结果。
