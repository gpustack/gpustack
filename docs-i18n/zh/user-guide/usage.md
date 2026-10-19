---
translation_status: translated
---

# 用量 { #usage }

**Usage** 页面报告 GPUStack 资源随时间变化的消耗情况，包括 LLM token、GPU/CPU 实例运行时长和存储容量，便于跟踪使用情况、将消耗归属到用户，并评估成本。

## 访问用量页面 { #accessing-the-usage-page }

1. 从左侧导航进入 `Usage` 页面（柱状图图标）。
2. 选择选项卡：`Summary`、`Tokens`、`GPU Instances`、`Storage` 或 `Resource Events`。
3. 设置 `date range` 及其他筛选条件，确定报表范围。

各选项卡如下：

- **Summary**——token、计算和存储的概览。
- **Tokens**——LLM token 消耗，按模型 / 用户 / API 密钥细分。
- **GPU Instances**——GPU/CPU 实例运行时长，按实例类型 / 实例 / 用户细分。
- **Storage**——存储容量用量，按存储 / 用户细分。
- **Resource Events**——计算和存储数值背后的生命周期审计日志。

## 计量范围 { #what-gets-metered }

| 资源 | 计量时段 | 分桶粒度 |
| --- | --- | --- |
| Token | 推理请求通过网关提供服务时。 | 每日 |
| GPU/CPU 实例 | 实例处于运行中（计量）阶段时。**Stopped** 或已删除实例不累计用量。 | 每小时 |
| 存储 | 从创建到删除，无论是否挂载均计量容量。 | 每小时 |

## 可见范围 { #visibility }

GPUStack 有两种角色：**Admin** 和 **User**（参见[用户管理](user-management.md)）：

- **Admin**——可以查看所有用户的用量，并通过 **Filter by user** 控件缩小查看范围。
- **User**——只能查看自己的用量，无法使用按用户细分的视图。

## 通用控件 { #common-controls }

除使用独立筛选器的 Resource Events 外，各选项卡共享相同控件：

- **Date range**——报表时间范围，默认为最近 30 天。
- **Filter by user**——限制为指定用户（仅管理员可用）。Tokens 选项卡还提供 **Filter by model** 和 **Filter by API key**。
- **Refresh**——使用当前筛选条件重新获取数据。
- **Metric**——趋势图和 KPI 排序使用的指标（例如 GPU Hours 或 Instance Hours）。
- **Group by**——将趋势图拆分为每组一条数据序列（例如按实例类型分组）。
- **Granularity**——趋势分桶粒度。GPU Instances 和 Storage 支持 **Hour / Day / Week / Month**；Tokens 仅支持 **Day / Week / Month**（token 用量按日汇总，因此没有小时桶）。
- **Export**（下载图标，在 Tokens / GPU Instances / Storage 中可用）——打开当前筛选后明细的预览，并支持下载。

## 指标定义 { #metric-definitions }

| 指标 | 含义 |
| --- | --- |
| **Input Tokens** | 提示词 token。**Input Cached Tokens** 是其中由提示词缓存提供的子集。 |
| **Output Tokens** | 生成的补全 token。 |
| **Total Tokens** | 输入 + 输出。 |
| **Instance Hours** | 实例的实际运行时长：Σ(uptime)，与实例包含的卡数无关。 |
| **GPU Hours** | 加速器运行时长：Σ(uptime × GPU 卡数)。一个双 GPU 实例运行 1 小时 = 2 GPU Hours，但只有 1 Instance Hour。 |
| **GB-Days** | 随时间累计的已分配存储：容量（GB）× 保留天数。 |
| **GB-Hours** | 与 GB-Days 相同，以小时为单位表示。 |
| **Active Instances** / **Active Storages** / **Active Users** | 时间范围内仍存在（未标记为 **Deleted**）的资源。 |
| **Last Active** | 资源最近一次累计用量的时间桶。 |

!!! note

    已删除资源仍保留历史用量：已删除的模型、实例或存储仍会出现在表格中（标记为 **Deleted**），并计入总量，但不再计为 **Active**。

## 概览 { #summary }

![用量概览](../assets/usage/usage-summary.png)

Summary 选项卡依次展示 **Tokens**、**Compute** 和 **Storage** 三个部分。每个部分都包含关键统计行、按类型细分的环形图，以及所选时间范围内的趋势图，无需切换选项卡即可快速了解三类资源。

## Token { #tokens }

![Token 用量](../assets/usage/usage-tokens.png)

显示 LLM 推理的 token 消耗。顶部 KPI 包括 **Input / Output / Total Tokens**、**API Requests** 和 **Models Used**。

底部明细表按三种方式分组：

- **Models**——每个模型的 Input Tokens / Input Cached Tokens / Output Tokens / Total Tokens、API Requests、Last Active。
- **Users**——按用户（仅管理员可用）。
- **API Keys**——按 API 密钥。

## GPU 实例 { #gpu-instances }

![GPU 实例用量](../assets/usage/usage-gpu-instances.png)

显示 GPU 和 CPU 实例的运行时长。KPI 包括 **GPU Hours**、**Instance Hours**、**Active Instances** 和 **Active Users**。选择 **GPU Hours** 或 **Instance Hours** 作为图表指标，并按**实例类型**、**实例**或**用户**对趋势分组。

明细表按以下方式分组：

- **Instance Types**——每个实例类型的 GPU Hours、Instance Hours、Active Instances、Last Active。
- **Instances**——按单个实例。
- **Users**——按用户（仅管理员可用）。

!!! note

    CPU 实例没有加速器，因此会计入 **Instance Hours**，不会计入 **GPU Hours**。

## 存储 { #storage }

![存储用量](../assets/usage/usage-storage.png)

显示存储容量用量。KPI 包括 **GB-Days**、**GB-Hours**、**Active Storages** 和 **Active Users**。选择 **GB-Days** 或 **GB-Hours** 作为图表指标，并按**存储**或**用户**分组。

明细表按以下方式分组：

- **Storage**——每个卷的 Type、Capacity、GB-Days、GB-Hours、Last Active。
- **Users**——按用户（仅管理员可用）。

!!! note

    由于存储从创建到删除均计量，无论是否挂载，已存在但空闲的存储仍会累计 GB-Days/GB-Hours。

## 资源事件 { #resource-events }

![用量资源事件](../assets/usage/usage-resource-events.png)

这是计算和存储数值背后的生命周期审计日志，每次影响计量的状态转换对应一行，可用于解释数值的来源。

筛选条件包括**日期范围**、**资源类型**（GPU Instance / Storage）、**事件类型**，以及模糊**资源名称**搜索。

## 时区 { #timezone }

Usage 页面中的所有日期和时间，包括趋势时间桶、**Last Active** 和资源事件时间，都采用统一且可配置的**汇总时区**显示，确保每个选项卡使用相同的日历边界。它由 GPUStack Server 环境变量 `GPUSTACK_TIMEZONE` 控制（旧版 `GPUSTACK_USAGE_ROLLUP_TIMEZONE` 仍作为已弃用的别名生效），默认使用 GPUStack Server 操作系统的本地时区。详情及夏令时说明参见[环境变量](../environment-variables.md#usage-tracking-configuration)。

## 数据保留 { #data-retention }

Token、计算和存储汇总数据保留约 13 个月，之后由后台任务迁移到归档表。保留时间和归档计划可通过 `GPUSTACK_*_RETENTION_MONTHS` 和 `GPUSTACK_*_ARCHIVE_CRON` 环境变量配置。
