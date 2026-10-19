---
translation_status: translated
---

# 架构 { #architecture }

下图展示了 GPUStack 的整体架构。

![GPUStack v2 架构](assets/gpustack-v2-architecture.png)

下图详细展示了内部组件及其交互关系。

![GPUStack v2 组件](assets/gpustack-v2-components.png)

### Server { #server }

GPUStack Server 包含以下组件：

- **API Server：** 提供 RESTful 接口，供客户端与系统交互，并负责身份认证和授权。
- **调度器：** 负责将模型实例分配给 Worker。
- **控制器：** 管理系统中的资源状态。例如，负责模型实例的发布和扩缩容，使其达到期望的副本数量。

### Worker { #worker }

GPUStack Worker 包含以下组件：

- **GPUStack Runtime：** 检测 GPU 设备，并与容器运行时交互以部署模型实例。
- **推理服务管理器：** 管理 Worker 上模型实例的生命周期。
- **指标导出器：** 导出模型实例及其性能指标。

### AI 网关 { #ai-gateway }

AI 网关处理来自客户端的 API 请求，根据请求的模型将流量路由到相应的模型实例。GPUStack 使用 [Higress](https://github.com/alibaba/higress) 实现 API 路由和负载均衡。

### SQL 数据库 { #sql-database }

GPUStack Server 连接 SQL 数据库以存储数据。GPUStack 默认使用内置 PostgreSQL，也可配置外部数据库。兼容的数据库及已验证版本，请参阅[数据库要求](installation/requirements.md#database-requirements)。

### 推理服务 { #inference-server }

推理服务是执行推理任务的后端。GPUStack 内置支持 [vLLM](https://github.com/vllm-project/vllm)、[SGLang](https://github.com/sgl-project/sglang)、[昇腾 MindIE](https://www.hiascend.com/en/software/mindie) 和 [VoxBox](https://github.com/gpustack/vox-box)，也支持添加自定义推理后端。

### Ray { #ray }

[Ray](https://ray.io) 是一个分布式计算框架，GPUStack 使用它运行分布式 vLLM。GPUStack 按需启动 Ray 集群，以跨多个 Worker 运行分布式 vLLM。
