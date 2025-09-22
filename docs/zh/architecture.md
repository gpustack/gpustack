# 架构

下图展示了 GPUStack 的架构：

![gpustack 架构](../assets/gpustack-architecture.png)

### 服务器

GPUStack 服务器由以下组件组成：

- **API 服务器**：为客户端提供 RESTful 接口以与系统交互，并处理身份验证与授权。
- **调度器**：负责将模型实例分配给工作节点。
- **模型控制器**：管理模型实例的发布与扩缩容，以匹配期望的模型副本数。
- **HTTP 代理**：将推理 API 请求路由到工作节点。

### 工作节点

GPUStack 的工作节点负责：

- 运行分配给该节点的模型实例的推理服务器。
- 向服务器报告状态。
- 将推理 API 请求路由到后端推理服务器。

### SQL 数据库

GPUStack 服务器连接到 SQL 数据库作为数据存储。GPUStack 默认使用 SQLite，但你也可以将其配置为使用外部的 PostgreSQL 或 MySQL。

### 推理服务器

推理服务器是执行推理任务的后端。GPUStack 支持将 [vLLM](https://github.com/vllm-project/vllm)、[Ascend MindIE](https://www.hiascend.com/en/software/mindie)、[llama-box](https://github.com/gpustack/llama-box) 和 [vox-box](https://github.com/gpustack/vox-box) 作为推理服务器。

### RPC 服务器

RPC 服务器使得可以在远程主机上运行 llama-box 后端。推理服务器与一个或多个 RPC 服务器实例通信，将计算卸载到这些远程主机。此设置允许在多个工作节点之间进行分布式 LLM 推理，即使单个资源受限，系统也能加载更大的模型。

### Ray 主节点/工作节点

[Ray](https://ray.io) 是一个分布式计算框架，GPUStack 利用它来运行分布式 vLLM。用户可以在 GPUStack 中启用 Ray 集群，以在多个工作节点上运行 vLLM。默认情况下，该功能处于禁用状态。