# GPUStack 多服务端调度改造项目

## 项目概述

本项目对 GPUStack 开源项目进行了改造，增加了多服务端调度支持，使 GPUStack 能够在多 Server 环境下实现协调的模型调度、高可用部署和负载均衡。

## 改造内容

### 1. 新增组件

#### 1.1 协调服务 (`gpustack/coordinator/`)
- **CoordinatorService**: 分布式协调服务核心类
- **LocalCoordinatorService**: 本地协调服务（单 Server 模式）
- **DistributedCoordinatorService**: 分布式协调服务（使用外部协调服务）
- **ServerInfo**: Server 信息数据模型
- **DistributedLock**: 分布式锁信息模型

**核心功能**:
- Server 注册与注销
- 分布式锁管理
- 心跳检测
- Worker 状态同步
- 负载信息收集

#### 1.2 分布式调度器 (`gpustack/scheduler/`)
- **DistributedScheduler**: 分布式调度器
- 支持分布式锁防止重复调度
- 协调服务集成
- 多调度模式支持（local/distributed/auto）

#### 1.3 Server 协调模块 (`gpustack/server/`)
- **MultiServerCoordinator**: 多 Server 协调器
- **ServerStateManager**: Server 状态管理器
- **WorkerFederationManager**: Worker 联邦管理器
- **coordinator_routes.py**: 协调服务 API 端点

#### 1.4 Worker 联邦 (`gpustack/worker/`)
- **WorkerFederation**: Worker 联邦管理器
- **MultiServerWorkerSelector**: 多 Server Worker 选择器
- 支持多 Server 注册
- 自动故障转移

### 2. 配置改造

#### 新增配置项

```python
# 多Server配置
server_id: Optional[str] = None                    # Server唯一标识
server_urls: Optional[List[str]] = []              # 所有Server URL列表
coordinator_url: Optional[str] = None              # 协调服务URL
scheduling_mode: str = "auto"                      # 调度模式
heartbeat_interval: int = 15                       # 心跳间隔
server_timeout: int = 60                           # Server超时时间
lock_timeout: int = 30                             # 锁超时时间
distributed_scheduling: bool = True               # 是否启用分布式调度
schedule_lock_timeout: int = 60                   # 调度锁超时时间
```

#### 环境变量支持

```bash
GPUSTACK_SERVER_ID=server-01
GPUSTACK_SERVER_URLS=http://server1:30080,http://server2:30080
GPUSTACK_COORDINATOR_URL=http://etcd:2379
GPUSTACK_SCHEDULING_MODE=distributed
GPUSTACK_DISTRIBUTED_SCHEDULING=true
```

### 3. API 端点

#### 协调服务 API

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/v1/coordinator/servers` | GET | 获取所有 Server |
| `/api/v1/coordinator/servers/register` | POST | 注册 Server |
| `/api/v1/coordinator/servers/unregister/{server_id}` | POST | 注销 Server |
| `/api/v1/coordinator/heartbeat` | POST | Server 心跳 |
| `/api/v1/coordinator/cluster` | GET | 获取集群信息 |
| `/api/v1/coordinator/locks/acquire` | POST | 获取分布式锁 |
| `/api/v1/coordinator/locks/release` | POST | 释放分布式锁 |
| `/api/v1/coordinator/locks` | GET | 获取所有锁 |
| `/api/v1/coordinator/health` | GET | 健康检查 |

## 项目结构

```
gpustack/
├── coordinator/
│   ├── __init__.py          # 协调服务核心实现
│   └── coordinator.py       # (已集成到 __init__.py)
├── scheduler/
│   ├── scheduler.py         # 原有调度器（保持兼容）
│   ├── distributed_scheduler.py  # 新增分布式调度器
│   └── queue.py             # 调度队列
├── server/
│   ├── server.py            # 原有 Server 类
│   ├── multi_server.py      # 新增多 Server 协调模块
│   └── coordinator_routes.py # 协调服务 API 端点
├── worker/
│   ├── worker.py            # 原有 Worker
│   └── worker_federation.py # 新增 Worker 联邦
├── schemas/
│   ├── servers.py           # 新增 Server 相关 Schema
│   └── workers.py           # 原有 Worker Schema
├── config/
│   └── config.py            # 配置类（已更新）
├── examples/
│   └── multi_server/        # 配置示例
└── docs/
    └── multi_server_deployment.md  # 部署文档
```

## 部署架构

```
┌─────────────────────────────────────────────────────────────┐
│                      GPUStack 集群                          │
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐               │
│  │ Server 1 │◄─►│ Server 2 │◄─►│ Server N │               │
│  │ :30080   │   │ :30080   │   │ :30080   │               │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘               │
│       │              │              │
│       └──────────────┼──────────────┘                      │
│                      │                                     │
│       ┌──────────────┼──────────────┐                      │
│       │              │              │                      │
│  ┌────▼─────┐  ┌────▼─────┐  ┌────▼─────┐                │
│  │ Worker 1 │  │ Worker 2 │  │ Worker N │                │
│  └──────────┘  └──────────┘  └──────────┘                │
│                                                             │
└─────────────────────────────────────────────────────────────┘

共享数据库: PostgreSQL (:5432)
```

## 使用方法

### 1. 单 Server 模式（默认）

```bash
gpustack start --config config.yaml
```

### 2. 多 Server 模式

#### 启动 Server 1

```yaml
# server1.yaml
server_id: "server-01"
server_urls:
  - "http://192.168.1.10:30080"
  - "http://192.168.1.11:30080"
  - "http://192.168.1.12:30080"
scheduling_mode: "distributed"
distributed_scheduling: true
```

```bash
gpustack start --config server1.yaml
```

#### 启动 Server 2 和 Server 3

```bash
gpustack start --config server2.yaml
gpustack start --config server3.yaml
```

### 3. 查看集群状态

```bash
curl http://server1:30080/api/v1/coordinator/cluster

# 返回示例
{
  "servers": [...],
  "total_workers": 10,
  "total_load": 0.35,
  "active_servers": 3
}
```

## 技术特性

### 1. 分布式锁机制

调度器使用分布式锁确保同一模型实例不会被多个 Server 同时调度：

```python
# 获取锁
acquired = await coordinator.acquire_lock(
    lock_id=f"schedule_lock_{instance_id}",
    owner=server_id,
    timeout=60
)

# 释放锁
await coordinator.release_lock(lock_id, server_id)
```

### 2. 心跳检测

所有 Server 定期发送心跳，超时（默认 60 秒）后被标记为不活跃。

### 3. 负载均衡

调度器根据 Server 负载选择最优 Server：

```python
load_score = worker_factor * 0.4 + load_factor * 0.6
```

### 4. 故障转移

当活跃 Server 故障时：
- Worker 自动切换到备用 Server
- 调度任务自动转移到其他活跃 Server
- 故障 Server 恢复后自动重新加入集群

## 兼容性

- ✅ 保持与原有 API 100% 兼容
- ✅ 原有配置无需修改即可运行
- ✅ 支持渐进式迁移（逐个升级 Server）
- ✅ 支持与原有单 Server 实例共存

## 测试

### 运行单元测试

```bash
pytest tests/scheduler/test_distributed_scheduler.py -v
pytest tests/coordinator/ -v
```

### 运行集成测试

```bash
# 启动测试集群
docker-compose -f examples/multi_server/docker-compose.test.yaml up -d

# 运行测试
pytest tests/integration/test_multi_server.py -v
```

## 性能优化

1. **锁超时**：调度锁默认 60 秒超时，避免死锁
2. **心跳间隔**：可配置的心跳间隔（默认 15 秒）
3. **批量同步**：Worker 状态批量同步减少网络开销
4. **连接池**：HTTP 连接池减少连接建立开销

## 安全考虑

1. **认证**：所有 API 端点需要认证
2. **加密**：支持 TLS 加密通信
3. **隔离**：每个 Server 有唯一标识
4. **审计**：记录所有协调操作

## 常见问题

### Q: 如何判断是否需要多 Server 模式？

A: 当满足以下条件时考虑多 Server 模式：
- 需要高可用性
- 单 Server 负载过高
- 需要跨地域部署
- Worker 数量超过 20 个

### Q: 多 Server 部署需要几个 Server？

A: 建议至少 3 个 Server 实现高可用：
- 1 个故障时仍可正常工作
- 2 个故障时服务降级但不中断

### Q: 如何从单 Server 迁移到多 Server？

A: 渐进式迁移：
1. 升级所有 Worker 到支持联邦的版本
2. 部署新的 Server 实例
3. 更新配置启用分布式调度
4. 原 Server 自动加入集群

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 许可证

本改造项目遵循 GPUStack 原项目的 Apache 2.0 许可证。

## 参考资料

- [GPUStack GitHub](https://github.com/gpustack/gpustack)
- [部署文档](docs/multi_server_deployment.md)
- [配置示例](examples/multi_server/config_examples.sh)
