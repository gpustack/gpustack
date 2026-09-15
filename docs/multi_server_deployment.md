# GPUStack 多服务端调度功能使用指南

## 概述

GPUStack 多服务端调度功能允许您部署多个 Server 实例来实现：
- **高可用性**：单个 Server 故障不会影响整体服务
- **负载均衡**：模型调度负载分散到多个 Server
- **水平扩展**：根据需求增加更多 Server 实例
- **容灾备份**：Worker 可以自动故障转移到其他 Server

## 架构设计

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Server 1  │◄───►│   Server 2  │◄───►│   Server N  │
│  (调度器)    │     │  (调度器)    │     │  (调度器)    │
│  端口:30080  │     │  端口:30081 │     │  端口:30082 │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
┌──────▼──────┐     ┌──────▼──────┐     ┌──────▼──────┐
│   Worker 1  │     │   Worker 2  │     │   Worker N  │
└─────────────┘     └─────────────┘     └─────────────┘
```

## 配置方法

### 1. 基础配置（单 Server 模式）

默认情况下，GPUStack 以单 Server 模式运行：

```yaml
# config.yaml
port: 80
api_port: 30080
database_url: "postgresql://root@127.0.0.1:5432/gpustack"
```

### 2. 多 Server 配置

#### Server 1 配置

```yaml
# server1.yaml
port: 80
api_port: 30080
database_url: "postgresql://root@127.0.0.1:5432/gpustack"

# 多Server配置
server_id: "server-01"  # 可选，不指定则自动生成
server_urls:
  - "http://server1.example.com:30080"
  - "http://server2.example.com:30080"
  - "http://server3.example.com:30080"

# 调度配置
scheduling_mode: "distributed"  # local, distributed, auto
distributed_scheduling: true
schedule_lock_timeout: 60

# 心跳配置
heartbeat_interval: 15
server_timeout: 60
lock_timeout: 30
```

#### Server 2 配置

```yaml
# server2.yaml
port: 80
api_port: 30080
database_url: "postgresql://root@127.0.0.1:5432/gpustack"

# 多Server配置
server_id: "server-02"
server_urls:
  - "http://server1.example.com:30080"
  - "http://server2.example.com:30080"
  - "http://server3.example.com:30080"

scheduling_mode: "distributed"
distributed_scheduling: true
schedule_lock_timeout: 60
```

#### Server 3 配置

```yaml
# server3.yaml
port: 80
api_port: 30080
database_url: "postgresql://root@127.0.0.1:5432/gpustack"

# 多Server配置
server_id: "server-03"
server_urls:
  - "http://server1.example.com:30080"
  - "http://server2.example.com:30080"
  - "http://server3.example.com:30080"

scheduling_mode: "distributed"
distributed_scheduling: true
schedule_lock_timeout: 60
```

### 3. 使用外部协调服务（可选）

如果需要真正的分布式协调，可以使用 etcd 或 Consul：

```yaml
# 使用 etcd
coordinator_url: "http://etcd.example.com:2379"

# 或使用 Consul
coordinator_url: "http://consul.example.com:8500"
```

## 启动顺序

### 1. 启动数据库

确保所有 Server 使用同一个数据库：

```bash
# 使用 Docker 启动 PostgreSQL
docker run -d \
  --name gpustack-db \
  -p 5432:5432 \
  -e POSTGRES_PASSWORD=your_password \
  postgres:15
```

### 2. 启动第一个 Server

```bash
gpustack start --config server1.yaml
```

### 3. 启动其他 Server

```bash
gpustack start --config server2.yaml
gpustack start --config server3.yaml
```

## 验证多 Server 部署

### 1. 检查 Server 状态

```bash
# 访问任一 Server 的 API
curl http://server1.example.com:30080/api/v1/coordinator/servers

# 返回示例
{
  "servers": [
    {
      "id": "server-01",
      "name": "gpustack-server-01",
      "address": "server1.example.com",
      "api_port": 30080,
      "status": "active",
      "last_heartbeat": "2026-01-27T13:00:00Z",
      "current_load": 0.3,
      "worker_count": 5
    },
    {
      "id": "server-02",
      "name": "gpustack-server-02",
      "address": "server2.example.com",
      "api_port": 30080,
      "status": "active",
      "last_heartbeat": "2026-01-27T13:00:00Z",
      "current_load": 0.2,
      "worker_count": 3
    }
  ],
  "count": 2
}
```

### 2. 检查集群状态

```bash
curl http://server1.example.com:30080/api/v1/coordinator/cluster

# 返回示例
{
  "servers": [...],
  "total_workers": 8,
  "total_load": 0.25,
  "active_servers": 2,
  "timestamp": "2026-01-27T13:00:00Z"
}
```

## Worker 配置

### Worker 向多个 Server 注册

```yaml
# worker.yaml
server_url: "http://server1.example.com:30080"

# 可选：指定多个 Server URL 用于故障转移
additional_server_urls:
  - "http://server2.example.com:30080"
  - "http://server3.example.com:30080"
```

## 调度策略

### 1. 调度模式

- **local**：仅本地调度，不参与分布式调度
- **distributed**：完全分布式调度，使用协调服务
- **auto**：自动模式，单 Server 时使用本地调度，多 Server 时使用分布式调度

### 2. 分布式锁

多 Server 调度使用分布式锁防止重复调度：

```
锁名称格式：schedule_lock_{instance_id}
默认超时：60 秒
```

### 3. 负载均衡

调度器会自动选择负载最低的 Server 进行调度：

```
负载计算公式：
负载分数 = Worker数量因子 × 0.4 + 当前负载 × 0.6
```

## 故障转移

### Server 故障

当一个 Server 故障时：
1. 其他 Server 会在 60 秒（server_timeout）后将其标记为不活跃
2. 调度任务会自动转移到活跃的 Server
3. Worker 会自动故障转移到备用 Server

### 恢复

当故障 Server 恢复后：
1. 它会重新注册到协调服务
2. 自动参与调度任务
3. 负载会逐渐平衡

## 监控和日志

### 查看协调服务状态

```bash
# 检查所有 Server
curl http://server1.example.com:30080/api/v1/coordinator/servers

# 检查分布式锁
curl http://server1.example.com:30080/api/v1/coordinator/locks
```

### 日志位置

- 协调服务日志：`/var/lib/gpustack/logs/coordinator.log`
- 调度器日志：`/var/lib/gpustack/logs/scheduler.log`

## 最佳实践

1. **数据库**：使用外部 PostgreSQL 数据库，不要使用嵌入式 SQLite
2. **负载均衡**：在生产环境中使用负载均衡器（如 Nginx）分发 API 请求
3. **监控**：配置监控告警，当 Server 数量低于预期时通知
4. **网络**：确保所有 Server 之间网络互通
5. **时钟同步**：所有 Server 使用 NTP 同步时钟

## 故障排除

### 问题：Server 无法注册

1. 检查数据库连接
2. 检查网络连通性
3. 查看日志：`journalctl -u gpustack-server`

### 问题：Worker 无法注册

1. 检查 Worker 配置的 server_url
2. 检查认证令牌
3. 查看 Worker 日志

### 问题：调度失败

1. 检查分布式锁状态
2. 确认所有 Server 时间同步
3. 查看调度器日志
