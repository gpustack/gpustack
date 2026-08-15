# GPUStack 多Server部署配置示例
# 文件路径: examples/multi_server/

# ============================================
# Server 1 配置 (主Server)
# ============================================
cat > server1.yaml << 'EOF'
# GPUStack Server 1 配置

# 基础配置
port: 80
api_port: 30080
database_url: "postgresql://postgres@127.0.0.1:5432/gpustack?sslmode=disable"

# 多Server配置
server_id: "server-01"
server_urls:
  - "http://192.168.1.10:30080"
  - "http://192.168.1.11:30080"
  - "http://192.168.1.12:30080"

# 调度配置
scheduling_mode: "distributed"  # local | distributed | auto
distributed_scheduling: true
schedule_lock_timeout: 60

# 心跳配置
heartbeat_interval: 15
server_timeout: 60
lock_timeout: 30

# 数据目录
data_dir: "/var/lib/gpustack"
cache_dir: "/var/cache/gpustack"

# 日志配置
log_dir: "/var/log/gpustack"

# 可选：使用外部协调服务
# coordinator_url: "http://etcd.example.com:2379"
EOF

# ============================================
# Server 2 配置
# ============================================
cat > server2.yaml << 'EOF'
# GPUStack Server 2 配置

# 基础配置
port: 80
api_port: 30080
database_url: "postgresql://root@127.0.0.1:5432/gpustack?sslmode=disable"

# 多Server配置
server_id: "server-02"
server_urls:
  - "http://192.168.1.10:30080"
  - "http://192.168.1.11:30080"
  - "http://192.168.1.12:30080"

# 调度配置
scheduling_mode: "distributed"
distributed_scheduling: true
schedule_lock_timeout: 60

# 心跳配置
heartbeat_interval: 15
server_timeout: 60
lock_timeout: 30

# 数据目录
data_dir: "/var/lib/gpustack"
cache_dir: "/var/cache/gpustack"

log_dir: "/var/log/gpustack"
EOF

# ============================================
# Server 3 配置
# ============================================
cat > server3.yaml << 'EOF'
# GPUStack Server 3 配置

# 基础配置
port: 80
api_port: 30080
database_url: "postgresql://root@127.0.0.1:5432/gpustack?sslmode=disable"

# 多Server配置
server_id: "server-03"
server_urls:
  - "http://192.168.1.10:30080"
  - "http://192.168.1.11:30080"
  - "http://192.168.1.12:30080"

# 调度配置
scheduling_mode: "distributed"
distributed_scheduling: true
schedule_lock_timeout: 60

# 心跳配置
heartbeat_interval: 15
server_timeout: 60
lock_timeout: 30

# 数据目录
data_dir: "/var/lib/gpustack"
cache_dir: "/var/cache/gpustack"

log_dir: "/var/log/gpustack"
EOF

# ============================================
# Worker 配置示例
# ============================================
cat > worker.yaml << 'EOF'
# GPUStack Worker 配置

# 主Server URL（必填）
server_url: "http://192.168.1.10:30080"

# 认证令牌（从Server获取）
token: "your-registration-token"

# 可选：备用Server URL列表
additional_server_urls:
  - "http://192.168.1.11:30080"
  - "http://192.168.1.12:30080"

# Worker网络配置
worker_ip: "192.168.1.20"
worker_ifname: "eth0"
worker_name: "gpu-worker-01"

# Worker端口配置
worker_port: 40050

# 数据目录
data_dir: "/var/lib/gpustack-worker"
EOF

# ============================================
# Docker Compose 部署示例
# ============================================
cat > docker-compose.yaml << 'EOF'
version: '3.8'

services:
  # PostgreSQL 数据库
  db:
    image: postgres:15
    container_name: gpustack-db
    environment:
      POSTGRES_PASSWORD: gpustack_secret
      POSTGRES_DB: gpustack
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  # Server 1
  server1:
    image: gpustack/gpustack:latest
    container_name: gpustack-server1
    command: start --config /etc/gpustack/server1.yaml
    volumes:
      - ./server1.yaml:/etc/gpustack/server1.yaml:ro
      - server1_data:/var/lib/gpustack
      - server1_cache:/var/cache/gpustack
      - server1_log:/var/log/gpustack
    ports:
      - "80:80"
      - "30080:30080"
    depends_on:
      - db
    restart: unless-stopped
    environment:
      - GPUSTACK_DATABASE_URL=postgresql://postgres:gpustack_secret@db:5432/gpustack

  # Server 2
  server2:
    image: gpustack/gpustack:latest
    container_name: gpustack-server2
    command: start --config /etc/gpustack/server2.yaml
    volumes:
      - ./server2.yaml:/etc/gpustack/server2.yaml:ro
      - server2_data:/var/lib/gpustack
      - server2_cache:/var/cache/gpustack
      - server2_log:/var/log/gpustack
    ports:
      - "30081:30080"
    depends_on:
      - db
    restart: unless-stopped
    environment:
      - GPUSTACK_DATABASE_URL=postgresql://root:gpustack_secret@db:5432/gpustack

  # Server 3
  server3:
    image: gpustack/gpustack:latest
    container_name: gpustack-server3
    command: start --config /etc/gpustack/server3.yaml
    volumes:
      - ./server3.yaml:/etc/gpustack/server3.yaml:ro
      - server3_data:/var/lib/gpustack
      - server3_cache:/var/cache/gpustack
      - server3_log:/var/log/gpustack
    ports:
      - "30082:30080"
    depends_on:
      - db
    restart: unless-stopped
    environment:
      - GPUSTACK_DATABASE_URL=postgresql://root:gpustack_secret@db:5432/gpustack

  # 负载均衡器 (Nginx)
  nginx:
    image: nginx:alpine
    container_name: gpustack-nginx
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    ports:
      - "8080:80"
    depends_on:
      - server1
      - server2
      - server3
    restart: unless-stopped

volumes:
  pgdata:
  server1_data:
  server1_cache:
  server1_log:
  server2_data:
  server2_cache:
  server2_log:
  server3_data:
  server3_cache:
  server3_log:
EOF

# ============================================
# Nginx 配置示例
# ============================================
cat > nginx.conf << 'EOF'
worker_processes auto;

events {
    worker_connections 1024;
}

http {
    upstream gpustack_servers {
        least_conn;
        server server1:30080 weight=1;
        server server2:30080 weight=1;
        server server3:30080 weight=1;
    }

    server {
        listen 80;
        server_name gpustack.example.com;

        location / {
            proxy_pass http://gpustack_servers;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket 支持
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
EOF

echo "配置文件创建完成！"
echo ""
echo "使用说明："
echo "1. 修改 server1.yaml, server2.yaml, server3.yaml 中的 IP 地址"
echo "2. 启动数据库: docker-compose up -d db"
echo "3. 启动所有Server: docker-compose up -d server1 server2 server3"
echo "4. 查看状态: curl http://localhost:30080/api/v1/coordinator/servers"
