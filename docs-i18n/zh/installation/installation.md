# 通过 Docker 安装 { #installation-via-docker }

## 前提条件 { #prerequisites }

**GPUStack Server：**

- 必须安装 [Docker](https://docs.docker.com/engine/install/)，也支持 Docker Desktop（Windows 和 macOS）。

**GPUStack Worker：**

- 必须安装 [Docker](https://docs.docker.com/engine/install/)，**不支持** Docker Desktop。
- GPUStack Worker 仅支持 Linux。如果使用 Windows，请考虑使用 WSL2，并避免使用 Docker Desktop。GPUStack Worker 不支持 macOS。
- 确保已安装适合硬件的 GPU 驱动和容器工具包。详细信息见[安装要求](./requirements.md)。

## 安装 GPUStack Server { #install-gpustack-server }

运行以下命令，使用 Docker 安装并启动 GPUStack Server：

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    -p 80:80 \
    --volume gpustack-data:/var/lib/gpustack \
    gpustack/gpustack
```

!!! note

    GPUStack v2 对所有 GPU 设备类型使用同一个统一容器镜像。

## 启动 { #startup }

查看 GPUStack 容器日志：

```bash
sudo docker logs -f gpustack
```

GPUStack Server 启动后，在浏览器中打开 `http://your_host_ip` 访问 GPUStack UI。

使用用户名 `admin` 和默认密码登录。通过以下命令获取初始密码：

```bash
sudo docker exec -it gpustack \
    cat /var/lib/gpustack/initial_admin_password
```

## 添加 GPU 集群和 Worker { #add-gpu-clusters-and-worker-nodes }

按照界面 `Clusters` 和 `Workers` 页面的指引添加 GPU 集群和 Worker。

## 自定义配置 { #custom-configuration }

以下内容介绍启动 GPUStack Server 容器时的自定义配置示例。完整可用选项见 [CLI 参考](../cli-reference/start.md)。

### 使用自定义证书启用 HTTPS { #enable-https-with-custom-certificate }


```diff
 sudo docker run -d --name gpustack \
     ...
     -p 80:80 \
+    -p 443:443 \
     --volume gpustack-data:/var/lib/gpustack \
+    --volume /path/to/cert_files:/path/to/cert_files:ro \
+    -e GPUSTACK_SSL_KEYFILE=/path/to/cert_files/your_domain.key \
+    -e GPUSTACK_SSL_CERTFILE=/path/to/cert_files/your_domain.crt \
     gpustack/gpustack
     ...
```

`GPUSTACK_SSL_CERTFILE` 应先包含服务器证书，再包含所有中间证书。如果服务器证书由私有 CA 签发，还应添加 PEM 格式的 CA 证书包，使 Worker 和基准测试容器能够自动验证服务器：

```diff
 sudo docker run -d --name gpustack \
     ...
+    -e GPUSTACK_SSL_CA_CERTFILE=/path/to/cert_files/ca-bundle.crt \
     gpustack/gpustack
     ...
```

对于自签名服务器证书，如果未单独配置 CA 证书包，则使用证书文件本身。由公共 CA 签发的证书无需额外配置 CA。

注册命令包含引导 CA 证书包的 SHA-256 校验和。Worker 在安装未经 TLS 验证下载的证书包前，会检查此校验和。这不是证书固定：如果服务器已通过常规 TLS 验证，Worker 会使用现有信任库，不会下载或检查引导证书包。

Worker 首先尝试将验证过的 CA 安装到系统信任库。如果无法更新该信任库，或者 Worker 的 TLS 客户端使用其他证书包，Worker 会通过 `SSL_CERT_FILE` 为自身及其子进程使用临时合并的证书包。现有的运维人员提供的证书会保留；替换注入的引导 CA 时，不会保留先前注入的证书包。

如果 Worker 始终无法验证服务器证书，最后可以在启动该 Worker 时使用 `--insecure-tls`（或 `GPUSTACK_INSECURE_TLS=true`），跳过与服务器连接时的证书验证。流量仍会加密，但不再受到防截获保护，因此仅应在可信网络中使用。参见 [CLI 参考](../cli-reference/start.md)。

### 使用外部数据库 { #using-an-external-database }

默认情况下，GPUStack 使用内嵌 PostgreSQL 数据库。如需使用 PostgreSQL 或 MySQL 等外部数据库，请在启动 GPUStack 容器时设置 `GPUSTACK_DATABASE_URL` 环境变量，或使用 `--database-url` 参数。兼容数据库和已验证版本列表见[数据库要求](requirements.md#database-requirements)。

```diff
 sudo docker run -d --name gpustack \
     ...
     --volume gpustack-data:/var/lib/gpustack \
+    -e GPUSTACK_DATABASE_URL="postgresql://username:password@host:port/dbname" \
     gpustack/gpustack
     ...
```

### 配置外部 GPUStack Server URL { #configure-external-server-url }

如果通过云提供方创建 Worker，请设置用于 Worker 注册的外部 GPUStack Server URL，确保 Worker 能够正确连接 GPUStack Server。

```diff
sudo docker run -d --name gpustack \
    ...
+   -e GPUSTACK_SERVER_EXTERNAL_URL="https://your_external_server_url" \
    gpustack/gpustack
    ...
```

### 添加受信任的 CA { #additional-trusted-cas }

如果 GPUStack 需要访问使用私有或企业 CA 签发证书的服务，例如自托管身份提供方、Hugging Face 镜像或内部 API 端点，请将 CA 证书挂载到容器的 `/usr/local/share/ca-certificates/` 目录。GPUStack 会在启动时自动导入挂载的 CA 证书，并将其添加到系统信任库。

```diff
 sudo docker run -d --name gpustack \
     ...
     --volume gpustack-data:/var/lib/gpustack \
+    --volume /path/to/custom-root-ca.crt:/usr/local/share/ca-certificates/custom-root-ca.crt:ro \
     gpustack/gpustack
     ...
```

!!! note

    CA 证书必须采用 PEM 编码，并使用 `.crt` 扩展名。可以添加多个 `--volume` 参数以挂载多个 CA 证书。

## 通过 Docker Compose 安装 { #installation-via-docker-compose }

### 前提条件 { #prerequisites_1 }

- 必须安装 [Docker Compose](https://docs.docker.com/compose/install/)。
- [所需端口](./requirements.md#port-requirements)必须可用。

### 部署 { #deployment }

Docker Compose 文件和配置文件维护在 [GPUStack 仓库](https://github.com/gpustack/gpustack/tree/main/docker-compose)中。

运行以下命令，克隆最新稳定版本：

```bash
LATEST_TAG=$(
    curl -s "https://api.github.com/repos/gpustack/gpustack/releases" \
    | grep '"tag_name"' \
    | sed -E 's/.*"tag_name": "([^"]+)".*/\1/' \
    | grep -Ev 'rc|beta|alpha|preview' \
    | head -1
)
echo "Latest stable release: $LATEST_TAG"
git clone -b "$LATEST_TAG" https://github.com/gpustack/gpustack.git
cd gpustack/docker-compose
```

启动 GPUStack Server：

```bash
sudo docker compose -f docker-compose.server.yaml up -d
```

GPUStack Server 启动后，在浏览器中打开 `http://your_host_ip` 访问 GPUStack UI。

使用用户名 `admin` 和默认密码登录。通过以下命令获取初始密码：

```bash
sudo docker exec -it gpustack-server cat /var/lib/gpustack/initial_admin_password
```

内置和外部可观测性选项见[可观测性](../user-guide/observability.md)。
