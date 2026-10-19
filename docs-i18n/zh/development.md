---
translation_status: translated
---

# 开发指南 { #development-guide }

## 前置条件 { #prerequisites }

1. 安装 Python（版本 3.10 至 3.12）。
2. 运行数据库（PostgreSQL 或 MySQL）。

例如，使用 Docker 运行 PostgreSQL 数据库：
```bash
docker run --name my-postgres -e POSTGRES_PASSWORD=mysecretpassword -p 5432:5432 -d --restart=always postgres
```

## 配置环境 { #set-up-environment }
```bash
make install
```

## 运行 { #run }

开发时使用 `disabled` 网关模式，并连接你的数据库：
```bash
uv run gpustack start --database-url postgresql://postgres:mysecretpassword@localhost:5432/postgres --gateway-mode disabled --api-port 80
```

## 构建 { #build }
```bash
make build
```

在 `dist` 中查看构建产物。

## 测试 { #test }
```bash
make test
```

## 打包 { #package }

构建容器镜像需要安装带有 [Buildx](https://docs.docker.com/build/install-buildx/) 插件的 Docker：
```bash
make package
```

如果 `docker buildx version` 执行失败，请使用包管理器安装插件（`apt-get install docker-buildx-plugin` 或 `yum install docker-buildx-plugin`），或将发布的二进制文件放到指定位置：
```bash
mkdir -p /usr/local/lib/docker/cli-plugins
VER=$(curl -fsSL https://api.github.com/repos/docker/buildx/releases/latest | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p')
ARCH=$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')
curl -fsSL "https://github.com/docker/buildx/releases/download/${VER}/buildx-${VER}.linux-${ARCH}" \
  -o /usr/local/lib/docker/cli-plugins/docker-buildx
chmod +x /usr/local/lib/docker/cli-plugins/docker-buildx
docker buildx version
```

未经身份认证的 GitHub API 对每个 IP 的限制为每小时 60 次请求。如果触发限流，`VER` 会为空，下载会返回 404；请从[发布页面](https://github.com/docker/buildx/releases)选择版本并手动设置。

使用 `sudo` 打包时，应将插件安装到 `/usr/local/lib/docker/cli-plugins`，而不是 `~/.docker/cli-plugins`，否则 `HOME` 的变化会导致 Docker CLI 无法找到插件。

## 更新依赖 { #update-dependencies }
```bash
uv add <something>
```

或
```bash
uv add --dev <something>
```

用于开发和测试依赖。
