---
translation_status: translated
---

# 从 v0.7 及更早版本迁移到 v2 { #migrating-from-v07-and-earlier-versions-to-v2 }

!!! note

    从 v2.0.0 起，GPUStack Worker 官方仅支持 Linux。如果使用 Windows 或 macOS，请将数据目录迁移到 Linux 系统后执行迁移。

    在 Windows 和 macOS 上，仍可通过 Docker Desktop 运行 GPUStack Server（不包含内置 Worker）。

## 迁移前准备 { #before-migration }

### 不兼容变更 { #breaking-changes }

#### 1. 移除 Ollama 模型来源（自 v0.7.x 起） { #1-removal-of-ollama-model-source-since-v07x }

- **变更：** 从 0.7 版本起，GPUStack 不再支持 `ollama` 模型来源。
- **影响：** 升级过程中不会保留来源为 `ollama` 的模型、模型文件和模型实例。
- **所需操作：** 如果从 v0.7 之前的版本升级，且当前部署了来自 `ollama` 的模型，必须在升级前手动迁移这些模型。\
  建议使用以下受支持的来源重新部署受影响的模型：
    - Hugging Face
    - ModelScope
    - 本地路径

    可在开始升级之前，通过 **GPUStack UI** 重新部署模型以完成迁移。

### 备份数据 { #backup-your-data }

!!! warning

      **先备份：** 强烈建议在开始 Server 迁移前备份数据库。

      对于 v0.7 及更早版本的默认安装，请停止 GPUStack Server，并备份容器内以下位置的数据目录：
      ```
      /var/lib/gpustack
      ```

开始迁移前，请先阅读[安装要求](./installation/requirements.md)。

如果在 v0.7.1 之前**未使用 Docker** 运行 GPUStack（例如通过 pip install 或安装脚本安装），请先按照 Docker Engine [安装指南](https://docs.docker.com/engine/install/)安装 Docker，再进行迁移。

如果在 v0.7.1 之前使用 GPU 加速推理，请在安装 Docker 后，检查是否需要安装对应加速器运行时的 Container Toolkit 或 Container Runtime。可按照**安装要求**中的步骤检查并安装。

## 迁移步骤 { #migration-steps }

### 确认旧数据目录 { #identify-your-legacy-data-directory }

找到原有 GPUStack 安装使用的数据目录。默认路径为：
```
/var/lib/gpustack
```

其他安装方式请参阅[相关说明](faq.md/#where-are-gpustacks-data-stored)以定位数据目录。以下步骤将该路径表示为 `${your-data-dir}`。

由于 v2.0.0 的架构变化，`postgres`、`s6-supervisor` 等新组件可能以非 root 用户读写 `${your-data-dir}` 中的文件。请确保所有所需用户都能访问 `${your-data-dir}`。可使用以下脚本调整权限，但运行前应了解其安全风险，并准确理解脚本执行的操作。
```bash
dir="${your-data-dir}"
chmod a+rx "${dir}/log" || true
chmod a+rx $dir
while [ "$dir" != "/" ]; do
  chmod a+x "$dir"
  dir=$(dirname "$dir")
done
```

### 使用 Docker 迁移 Server { #migrate-server-using-docker }

从 v2.0.0 起，无需再为 Server 或 Worker 镜像指定 GPU 计算平台或版本（例如 `-cpu`、`-cuda12.8` 或 `-rocm`），请使用统一镜像：`gpustack/gpustack:latest`。

#### 内置数据库迁移（SQLite → PostgreSQL） { #embedded-database-migration-sqlite-postgresql }

在 v0.7 及更早版本中，GPUStack 默认使用内置 SQLite 数据库存储管理数据。从 v2.0.0 起，GPUStack 不再支持 SQLite，默认改用内置 PostgreSQL，以提高性能和可扩展性。

启动 GPUStack 时设置 `GPUSTACK_DATA_MIGRATION=true`，以启用内置数据库迁移。将 `${your-data-dir}` 替换为包含原始 SQLite 数据库及相关文件的旧数据目录：
```bash
sudo docker run -d --name gpustack \
  --restart=unless-stopped \
  --privileged \
  --network=host \
  --env GPUSTACK_DATA_MIGRATION=true \
  --volume /var/run/docker.sock:/var/run/docker.sock \
  --volume ${your-data-dir}:/var/lib/gpustack \
  --runtime nvidia \
  gpustack/gpustack:latest
```

如果 Docker 报错 `docker: Error response from daemon: unknown or invalid runtime name: nvidia`，请先检查[加速器运行时要求](./installation/requirements.md#accelerator-runtime-requirements)，再参照[其他 GPU 架构](#other-gpu-architectures)，将 `--runtime nvidia` 参数替换为对应运行时。

数据库迁移时也支持通过 `--data-dir` 或 `GPUSTACK_DATA_DIR` 自定义数据目录。命令调整如下：
```diff
...
-  --volume ${your-data-dir}:/var/lib/gpustack
+  --volume ${your-data-dir}:${your-data-dir}
...
+   gpustack/gpustack:latest \
+   --data-dir ${your-data-dir}
```

#### 外部数据库迁移 { #external-database-migration }

GPUStack 支持使用外部数据库存储管理数据。如果此前使用外部数据库部署 GPUStack，请通过以下命令启动 Server：
```bash
sudo docker run -d --name gpustack-server \
  --restart=unless-stopped \
  --privileged \
  --network=host \
  --volume /var/run/docker.sock:/var/run/docker.sock \
  --volume ${your-data-dir}:/var/lib/gpustack \
  --runtime nvidia \
  gpustack/gpustack \
  --database-url ${your-database-url}
```

### 使用 Docker 迁移 Worker { #migrate-workers-using-docker }

对于 Worker 节点，请将 `${your-data-dir}` 替换为旧 Worker 数据目录路径，并使用以下命令：
```bash
sudo docker run -d --name gpustack-worker \
  --restart=unless-stopped \
  --privileged \
  --network=host \
  --volume ${your-data-dir}:/var/lib/gpustack \
  --volume /var/run/docker.sock:/var/run/docker.sock \
  --runtime nvidia \
  gpustack/gpustack \
  --server-url ${server-url} \
  --token ${token}
```

请确保 Docker 命令包含 `--volume /var/run/docker.sock:/var/run/docker.sock` 和 `--runtime nvidia`，旧版本无需这两个参数。对于不同加速器运行时，请参阅[其他 GPU 架构](#other-gpu-architectures)，使用对应选项替换 `--runtime nvidia`。

这会使用已有数据启动 GPUStack Worker，并连接到指定 Server。

### 其他 GPU 架构 { #other-gpu-architectures }

对于 NVIDIA 之外的架构（例如 AMD、昇腾），迁移流程相同。请通过[加速器运行时要求](./installation/requirements.md#accelerator-runtime-requirements)确认 GPU 架构受支持。

例如，在配备 AMD GPU 的宿主机上运行 Server 和 Worker 时，将 docker run 命令调整为：
```diff
  sudo docker run -d --name gpustack \
  ...
- --runtime nvidia \
+ --runtime amd \
  ...
```

### 重建模型实例 { #recreating-model-instances }

升级完成后，已有模型实例可能一直处于 `Starting` 状态。此时重建模型实例即可恢复正常运行。

### 从 llama-box 迁移 { #migration-from-llama-box }

如果此前使用 llama-box 推理后端，请注意自 v2.0.0 起不再支持 llama-box。请通过自定义推理后端使用 llama.cpp 替代。

1. 在推理后端页面创建 `llama.cpp` 自定义后端。llama.cpp 的配置请参阅[相关文档](./user-guide/inference-backend-management.md#enable-community-inference-backend)。
2. 进入部署页面，修改原本使用 llama-box 启动的模型，将后端改为 llama.cpp。
3. 保存后重建模型实例。

!!! Note

    自定义推理后端目前不支持跨多个 Worker 的分布式推理。

### 从自定义后端版本迁移 { #migration-from-custom-backend-versions }

如果在 v2.0.0 之前使用自定义后端版本，请注意这些版本依赖 Python 虚拟环境，而从 v2.0.0 起**不再支持**这种方式。所有推理后端现在都在容器环境中运行。

如需继续使用模型，必须通过以下一种方式**重新创建模型**：

**方案 1 - 使用内置后端版本：**

GPUStack v2.0.0+ 提供多个预配置的内置推理后端版本。重新创建模型部署，并选择最符合模型要求的内置后端版本。

**方案 2 - 向内置后端添加自定义版本：**

如果内置版本均无法满足需求，可通过添加自定义版本扩展内置推理后端。详细步骤请参阅：[向内置 vLLM 推理后端添加自定义版本](user-guide/inference-backend-management.md#example-add-a-custom-version-to-the-built-in-vllm-inference-backend)。
