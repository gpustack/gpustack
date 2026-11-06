---
translation_status: translated
---

# 使用自定义推理后端 { #using-custom-inference-backends }

本指南介绍如何在 GPUStack 中添加自定义推理后端，包括使用经过验证的社区配置和从头创建配置。

参数说明见[用户指南](../user-guide/inference-backend-management.md)。

## 后端类型 { #backend-types }

GPUStack 支持三类推理后端：

- **内置**：由 GPUStack 维护的预配置后端，例如 vLLM、MindIE、VoxBox、SGLang，会自动针对不同硬件优化。
- **社区**：经过预先验证的自定义后端配置，本质上是标记为“community”的 CustomBackends，用于简化手动配置。
- **自定义**：使用自定义 Docker 镜像和命令自行配置的后端。

## 使用社区后端 { #using-community-backends }

社区后端是添加常用推理引擎的快捷方式。

**步骤：**

1. 打开 Inference Backend 页面，点击“Add Backend”。
2. 选择“Community”。
3. 浏览“Community Backend Marketplace”，启用所需后端。

## 创建自定义后端 { #creating-custom-backends }

### 基本步骤 { #core-steps }

1. 准备所需推理后端的 Docker 镜像。
2. 理解镜像的 ENTRYPOINT 或 CMD，以确定启动命令。
3. 在 Inference Backend 页面添加配置。
4. 部署模型并选择新添加的后端。

### 示例：TensorRT-LLM { #example-tensorrt-llm }

下面以 TensorRT-LLM 为例，说明如何添加和使用推理后端。

> 以下示例用于演示功能，并非经过性能优化的配置。如需优化性能，请参考各后端官方文档进行调优。

1. 从 TensorRT-LLM 文档链接的[发布页面](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release)查找所需镜像。
2. TensorRT-LLM 镜像必须通过 `trtllm-serve` 启动推理服务，否则会启动交互式 shell。`run_command` 支持 `{{model_path}}` 和 `{{port}}` 等占位符，也可使用 `{{model_name}}`、`{{worker_ip}}`、`{{gpu_count}}`、`{{gpu_ids}}`。部署调度到 Worker 时，这些占位符会替换为实际值。`{{gpu_count}}` 和 `{{gpu_ids}}` 表示调度目标 Worker 上分配的 GPU；自定义后端在单个 Worker 上运行。
3. 在 Inference Backend 页面添加配置，支持导入 YAML。例如：

```yaml
backend_name: TensorRT-LLM-custom
default_version: 1.2.0rc0
version_configs:
  1.2.0rc0:
    image_name: nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc0
    run_command: 'trtllm-serve {{model_path}} --host 0.0.0.0 --port {{port}}'
    custom_framework: cuda
```

4. 在 Deployments 页面选择刚添加的后端并部署模型。

![选择自定义推理后端部署模型](../assets/tutorials/using-custom-backend/deploy-by-custom-backend.png)

**结果**

推理后端服务启动后，可以看到 model_instance 状态变为 RUNNING。

![自定义推理后端的模型实例正在运行](../assets/tutorials/using-custom-backend/custom-backend-running.png)

可以在 Playground 中与模型对话。

![在 Playground 中与使用自定义推理后端的模型对话](../assets/tutorials/using-custom-backend/use-custom-backend-in-playground.png)

## 高级配置 { #advanced-configuration }

### 使用环境变量 { #using-environment-variables }

环境变量可以灵活配置参数，避免在命令中硬编码值：

```yaml
backend_name: advanced-backend-custom
default_env:
  CACHE_DIR: /models/cache
  LOG_LEVEL: info
version_configs:
  v1:
    image_name: my-backend:v1
    custom_framework: cuda
    run_command: 'serve {{model_path}} --cache {{CACHE_DIR}} --log-level {{LOG_LEVEL}} --port {{port}}'
    env:
      LOG_LEVEL: debug  # Override for this version
```

此示例中：

- 在后端层级定义了 `CACHE_DIR` 和 `LOG_LEVEL`。
- 版本 `v1` 将 `LOG_LEVEL` 覆盖为 `debug`。
- 命令通过 `{{VAR_NAME}}` 语法引用这两个变量。

### 自定义 Entrypoint { #custom-entrypoint }

镜像需要自定义初始化时，可以覆盖容器默认 Entrypoint。后端和版本层级都可以配置 Entrypoint：

```yaml
backend_name: custom-entry-backend-custom
default_entrypoint: /usr/local/bin/default-init
version_configs:
  v1:
    image_name: my-backend:v1
    custom_framework: cuda
    run_command: 'serve {{model_path}} --port {{port}}'
  v2:
    image_name: my-backend:v2
    custom_framework: cuda
    entrypoint: /usr/local/bin/v2-init  # Version-specific entrypoint overrides default
    run_command: 'serve {{model_path}} --port {{port}}'
```
