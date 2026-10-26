---
translation_status: translated
---

# 推理后端管理 { #inference-backend-management }

GPUStack 允许管理员配置推理后端及后端版本。

本文是 Inference Backend 页面的操作指南。内置后端及其支持的能力，参见[内置推理后端](built-in-inference-backends.md)。

自定义后端的配置指导和经过验证可用的示例，参见[自定义推理后端](../tutorials/using-custom-backends.md)。

## 后端来源 { #backend-sources }

GPUStack 支持三类推理后端：

- **Built-in**：GPUStack 维护的预配置后端（例如 vLLM、MindIE、VoxBox），不可删除。
- **Community**：社区后端市场共享的后端，可按需启用。
- **Custom**：通过自己的配置创建的后端，可自由添加、编辑和删除。

内置和社区列表可通过 OTA 更新，无需升级 GPUStack。也可以将任一列表指向自己维护的文档。参见 [OTA 内容更新](ota-content-updates.md)。

## 参数说明 { #parameter-description }

| 参数名称 | 描述 | 必填 |
| --- | --- | --- |
| Name | 推理后端名称 | 是 |
| Health Check Path | 用于确认后端已启动且能响应的健康检查路径。默认值：/v1/models（兼容 OpenAI）。 | 否 |
| Default Execution Command | 容器启动命令/参数。例如（vLLM）：`vllm serve {{model_path}} --port {{port}} --served-model-name {{model_name}} --host {{worker_ip}}`。部署调度到 Worker 时，会自动替换 `{{model_path}}`、`{{model_name}}`、`{{port}}`、`{{worker_ip}}`、`{{gpu_count}}`、`{{gpu_ids}}` 和 `{{VAR_NAME}}`（用于环境变量）占位符；替换后，参数按 POSIX 风格拆分。`{{gpu_count}}` 和 `{{gpu_ids}}` 表示单个已调度 Worker 上分配的 GPU（自定义后端不支持跨 Worker 推理）。包含空格的值请加引号，避免使用 shell 运算符。 | 否 |
| Default Entrypoint | 容器入口点覆盖值。设置后，会替换此后端的镜像入口点。参数按 POSIX 风格拆分。 | 否 |
| Default Environment Variables | 为此后端所有版本设置的环境变量，可通过 `{{VAR_NAME}}` 语法在命令中引用。版本级环境变量优先。 | 否 |
| Default Backend Parameters | 部署时预填 Advanced Backend Parameters 部分；启动前可调整。 | 否 |
| Flag Format | 控制组装后端启动命令时，各个 `--key value` 参数对的呈现方式：`Space Separated (--key value)` 或 `Equal Sign (--key=value)`。留空时禁用规范化，按原样保留输入参数。多值选项始终以空格分隔。参见[参数格式](#flag-format)。 | 否 |
| Common Backend Parameters | 此后端的常用参数列表，部署时显示为 `Backend Parameters` 输入框的建议，便于快速输入。与 `Default Backend Parameters` 不同，这些参数仅供快速选择，打开部署表单时不会预填，不影响运行时行为。 | 否 |
| Description | 描述 | 否 |
| Version Configs | 配置此后端的可用版本 | 是 |
| Default Version | 部署时预选的版本。如果未选择版本，将使用此版本的镜像。 | 否 |

## 环境变量 { #environment-variables }

可以在两个层级定义环境变量：

- **后端级**（`Default Environment Variables`）：适用于此后端的所有版本。
- **版本级**（`Environment Variables`）：特定于某个版本，覆盖后端级变量。

可以通过 `{{VAR_NAME}}` 语法在命令中引用环境变量。

**示例：**

```yaml
backend_name: my-backend-custom
default_env:
  MODEL_CACHE_DIR: /cache
  LOG_LEVEL: info
version_configs:
  v1:
    image_name: my-image:v1
    custom_framework: cuda
    run_command: "serve {{model_path}} --cache {{MODEL_CACHE_DIR}} --log-level {{LOG_LEVEL}} --port {{port}}"
    env:
      MODEL_CACHE_DIR: /custom-cache # Overrides backend-level value
```

此示例中：

- `MODEL_CACHE_DIR` 在后端级设置为 `/cache`。
- 版本 `v1` 将它覆盖为 `/custom-cache`。
- 所有版本的 `LOG_LEVEL` 都保持为 `info`。
- 两个变量都通过 `{{VAR_NAME}}` 语法在命令中引用。

Version Configs 参数说明：

| 参数名称 | 描述 | 必填 |
| --- | --- | --- |
| Version | 部署时显示在 Backend Version 下拉菜单中的版本名称 | 是 |
| Image Name | 后端的容器镜像名称（例如 `ghcr.io/org/image:tag`） | 是 |
| Framework<br/>(custom_framework) | 后端框架（内部标识符：`custom_framework`）。部署和调度会根据支持的框架进行筛选。 | 是 |
| Environment Variables | 此版本专用的环境变量，覆盖后端级 `Default Environment Variables`。可通过 `{{VAR_NAME}}` 语法在 `Execution Command` 中引用。 | 否 |
| Entrypoint | 版本专用的容器入口点覆盖值。未设置时使用 `Default Entrypoint`。参数按 POSIX 风格拆分。 | 否 |
| Execution Command | 版本专用的启动命令。未设置时使用 Default Execution Command。解析和拆分规则与 `Default Execution Command` 相同。 | 否 |

## 参数格式 { #flag-format }

`Flag Format` 控制 GPUStack 组装后端启动命令时，各个 `--key value` 参数对的呈现方式。它在后端级配置，适用于使用此后端的所有模型部署，并在 Worker 组装容器启动命令时生效。

可用值：

- `Space Separated (--key value)`：以空格分隔，呈现为 `--key value`。
- `Equal Sign (--key=value)`：以 `=` 连接，呈现为 `--key=value`。
- 留空（默认）：不进行规范化，保留 `Backend Parameters` 中输入的原样参数。

如果推理引擎的命令行解析器只接受其中一种形式，可用此字段强制采用一致的参数风格，无需逐个手动改写参数。

**示例：**

将后端的 `Flag Format` 设为 `Equal Sign (--key=value)` 后，使用此后端的部署若具有以下 `Backend Parameters`：

```bash
--tensor-parallel-size 2 --max-model-len 8192 --lora-modules a=/path1 b=/path2
```

启动命令会被规范化为：

```bash
vllm serve Qwen/Qwen3.5-0.8B --tensor-parallel-size=2 --max-model-len=8192 --lora-modules a=/path1 b=/path2
```

`--tensor-parallel-size` 和 `--max-model-len` 转换为等号形式，`--lora-modules` 因包含多个值，仍保留空格分隔形式。

!!! note

    - **多值选项**（例如 `--lora-modules`）始终以空格分隔，不受所选格式影响。将其转换为 `--key=value` 形式会改变 argparse 语义，因为后续值通常会覆盖先前值。
    - **单短横线短选项**（例如 llama.cpp 的 `-n`、`-ngl` 和 `-m`）逐字保留，不会被强制改为 `--n` 等双短横线形式。

## 添加自定义推理后端 { #add-custom-inference-backend }

1. 单击右上角的 "Add Backend" 按钮。
2. 可以填写表单或粘贴 YAML 定义来添加自定义推理后端。字段含义参见上方参数说明。
3. 后端名称创建后不可修改。自定义后端名称必须以 "-custom" 结尾（表单已预填）。
4. 单击 "Save" 提交。

添加自定义推理后端有两种方式：

- 通过 UI 表单：进入 **Model Service > Inference Backends** 页面，单击 **Add Custom Inference Backend** 按钮。
- 通过 YAML 配置：导入包含后端配置的 YAML 文件。

### 启用社区推理后端 { #enable-community-inference-backend }

这些后端本质上是带有 "community" 来源标签的自定义后端，让你无需手动配置即可快速创建自定义后端。

1. 在 Inference Backend 页面，单击右上角的 "Add Backend" 按钮。
2. 选择 "Community"，浏览市场中的可用社区后端。
3. 找到要使用的后端，在卡片操作菜单中单击 "Enable"。
4. 启用后，该后端即可用于模型部署。要禁用它，请从 Inference Backend 页面删除此后端。

## 编辑推理后端或添加自定义版本 { #edit-inference-backend-or-add-custom-version }

1. 在 Inference Backend 页面找到目标后端，从卡片右上角的下拉菜单中选择 "Edit"。
2. 修改后端属性（名称不可更改），或添加新版本。
3. 对于内置后端，自定义版本必须以 "-custom" 结尾（表单已预填）。
4. 单击 "Save" 提交。

### 示例：为内置 vLLM 推理后端添加自定义版本 { #example-add-a-custom-version-to-the-built-in-vllm-inference-backend }

1. 在 Inference Backend 页面找到 vLLM 推理后端，从卡片右上角的下拉菜单中选择 "Edit"。
2. 在 Version Configs 部分单击 "Add Version"。
3. 按以下方式填写字段：

- Version：`0.19.0`
- Image Name：`vllm/vllm-openai:v0.19.0`
- Framework：`cuda`
- Override Image Entrypoint：`vllm serve`
- Execution Command：`{{model_path}} --host {{worker_ip}} --port {{port}} --served-model-name {{model_name}}`

4. 单击 "Save" 提交。

!!! note

    vLLM 自 v0.11.1 起更改了 Docker 镜像的入口点。因此，添加 vLLM v0.11.1 或更高版本的自定义版本时，必须指定 `Override Image Entrypoint` 和 `Execution Command` 字段，否则模型无法启动。如果使用较新版本的 `gpustack/runner` 镜像，则无需设置 `Execution Command` 字段。

    对于通过 `Override Image Entrypoint` 提供可执行程序的后端（例如 vLLM 和 SGLang），`Execution Command` 只能包含参数。不要在 `Execution Command` 中放入可执行程序（例如 `vllm serve` 或 `python -m vllm ...`），否则它会被追加到入口点后，产生类似 `vllm serve python -m vllm ...` 的重复前缀。

### 示例：为内置 SGLang 推理后端添加自定义版本 { #example-add-a-custom-version-to-the-built-in-sglang-inference-backend }

1. 在 Inference Backend 页面找到 SGLang 推理后端，从卡片右上角的下拉菜单中选择 "Edit"。
2. 在 Version Configs 部分单击 "Add Version"。
3. 按以下方式填写字段：

- Version：`0.5.10`
- Image Name：`lmsysorg/sglang:v0.5.10`
- Framework：`cuda`
- Override Image Entrypoint：`sglang serve`
- Execution Command：`--model-path {{model_path}} --host {{worker_ip}} --port {{port}}`

4. 单击 "Save" 提交。

## 删除自定义推理后端 { #delete-custom-inference-backend }

1. 在 Inference Backend 页面找到目标后端，从卡片右上角的下拉菜单中选择 "Delete"。
2. 内置后端不可删除。
3. 在确认对话框中单击 "Delete"。

## 列出推理后端版本 { #list-versions-of-inference-backend }

在 Inference Backend 页面单击后端卡片任意位置（操作按钮除外），打开弹窗，浏览所有内置版本和自定义添加的版本。

## 灵活测试部署 { #flexible-testing-deployment }

使用此模式可以快速验证或调整镜像和启动命令，无需编辑后端定义。

1. 进入 Deployments 页面，单击 "Deploy Model" 按钮，选择任意模型来源。
2. 在 Basic 选项卡中打开 "Backend" 下拉菜单，从 "Built-in" 部分选择 "Custom"。
3. 将出现两个字段：`image_name` 和 `run_command`。它们仅为本次部署覆盖后端配置。
4. 检查其余必填设置并提交部署。

## 自定义推理后端的限制 { #limitations-of-custom-inference-backends }

自定义推理后端不支持跨多个 Worker 的分布式推理。
