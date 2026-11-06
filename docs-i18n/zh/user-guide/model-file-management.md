---
translation_status: translated
---

# 模型文件管理 { #model-file-management }

GPUStack 允许管理员下载和管理模型文件。

## 添加模型文件 { #add-model-file }

GPUStack 目前支持来自 [Hugging Face](https://huggingface.co)、[ModelScope](https://modelscope.cn) 和本地路径的模型。要添加模型文件，请进入 `Model Files` 页面。

### 添加 Hugging Face 模型 { #add-a-hugging-face-model }

1. 单击 `Add Model File` 按钮，从下拉菜单中选择 `Hugging Face`。
2. 使用左上角的搜索栏按名称查找模型，例如 `Qwen/Qwen3-0.6B`。
3. _（可选）_ 对于 GGUF 模型，从 `Available Files` 中选择所需量化格式。
4. 选择用于下载模型文件的目标 Worker。
5. _（可选）_ 指定 `Local Directory`，将模型下载到自定义路径，而不是 GPUStack 缓存目录。
6. 单击 `Save` 按钮。

### 添加 ModelScope 模型 { #add-a-modelscope-model }

1. 单击 `Add Model File` 按钮，从下拉菜单中选择 `ModelScope`。
2. 使用左上角的搜索栏按名称查找模型，例如 `Qwen/Qwen3-0.6B`。
3. _（可选）_ 对于 GGUF 模型，从 `Available Files` 中选择所需量化格式。
4. 选择用于下载模型文件的目标 Worker。
5. _（可选）_ 指定 `Local Directory`，将模型下载到自定义路径，而不是 GPUStack 缓存目录。
6. 单击 `Save` 按钮。

### 添加本地路径模型 { #add-a-local-path-model }

可以从本地路径添加模型。该路径可以是 Worker 上的目录（例如 Hugging Face 模型文件夹）或文件（例如 GGUF 模型）。

1. 单击 `Add Model File` 按钮，从下拉菜单中选择 `Local Path`。
2. 输入 `Model Path`。
3. 选择目标 Worker。
4. 单击 `Save` 按钮。

## 重试下载 { #retry-download }

模型文件下载失败或下载速度长时间停留在很低水平时，可以重试：

1. 进入 `Model Files` 页面。
2. 找到模型文件。
3. 单击 `Operations` 列中的省略号按钮，选择 `Retry Download`。
4. GPUStack 会尝试从指定来源重新下载模型文件。

## 部署模型 { #deploy-model }

可以从模型文件部署模型。由于模型存储在指定 Worker 上，GPUStack 会添加使用 `worker-name` 键的 Worker 选择器，以确保正确调度。

!!! tip

    如果希望模型可以跨节点进行故障转移，请确保集群中的所有节点都能通过相同路径访问模型文件，并手动从 Worker 选择器中移除 `worker-name` 标签。

1. 进入 `Model Files` 页面。
2. 找到要部署的模型文件。
3. 单击 `Operations` 列中的 `Deploy` 按钮。
4. 检查或调整 `Name`、`Backend`、`Backend Version`、`Replicas` 及其他部署参数。
5. 单击 `Save` 按钮。

## 删除模型文件 { #delete-model-file }

1. 进入 `Model Files` 页面。
2. 找到要删除的模型文件。
3. 单击 `Operations` 列中的省略号按钮，选择 `Delete`。
4. _（可选）_ 勾选 `Also delete the file from disk` 选项。
5. 单击 `Delete` 按钮确认。
