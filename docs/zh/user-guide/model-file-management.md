# 模型文件管理

GPUStack 允许管理员下载和管理模型文件。

## 添加模型文件

GPUStack 目前支持来自 [Hugging Face](https://huggingface.co)、[ModelScope](https://modelscope.cn) 以及本地路径的模型。要添加模型文件，请前往 `Model Files` 页面。

### 添加 Hugging Face 模型

1. 点击 `Add Model File` 按钮，并在下拉菜单中选择 `Hugging Face`。
2. 使用左上角的搜索栏按名称查找模型，例如 `Qwen/Qwen2.5-0.5B-Instruct`。若只搜索 GGUF 模型，勾选 `GGUF` 复选框。
3. _(可选)_ 对于 GGUF 模型，可在 `Available Files` 中选择所需的量化格式。
4. 选择目标工作节点以下载模型文件。
5. _(可选)_ 指定 `Local Directory`，将模型下载到自定义路径，而非 GPUStack 缓存目录。
6. 点击 `Save` 按钮。

### 添加 ModelScope 模型

1. 点击 `Add Model File` 按钮，并在下拉菜单中选择 `ModelScope`。
2. 使用左上角的搜索栏按名称查找模型，例如 `Qwen/Qwen2.5-0.5B-Instruct`。若只搜索 GGUF 模型，勾选 `GGUF` 复选框。
3. _(可选)_ 对于 GGUF 模型，可在 `Available Files` 中选择所需的量化格式。
4. 选择目标工作节点以下载模型文件。
5. _(可选)_ 指定 `Local Directory`，将模型下载到自定义路径，而非 GPUStack 缓存目录。
6. 点击 `Save` 按钮。

### 添加本地路径模型

你可以从本地路径添加模型。该路径可以是目录（例如 Hugging Face 模型文件夹）或文件（例如 GGUF 模型），且需要位于目标工作节点上。

1. 点击 `Add Model File` 按钮，并在下拉菜单中选择 `Local Path`。
2. 输入 `Model Path`。
3. 选择目标工作节点。
4. 点击 `Save` 按钮。

## 重试下载

如果模型文件下载失败，你可以重试下载：

1. 前往 `Model Files` 页面。
2. 找到状态为错误的模型文件。
3. 点击 `Operations` 列中的省略号按钮，选择 `Retry Download`。
4. GPUStack 将尝试从指定来源再次下载该模型文件。

## 部署模型

可从模型文件部署模型。由于模型存储在特定工作节点上，GPUStack 会使用 `worker-name` 键添加工作节点选择器，以确保正确调度。

1. 前往 `Model Files` 页面。
2. 找到想要部署的模型文件。
3. 点击 `Operations` 列中的 `Deploy` 按钮。
4. 查看或调整 `Name`、`Replicas` 及其他部署参数。
5. 点击 `Save` 按钮。

## 删除模型文件

1. 前往 `Model Files` 页面。
2. 找到要删除的模型文件。
3. 点击 `Operations` 列中的省略号按钮，选择 `Delete`。
4. _(可选)_ 勾选 `Also delete the file from disk` 选项。
5. 点击 `Delete` 按钮确认。