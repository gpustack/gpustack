---
translation_status: translated
---

# 使用视觉语言模型 { #using-vision-language-models }

**视觉语言模型（VLM）**可以同时处理视觉（图像）和语言（文本）数据，适用于图像描述、视觉问答等应用。本指南介绍如何在 GPUStack 中部署和使用 VLM。

这些模型的部署和调用流程相近，主要区别在于部署时需要设置的参数。参数说明见[后端参数](../user-guide/built-in-inference-backends.md#parameters-reference_1)。

本指南涵盖以下模型：

- **Qwen3-VL**
- **Llama3.2-Vision**
- **Pixtral**
- **Phi3.5-Vision**

## 前提条件 { #prerequisites }

开始前，请确认：

- 有一台配备一张或多张 GPU 的 Linux 机器，总显存至少 30 GB。本指南使用仅支持 Linux 的 vLLM 后端。
- 可以访问 Hugging Face，并持有 [Hugging Face API key](https://huggingface.co/settings/tokens)，以下载模型文件。
- 已取得上述模型在 Hugging Face 上的访问权限。[Llama3.2-VL](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) 和 [Pixtral](https://huggingface.co/mistralai/Pixtral-12B-2409) 是受限模型，需要先申请访问权限。

!!! note

    本指南使用配备一张 H100（80GB）GPU 的 Ubuntu 节点。

## 步骤 1：安装 GPUStack { #step-1-install-gpustack }

请按照[安装文档](../installation/requirements.md)安装 GPUStack。

## 步骤 2：登录 GPUStack 界面 { #step-2-log-in-to-gpustack-ui }

GPUStack Server 启动后，运行以下命令获取默认管理员密码：

```bash
docker exec gpustack cat /var/lib/gpustack/initial_admin_password
```

在浏览器中访问 `http://your_host_ip`，使用默认用户名 `admin` 和上面获取的密码登录 GPUStack 界面。

## 步骤 3：使用 vLLM 部署视觉语言模型 { #step-3-deploy-vision-language-models-with-vllm }

### 部署 Qwen3-VL { #deploy-qwen3-vl }

1. 在 GPUStack 界面中打开 `Deployments` 页面。
2. 点击 `Deploy Model`，在下拉菜单中选择 `Hugging Face`。
3. 在搜索栏中搜索 `Qwen/Qwen3-VL-4B-Instruct`。
4. 点击 `Save`。只要 GPU 资源充足，默认配置通常即可运行。

### 部署 Llama3.2-Vision { #deploy-llama32-vision }

1. 在 GPUStack 界面中打开 `Deployments` 页面。
2. 点击 `Deploy Model`，在下拉菜单中选择 `Hugging Face`。
3. 在搜索栏中搜索 `meta-llama/Llama-3.2-11B-Vision-Instruct`。
4. 展开配置中的 `Advanced`，滚动到 `Backend Parameters`。
5. 多次点击 `Add Parameter`，添加以下参数：

- `--enforce-eager`
- `--max-num-seqs=16`
- `--max-model-len=8192`

6. 点击 `Save`。

### 部署 Pixtral { #deploy-pixtral }

1. 在 GPUStack 界面中打开 `Deployments` 页面。
2. 点击 `Deploy Model`，在下拉菜单中选择 `Hugging Face`。
3. 在搜索栏中搜索 `mistralai/Pixtral-12B-2409`。
4. 展开配置中的 `Advanced`，滚动到 `Backend Parameters`。
5. 多次点击 `Add Parameter`，添加以下参数：

- `--tokenizer-mode=mistral`
- `--limit-mm-per-prompt=image=4`

6. 点击 `Save`。

### 部署 Phi3.5-Vision { #deploy-phi35-vision }

1. 在 GPUStack 界面中打开 `Deployments` 页面。
2. 点击 `Deploy Model`，在下拉菜单中选择 `Hugging Face`。
3. 在搜索栏中搜索 `microsoft/Phi-3.5-vision-instruct`。
4. 展开配置中的 `Advanced`，滚动到 `Backend Parameters`。
5. 点击 `Add Parameter`，添加以下参数：

- `--trust-remote-code`

6. 点击 `Save`。

## 步骤 4：使用视觉语言模型 { #step-4-interact-with-vision-language-models }

1. 在 GPUStack 界面中打开 `Chat` 页面。
2. 在右上角下拉菜单中选择已部署的模型。
3. 点击文本输入框上方的 `Upload Image` 并上传图片。
4. 输入提示词，例如“Describe the image.”。
5. 点击 `Submit` 生成结果。

![在 Playground 中使用视觉语言模型](../assets/using-models/using-vision-language-models/playground-vl.png)
