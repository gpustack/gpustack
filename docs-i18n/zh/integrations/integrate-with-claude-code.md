---
translation_status: translated
---

# 与 Claude Code 集成 { #integrate-with-claude-code }

Claude Code 是 Anthropic 提供的智能体编程工具。GPUStack 上的模型部署兼容 Anthropic API，因此可以将 Claude Code 连接到 GPUStack，用于代码生成。本指南通过让 Claude Code 创建一个 Flappy Bird 游戏，介绍接入步骤并验证集成结果。

## 前提条件 { #prerequisites }

- 一张或多张 GPU，总显存至少 100 GB。
- GPUStack 已安装并运行。
- 可以访问 Hugging Face 或 ModelScope，以下载模型文件。

!!! note

    本指南使用一张 NVIDIA H200 GPU，以 FP8 量化运行 `Qwen3-Coder-Next` 模型。

## 部署模型 { #deploy-the-model }

1. 在 GPUStack 界面中打开 **Model Catalog** 页面。
2. 搜索 `Qwen3-Coder-Next`，使用默认配置部署。

   ![模型目录](../assets/integrations/claude-code/catalog.png)

3. 部署完成后，即可通过 Anthropic 兼容 API 进行推理。

   ![模型已就绪](../assets/integrations/claude-code/model-ready.png)

## 创建 API 密钥 { #create-an-api-key }

1. 打开 **Access Control** > **API Keys** 页面，点击 **Add API Key**。
2. 输入名称，点击 **Save**。
3. 复制 API 密钥并保存，供后续使用。

   ![API 密钥](../assets/integrations/claude-code/api-key.png)

## 安装 Claude Code { #install-claude-code }

按照 [Claude Code 文档](https://github.com/anthropics/claude-code)在系统中安装 Claude Code。

## 安装 CC-Switch（可选） { #install-cc-switch-optional }

可使用 CC-Switch 或类似工具，在不同模型提供商之间切换。此步骤可选，也可以手动配置 Claude Code 或使用其他工具。

按照 [CC-Switch 文档](https://github.com/farion1231/cc-switch)完成安装。

## 为 Claude Code 配置 GPUStack { #configure-claude-code-with-gpustack }

1. 打开 CC-Switch，添加自定义提供商，填写：

   - **Provider Name**：`GPUStack`
   - **API Endpoint**：GPUStack Server 的 URL
   - **API Key**：前面创建的 API 密钥

2. 将所有模型配置为使用 `qwen3-coder-next`。

   ![CC-Switch 配置](../assets/integrations/claude-code/cc-switch.png)

## 测试集成 { #test-the-integration }

1. 创建新目录并启动 Claude Code：

   ```bash
   mkdir flappybird && cd flappybird
   claude
   ```

2. 使用 `/status` 命令验证配置。
3. 尝试一个示例任务，例如让 Claude Code 创建 Flappy Bird 游戏：

   ```
   Write a Flappy Bird game.
   ```

   ![Claude Code 生成结果](../assets/integrations/claude-code/claude.png)

4. Claude Code 完成后，在浏览器中打开生成的 HTML 文件，测试运行结果。

   ![Flappy Bird 游戏](../assets/integrations/claude-code/flappybird.gif)

## 总结 { #conclusion }

本指南介绍了将 Claude Code 接入 GPUStack，并使用它生成 Flappy Bird 游戏的完整流程。可以继续尝试更复杂的编程任务，并通过 GPUStack 提供模型推理服务。
