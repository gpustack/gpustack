---
translation_status: translated
---

## 聊天 Playground { #chat-playground }

与聊天补全 API 交互。以下是示例截图：

![Playground 截图](../../assets/quick-start/quick-chat.png)

### 提示词 { #prompts }

可以在 Playground 左侧调整提示词消息。消息包含三种角色：system、user 和 assistant。

- **System**：通常是预定义的指令或指导，用于设置上下文、定义行为，或对模型生成响应的方式施加特定约束。
- **User**：由用户（与 LLM 交互的人）提供的输入或问题。
- **Assistant**：LLM 生成的响应。

#### 编辑系统消息 { #edit-system-message }

可以在 Playground 顶部添加和编辑系统消息。

#### 编辑用户和助手消息 { #edit-user-and-assistant-messages }

要添加用户或助手消息，请单击 `Add` 按钮。

要移除用户或助手消息，请单击消息右上角的减号按钮。

要更改消息角色，请单击消息开头的 `User` 或 `Assistant` 文本。

### 上传图片 { #upload-image }

单击 `Upload Image` 按钮，将图片添加到提示词中。

### 上传音频 { #upload-audio }

单击 `Upload Audio` 按钮，将音频文件添加到提示词中。

### 清空提示词 { #clear-prompts }

单击 `Clear` 按钮清空所有提示词。

### 选择模型 { #select-model }

单击 Playground 右上角的模型下拉菜单，选择 GPUStack 中可用的模型。模型部署方法参见[模型部署管理](../model-deployment-management.md)。

### 自定义参数 { #customize-parameters }

可以在 `Parameters` 部分自定义补全参数。

### 执行补全 { #do-completion }

单击 `Submit` 按钮或按 Enter 键执行补全。

### 查看代码 { #view-code }

尝试不同提示词和参数后，单击 `View Code` 按钮，查看如何通过代码使用相同输入调用 API。提供 `curl`、`Python` 和 `Node.js` 代码示例。

## 对比 Playground { #compare-playground }

可以在 Playground 中比较多个模型。以下是示例截图：

![对比 Playground 截图](../../assets/playground/compare-playground.png)

### 对比模式 { #comparison-mode }

单击对比视图按钮，选择参与对比的模型数量，支持 2、3、4 和 6 个模型。

### 提示词 { #prompts_1 }

可以按照与聊天 Playground 相同的方式调整提示词消息。

### 上传图片 { #upload-image_1 }

单击 `Upload Image` 按钮，将图片添加到提示词中。

### 上传音频 { #upload-audio_1 }

单击 `Upload Audio` 按钮，将音频文件添加到提示词中。

### 清空提示词 { #clear-prompts_1 }

单击 `Clear` 按钮清空所有提示词。

### 选择模型 { #select-model_1 }

单击各模型面板左上角的模型下拉菜单，选择 GPUStack 中可用的模型。

### 自定义参数 { #customize-parameters_1 }

单击各模型的设置按钮，自定义补全参数。
