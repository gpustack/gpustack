# 聊天演练场

与聊天补全 API 交互。下面是一个示例截图：

![演练场截图](../../assets/playground/chat.png)

## 提示

你可以在演练场左侧调整提示消息。提示消息有三种角色类型：system、user 和 assistant。

- **System**: 通常是预先定义的指令或指导，用于设定上下文、定义行为，或对模型生成响应的方式施加特定约束。
- **User**: 由用户（与 LLM 交互的人）提供的输入或问题。
- **Assistant**: 由 LLM 生成的响应。

## 编辑 System 消息

你可以在演练场顶部添加并编辑 system 消息。

## 编辑 User 和 Assistant 消息

要添加 user 或 assistant 消息，点击 `New Message` 按钮。

要移除 user 或 assistant 消息，点击消息右侧的减号按钮。

要更改消息的角色，点击消息开头的 `User` 或 `Assistant` 文本。

## 上传图片

点击 `Upload Image` 按钮可将图片添加到提示。

## 清空提示

点击 `Clear` 按钮可清空所有提示。

## 选择模型

你可以点击演练场右上角的模型下拉框选择 GPUStack 中可用的模型。请参阅[模型管理](../model-deployment-management.md)了解如何管理模型。

## 自定义参数

你可以在 `Parameters` 区域自定义补全参数。

## 执行补全

点击 `Submit` 按钮即可进行补全。

## 查看代码

当你用提示和参数试验完成后，可以点击 `View Code` 按钮查看如何用相同输入在代码中调用 API。提供了 `curl`、`Python` 和 `Node.js` 的代码示例。

# 对比演练场

你可以在演练场中对比多个模型。下面是一个示例截图：

![对比演练场截图](../../assets/compare-playground-screenshot.png)

## 比较模式

你可以通过点击比较视图按钮选择要对比的模型数量，支持 2、3、4 和 6 个模型的对比。

## 提示

你可以像在聊天演练场中那样调整提示消息。

## 上传图片

点击 `Upload Image` 按钮可将图片添加到提示。

## 清空提示

点击 `Clear` 按钮可清空所有提示。

## 选择模型

你可以点击每个模型面板左上角的模型下拉框选择 GPUStack 中可用的模型。

## 自定义参数

你可以点击每个模型的设置按钮自定义补全参数。