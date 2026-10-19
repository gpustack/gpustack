---
translation_status: translated
---

# 使用图像生成模型 { #using-image-generation-models }

GPUStack 支持部署和运行**图像生成模型**。本指南介绍如何部署 **Z-Image-Turbo** 文生图模型，并在 GPUStack 界面中通过文本提示词生成图片。

## 前提条件 { #prerequisites }

开始前，请确认：

- 至少有一张显存不低于 24 GB 的 GPU，例如 NVIDIA RTX 4090。
- GPUStack 已安装并运行；如未完成，请参考[快速入门](../quickstart.md)。
- 可以访问 Hugging Face 或 ModelScope，以下载模型文件。

## 步骤 1：部署 Z-Image-Turbo 模型 { #step-1-deploy-the-z-image-turbo-model }

按以下步骤部署模型：

1. 在 GPUStack 界面中打开 `Model Catalog` 页面。
2. 在分类筛选中选择 `Image`。
3. 点击 `Z-Image-Turbo` 模型卡片，进入部署页面。
4. 保持默认设置，点击 `Save` 部署。

![部署模型](../assets/using-models/using-image-generation-models/deploy-model.png)

部署后，可在 `Deployments` 页面查看状态。部署成功后，点击部署项的省略号图标，选择 `Open in Playground`，即可在 Playground 中使用模型。

![模型列表](../assets/using-models/using-image-generation-models/model-list.png)

## 步骤 2：使用模型生成图像 { #step-2-use-the-model-for-image-generation }

在 `Image` Playground 中：

1. 在 `Size` 下拉菜单中选择所需图片分辨率。
2. 输入描述目标图片的提示词。可点击 `Random Prompt` 获取随机提示词作为参考。
3. 点击 `Submit` 生成图片。

生成的图片会显示在界面中。结果会随随机性、种子和提示词细节而变化。

![生成结果](../assets/using-models/using-image-generation-models/image-playground.png)

## 总结 { #conclusion }

完成上述配置后，即可通过 Z-Image-Turbo 将文本提示词转换为图片。尝试不同提示词和设置，可以生成丰富多样的图像。
