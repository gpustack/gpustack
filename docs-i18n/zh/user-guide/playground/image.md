---
translation_status: translated
---

# 图像 Playground { #image-playground }

图像 Playground 是用于测试和探索 GPUStack 图像生成 API 的专用空间。用户可以交互式探索不同模型的能力，自定义参数，并查看便于集成 API 的代码示例。

## 生成图像 { #generate-image }

### 提示词 { #prompt }

可以输入或随机生成提示词，然后单击 `Submit` 按钮生成图像。

![图像生成示例](../../assets/playground/create-image.png)

## 编辑图像 { #edit-image }

上传图像，并通过涂抹标出要修改的区域。然后输入提示词并单击 `Submit`。如果没有涂抹任何区域，将修改整张图像。

![图像编辑示例](../../assets/playground/image-edit.png)

### 保存遮罩 { #save-mask }

单击 `Save Mask`，将涂抹区域保存为独立图像。

### 下载图像 { #download-image }

单击 `Download Image` 保存编辑后的图像。

## 选择模型 { #select-model }

单击 Playground 界面右上角的模型下拉菜单，选择 GPUStack 中可用的模型。

## 参数 { #parameters }

| 参数              | 默认值     | 描述 |
| ----------------- | ---------- | --- |
| `Size`            | `512x512`  | 生成图像的尺寸，格式为 'widthxheight'。 |
| `Seed`            | （空）     | 随机种子。 |

## 查看代码 { #view-code }

尝试不同提示词和参数后，单击 `View Code` 按钮，查看如何使用相同输入调用 API。提供 `curl`、`Python` 和 `Node.js` 代码示例。
