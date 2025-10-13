# 图像 Playground

图像 Playground 是用于测试和试验 GPUStack 图像生成 API 的专用空间。它允许用户以交互方式探索不同模型的能力、自定义参数，并查看代码示例以无缝集成 API。

## 生成图像

## 提示词

您可以输入或随机生成提示词，然后点击 Submit 按钮来生成图像。

![这里是一张图片](../../assets/playground/create-image-01.png)

## 清空提示词

点击 `Clear` 按钮以重置提示词并移除生成的图像。

## 预览

将 `Preview` 参数设置为 `Faster` 选项，可实时查看图像生成进度。

![这里是一张图片](../../assets/playground/create-image-02.png)

## 编辑图像

上传一张图像，并通过涂抹的方式高亮需要修改的区域。然后输入提示词并 `submit`。如果未涂抹任何区域，则会修改整张图像。

![这里是一张图片](../../assets/playground/image-edit-01.png)

## 保存遮罩

点击 `Save Mask` 将涂抹区域另存为一张独立图像。

## 下载图像

点击 `Download Image` 保存已编辑的图像。


## 预览

在编辑图像时，您可以启用 `Preview` 以实时查看变化。


## 选择模型

您可以在 Playground 界面右上角点击模型下拉菜单，选择 GPUStack 中可用的模型。

## 自定义参数

您可以通过在两种 API 样式间切换来自定义图像生成参数：

1. **OpenAI 兼容模式**。
2. **高级模式**。

![图像参数](../../assets/playground/api-style.png)

### 高级参数



| 参数               | 默认值     | 说明                                                                                                                                                              |
| ------------------ | ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Counts`           | `1`        | 要生成的图像数量。                                                                                                                                                 |
| `Size`             | `512x512`  | 生成图像的尺寸，格式为“宽x高”。                                                                                                                                    |
| `Sample Method`    | `euler_a`  | 图像生成的采样器算法。可选值包括 'euler_a'、'euler'、'heun'、'dpm2'、'dpm++2s_a'、'dpm++2m'、'dpm++2mv2'、'ipndm'、'ipndm_v' 和 'lcm'。                           |
| `Schedule Method`  | `discrete` | 噪声调度方法。                                                                                                                                                     |
| `Sampling Steps`   | `10`       | 执行的采样步数。更高的数值可能提升图像质量，但会增加处理时间。                                                                                                      |
| `Guidance`         | `3.5`      | 无分类器引导（classifier-free guidance）的强度。值越高，对提示词的遵从度越高。                                                                                     |
| `CFG Scale`        | `4.5`      | 无分类器引导（classifier-free guidance）的强度。值越高，对提示词的遵从度越高。                                                                                     |
| `Negative Prompt`  | (empty)    | 用于指定图像中应避免出现的内容的负面提示词。                                                                                                                       |
| `Preview`          | `Faster`   | 控制图像生成过程的显示方式。可选值包括 'Faster'、'Normal'、'None'。                                                                                                |
| `Seed`             | (empty)    | 随机种子。                                                                                                                                                         |

!!! note

    最大图像尺寸受模型的部署设置限制。请参见下图：

![图像尺寸设置](../../assets/playground/image-size.png)

## 查看代码

在尝试不同的提示词与参数后，点击 `View Code` 按钮以查看如何使用相同输入调用 API。代码示例提供了 `curl`、`Python` 和 `Node.js` 版本。