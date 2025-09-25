---
hide:
  - toc
---

# gpustack draw

使用扩散模型生成图像。

```bash
gpustack draw [model] [prompt]
```

## 位置参数

| 名称   | 说明                                     |
| ------ | ---------------------------------------- |
| model  | 用于生成图像的模型。                     |
| prompt | 用于生成图像的文本提示词。               |

`model` 可以是以下任一形式：

1. GPUStack 模型的名称。使用前需要先在 GPUStack 中创建该模型。
2. 以 Ollama 风格引用的 Hugging Face GGUF 扩散模型。使用此选项时，如果该模型尚不可用，将会自动部署。若未指定标签，默认使用 `Q4_0`。示例：

   - `hf.co/gpustack/stable-diffusion-v3-5-large-turbo-GGUF`
   - `hf.co/gpustack/stable-diffusion-v3-5-large-turbo-GGUF:FP16`
   - `hf.co/gpustack/stable-diffusion-v3-5-large-turbo-GGUF:stable-diffusion-v3-5-large-turbo-Q4_0.gguf`


## 配置项

| <div style="width:180px">参数</div> | <div style="width:100px">默认值</div> | 说明                                                                                         |
| ----------------------------------- | -------------------------------------- | -------------------------------------------------------------------------------------------- |
| `--size` value                      | `512x512`                              | 生成图像的尺寸，格式为 `宽x高`。                                                             |
| `--sampler` value                   | `euler`                                | 采样方法。可选：euler_a、euler、heun、dpm2、dpm++2s_a、dpm++2m、lcm 等。                     |
| `--sample-steps` value              | （空）                                 | 采样步数。                                                                                   |
| `--cfg-scale` value                 | （空）                                 | 无分类器引导（CFG）系数，用于在提示词遵从与创造性之间平衡。                                   |
| `--seed` value                      | （空）                                 | 随机数种子。有助于结果可复现。                                                               |
| `--negative-prompt` value           | （空）                                 | 反向提示词：用于指定图像中应避免的内容。                                                     |
| `--output` value                    | （空）                                 | 保存生成图像的路径。                                                                         |
| `--show`                            | `False`                                | 若为 True，则在默认图像查看器中打开生成的图像。                                              |
| `-d`, `--debug`                     | `False`                                | 启用调试模式。                                                                               |