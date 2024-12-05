---
hide:
  - toc
---

# gpustack draw

Generate an image with a diffusion model.

```bash
gpustack draw [model] [prompt]
```

## Positional Arguments

| Name   | Description                              |
| ------ | ---------------------------------------- |
| model  | The model to use for image generation.   |
| prompt | Text prompt to use for image generation. |

The `model` can be either of the following:

1. Name of a GPUStack model. You need to create a model in GPUStack before using it here.
2. Reference to a Hugging Face GGUF diffusion model in Ollama style. When using this option, the model will be deployed if it is not already available. When not specified the default `Q4_0` tag is used. Examples:

   - `hf.co/gpustack/stable-diffusion-v3-5-large-turbo-GGUF`
   - `hf.co/gpustack/stable-diffusion-v3-5-large-turbo-GGUF:FP16`
   - `hf.co/gpustack/stable-diffusion-v3-5-large-turbo-GGUF:stable-diffusion-v3-5-large-turbo-Q4_0.gguf`

## Configurations

| <div style="width:180px">Flag</div> | <div style="width:100px">Default</div> | Description                                                                                 |
| ----------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------- |
| `--size` value                      | `512x512`                              | Size of the image to generate, specified as `widthxheight`.                                 |
| `--sampler` value                   | `euler`                                | Sampling method. Options include: euler_a, euler, heun, dpm2, dpm++2s_a, dpm++2m, lcm, etc. |
| `--sample-steps` value              | (Empty)                                | Number of sampling steps.                                                                   |
| `--cfg-scale` value                 | (Empty)                                | Classifier-free guidance scale for balancing prompt adherence and creativity.               |
| `--seed` value                      | (Empty)                                | Seed for random number generation. Useful for reproducibility.                              |
| `--negative-prompt` value           | (Empty)                                | Text prompt for what to avoid in the image.                                                 |
| `--output` value                    | (Empty)                                | Path to save the generated image.                                                           |
| `--show`                            | `False`                                | If True, opens the generated image in the default image viewer.                             |
| `-d`, `--debug`                     | `False`                                | Enable debug mode.                                                                          |
