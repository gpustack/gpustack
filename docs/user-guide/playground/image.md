# Image Playground

The Image Playground is a dedicated space for testing and experimenting with GPUStack’s image generation APIs. It allows users to interactively explore the capabilities of different models, customize parameters, and review code examples for seamless API integration.

## Prompt

You can input or randomly generate a prompt, then click the Submit button to generate an image.

![here a image](../../assets/playground/image-prompt.png)

## Clear Prompt

Click the `Clear` button to reset the prompt and remove the generated image.

## Select Model

You can select available models in GPUStack by clicking the model dropdown at the top-right corner of the playground UI.

## Customize Parameters

You can customize the image generation parameters by switching between two API styles:

1. **OpenAI-compatible mode**.
2. **Advanced mode**.

![image-parameter](../../assets/playground/image-params.png)

### Advanced Parameters

| Parameter         | Default    | Description                                                                                                                                                         |
| ----------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Counts`          | `1`        | Number of images to generate.                                                                                                                                       |
| `Size`            | `512x512`  | The size of the generated image in 'widthxheight' format.                                                                                                           |
| `Sampler`         | `euler_a`  | The sampler algorithm for image generation. Options include 'euler_a', 'euler', 'heun', 'dpm2', 'dpm++2s_a', 'dpm++2m', 'dpm++2mv2', 'ipndm', 'ipndm_v', and 'lcm'. |
| `Schedule`        | `discrete` | The noise scheduling method.                                                                                                                                        |
| `Sampler Steps`   | `10`       | The number of sampling steps to perform. Higher values may improve image quality at the cost of longer processing time.                                             |
| `CFG Scale`       | `4.5`      | The scale for classifier-free guidance. A higher value increases adherence to the prompt.                                                                           |
| `Negative Prompt` | (empty)    | A negative prompt to specify what the image should avoid.                                                                                                           |
| `Seed`            | (empty)    | Random seed.                                                                                                                                                        |

!!! note

    The maximum image size is restricted by the model's deployment settings. See the diagram below:

![image-size-setting](../../assets/playground/image-size.png)

## View Code

After experimenting with prompts and parameters, click the `View Code` button to see how to call the API with the same inputs. Code examples are provided in `curl`, `Python`, and `Node.js`.
