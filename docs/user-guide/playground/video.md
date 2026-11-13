# Video Playground

The Video Playground lets you test GPUStack's video generation APIs, customize parameters, play generated videos, and review code examples for API integration.

## Generate Video

### Prompt

Enter a prompt or use the random prompt button, then click `Submit` to generate a video.

### Generation Progress

A loading animation and `Generating…` appear over the video area while the task is running. The page checks the task status every five seconds. A percentage is shown only when the inference engine reports progress greater than zero; some engines report no intermediate progress.

The video becomes available for playback after generation and download finish. Playback during generation is not supported.

### Switch Pages

While a generation task is being tracked, navigating to another page preserves the form and task. Status checks pause when you leave the Video page or put the browser tab in the background. Returning to the active Video page immediately checks the same task and resumes polling.

This state is kept within the current browser page. Refreshing or closing the browser page does not preserve it.

### Stop Request

Click `Stop` to cancel the current browser request, stop status checks, and exit the loading state. An accepted generation job can continue on the inference engine: this button does not send an engine cancellation request. Submitting again creates a new task.

## Select Model

Select an available model from the model dropdown at the top-right corner of the playground UI. Only models in the `video` category are listed. If a model is missing, make sure it is running and its deployment and route use the correct category.

Use a compatible SGLang Diffusion or vLLM-Omni deployment. Video understanding models use the `llm` category and the Chat Playground.

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `Duration` | `4` seconds | Requested video duration. The page offers `4`, `8`, and `12` seconds; accepted values depend on the model and engine. |
| `Resolution` | `720x1280` | Output size in `widthxheight` format. Available options and the initial value may change with model metadata. |
| `Reference Image` | (empty) | Optional reference image in advanced settings. Upload a PNG/JPEG image or enter an HTTP(S) image URL. |

Use the clear button to remove a reference image or URL. Reference-image support depends on the selected model; choosing the `video` category does not make a text-to-video model accept image input.

## View Code

After experimenting with prompts and parameters, click `View Code` to see how to call the API with the same inputs. Code examples are provided in `curl`, `Python`, and `Node.js`.

The examples create an asynchronous video job, poll its status, and download the completed video. See [Video generation](../../integrations/inference-apis.md#video-generation) for the API endpoints.
