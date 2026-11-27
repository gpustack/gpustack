# Video Playground

The Video Playground lets you test GPUStack's video generation APIs, customize parameters, play generated videos, and review code examples for API integration.

## Generate Video

### Prompt

Enter a prompt and click **Submit** to start an asynchronous video task. The page tracks only the latest saved task and keeps the prompt in the input field for reuse. If you submit again after a status error, the earlier task can continue on the inference engine.

### Generation Progress

While the page is visible, the Playground checks the current task every five seconds. Each check can take up to five seconds. The result area shows whether the task is queued or generating and displays a percentage when the engine reports positive generation progress. If a status check fails or times out, the Playground hides the queued or generating display and stops this round of polling. The error appears below the result area and above the prompt, and you can submit again. An HTTP 404 from the status check also clears the latest record from this browser. Other status errors keep the record so you can check the same task again by refreshing or returning to the page.

### Switch Pages

The Playground saves the latest video ID and submitted parameters for each account in this browser. After a refresh, it restores the form and checks that ID. Leaving the page or hiding the tab pauses status checks. Returning checks the saved task again.

### Current Task { #tasks }

The result area shows the current task. When you refresh with a completed task, the player appears once the video has been downloaded and its dimensions are available. Video availability depends on the instance that created the task and how long the engine keeps it.

### Stop Request

Click **Cancel generation** beside the prompt to cancel immediately. While waiting for a video ID, this only cancels the browser's create request. If the engine already received the request, generation can continue. Once the Playground has an ID, it forwards the cancellation request to the inference engine. A successful cancellation clears the browser record and returns the result area to its empty state. The engine controls file cleanup.

**Clear video record**, next to the random prompt button, removes the saved video ID, submitted parameters, and preview only from this browser after you confirm.

## Select Model

Select an available model from the dropdown in the Parameters panel. Only models in the `video` category are listed. If a model is missing, make sure it is running and its deployment and route use the correct category.

Use a compatible SGLang Diffusion or vLLM-Omni deployment. Video understanding models use the `llm` category and the Chat Playground.

## Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `Duration (seconds)` | Model default or `4` | If the model provides duration metadata, enter a whole number within its range, such as 4–15 for MiniMax H3. Otherwise, choose `4`, `8`, or `12`. The generated video may be slightly longer or shorter due to frame alignment. |
| `Resolution` | `832x480` or model preset | Output size in `widthxheight` format. Choosing a preset sends its exact dimensions. |
| `Seed` | Random | Turn off **Random Seed** to enter a fixed seed. |
| `Reference Image` | (empty) | Optional reference image in advanced settings. Upload a PNG/JPEG image or enter an HTTP(S) image URL. |

Use the clear button to remove a reference image or URL. Reference-image support depends on the selected model; choosing the `video` category does not make a text-to-video model accept image input.

Choose the aspect ratio using tiles, then choose a resolution level with buttons. Presets include 480p, 720p, 1080p and 4k where available, with exact width and height shown when hovering an option. Model metadata can add dimensions or limit sizes; these are UI presets, not a guarantee that an engine accepts every size. There is no 21:9 preset. Choose `Custom` under Aspect ratio to adjust width and height with sliders or numeric inputs. The initial size uses model metadata when available; otherwise it starts at `832x480` (16:9), subject to model size limits.

## Model Capabilities

For MiniMax H3, model metadata provides the tasks supported by the loaded partition, a default resolution, and a duration range of 4–15 seconds. The Playground defaults to `t2va` when that task is available. The server recognizes the model artifact and loaded partition for both Catalog and direct Hugging Face deployments. Existing deployments receive capability metadata through background reconciliation.

When creating an asynchronous MiniMax H3 task from the Playground, the UI translates its controls (`task`, `size`, `seconds`, and reference media) into the selected inference target's native parameters before sending `POST /v1/videos`. SGLang uses `task`, `conditions`, and `target`. vLLM-Omni uses `extra_params` and engine-specific reference fields (`image_reference`, `video_reference`, `audio_reference`, `frame_indices`).

The Playground's `Duration (seconds)` input accepts whole seconds. Native duration parameters in Advanced Parameters can express fractions when the engine supports them.

Available task types follow the deployed model's capability metadata. A single available task is selected automatically and its selector is hidden. With multiple tasks, choose a task in advanced settings.

- `t2va`: reference controls are hidden, and requests and code examples omit all reference fields.
- `fl2va`: upload an image for the first frame, the last frame, or both.
- `ref2va`: provide one reference image or video using a URL or upload. Use the API for multiple references or separate audio conditioning.

Changing the task or model clears reference inputs. Image uploads accept PNG/JPEG up to 1 MiB and videos MP4/MOV up to 2 MiB per file.

## View Code

After experimenting with prompts and parameters, click `View Code` to see how to call the API with the same inputs. Code examples are provided in `curl`, `Python`, and `Node.js`.

The examples create an asynchronous video job, poll its status, and download the completed video. See [Video generation](../../integrations/inference-apis.md#video-generation) for the API endpoints.
