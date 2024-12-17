# Image Generation APIs

GPUStack provides APIs for generating images given a prompt and/or an input image when running diffusion models.

!!! note

    The image generation APIs are only available when using the [llama-box](./inference-backends.md#llama-box) inference backend.

## Supported Models

The following models are available for image generation:

!!! tip

      Please use the converted GGUF models provided by GPUStack. Check the model link for more details.

- stabilityai/stable-diffusion-3.5-large-turbo [[Hugging Face]](https://huggingface.co/gpustack/stable-diffusion-v3-5-large-turbo-GGUF), [[ModelScope]](https://modelscope.cn/models/gpustack/stable-diffusion-v3-5-large-turbo-GGUF)
- stabilityai/stable-diffusion-3.5-large [[Hugging Face]](https://huggingface.co/gpustack/stable-diffusion-v3-5-large-GGUF), [[ModelScope]](https://modelscope.cn/models/gpustack/stable-diffusion-v3-5-large-GGUF)
- stabilityai/stable-diffusion-3.5-medium [[Hugging Face]](https://huggingface.co/gpustack/stable-diffusion-v3-5-medium-GGUF), [[ModelScope]](https://modelscope.cn/models/gpustack/stable-diffusion-v3-5-medium-GGUF)
- stabilityai/stable-diffusion-3-medium [[Hugging Face]](https://huggingface.co/gpustack/stable-diffusion-v3-medium-GGUF), [[ModelScope]](https://modelscope.cn/models/gpustack/stable-diffusion-v3-medium-GGUF)
- TencentARC/FLUX.1-mini [[Hugging Face]](https://huggingface.co/gpustack/FLUX.1-mini-GGUF), [[ModelScope]](https://modelscope.cn/models/gpustack/FLUX.1-mini-GGUF)
- Freepik/FLUX.1-lite [[Hugging Face]](https://huggingface.co/gpustack/FLUX.1-lite-GGUF), [[ModelScope]](https://modelscope.cn/models/gpustack/FLUX.1-lite-GGUF)
- black-forest-labs/FLUX.1-dev [[Hugging Face]](https://huggingface.co/gpustack/FLUX.1-dev-GGUF), [[ModelScope]](https://modelscope.cn/models/gpustack/FLUX.1-dev-GGUF)
- black-forest-labs/FLUX.1-schnell [[Hugging Face]](https://huggingface.co/gpustack/FLUX.1-schnell-GGUF), [[ModelScope]](https://modelscope.cn/models/gpustack/FLUX.1-schnell-GGUF)
- stabilityai/sdxl-turbo [[Hugging Face]](https://huggingface.co/gpustack/stable-diffusion-xl-1.0-turbo-GGUF), [[ModelScope]](https://modelscope.cn/models/gpustack/stable-diffusion-xl-1.0-turbo-GGUF)
- stabilityai/stable-diffusion-xl-refiner-1.0 [[Hugging Face]](https://huggingface.co/gpustack/stable-diffusion-xl-refiner-1.0-GGUF), [[ModelScope]](https://modelscope.cn/models/gpustack/stable-diffusion-xl-refiner-1.0-GGUF)
- stabilityai/stable-diffusion-xl-base-1.0 [[Hugging Face]](https://huggingface.co/gpustack/stable-diffusion-xl-base-1.0-GGUF), [[ModelScope]](https://modelscope.cn/models/gpustack/stable-diffusion-xl-base-1.0-GGUF)
- stabilityai/sd-turbo [[Hugging Face]](https://huggingface.co/gpustack/stable-diffusion-v2-1-turbo-GGUF), [[ModelScope]](https://modelscope.cn/models/gpustack/stable-diffusion-v2-1-turbo-GGUF)
- stabilityai/stable-diffusion-2-1 [[Hugging Face]](https://huggingface.co/gpustack/stable-diffusion-v2-1-GGUF), [[ModelScope]](https://modelscope.cn/models/gpustack/stable-diffusion-v2-1-GGUF)
- stable-diffusion-v1-5/stable-diffusion-v1-5 [[Hugging Face]](https://huggingface.co/gpustack/stable-diffusion-v1-5-GGUF), [[ModelScope]](https://modelscope.cn/models/gpustack/stable-diffusion-v1-5-GGUF)
- CompVis/stable-diffusion-v1-4 [[Hugging Face]](https://huggingface.co/gpustack/stable-diffusion-v1-4-GGUF), [[ModelScope]](https://modelscope.cn/models/gpustack/stable-diffusion-v1-4-GGUF)

## API Details

The image generation APIs adhere to OpenAI API specification. While OpenAI APIs for image generation are simple and opinionated, GPUStack extends these capabilities with additional features.

### Create Image

#### Streaming

This image generation API supports streaming responses to return the progressing of the generation. To enable streaming, set the `stream` parameter to `true` in the request body. Example:

```
REQUEST : (application/json)
{
  "n": 1,
  "response_format": "b64_json",
  "size": "512x512",
  "prompt": "A lovely cat",
  "quality": "standard",
  "stream": true,
  "stream_options": {
    "include_usage": true, // return usage information
  }
}

RESPONSE : (text/event-stream)
data: {"created":1731916353,"data":[{"index":0,"object":"image.chunk","progress":10.0}], ...}
...
data: {"created":1731916371,"data":[{"index":0,"object":"image.chunk","progress":50.0}], ...}
...
data: {"created":1731916371,"data":[{"index":0,"object":"image.chunk","progress":100.0,"b64_json":"..."}], "usage":{"generation_per_second":...,"time_per_generation_ms":...,"time_to_process_ms":...}, ...}
data: [DONE]
```

#### Advanced Options

This image generation API supports additional options to control the generation process. The following options are available:

```
REQUEST : (application/json)
{
  "n": 1,
  "response_format": "b64_json",
  "size": "512x512",
  "prompt": "A lovely cat",
  "sampler": "euler",      // required, select from euler_a;euler;heun;dpm2;dpm++2s_a;dpm++2m;dpm++2mv2;ipndm;ipndm_v;lcm
  "schedule": "default",   // optional, select from default;discrete;karras;exponential;ays;gits
  "seed": null,            // optional, random seed
  "cfg_scale": 4.5,        // optional, for sampler, the scale of classifier-free guidance in the output phase
  "sample_steps": 20,      // optional, number of sample steps
  "negative_prompt": "",   // optional, negative prompt
  "stream": true,
  "stream_options": {
    "include_usage": true, // return usage information
  }
}

RESPONSE : (text/event-stream)
data: {"created":1731916353,"data":[{"index":0,"object":"image.chunk","progress":10.0}], ...}
...
data: {"created":1731916371,"data":[{"index":0,"object":"image.chunk","progress":50.0}], ...}
...
data: {"created":1731916371,"data":[{"index":0,"object":"image.chunk","progress":100.0,"b64_json":"..."}], "usage":{"generation_per_second":...,"time_per_generation_ms":...,"time_to_process_ms":...}, ...}
data: [DONE]
```

### Create Image Edit

#### Streaming

This image generation API supports streaming responses to return the progressing of the generation. To enable streaming, set the `stream` parameter to `true` in the request body. Example:

```
REQUEST: (multipart/form-data)
n=1
response_format=b64_json
size=512x512
prompt="A lovely cat"
quality=standard
image=...                         // required
mask=...                          // optional
stream=true
stream_options_include_usage=true // return usage information

RESPONSE : (text/event-stream)
CASE 1: correct input image
  data: {"created":1731916353,"data":[{"index":0,"object":"image.chunk","progress":10.0}], ...}
  ...
  data: {"created":1731916371,"data":[{"index":0,"object":"image.chunk","progress":50.0}], ...}
  ...
  data: {"created":1731916371,"data":[{"index":0,"object":"image.chunk","progress":100.0,"b64_json":"..."}], "usage":{"generation_per_second":...,"time_per_generation_ms":...,"time_to_process_ms":...}, ...}
  data: [DONE]
CASE 2: illegal input image
  error: {"code": 400, "message": "Invalid image", "type": "invalid_request_error"}
```

#### Advanced Options

This image generation API supports additional options to control the generation process. The following options are available:

```
REQUEST: (multipart/form-data)
n=1
response_format=b64_json
size=512x512
prompt="A lovely cat"
image=...                         // required
mask=...                          // optional
sampler=euler                     // required, select from euler_a;euler;heun;dpm2;dpm++2s_a;dpm++2m;dpm++2mv2;ipndm;ipndm_v;lcm
schedule=default                  // optional, select from default;discrete;karras;exponential;ays;gits
seed=null                         // optional, random seed
cfg_scale=4.5                     // optional, for sampler, the scale of classifier-free guidance in the output phase
sample_steps=20                   // optional, number of sample steps
negative_prompt=""                // optional, negative prompt
stream=true
stream_options_include_usage=true // return usage information

RESPONSE : (text/event-stream)
CASE 1: correct input image
  data: {"created":1731916353,"data":[{"index":0,"object":"image.chunk","progress":10.0}], ...}
  ...
  data: {"created":1731916371,"data":[{"index":0,"object":"image.chunk","progress":50.0}], ...}
  ...
  data: {"created":1731916371,"data":[{"index":0,"object":"image.chunk","progress":100.0,"b64_json":"..."}], "usage":{"generation_per_second":...,"time_per_generation_ms":...,"time_to_process_ms":...}, ...}
  data: [DONE]
CASE 2: illegal input image
  error: {"code": 400, "message": "Invalid image", "type": "invalid_request_error"}
```

## Usage

The followings are examples using the image generation APIs:

### curl (Create Image)

```bash
export GPUSTACK_API_KEY=myapikey
curl http://myserver/v1-openai/image/generate \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $GPUSTACK_API_KEY" \
    -d '{
        "n": 1,
        "response_format": "b64_json",
        "size": "512x512",
        "prompt": "A lovely cat",
        "quality": "standard",
        "stream": true,
        "stream_options": {
        "include_usage": true
        }
    }'

```

### curl (Create Image Edit)

```bash
export GPUSTACK_API_KEY=myapikey
curl http://myserver/v1-openai/image/edit \
    -H "Authorization: Bearer $GPUSTACK_API_KEY" \
    -F image="@otter.png" \
    -F mask="@mask.png" \
    -F prompt="A lovely cat" \
    -F n=1 \
    -F size="512x512"
```
