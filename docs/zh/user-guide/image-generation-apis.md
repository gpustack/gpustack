# 图像生成 API

GPUStack 提供在运行扩散模型时，根据提示词和/或输入图像生成图像的 API。

!!! note

    仅在使用 [llama-box](./inference-backends.md#llama-box) 推理后端时，图像生成 API 可用。

## 支持的模型

下列模型可用于图像生成：

!!! tip

      请使用 GPUStack 提供的已转换 GGUF 模型。查看模型链接了解详情。

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

## API 详情

图像生成 API 遵循 OpenAI API 规范。尽管 OpenAI 的图像生成 API 简洁且带有固定约定，GPUStack 在此基础上扩展了更多功能。

### 创建图像

#### 流式传输

此图像生成 API 支持流式响应，用于返回生成进度。要启用流式传输，在请求体中将 `stream` 参数设置为 `true`。示例：

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
    "include_usage": true, // 返回用量信息
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

#### 高级选项

此图像生成 API 提供用于控制生成过程的附加选项。可用选项如下：

```
REQUEST : (application/json)
{
  "n": 1,
  "response_format": "b64_json",
  "size": "512x512",
  "prompt": "A lovely cat",
  "sampler": "euler",      // 必填，从 euler_a;euler;heun;dpm2;dpm++2s_a;dpm++2m;dpm++2mv2;ipndm;ipndm_v;lcm 中选择
  "schedule": "default",   // 可选，从 default;discrete;karras;exponential;ays;gits 中选择
  "seed": null,            // 可选，随机种子
  "cfg_scale": 4.5,        // 可选，对采样器而言，输出阶段的无分类器引导系数
  "sample_steps": 20,      // 可选，采样步数
  "negative_prompt": "",   // 可选，负面提示词
  "stream": true,
  "stream_options": {
    "include_usage": true, // 返回用量信息
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

### 创建图像编辑

#### 流式传输

此图像生成 API 支持流式响应，用于返回生成进度。要启用流式传输，在请求体中将 `stream` 参数设置为 `true`。示例：

```
REQUEST: (multipart/form-data)
n=1
response_format=b64_json
size=512x512
prompt="A lovely cat"
quality=standard
image=...                         // 必填
mask=...                          // 可选
stream=true
stream_options_include_usage=true // 返回用量信息

RESPONSE : (text/event-stream)
CASE 1: 正确的输入图像
  data: {"created":1731916353,"data":[{"index":0,"object":"image.chunk","progress":10.0}], ...}
  ...
  data: {"created":1731916371,"data":[{"index":0,"object":"image.chunk","progress":50.0}], ...}
  ...
  data: {"created":1731916371,"data":[{"index":0,"object":"image.chunk","progress":100.0,"b64_json":"..."}], "usage":{"generation_per_second":...,"time_per_generation_ms":...,"time_to_process_ms":...}, ...}
  data: [DONE]
CASE 2: 非法的输入图像
  error: {"code": 400, "message": "Invalid image", "type": "invalid_request_error"}
```

#### 高级选项

此图像生成 API 提供用于控制生成过程的附加选项。可用选项如下：

```
REQUEST: (multipart/form-data)
n=1
response_format=b64_json
size=512x512
prompt="A lovely cat"
image=...                         // 必填
mask=...                          // 可选
sampler=euler                     // 必填，从 euler_a;euler;heun;dpm2;dpm++2s_a;dpm++2m;dpm++2mv2;ipndm;ipndm_v;lcm 中选择
schedule=default                  // 可选，从 default;discrete;karras;exponential;ays;gits 中选择
seed=null                         // 可选，随机种子
cfg_scale=4.5                     // 可选，对采样器而言，输出阶段的无分类器引导系数
sample_steps=20                   // 可选，采样步数
negative_prompt=""                // 可选，负面提示词
stream=true
stream_options_include_usage=true // 返回用量信息

RESPONSE : (text/event-stream)
CASE 1: 正确的输入图像
  data: {"created":1731916353,"data":[{"index":0,"object":"image.chunk","progress":10.0}], ...}
  ...
  data: {"created":1731916371,"data":[{"index":0,"object":"image.chunk","progress":50.0}], ...}
  ...
  data: {"created":1731916371,"data":[{"index":0,"object":"image.chunk","progress":100.0,"b64_json":"..."}], "usage":{"generation_per_second":...,"time_per_generation_ms":...,"time_to_process_ms":...}, ...}
  data: [DONE]
CASE 2: 非法的输入图像
  error: {"code": 400, "message": "Invalid image", "type": "invalid_request_error"}
```

## 用法

以下示例展示了如何使用图像生成 API：

### curl（创建图像）

```bash
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1-openai/image/generate \
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

### curl（创建图像编辑）

```bash
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1-openai/image/edit \
    -H "Authorization: Bearer $GPUSTACK_API_KEY" \
    -F image="@otter.png" \
    -F mask="@mask.png" \
    -F prompt="A lovely cat" \
    -F n=1 \
    -F size="512x512"
```