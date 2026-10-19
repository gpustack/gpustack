---
translation_status: translated
---

# 模型目录 { #model-catalog }

模型目录是经过 GPUStack 调优的模型索引。

## 浏览模型 { #browse-models }

进入 `Catalog` 页面即可浏览模型目录。可以按名称和类别筛选模型。以下截图展示模型目录页面：

![模型目录](../assets/model-catalog.png)

## 从目录部署模型 { #deploy-a-model-from-the-catalog }

单击模型卡片即可从模型目录部署模型。页面会显示模型部署配置，可以检查和自定义部署配置，然后单击 `Save` 按钮部署模型。

## 自定义模型目录 { #customize-model-catalog }

可以在 GPUStack Server 配置中使用 `--model-catalog-file` 参数提供 YAML 文件，以自定义模型目录。它接受本地文件路径或 URL。结构定义可参考[内置模型目录文件](https://github.com/gpustack/gpustack/blob/main/gpustack/assets/model-catalog.yaml)。

要在不重启 GPUStack Server 的情况下自定义目录，请改为配置目录来源：自定义文档会替换随包提供的目录，抽屉中也会提供该目录，供你作为编辑起点。参见 [OTA 内容更新](ota-content-updates.md)。

以下是自定义模型目录 YAML 文件的示例：

```yaml
draft_models:
- name: Qwen3-8B-EAGLE3
  algorithm: eagle3
  source: huggingface
  huggingface_repo_id: Tengyunw/qwen3_8b_eagle3
model_sets:
- name: Deepseek R1 0528
  description: DeepSeek-R1-0528 is a minor version of the DeepSeek R1 model that features enhanced reasoning depth and inference capabilities. These improvements are achieved through increased computational resources and algorithmic optimizations applied during post-training. The model delivers strong performance across a range of benchmark evaluations, including mathematics, programming, and general logic, with overall capabilities approaching those of leading models such as O3 and Gemini 2.5 Pro.
  home: https://www.deepseek.com
  icon: /static/catalog_icons/deepseek.png
  categories:
    - llm
  capabilities:
    - context/128K
  size: 671
  licenses:
    - mit
  release_date: "2025-05-28"
  specs:
    - mode: throughput
      quantization: FP8
      gpu_filters:
        vendor: nvidia
        compute_capability: ">=9.0" # Hopper or later
      source: huggingface
      huggingface_repo_id: deepseek-ai/DeepSeek-R1-0528
      backend: SGLang
      backend_parameters:
        - --enable-dp-attention
        - --context-length=32768
    - mode: standard
      quantization: FP8
      source: huggingface
      huggingface_repo_id: deepseek-ai/DeepSeek-R1-0528
      backend: vLLM
      backend_parameters:
        - --max-model-len=32768
```

### 在隔离环境中使用模型目录 { #using-model-catalog-in-air-gapped-environments }

内置模型目录从 Hugging Face 或 ModelScope 获取模型。如果在无法访问互联网的隔离环境中使用 GPUStack，可以自定义模型目录，改用本地路径模型来源。以下是示例：

```yaml
model_sets:
- name: Deepseek R1 0528
  description: DeepSeek-R1-0528 is a minor version of the DeepSeek R1 model that features enhanced reasoning depth and inference capabilities. These improvements are achieved through increased computational resources and algorithmic optimizations applied during post-training. The model delivers strong performance across a range of benchmark evaluations, including mathematics, programming, and general logic, with overall capabilities approaching those of leading models such as O3 and Gemini 2.5 Pro.
  home: https://www.deepseek.com
  icon: /static/catalog_icons/deepseek.png
  categories:
    - llm
  capabilities:
    - context/128K
  size: 671
  licenses:
    - mit
  release_date: "2025-05-28"
  specs:
    - mode: throughput
      quantization: FP8
      gpu_filters:
        vendor: nvidia
        compute_capability: ">=9.0" # Hopper or later
      source: local_path
      local_path: /path/to/DeepSeek-R1-0528
      backend: SGLang
      backend_parameters:
        - --enable-dp-attention
        - --context-length=32768
    - mode: standard
      quantization: FP8
      source: local_path
      # assuming you have /path/to/DeepSeek-R1-0528 directory
      local_path: /path/to/DeepSeek-R1-0528
      backend: vLLM
      backend_parameters:
        - --max-model-len=32768
```

### 模型目录结构 { #model-catalog-schema }

模型目录 YAML 文件包含两个主要部分：`draft_models` 和 `model_sets`。

- `draft_models`：用于推测解码的草稿模型列表。
- `model_sets`：经过测试和优化的模型集合列表。

每个草稿模型包含以下字段：

| 字段 | 类型 | 描述 |
| --- | --- | --- |
| name | string | 草稿模型名称。 |
| algorithm | string | 模型使用的推测解码算法。目前仅支持 eagle3。 |
| source | string | 模型来源（例如 huggingface、model_scope）。 |
| huggingface_repo_id | string | 模型的 Hugging Face 仓库 ID（source 为 huggingface 时使用）。 |
| model_scope_model_id | string | 模型的 ModelScope 仓库 ID（source 为 model_scope 时使用）。 |

每个模型集合包含以下字段：

| 字段 | 类型 | 描述 |
| --- | --- | --- |
| name | string | 模型名称。 |
| description | string | 模型的简要描述。 |
| home | string | 模型主页 URL。 |
| icon | string | 模型图标 URL。 |
| categories | list of str | 模型所属类别列表。 |
| capabilities | list of str | 模型能力列表。 |
| size | int | 模型大小，以十亿参数为单位。 |
| licenses | list of str | 模型许可证列表。 |
| release_date | string | 模型发布日期，格式为 YYYY-MM-DD。 |
| specs | list of spec | 模型部署规格列表。 |

每个部署规格包含以下字段：

| 字段 | 类型 | 描述 |
| --- | --- | --- |
| mode | string | GPUStack 为不同使用场景提供常规和优化模式，包括吞吐量、延迟和标准场景。用户也可以按需定义自定义模式。 |
| quantization | string | 量化类型（例如 FP16、FP8、INT8）。 |
| gpu_filters | dict | 用于指定兼容 GPU 的 GPU 筛选条件。 |

部署规格的其他字段与模型 API 字段类似。更多详情参见 [API 参考](../api-reference.md)。
