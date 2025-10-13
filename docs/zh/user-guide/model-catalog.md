# 模型目录

模型目录是热门模型的索引，帮助你快速查找并部署模型。

## 浏览模型

你可以通过进入 `Catalog` 页面浏览模型目录。你可以按名称和类别筛选模型。下图展示了模型目录页面：

![模型目录](../../assets/model-catalog.png)

## 从目录部署模型

在模型目录中点击某个模型卡片即可部署模型。随后会出现模型部署配置页面。你可以查看并自定义部署配置，然后点击 `Save` 按钮完成部署。

## 自定义模型目录

你可以在启动 GPUStack 服务器时，通过 `--model-catalog-file` 参数提供一个 YAML 文件来自定义模型目录。该参数可接受本地文件路径或 URL。关于其模式（schema），可参考内置的模型目录文件[此处](https://github.com/gpustack/gpustack/blob/main/gpustack/assets/model-catalog.yaml)。该文件包含一组模型集合，每个集合含有模型元数据以及用于部署配置的模板。

下面是模型目录文件中的一个模型集合示例：

```yaml
- name: Llama3.2
  description: Llama 3.2 多语言大语言模型（LLMs）集合包含 1B 和 3B 参数规模的预训练与指令微调生成式模型（文本输入/文本输出）。Llama 3.2 的指令微调纯文本模型针对多语言对话场景进行了优化，包括具备代理能力的检索与摘要任务。在常见行业基准上，它们优于许多开源与闭源聊天模型。
  home: https://www.llama.com/
  icon: /static/catalog_icons/meta.png
  categories:
    - llm
  capabilities:
    - context/128k
    - tools
  sizes:
    - 1
    - 3
  licenses:
    - llama3.2
  release_date: "2024-09-25"
  order: 2
  templates:
    - quantizations:
        - Q3_K_L
        - Q4_K_M
        - Q5_K_M
        - Q6_K_L
        - Q8_0
        - f16
      source: huggingface
      huggingface_repo_id: bartowski/Llama-3.2-{size}B-Instruct-GGUF
      huggingface_filename: "*-{quantization}*.gguf"
      replicas: 1
      backend: llama-box
      cpu_offloading: true
      distributed_inference_across_workers: true
    - quantizations: ["BF16"]
      source: huggingface
      huggingface_repo_id: unsloth/Llama-3.2-{size}B-Instruct
      replicas: 1
      backend: vllm
      backend_parameters:
        - --enable-auto-tool-choice
        - --tool-call-parser=llama3_json
        - --chat-template={data_dir}/chat_templates/tool_chat_template_llama3.2_json.jinja
```

### 在隔离网络环境中使用模型目录

内置的模型目录会从 Hugging Face 或 ModelScope 获取模型。如果你在无互联网访问的隔离网络环境中使用 GPUStack，可以通过自定义模型目录改为使用本地路径作为模型来源。示例如下：

```yaml
- name: Llama3.2
  description: Llama 3.2 多语言大语言模型（LLMs）集合包含 1B 和 3B 参数规模的预训练与指令微调生成式模型（文本输入/文本输出）。Llama 3.2 的指令微调纯文本模型针对多语言对话场景进行了优化，包括具备代理能力的检索与摘要任务。在常见行业基准上，它们优于许多开源与闭源聊天模型。
  home: https://www.llama.com/
  icon: /static/catalog_icons/meta.png
  categories:
    - llm
  capabilities:
    - context/128k
    - tools
  sizes:
    - 1
    - 3
  licenses:
    - llama3.2
  release_date: "2024-09-25"
  order: 2
  templates:
    - quantizations:
        - Q3_K_L
        - Q4_K_M
        - Q5_K_M
        - Q6_K_L
        - Q8_0
        - f16
      source: local_path
      # 假设你已将所有 GGUF 模型文件放在 /path/to/the/model/directory
      local_path: /path/to/the/model/directory/Llama-3.2-{size}B-Instruct-{quantization}.gguf
      replicas: 1
      backend: llama-box
      cpu_offloading: true
      distributed_inference_across_workers: true
    - quantizations: ["BF16"]
      source: local_path
      # 假设你同时拥有 /path/to/Llama-3.2-1B-Instruct 与 /path/to/Llama-3.2-3B-Instruct 目录
      local_path: /path/to/Llama-3.2-{size}B-Instruct
      replicas: 1
      backend: vllm
      backend_parameters:
        - --enable-auto-tool-choice
        - --tool-call-parser=llama3_json
        - --chat-template={data_dir}/chat_templates/tool_chat_template_llama3.2_json.jinja
```

### 模板变量

以下模板变量可用于部署配置：

- `{size}`：以十亿参数为单位的模型规模。
- `{quantization}`：模型的量化方法。
- `{data_dir}`：GPUStack 数据目录路径。