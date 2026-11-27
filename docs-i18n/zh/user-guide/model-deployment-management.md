---
translation_status: translated
---

# 模型部署管理 { #model-deployment-management }

进入 `Models - Deployments` 页面即可管理 GPUStack 中的模型部署。一个模型部署包含一个或多个模型实例副本。部署时，GPUStack 自动根据模型元数据计算实例所需资源，并将其调度到可用 Worker 上。

## 部署模型 { #deploy-model }

目前支持来自 [Hugging Face](https://huggingface.co)、[ModelScope](https://modelscope.cn) 和本地路径的模型。

### 部署 Hugging Face 模型 { #deploying-a-hugging-face-model }

1. 单击 `Deploy Model` 按钮，然后从下拉菜单中选择 `Hugging Face`。

2. 使用左上角搜索栏按名称从 `Hugging Face` 搜索模型，例如 `Qwen/Qwen3-0.6B`。

3. 根据需要调整 `Name`、`Cluster`、`Backend`、`Backend Version` 和 `Replicas`。

4. 如需性能配置，展开 `Performance` 部分。详情参见[性能相关配置](#performance-related-configuration)。

5. 如需调度配置，展开 `Scheduling` 部分。详情参见[调度配置](#scheduling-configuration)。

6. 如需高级配置，展开 `Advanced` 部分。详情参见[高级配置](#advanced-configuration)。

7. 单击 `Save` 按钮。

### 部署 ModelScope 模型 { #deploying-a-modelscope-model }

1. 单击 `Deploy Model` 按钮，然后从下拉菜单中选择 `ModelScope`。

2. 使用左上角搜索栏按名称从 `ModelScope` 搜索模型，例如 `Qwen/Qwen3-0.6B`。

3. 根据需要调整 `Name`、`Cluster`、`Backend`、`Backend Version` 和 `Replicas`。

4. 如需性能配置，展开 `Performance` 部分。详情参见[性能相关配置](#performance-related-configuration)。

5. 如需调度配置，展开 `Scheduling` 部分。详情参见[调度配置](#scheduling-configuration)。

6. 如需高级配置，展开 `Advanced` 部分。详情参见[高级配置](#advanced-configuration)。

7. 单击 `Save` 按钮。

### 部署本地路径模型 { #deploying-a-local-path-model }

可以从本地路径部署模型。模型路径可以是 Worker 上的目录（例如已下载的 Hugging Face 模型目录）或文件（例如 GGUF 模型文件），适合隔离环境。

!!! note

    1. GPUStack 使用模型文件估算资源需求。如果 GPUStack Server 无法访问模型路径，会尝试从 Worker 访问。
    2. GPUStack 不会自动同步模型文件。必须确保目标 Worker 能够访问模型路径（例如使用 NFS、rsync 等）。也可以通过 Worker 选择器配置，将模型部署到指定 Worker。

部署本地路径模型：

1. 单击 `Deploy Model` 按钮，然后从下拉菜单中选择 `Local Path`。

2. 填写部署的 `Name`。

3. 填写 `Model Path`。

4. 根据需要调整 `Cluster`、`Backend`、`Backend Version` 和 `Replicas`。

5. 如需性能配置，展开 `Performance` 部分。详情参见[性能相关配置](#performance-related-configuration)。

6. 如需调度配置，展开 `Scheduling` 部分。详情参见[调度配置](#scheduling-configuration)。

7. 如需高级配置，展开 `Advanced` 部分。详情参见[高级配置](#advanced-configuration)。

8. 单击 `Save` 按钮。

### 后端 { #backend }

GPUStack 目前支持 vLLM、SGLang、MindIE 和 VoxBox 等内置后端。

更多详情参见[推理后端](built-in-inference-backends.md)。

### 后端版本 { #backend-version }

选择后端版本。可用版本取决于所选后端。此选项有助于确保兼容性，或使用特定后端版本引入的功能。

对于内置 vLLM 或 SGLang 后端，可在部署配置中设置 `image_name`，使用 runner 目录未收录的容器镜像。此时请不要设置 `backend_version`，自定义镜像与后端版本不能同时选择。vLLM 部署使用自定义镜像时，分布式执行默认采用 `mp`。如果镜像包含 Ray 且希望使用它，请在 `backend_parameters` 中设置 `--distributed-executor-backend=ray`。

对于内置 vLLM 或 SGLang 后端，`run_command` 提供不含可执行程序的初始启动参数；GPUStack 仍会添加其管理的参数，包括服务主机和端口。对于自定义后端，`run_command` 指定包含可执行程序的完整命令。示例参见[使用自定义推理后端](../tutorials/using-custom-backends.md)。

## 编辑模型部署 { #edit-model-deployment }

1. 在部署列表页找到要编辑的模型部署。
2. 单击 `Operations` 列中的 `Edit` 按钮。
3. 根据需要更新属性，例如修改 `Replicas` 进行扩容或缩容。
4. 单击 `Save` 按钮。

!!! note

    编辑模型部署后，配置不会应用到已有模型实例。需要删除已有模型实例，GPUStack 会按更新后的模型配置重新创建实例。

## 停止模型部署 { #stop-model-deployment }

停止模型部署会将目标副本数设为零，删除实例并释放资源。对于启用定时伸缩的部署，还会设置 `scaling_schedule.paused` 以暂停计划执行。启用设置、基准副本数和规则均保留。编辑或回滚时，只要保持计划启用，就会保留其暂停状态；普通部署仅通过副本伸缩控制，不存在暂停状态。

1. 在部署列表页找到要停止的模型部署。
2. 单击 `Operations` 列中的省略号按钮，然后选择 `Stop`。
3. 确认操作。

## 启动模型部署 { #start-model-deployment }

启用定时伸缩时，启动会清除 `scaling_schedule.paused`，GPUStack 会按当前时间评估计划，不会重放错过的时间窗口。如果基准副本数为零，在非活跃窗口内目标副本数仍可能为零；此时部署是在等待计划，而非暂停。对于此类部署，操作标签显示为 `Resume Schedule`。

未启用计划时，启动会将目标副本数设为一。禁用定时伸缩是独立的配置选择。在列表中直接编辑副本数，经确认后会切换为手动伸缩。

1. 在部署列表页找到要启动的模型部署。
2. 单击 `Operations` 列中的省略号按钮，然后选择 `Start`。

## 删除模型部署 { #delete-model-deployment }

1. 在部署列表页找到要删除的模型部署。
2. 单击 `Operations` 列中的省略号按钮，然后选择 `Delete`。
3. 确认删除。

## 导出和导入部署 { #export-and-import-deployments }

可将部署导出为包含完整配置的 YAML 文件，再导入该文件重新创建部署。典型用途包括：升级或迁移前备份、重新安装 GPUStack 后恢复、将部署配置纳入版本控制、与团队共享部署模板，或在另一个集群中复现相同部署。

### 导出部署 { #export-deployments }

1. 在部署列表页找到要导出的模型部署。
2. 单击 `Operations` 列中的省略号按钮，然后选择 `Export YAML`。
3. 要同时导出多个部署，先选择它们，再从列表上方的批量操作下拉菜单中选择 `Export YAML`。

浏览器会下载 YAML 文件；单个部署的文件名为 `<deployment-name>.yaml`，多个部署为 `gpustack-deployments-<timestamp>.yaml`。

文件中每个 YAML 文档对应一个部署，以 `---` 分隔。每个文档包含创建部署的请求体及其所属集群，去掉 GPUStack Server 生成的字段：

```yaml
# Exported from GPUStack v2.3.0 at 2026-09-07T10:00:00Z
name: qwen3-8b
cluster_name: production
source: huggingface
huggingface_repo_id: Qwen/Qwen3-8B
replicas: 2
placement_strategy: binpack
worker_selector:
  zone: a
gpu_selector:
  gpu_ids:
  - worker-1:cuda:0
  gpus_per_replica: 1
backend: vLLM
backend_version: 0.11.0
backend_parameters:
- --max-model-len=32768
env:
  HF_TOKEN: hf_xxx
enable_model_route: true
---
name: bge-m3
cluster_name: staging
source: huggingface
huggingface_repo_id: BAAI/bge-m3
replicas: 1
```

由于每个部署都从左侧顶格开始，可以直接从文件中复制单个部署，或手动编写，而无需重新调整缩进。导入也接受单个 YAML 列表形式的文件，这是较早 GPUStack 版本采用的导出格式。

#### 字段参考 { #field-reference }

文件中的每个字段都对应部署表单中的设置。下表按导出文件中的顺序列出字段及其表单名称。字段含义在对应设置处说明，右列提供相应链接。

| 字段 | 部署表单中的名称 |
| --- | --- |
| `name` | `Name` |
| `cluster_name` | `Cluster`，以集群名称而非 ID 写入 |
| `source` | `Source` |
| `huggingface_repo_id`, `huggingface_filename` | Hugging Face 模型的 `Repo ID` 和 `File Name` |
| `model_scope_model_id`, `model_scope_file_path` | ModelScope 模型的 `Repo ID` 和 `File Name` |
| `local_path` | 本地路径模型的 `Model Path` |
| `replicas` | `Replicas` |
| `categories` | [`Model Category`](#model-category) |
| `placement_strategy` | `Scheduling` → [`Auto`](#auto)，放置策略 |
| `cpu_offloading` | [`Allow CPU Offloading`](#allow-cpu-offloading) |
| `distributed_inference_across_workers` | [`Allow Distributed Inference Across Workers`](#allow-distributed-inference-across-workers) |
| `worker_selector` | `Scheduling` → [`Manual`](#manual) → `Selector` |
| `gpu_selector` | `Scheduling` → `Specify GPU` |
| `gpu_type_selector` | `Scheduling` → `Specify GPU Type` |
| `backend` | [`Backend`](#backend) |
| `backend_version` | [`Backend Version`](#backend-version) |
| `backend_parameters` | [`Backend Parameters`](#backend-parameters) |
| `image_name` | 内置或自定义后端的容器镜像；对于内置后端，与 `backend_version` 互斥 |
| `run_command` | 内置 vLLM 或 SGLang 的初始参数，或自定义后端的完整命令 |
| `native_anthropic_api` | `Native Anthropic API` |
| `env` | [`Environment Variables`](#environment-variables) |
| `restart_on_error` | [`Auto-Restart On Error`](#auto-restart-on-error) |
| `extended_kv_cache` | [`Extended KV Cache`](#extended-kv-cache) |
| `speculative_config` | `Performance` → 推测解码 |
| `scaling_schedule` | `Scheduled Scaling` |
| `generic_proxy` | [`Enable Generic Proxy`](#enable-generic-proxy) |
| `lora_list` | [`LoRA Adapters`](#lora-adapters) |
| `enable_model_route` | `Enable Model Route`，表单仅在创建部署时提供 |

最后一组字段 `gpu_selector`、`gpu_type_selector`、`extended_kv_cache`、`speculative_config`、`scaling_schedule` 和 `lora_list` 包含配置块，而不是单个值；此处不展开块结构。请先在部署上配置相应设置，再导出并复制文件中的配置块。这比手工抄写更快，也是确保内容正确的可靠方式。

`description` 和 `distributable` 会随文件传递，但表单中没有对应控件。

文件不包含以下内容：ID、时间戳、就绪副本数和暂停状态等运行时状态、调度器推导的元数据、所属组织、访问策略和 LoRA 运行时路径。启用定时伸缩时，导出的副本数是配置的基准值，而非当前窗口的目标值。GPUStack Server 会在导入时重新生成运行时状态；覆盖一个已启用且暂停计划的现有部署时，会保留其暂停状态。

可以同时选择来自多个集群的部署：每个条目记录自己的 `cluster_name`，因此一个文件就能描述全部部署并完整导入。集群按名称而非 ID 写入，所以即使目标 GPUStack 的集群编号不同（例如重新安装后），文件仍可正确恢复。

但是，一个文件只能涵盖一个组织，无论其中涉及多少集群。部署名称在组织内唯一，而非跨组织唯一，因此跨组织的选择会被拒绝；只有查看 `All` 的平台管理员能进行此类选择，请按组织分别导出。

部署曾经运行的位置绝不会被导出。自动调度的部署在文件中完全没有 `gpu_selector`，导入后会根据目标集群的可用资源重新调度。手动选择的 GPU 代表你的配置意图，而非调度结果，因此会按原样导出，参见下方关于导入到不同集群的说明。

!!! warning

    环境变量会按输入原值导出，可能包含 `HF_TOKEN` 等凭据。不要将导出文件提交到公共仓库；共享前请移除或替换敏感值。

### 导入部署 { #import-deployments }

1. 单击 `Deploy Model` 按钮，然后从下拉菜单中选择 `YAML File`。
2. 选择 `.yaml` 文件。文件是起点，不会先从集群读取内容；系统会根据文件中的部署名称查找现有部署，填充差异视图的只读一侧。选择第二个文件会完整替换第一个文件及其所有编辑。
3. 确定部署的目标位置。抽屉默认的 `Follow the file` 会将每个条目放到其 `cluster_name` 指定的集群，可一次恢复横跨多个集群的备份。未指定集群的条目会回退到其现有部署所在集群。如果改为选择具体集群，则无论文件如何指定，所有条目都会放到该集群。
4. 检查差异。左侧为 GPUStack 当前保存的部署，右侧为将要导入的内容。旁边的列表首先显示完整文档，再逐一列出各部署，并标记 `Create`、`Update` 或 `Unchanged`；`Update` 还会显示将更改的字段数。选择单个条目可单独编辑该部署，选择第一行可同时编辑所有部署。
5. 修正列表中标红的内容。原因会显示在差异视图上方；编辑时会重新检查文档，在停止输入片刻后或离开编辑器时立即触发。
6. 单击 `Import`。如果计划替换任何现有部署，需要确认一次，对话框会列出所有将被替换的部署。所有内容一次性写入；任何条目失败，都不会写入文件中的任何内容。

文档不会产生任何写入时，`Import` 保持禁用。重新导入未修改的导出文件就是这种情况：每个条目都显示为 `Unchanged`，只有你做出的更改才会形成导入操作。

文件本身不会上传，只有编辑器中的文本随导入请求发送。

导入时遵循以下规则：

- **覆盖操作有明确意图且范围受限。** 条目名称匹配现有部署时，仅在导入时确认替换，且该部署已**停止**的情况下才会覆盖。停止意味着副本数已缩容到零，且没有仍在运行的实例。缩容会立即返回，但实例关闭需要一段时间，因此紧接着导入时可能会要求稍候。部署名称在整个组织内唯一，所以条目中的名称总能明确识别一个部署，与其所在集群无关。
- **集群是条目的普通字段。** 在编辑器中修改 `cluster_name`，或为导入选择其他集群，覆盖操作就会将部署移到该集群。由于只能覆盖**已停止**的部署，无需迁移实例；后续新实例会放到当时指定的集群。请同步更新 `gpu_selector` 和 `worker_selector` 以匹配新集群，或移除它们。
- **覆盖是替换部署，而不是合并。** 文件表示期望状态，因此从文件中删除的字段会恢复默认值。移除 `gpu_selector` 块即可使部署恢复自动调度。计划标记为 `Unchanged` 的条目完全不会写入。
- **模型路由遵循 `enable_model_route`。** 覆盖时将其设为 `false`，会删除此部署创建的路由及其 LoRA 子路由。从文件中移除某个 LoRA 适配器，只会删除该适配器的路由，其他路由不受影响。无论哪种情况，如果要删除的路由还服务于其他部署，条目会被拒绝，请先解除其他目标。
- **副本数来自文件。** 如果目标环境的规模与源环境不同，请在编辑器中修改 `replicas`。手动选择 GPU 时，请像编辑部署一样，检查 `gpu_selector` 中仍有足够 GPU 满足新的副本数。启用定时伸缩时，此数值设置 `scaling_schedule.baseline_replicas`，因为副本数由计划控制。
- GPUStack 无法识别的字段会被拒绝，不会静默丢弃。通常表示文件来自更新的 GPUStack 版本；请移除错误中指出的字段并重试。
- **不存在的集群会明确报出，不会猜测。** 从另一个安装导入文件时，如果集群名称不同，每个 `cluster_name` 未知的条目都会被标记。请改成此处已有的集群名称，或选择具体集群放置所有部署。对于尚不存在的部署，如果条目完全没有指定集群，也会同样标记，因为没有可放置的位置。
- **文件决定目标组织。** 查看 `All` 的平台管理员没有选定组织，因此 `Follow the file` 会根据文件指定的集群确定组织。如果文件指定两个组织的集群，或没有指定此 GPUStack 中已有的集群，会被拒绝，不会默认放到某处。请为导入选择具体集群，或先切换到目标组织。
- 条目导入到不同于导出时的集群时，`gpu_selector` 下的 `gpu_ids` 和 `worker_selector` 仍指向原集群的 GPU 和 Worker。请改为新集群中的值，或移除它们，让调度器安排部署；否则导入会因找不到 GPU 而失败。

## 部署修订历史 { #deployment-revision-history }

创建模型部署或更改部署设置时，GPUStack 会记录配置修订，包括从 YAML 导入的更新。保存相同配置、暂停或恢复定时伸缩、重启实例，以及定时伸缩等自动更改，都不会创建修订。现有部署在首次打开历史记录或首次更新配置时会获得初始快照，更早的配置无法恢复。

1. 在模型部署列表中找到部署。
2. 打开其 `Operations` 菜单，选择 `Revision history`。
3. 选择修订，在只读 YAML 差异视图中与紧邻的上一修订比较。例如，选择 `v2` 时，左侧显示 `v1`，右侧显示 `v2`。最新修订仍可查看。对于初始修订，或上一修订已删除或被清理的修订，改为显示完整配置。
4. 在较旧修订的行中单击 `Roll back to this revision`。同一抽屉切换为标题为 `Roll back <name> to vN` 的预览，将当前部署配置与目标修订比较。检查回滚时，修订列表会隐藏。单击 `Back to history` 可返回先前选择和页码。要应用配置，直接在预览中单击 `Roll back`，不会再显示额外确认对话框。最新修订没有回滚操作。配置相同时，回滚被禁用。GPUStack Server 使用与正常编辑相同的规则验证配置。成功回滚且配置发生更改时，会创建新修订。

回滚会恢复部署设置，包括后端配置、环境变量、放置设置、副本数、定时伸缩和 LoRA 适配器。它会保留部署名称、描述、组织、集群、访问策略和修订保留设置。独立模型路由设置不会恢复；派生 LoRA 路由则跟随恢复后的适配器。

如果保存的快照缺少某字段，回滚会保留该字段当前值。回滚预览包含此保留值，以便与将提交的配置一致。快照中显式的 `null` 仍会清除字段。不兼容的配置 schema 变更需要进行历史迁移。

修订视图使用与 YAML 导入导出相同的配置格式：省略未设置的模型字段，字段按 schema 顺序排列。`false`、`0` 和空覆盖等显式值仍会显示。清除设置在差异视图中表现为移除该行。此格式化仅影响显示配置；回滚使用完整保存快照。

预览底部操作按钮旁的左侧提供默认勾选的 `Restart on rollback` 复选框。保存恢复后的配置后，UI 会请求重启部署，重新创建实例。取消勾选可只保存、不重启；配置变更需要重建实例时会显示黄色提示。回滚及任何请求的重启成功后，抽屉关闭，部署列表刷新。如果配置保存成功但重启失败，抽屉保持打开，报告两项结果，并提供 `Retry restart`，此操作不会重复回滚。重试成功后也会关闭抽屉并刷新列表。直接调用回滚 API 仅恢复配置。

最新修订表示最近保存的配置，不保证所有运行中的实例都已使用该配置。重启是异步的，不提供零停机切换。配置有效性在提交回滚时检查，因此如果引用的资源或支持的设置已变化，预览仍可能应用失败。

### 修订保留 { #revision-retention }

默认情况下，部署在最新修订之外保留 **10** 个较旧修订。部署表单未提供此设置，高级用户可通过 API 调整 `revision_history_limit`。设为 **0** 时仅保留最新修订。降低限制会立即从最旧条目开始移除超出的历史记录，不会创建配置修订。

历史抽屉不提供删除单个修订的操作，自动清理遵循部署的保留限制。API 允许删除较旧修订，但最新修订不能删除。删除部署会移除其所有修订。已移除的修订无法恢复。

历史记录与编辑部署具有相同的组织和访问限制，Worker 和集群 SYSTEM 主体无法访问。环境变量按原值存储，因此旧修订可能一直保留凭据，直到该修订被删除或自动清理。

## 查看模型实例 { #view-model-instance }

1. 在部署列表页找到要查看的模型部署。
2. 单击 `>` 符号查看部署的实例列表。

## 删除模型实例 { #delete-model-instance }

1. 在部署列表页找到要查看的模型部署。
2. 单击 `>` 符号查看部署的实例列表。
3. 找到要删除的模型实例。
4. 单击该模型实例 `Operations` 列中的省略号按钮，然后选择 `Delete`。
5. 确认删除。

!!! note

    模型实例删除后，GPUStack 会在需要时重新创建实例，以满足部署的预期副本数。

## 查看模型实例日志 { #view-model-instance-logs }

1. 在部署列表页找到要查看的模型部署。
2. 单击 `>` 符号查看部署的实例列表。
3. 找到要查看的模型实例。
4. 单击该模型实例 `Operations` 列中的 `View Logs` 按钮。

## 性能相关配置 { #performance-related-configuration }

GPUStack 提供以下配置选项来优化模型推理性能。

### 扩展 KV 缓存 { #extended-kv-cache }

可以启用扩展 KV 缓存，将 KV 缓存卸载到 CPU 内存或远程存储。此功能尤其适用于 GPU 内存有限但需要长上下文的环境。底层 GPUStack 会配置所选后端和框架支持的 KV 缓存卸载，例如 [LMCache](https://github.com/LMCache/LMCache) 或 HiCache。

可用选项：

- **RAM-to-VRAM Ratio**：用于 KV 缓存的系统 RAM 与 GPU VRAM 比例。例如，2.0 表示 RAM 中的缓存大小可以是 GPU VRAM 的两倍。
- **Maximum RAM Size**：存储在系统内存中的 KV 缓存最大大小（GiB）。设置后会覆盖 `RAM-to-VRAM Ratio`。
- **Size of Cache Chunks**：每个 KV 缓存 chunk 的 token 数。不适用于 CANN 上的 vLLM，因为其连接器卸载完整 KV 块，并按引擎块大小复用。

此功能仅适用于特定后端和框架。

#### 兼容性矩阵 { #compatibility-matrix }

| 后端 | 框架 |
| --- | --- |
| vLLM | CUDA, ROCm, CANN |
| SGLang | CUDA, ROCm, CANN |

对于 CANN 上的 vLLM，扩展 KV 缓存需要 vLLM Ascend v0.21.0 或更高版本，且不支持 Atlas 300I Duo（310P）。

## 调度配置 { #scheduling-configuration }

### 调度模式 { #schedule-mode }

#### 自动 { #auto }

GPUStack 根据当前资源可用情况，将模型实例自动调度到合适的 GPU/Worker。

- **放置策略**

Spread：使整个集群的资源相对均匀地分布在所有 Worker 上，可能使单个 Worker 产生更多资源碎片。

Binpack：优先考虑集群资源的整体利用率，减少 Worker/GPU 上的资源碎片。

- **Worker 选择器**

配置后，调度器会将模型实例部署到包含指定标签的 Worker。

1. 进入 `Workers` 页面，编辑目标 Worker。在标签部分添加自定义标签，为 Worker 分配标签。

2. 进入 `Deployments` 页面，单击 `Deploy Model` 按钮。展开 `Scheduling` 部分，在 `Worker Selector` 配置中输入此前分配的 Worker 标签。部署时，模型实例会根据这些标签分配到对应 Worker。

#### 手动 { #manual }

此调度类型允许用户指定模型实例部署在哪个 GPU 上。

- **GPU 选择器**

  从列表中选择一个或多个 GPU。资源允许时，模型实例会尝试部署到所选 GPU。

- **每副本 GPU 数**

Auto：系统自动计算每个副本的 GPU 数量，默认使用 2 的幂，并以所选 GPU 数量为上限。

Manual：从下拉菜单中选择每个副本使用的 GPU 数量。

## 高级配置 { #advanced-configuration }

GPUStack 支持针对模型部署进行定制配置。

### 模型类别 { #model-category }

模型类别便于组织和筛选模型。GPUStack 默认根据模型元数据自动检测类别，也可以从下拉列表中选择来自定义类别。

### 后端参数 { #backend-parameters }

输入运行模型时要自定义的后端参数。支持以下格式：

| 方式 | 示例 | 备注 |
| --- | --- | --- |
| 等号分隔 | `--hf-overrides={"architectures": ["NewModel"]}` | - |
| 空格分隔 | `--hf-overrides '{"architectures": ["NewModel"]}'` | 支持 `shell-like` 风格拆分（例如包含空格的值）。 |
| 独立字段 | `--max-model-length`, `8192` | 将参数名称和值作为两个独立条目输入。 |

支持的完整参数列表参见[推理后端](built-in-inference-backends.md)。

### 环境变量 { #environment-variables }

运行模型时使用的环境变量，在启动时传递给后端进程。

### LoRA 适配器 { #lora-adapters }

LoRA（Low-Rank Adaptation）是一种参数高效微调方法，通过加载小型适配器文件，使基础模型适应特定领域，而无需重新训练全部权重。GPUStack 支持在单个已部署的基础 LLM 上挂载多个 LoRA 适配器，并自动为每个适配器创建一个模型路由。

!!! note

    LoRA 仅支持 `vLLM`、`SGLang` 和昇腾 MindIE（`Ascend MindIE`）后端。其他后端会忽略 `LoRA Adapters` 配置。

展开部署表单的 `Advanced` 部分，找到 `LoRA Adapters`，逐一添加适配器：

- 从下拉菜单选择适配器。列表支持搜索。如果未显示目标适配器，请先在 `Hugging Face` 或 `ModelScope` 确认模型确实提供该适配器，然后将其仓库 ID 粘贴到搜索框进行选择。
- `LoRA name`：输入不带前缀的适配器名称，例如 `alpaca`。

部署后，GPUStack 为每个 LoRA 适配器自动创建模型路由，名称为 `<base-model-name>:<adapter-name>`。要切换适配器，请将兼容 OpenAI 的 API 的 `model` 字段设置为对应路由名称。所有适配器共享同一 GPU 实例。GPUStack 会自动注入 `--enable-lora` 等后端启动参数，因此通常无需在 `Backend Parameters` 中重复设置。

!!! note

    更改已部署模型的 LoRA 配置后，必须重启对应实例，新配置才会生效。

完整流程（包括调用示例和各后端兼容性细节）参见[使用 LoRA 适配器提供模型服务](../tutorials/serving-with-lora-adapters.md)教程。

### 允许 CPU 卸载 { #allow-cpu-offloading }

!!! note

    仅自定义后端可用。

启用 CPU 卸载后，如果 GPU 资源不足，GPUStack 会分配 CPU 内存。必须正确配置推理后端，使其使用 CPU+GPU 混合推理或纯 CPU 推理。

### 允许跨 Worker 分布式推理 { #allow-distributed-inference-across-workers }

!!! note

    适用于 vLLM、SGLang 和 MindIE 后端。

启用跨多个 Worker 的分布式推理。主模型实例会与其他一个或多个 Worker 上的后端实例通信，将计算任务卸载给它们。

### 出错时自动重启 { #auto-restart-on-error }

启用后，模型实例遇到错误时会自动重启，以确保高可用性和可靠性。出错时，GPUStack 会采用指数退避策略自动尝试重启模型实例。重启尝试之间的延迟指数增长，最大间隔为 5 分钟。这种方式可以防止持续错误情况下的频繁重启使系统过载。

### 启用通用代理 { #enable-generic-proxy }

集成兼容 OpenAI 的 API 是常见做法，但用户在不同场景下可能有不同需求。GPUStack 也支持兼容 OpenAI 之外的任意推理 API，为 AI 应用开发提供更多灵活性。

启用 Generic Proxy 后，GPUStack 提供两种指定目标模型的方式。建议使用**基于路径的形式**；**基于请求头的形式**仅为向后兼容保留，已弃用。

#### 基于路径的形式（推荐） { #path-based-form-recommended }

将数值型模型路由 ID 追加到代理前缀，即 `/model/proxy/<model_route_id>/<upstream-path>`，GPUStack 会将请求分发到该路由的目标，无需额外请求头。

```bash
# Assume the model route id is 42.
curl http://<server-url>/model/proxy/42/embed \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <GPUSTACK_API_KEY>" \
  -d '{"inputs":["What is Deep Learning?", "Deep Learning is not..."]}'
```

网关在转发前会去掉 `/model/proxy/<id>`，因此上游推理服务器看到的是：

```bash
curl http://<inference-server-url>/embed \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"inputs":["What is Deep Learning?", "Deep Learning is not..."]}'
```

模型路由 ID 在重命名后保持不变，可在模型详情页查看，或通过 `GET /v2/model-routes` 获取。

#### 基于请求头的形式（已弃用） { #header-based-form-deprecated }

!!! warning "已弃用"
    `/model/proxy` + `X-GPUStack-Model` 形式仅为向后兼容保留，将在未来版本移除。请迁移到上方基于路径的形式。

```bash
curl http://<server-url>/model/proxy/embed \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <GPUSTACK_API_KEY>" \
  -H "X-GPUStack-Model: bge-m3" \
  -d '{"inputs":["What is Deep Learning?", "Deep Learning is not..."]}'
```

转发前会去掉路径前缀 `/model/proxy`。必须提供 `X-GPUStack-Model` 请求头，或 JSON 请求体中的 `model` 属性，让网关可以解析目标模型。上游推理服务器看到的请求如下：

```bash
curl http://<inference-server-url>/embed \
  -X POST \
  -H "Content-Type: application/json" \
  -H "X-GPUStack-Model: bge-m3" \
  -d '{"inputs":["What is Deep Learning?", "Deep Learning is not..."]}'
```
