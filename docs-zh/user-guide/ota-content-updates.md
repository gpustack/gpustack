---
translation_status: translated
---

# OTA 内容更新 { #ota-content-updates }

GPUStack 从官方 OTA 服务器获取模型目录、社区后端市场和内置后端版本，因此新模型和后端版本可以在不升级或重启的情况下提供给运行中的 GPUStack 服务。

更新默认开启，内容来自单个主机 `https://ota.gpustack.ai/latest`。如果 GPUStack 服务运行在受限网络中，只需允许访问该主机。也可以切换到由你自行维护、在该网络中可访问的文件。第三种方式是部署自己的 OTA 服务器副本，并在启动 GPUStack 时使用 `--ota-server-url` 指向它。这三类内容仍会自动更新，只是改为从你的副本读取。参见 [start](../cli-reference/start.md)。

来源在提供相应内容的页面上配置。模型目录位于 `Model Service > Catalog` 页面，两类后端内容位于 `Model Service > Inference Backends` 页面。它们都使用仅平台管理员可见的 `Manage Sources` 按钮。

## OTA 更新的内容 { #what-updates-over-the-air }

| 内容 | OTA 服务器发布的内容 |
| --- | --- |
| 模型目录 | 新增和修订的目录条目 |
| 社区后端市场 | 新增和修订的社区后端定义 |
| 内置后端版本 | vLLM、SGLang 等内置后端的新版本及其镜像 |

每类内容同时只由一个**来源**提供，一个来源文件包含该类的全部内容。保存时，它会替换当前提供的内容，而不是追加，因此新文件中缺失的条目也会停止提供。

| 来源 | 提供的内容 | 自动更新 |
| --- | --- | --- |
| `URL` | GPUStack Server 获取并存储的文件。 | 可选 |
| `Yaml File` | 粘贴的文件，仅模型目录支持。 | 否 |
| `Embedded` | 仅当前版本随包提供的内容，无需网络访问。 | 否 |

来源留空时，该类内容继续跟随 OTA 服务器，在 UI 中标记为 `Official`。配置自己的 URL 或 Yaml 文件后，该类内容将停止跟随 OTA 服务器，改用你提供的内容。

自定义 `Source URL` 必须满足以下条件：

- 必须使用 `http(s)`，且不能包含凭据。
- 必须直接指向文件本身，而不是仓库页面。
- 文件大小不能超过 4 MB，且必须采用 UTF-8 编码。

文件由 GPUStack Server 获取，Worker 无需访问该地址。

## 配置目录来源 { #configure-the-catalog-source }

1. 进入 `Model Service > Catalog` 页面。
2. 单击 `Manage Sources` 按钮。
3. 在 `Catalog Source` 抽屉中选择以下两种来源类型之一：

    - `URL`：填写 `Source URL`。

        ![截图：模型目录来源抽屉使用 URL 来源](../assets/ota-content-updates/1-catalog-source-url.png)

    - `Yaml File`：将内容粘贴到 `Content` 中，或单击 `Import` 将文件加载到编辑器。

        ![截图：模型目录来源抽屉中的 YAML 文件编辑器](../assets/ota-content-updates/2-catalog-source-yaml.png)

4. 单击 `Save`。

!!! note

    请从现有内容开始修改，不要使用空文件。在 `Yaml File` 中，单击 `Official File` 下载 OTA 服务器当前提供的内容。在 `URL` 中，`Source URL` 输入框已填入当前跟随的地址，可直接复制。

## 配置后端来源 { #configure-the-backend-sources }

两类后端内容独立更新，各有自己的来源。同一个抽屉通过两个面板分别配置：

1. 进入 `Model Service > Inference Backends` 页面。
2. 单击 `Manage Sources` 按钮。
3. 在 `Backend Source` 抽屉中展开所需面板：
    - `Built-in backend source`：内置后端（vLLM、SGLang、MindIE、VoxBox）的版本和镜像。
    - `Community backend source`：社区后端市场。
4. 选择 `URL` 并填写 `Source URL`。两类都只支持 URL，不支持粘贴内容。
5. 单击相应面板中的 `Save` 按钮。各面板独立保存。

![截图：后端来源抽屉中的两个面板均已展开](../assets/ota-content-updates/3-backend-source.png)

!!! warning

    如果更新会移除部署仍在使用的后端版本，保存会被拒绝，提示消息会列出受影响模型。请先将这些模型缩容到零，或迁移到新文件仍包含的版本，然后再次保存。

!!! note

    如果社区后端已启用，或包含手动添加的版本，即使新文件不再包含它，也会将其保留为自定义后端，确保其上的模型继续运行。如果某个来源再次发布它，它会恢复为社区后端，但需要自行重新启用。如果后端未启用且未添加自定义版本，则会被移除，因此此前在其上编辑的 `default_env` 不会在它恢复时还原。

## 控制更新计划 { #control-the-update-schedule }

每个来源都有独立的更新设置：

- `Auto-update`：是否按固定间隔自动检查更新。跟随 OTA 服务器时默认开启。切换到自己的 `URL` 后默认关闭，需要自行勾选。
- `Update Interval (hours)`：勾选 `Auto-update` 后显示。它是自动检查之间的间隔，默认为 12 小时，可以修改。
- `Update Now`：立即检查一次，无需等待更新间隔。

修改 `Source URL` 后，请单击 `Save`。它会保存新地址并立即获取内容。内容未变化时，不会重复下载。

!!! note

    OTA 服务器暂时不可达不会影响服务。已获取的内容会继续提供，错误会被记录，并在一小时内进行下一次尝试。从未访问过 OTA 服务器的集群会提供内置内容。内置内容是完整的，只是不包含当前 GPUStack 版本发布之后新增的内容。

## 停止更新并使用内置内容 { #stop-updates-and-serve-embedded-content }

取消勾选 `Auto-update` 只会停止检查，已获取的内容仍会继续提供。要完全停止使用已获取的内容，请选择 `Embedded` 并保存。该类内容随后只使用当前版本随包提供的内容，与全新离线安装相同。

例如，目录条目不适合你的集群，或某些后端版本的镜像无法在当前环境中拉取时，可以切换到 `Embedded`。

- **不会丢失配置。** 输入的 URL 和内容都会保留，只是不再使用。再次选择 `URL` 或 `Yaml File` 即可恢复。
- **无需网络。** 即使不再想使用的文件已经无法访问，也可以切换到 `Embedded`。
- 它仅影响这一类内容，在每个 GPUStack Server 上立即生效，无需重启。
- 单击 `Reset to Official Source` 可恢复跟随 OTA 服务器。这会清除你输入的内容。

!!! warning

    切换到 `Embedded` 后，可用的后端版本会减少，因此同样可能被拒绝。切换前，请确认没有部署仍在使用仅存在于已获取内容中的版本。
