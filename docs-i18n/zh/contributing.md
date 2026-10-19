---
translation_status: translated
---

# 为 GPUStack 做贡献 { #contributing-to-gpustack }

感谢你抽出时间为 GPUStack 做贡献！

请阅读并遵守[行为准则](./code-of-conduct.md)。

## 提交 issue { #filing-issues }

如果发现错误或遇到问题，请先搜索已有 issue，其他人可能遇到过相同问题，或我们已经在处理。

如果找不到相关内容，请提交 issue 联系我们。为帮助诊断和解决问题，请尽可能提供详细信息，包括：

- 软件：GPUStack 版本、安装方式、操作系统信息等。
- 硬件：节点信息、GPU 信息等。
- 复现步骤：尽可能详细地说明如何出现所报告的问题。
- 日志：提供相关日志，例如 Server 日志、Worker 日志等。

## 贡献代码 { #contributing-code }

开发环境的配置方法，请参阅[开发指南](./development.md)。

对于小问题，可以直接提交 PR。如果计划提交较大的 PR，以实现新功能或修复较复杂的问题，请先创建 issue，说明改动及其动机。如果处理的是错误，还请说明复现方法。

## 更新文档 { #updating-documentation }

如果需要更新文档，欢迎提交带有 `documentation` 标签的 issue，或直接提交 PR。

英文文档 `docs/` 是内容标准。简体中文、日语、俄语和土耳其语译文分别放在 `docs-i18n/zh/`、`docs-i18n/ja/`、`docs-i18n/ru/` 和 `docs-i18n/tr/` 下的相同相对路径中；图片等公共资源继承自 `docs/`。翻译标题时保留英文标题 ID（例如 `## 安装 { #installation }`），以保持深链和语言切换一致。

导航与主题标签在相应的 `docs-i18n/mkdocs/zh.yml`、`docs-i18n/mkdocs/ja.yml`、`docs-i18n/mkdocs/ru.yml` 和 `docs-i18n/mkdocs/tr.yml` 中翻译。翻译时保留命令、配置键、代码示例、链接和模型名称。没有译文，或元数据标记为 `translation_status: pending` 的页面，会显示英文原文并附上相应语言的提示。

运行 `make build-docs` 构建全部五种语言。使用 `mkdocs serve` 或 `make serve-docs` 实时预览全部五种语言；修改 `docs-i18n/` 下的译文或语言配置会自动重建预览。预览中的英文路径为 `/`，其他语言为 `/zh/`、`/ja/`、`/ru/` 和 `/tr/`。
`bash hack/build-docs.sh --offline` 将内置文档构建到 `gpustack/help/`，GPUStack 通过 `/help/` 和对应的 `/help/<language>/` 路径提供访问。每种语言拥有独立的本地搜索索引，日语搜索使用主题自带的分词器，可离线使用。

普通 `mkdocs build` 和 `mike deploy` 也会通过已配置的文档钩子构建全部语言。
多语言文件保留在同一个版本目录内，例如 `/latest/ja/` 或 `/2.3/ja/`。
请使用现有发布流程发布完整的站点产物；仅完成构建不会自动发布文档。
