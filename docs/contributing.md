# Contributing to GPUStack

Thanks for taking the time to contribute to GPUStack!

Please review and follow the [Code of Conduct](./code-of-conduct.md).

## Filing Issues

If you find any bugs or are having any trouble, please search the reported issue as someone may have experienced the same issue, or we are actively working on a solution.

If you can't find anything related to your issue, contact us by filing an issue. To help us diagnose and resolve, please include as much information as possible, including:

- Software: GPUStack version, installation method, operating system info, etc.
- Hardware: Node info, GPU info, etc.
- Steps to reproduce: Provide as much detail on how you got into the reported situation.
- Logs: Please include any relevant logs, such as server logs, worker logs, etc.

## Contributing Code

For setting up development environment, please refer to [Development Guide](./development.md).

If you're fixing a small issue, you can simply submit a PR. However, if you're planning to submit a bigger PR to implement a new feature or fix a relatively complex bug, please open an issue that explains the change and the motivation for it. If you're addressing a bug, please explain how to reproduce it.

## Updating Documentation

If you have any updates to our documentation, feel free to file an issue with the `documentation` label or make a pull request.

English documentation in `docs/` is the source of truth. Maintain Simplified Chinese, Japanese, Russian, and Turkish translations at matching paths under `docs-i18n/zh/`, `docs-i18n/ja/`, `docs-i18n/ru/`, and `docs-i18n/tr/`. Images and other shared assets are inherited from `docs/`. Preserve the English heading IDs in translated headings (for example, `## 安装 { #installation }`) so links and language switching remain consistent.

Translate navigation and theme labels in the corresponding `docs-i18n/mkdocs/zh.yml`, `docs-i18n/mkdocs/ja.yml`, `docs-i18n/mkdocs/ru.yml`, and `docs-i18n/mkdocs/tr.yml` files. Preserve commands, configuration keys, code examples, links, and model names when translating from English. Pages without a translation, or with `translation_status: pending` in their front matter, display the English source with a notice.

Run `make build-docs` to build all five languages. Preview them with live reload using `mkdocs serve` or `make serve-docs`; changes to translations and language configuration under `docs-i18n/` rebuild the preview. English is served at `/`; translated sites use `/zh/`, `/ja/`, `/ru/`, and `/tr/`.
`bash hack/build-docs.sh --offline` builds the bundled site at `gpustack/help/`, served by GPUStack at `/help/` and the corresponding `/help/<language>/` paths. Each language has its own local search index. Japanese search uses the tokenizer bundled with the documentation theme and works offline.

Ordinary `mkdocs build` and `mike deploy` also build all languages through the
configured documentation hooks. Versioned deployments keep translations inside
the same version directory, for example `/latest/ja/` or `/2.3/ja/`. Use the existing
release process to publish the complete generated site; a build alone does not
publish documentation.
