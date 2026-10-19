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

English documentation lives in `docs/`. Add Simplified Chinese translations at the
same relative paths under `docs-zh/`; images and other shared assets are inherited
from `docs/`. Preserve the English heading IDs in translated headings (for example,
`## 安装 { #installation }`) so links and language switching remain consistent.
Navigation labels and theme labels are translated in `mkdocs.zh.yml`. Pages without
a translation, or with `translation_status: pending` in their front matter, display
the English source with a notice.

Run `make build-docs` to build both languages, or `make serve-docs` to preview them
with live reload. English is served at `/` and Chinese at `/zh/` in the preview.
`bash hack/build-docs.sh --offline` builds the bundled site at `gpustack/help/`,
served by GPUStack at `/help/` and `/help/zh/`. Each language has its own search index.
