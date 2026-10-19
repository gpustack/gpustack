---
translation_status: translated
---

# 基准测试 { #benchmarking }

GPUStack 可以对运行中的模型实例执行基准测试。基准测试由 Worker 在专用基准测试容器镜像中执行，结果和日志存储在 Worker 上。

## 前提条件 { #prerequisites }

- 模型实例正在运行且状态正常。

## 创建基准测试 { #create-benchmark }

1. 进入 `Benchmarks` 页面。
2. 单击 `Add Benchmark`。
3. 选择实例并填写配置。
4. 单击 `Save` 按钮。

### ShareGPT 工作负载长度 { #sharegpt-workload-lengths }

对于内置 ShareGPT 数据集，可以选择设置输入 token 长度的最小值、最大值，或同时设置两者。GPUStack 会保留经分词器计算后 token 数满足指定限制的首轮提示词。输出 token 长度为可选项：设置后，所有保留的提示词都会请求相同的最大输出长度；未设置时，每条提示词使用数据集中参考答案的长度。如果没有提示词落在指定范围内，本次运行会在发送基准测试请求前失败。

## 查看基准测试结果 { #view-benchmark-results }

1. 进入 `Benchmarks` 页面。
2. 找到要查看的基准测试。
3. 单击基准测试名称，查看结果和配置快照。

## 导出基准测试结果 { #export-benchmark-results }

1. 进入 `Benchmarks` 页面。
2. 找到要导出的基准测试。
3. 选择一个或多个要导出的基准测试运行记录。
4. 单击 `Export` 按钮下载结果。

## 编辑基准测试 { #edit-benchmark }

1. 进入 `Benchmarks` 页面。
2. 找到要编辑的基准测试。
3. 单击 `Operations` 列中的 `Edit` 按钮。
4. 根据需要修改名称和描述。
5. 单击 `Save` 按钮。

## 删除基准测试 { #delete-benchmark }

1. 进入 `Benchmarks` 页面。
2. 找到要删除的基准测试。
3. 单击 `Operations` 列中的 `Delete` 按钮。
4. 确认删除。

## 使用私有 CA 的 HTTPS GPUStack Server 上的基准测试 { #benchmarks-on-an-https-server-with-a-private-ca }

基准测试容器会通过 GPUStack Server URL 向 GPUStack Server 报告进度，因此在 HTTPS 部署中必须验证 GPUStack Server 的证书。基准测试运行在独立镜像中，不会自行导入 CA，所以 Worker 会将自身信任的 CA 包传给它，其中包括挂载到 `/usr/local/share/ca-certificates/` 下的私有 CA（参见[额外受信任的 CA](../installation/installation.md#additional-trusted-cas)）。只要该 CA 在 Worker 上可用，就无需额外配置。

如果验证仍然失败，基准测试本身不受影响，因为负载是通过明文 HTTP 向模型实例生成的；但进度会一直停留在 0%，直到运行结束，Worker 日志中会记录原因。请在 Worker 上导入 GPUStack Server 的 CA 来解决此问题。
