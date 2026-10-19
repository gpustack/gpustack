---
translation_status: translated
---

# 管理模型路由 { #managing-model-routes }

本指南介绍模型路由的使用方法，涵盖常见场景及其配置方式。

## 部署模型时创建路由 { #create-route-when-deploying-model }

部署模型时，默认勾选 `Enable Model Route`，为该部署创建同名模型路由。部署完成后，用户即可通过相同名称访问模型。

## 通过模型别名升级模型 { #model-upgrade-with-model-alias }

模型发布新版本时，管理员可以在保持模型名称不变的情况下升级：部署新版本时禁用模型路由，再编辑现有路由切换流量。

1. 为新版本模型向 GPUStack 添加足够的 GPU 资源。
2. 部署新版本模型，在部署抽屉中取消勾选 `Enable Model Route`。
3. 在 `Routes` 页面找到指向旧版本模型的路由。
4. 编辑路由，将目标替换为新版本模型部署。
5. 此后发往该模型路由的新请求会转发给新版本模型。

## 同时使用自托管模型和公共 MaaS 模型 { #serve-model-for-self-host-models-and-public-maas-models }

自托管模型请求量上升时，可能出现延迟。如果没有可用于扩容的资源，可引入公共 MaaS。在模型路由中同时配置部署模型目标和提供商模型目标，并分配权重，即可由公共 MaaS 分担当前模型的访问负载。

1. 打开 `Providers` 页面。
2. 按需添加提供商，并选择要使用的模型。
3. 打开 `Routes` 页面，找到待编辑的模型。
4. 编辑该模型，为模型路由添加目标。
5. 同一路由中可以同时选择 GPUStack `Deployments` 中的模型和 `Providers` 中的模型作为目标。
6. 各目标的权重决定流量比例，默认权重为 100。例如，目标 A 权重为 100，目标 B 权重为 200，则 33% 的请求转发给 A，67% 转发给 B。

## 模型路由回退 { #model-route-fallback }

将公共 MaaS 模型设为路由目标很方便，但也可能产生较高费用。流量分配规则始终生效，即使自托管模型负载不高，也会按配置向公共 MaaS 转发流量。此时可以使用模型路由的 `Fallback` 功能。

1. 在 `Routes` 页面找到需要配置回退的模型路由。
2. 编辑路由，配置 `Fallback Route Target`。与其他路由目标一样，它可以来自 GPUStack `Deployments`，也可以来自 `Providers`。
3. 回退目标与流量分配策略互斥，因此不能为它配置权重。

## 通过模型路由代理 OpenAI 兼容推理服务 { #proxy-openai-compatible-inference-service-via-model-route }

已有推理服务，例如 ollama 或 lm-studio，如需使用 GPUStack 的代理、访问控制和 token 用量统计，可以创建使用自定义路径的 `OpenAI` Model Provider。

1. 打开 `Providers` 页面，点击 `Add Provider`。
2. 将类型设为 `OpenAI`，将 `Custom Base URL` 设为 OpenAI 兼容推理服务的地址，格式为 `http://<ip>:<port>/v1`。按需填写名称、API 密钥和说明。
3. 为提供商添加模型。如果推理服务支持 `/v1/models` API，界面会列出可选模型。
4. 点击 `Save`。
5. 在该提供商的 `Operations` 列中点击 `Add Route`。
6. 默认使用提供商的第一个模型预填路由配置，可按需调整。
7. 点击 `Save` 应用模型路由，此时 GPUStack 已开始代理该模型。
8. 在 `Operations` 列的 `Access Setting` 中授权对该路由的访问。
