---
translation_status: translated
---

# 决策 Playground { #decision-playground }

决策 Playground 用于测试根据给定情境回答类型化问题的决策模型。编辑 state 与 questions，对已部署的决策模型发起请求，查看类型化的答案，并获取用于 API 集成的代码示例。

## State { #state }

`State` 编辑器存放做出决策所依据的情境——纯文本，或 chat messages 这类 JSON 对象/数组。编辑器会实时校验 JSON；state 是运行的必填项。

## Questions { #questions }

`Questions` 编辑器存放针对 state 提出的问题，是一个将问题 id 映射到问题定义的 JSON 对象。使用 `+ choice`、`+ score`、`+ noul` 按钮插入可直接运行的模板：

- `choice`——从 `criteria`（选项名 → 描述）中选择一个选项。
- `score`——按 `criteria` 中的有序级别列表对 state 评分。
- `noul`——二元（是/否）问题。

编辑器会校验 JSON 并提供 schema 提示，但问题集保持开放——引擎专属字段会原样透传。

## Run { #run }

选择决策模型后，点击 `Run` 按钮或按 `Ctrl`/`Cmd` + `Enter`，将 state 和 questions 发送到 `/v1/systemone`。点击 `Stop` 取消正在运行的请求。

![决策 Playground](../../assets/playground/decision.png)

## Answers { #answers }

每个答案以其问题 id 返回，渲染为卡片：判定结果（`Choice`、`Score`，或 `noul` 的是/否标签）、`Confidence` 以及各选项的概率。卡片上方的摘要行显示回答的模型与 token 用量。无法识别的答案形态会回退为原始 JSON 视图。

## 选择模型 { #select-model }

点击 Playground UI 右上角的模型下拉框，选择 GPUStack 中可用的决策模型。列表中只显示 `decision` 类别的模型——如果模型缺失，请确认它正在运行且类别正确；必要时在模型设置中调整类别。

## 查看代码 { #view-code }

编辑好 state 和 questions 后，点击 `View Code` 按钮查看用相同输入调用 API 的方式。代码示例提供 `curl`、`Python` 和 `Node.js` 三种。
