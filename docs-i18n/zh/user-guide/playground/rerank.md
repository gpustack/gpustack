---
translation_status: translated
---

# Rerank Playground { #rerank-playground }

Rerank Playground 用于测试重排序模型，根据多条文本与查询的相关性重新排列文本。可以尝试不同输入文本、自定义参数，并查看集成 API 的代码示例。

## 添加文本 { #add-text }

在文档中添加多条文本，用于重排序。

## 批量输入文本 { #batch-input-text }

启用 `Batch Input Mode` 后，会按换行符将多行文本拆分为独立条目，便于高效处理多段文本。

## 清空 { #clear }

单击 `Clear` 按钮，重置文档和查询结果。

## 查询 { #query }

输入查询并单击 `Submit` 按钮，获取按查询相关性排序的文本列表。

![重排序](../../assets/playground/reranker.png)

## 选择模型 { #select-model }

单击 Playground 界面右上角的模型下拉菜单，选择 GPUStack 中可用的重排序模型。

## 自定义参数 { #customize-parameters }

在参数部分设置 `Top N`，指定要检索的匹配文本数量。

## 查看代码 { #view-code }

尝试不同输入文本和查询后，单击 `View Code` 按钮，查看如何使用相同输入调用 API。提供 `curl`、`Python` 和 `Node.js` 代码示例。
