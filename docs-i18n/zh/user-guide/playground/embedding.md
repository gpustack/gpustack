---
translation_status: translated
---

# Embedding Playground { #embedding-playground }

Embedding Playground 用于测试模型将文本转换为向量嵌入的能力。可以尝试多个文本输入、可视化向量嵌入，并查看集成 API 的代码示例。

## 添加文本 { #add-text }

添加至少两条文本，然后单击 `Submit` 按钮生成向量嵌入。

## 批量输入文本 { #batch-input-text }

启用 `Batch Input Mode` 后，会按换行符自动将多行文本拆分为独立条目，便于在一次操作中处理多段文本。

## 可视化 { #visualization }

使用 PCA（主成分分析）对向量嵌入结果降维，并在二维图上显示。结果可通过两种形式查看：

1. **Chart**——以图表形式显示 PCA 结果。
2. **JSON**——以 JSON 格式查看原始向量嵌入。

图中点之间的距离代表对应文本之间的相似程度。点越接近，相似度越高。

![向量嵌入](../../assets/playground/embedding.png)

## 清空 { #clear }

单击 `Clear` 按钮，重置文本条目并清空输出。

## 选择模型 { #select-model }

单击 Playground 界面右上角的模型下拉菜单，选择 GPUStack 中可用的模型。

## 查看代码 { #view-code }

尝试不同文本输入后，单击 `View Code` 按钮，查看如何使用相同输入调用 API。提供 `curl`、`Python` 和 `Node.js` 代码示例。
