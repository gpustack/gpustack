# 互換性チェック { #compatibility-check }

GPUStack はモデルのデプロイ前に互換性をチェックします。現在の GPUStack 環境とモデルの互換性に関する詳細情報を確認できます。次のチェックを実行します。

## 推論バックエンドの互換性 { #inference-backend-compatibility }

選択した推論バックエンドが、オペレーティングシステム、GPU、アーキテクチャなど、現在の環境と互換性を持つかを確認します。

## モデルの互換性 { #model-compatibility }

選択した推論バックエンドがモデルに対応しているかを判定します。対応するモデル形式とアーキテクチャ（`LlamaForCausalLM`、`Qwen3ForCausalLM` など）を確認します。このチェックは、組み込み推論バックエンドとその対応モデルに基づきます。カスタムバックエンドまたはカスタムバージョンを使用する場合は、このチェックをスキップします。

## スケジュール可能性のチェック { #schedulability-check }

現在の環境でモデルをスケジュールできるかを評価します。RAM や VRAM などの空きリソースと、設定されたスケジューリングルールを確認します。

### スケジューリングルール { #scheduling-rules }

Worker セレクター、GPU セレクター、スケジューリングポリシーなどのルールに基づき、現在の環境でモデルをスケジュールできるかを判定します。

### リソースのチェック { #resource-check }

モデルのデプロイに十分なシステムリソースがあることを確認します。GPUStack は必要なリソースを推定し、環境内の空きリソースと比較します。推定には次の方法を使用します。

1. **GGUF モデル**：GPUStack は [GGUF パーサー](https://github.com/gpustack/gguf-parser-go)を使用して、モデルに必要なリソースを推定します。
2. **その他のモデル**：次の式で VRAM 使用量を推定します。

$$
\text{VRAM} = \text{WEIGHT\_SIZE} \times 1.2 + \text{FRAMEWORK\_FOOTPRINT}
$$

- `WEIGHT_SIZE` は、バイト単位のモデルの重みのサイズです。
- `FRAMEWORK_FOOTPRINT` は、フレームワークのメモリオーバーヘッドを表す定数です。たとえば vLLM は CUDA グラフに数 GB の VRAM を使用する場合があります。
- 1.2 倍という係数は経験的な推定値です。詳細は[こちらの説明](https://blog.eleuther.ai/transformer-math/#total-inference-memory)を参照してください。

この式は概算であり、すべてのモデルで正確とは限りません。通常、必要な VRAM の下限に近い推定値となります。推定が不十分な場合は、Worker と GPU を手動で選択するか、高度なバックエンドパラメーターを調整して、より細かくスケジューリングできます。たとえば vLLM では、`--tensor-parallel-size` と `--pipeline-parallel-size` でモデルへの GPU 割り当てを制御できます。
