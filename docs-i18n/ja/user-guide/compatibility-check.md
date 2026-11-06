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

#### ルックアップテーブルのオフロード { #offloaded-lookup-tables }

!!! note

    ルックアップテーブルの自動推定は、対応済みの vLLM および SGLang アーキテクチャ、形式、オフロード設定のみを対象とします。ネイティブ MindIE、custom バックエンド、および未対応の vLLM Ascend の処理経路は対象外です。自動推定がデプロイの想定に合わない場合は、[リソース要件のカスタマイズ](#custom-resource-requirements)で調整できます。

Engram や PLE など、ルックアップテーブルを CPU メモリにオフロードする対応モデルでは、GPUStack はモデル設定と精度のメタデータからテーブルのサイズを推定します。上記の VRAM の式を適用する前に、オフロードするバイト数を `WEIGHT_SIZE` から差し引きます。また、各参加 Worker で、ローカル拡張 KV キャッシュ用 RAM に加えてテーブル用 RAM を予約します。

RAM の推定では、同じ Worker 上のデータ並列レプリカがテーブルを共有すると仮定し、並列構成に必要な独立したコピーを計上します。追加の余裕は含まれないため、実際のメモリ使用量が推定値を超える場合があります。テーブルのサイズやオフロード動作を特定できない場合は、全重みに基づく VRAM 推定を維持します。

#### リソース要件のカスタマイズ { #custom-resource-requirements }

自動推定がデプロイの実際の要件に合わない場合は、モデルの[リソース要件の環境変数](../environment-variables.md#model-deployment-configuration)で RAM と VRAM の要件をバイト単位で指定できます。RAM の指定値は、テーブルとローカルキャッシュのメモリを含む Worker ごとの総量の推定に置き換わります。これらの指定はスケジューリングに使用され、バックエンドの実際のメモリ使用量は設定しません。RAM と VRAM は個別に指定できます。指定されていないリソースには、各バックエンドの推定値が引き続き使用されます。

たとえば、Worker ごとに 64 GiB の RAM、モデル全体で 32 GiB の VRAM が必要と宣言するには、次のモデル環境変数を追加します（`1 GiB = 1024^3 bytes`）。VRAM の値は、使用する GPU 全体にわたるモデルの推定要件であり、GPU ごとの予約量ではありません。

```text
GPUSTACK_MODEL_RAM_CLAIM=68719476736
GPUSTACK_MODEL_VRAM_CLAIM=34359738368
```
