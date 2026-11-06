# 組み込みの推論バックエンド { #built-in-inference-backends }

GPUStack は次の推論バックエンドをサポートしています。

- [vLLM](#vllm)
- [SGLang](#sglang)
- [MindIE](#mindie)
- [VoxBox](#voxbox)

モデルのデプロイを開始すると、UI は次の条件に基づいてバックエンドを自動選択します。

- 既知の `Text-to-Speech` または `Speech-to-Text` モデルの場合は、`VoxBox` を使用します。
- 既知の `Image` モデルの場合は、`SGLang` を使用します。
- それ以外の場合は、`vLLM` を使用します。

この事前選択は、デプロイフォームのデフォルト値を設定するだけです。実際のデプロイでは、ユーザーがデプロイリクエストを送信するときに明示的に選択したバックエンドを使用します。上記のサポート対象バックエンドから選択できます。

!!! Note

    サポート対象のすべての推論バックエンドについて、ビルド済みの Docker イメージを [DockerHub](https://hub.docker.com/r/gpustack/runner) で公開しています。モデルのデプロイ時に、システムが対応するイメージを自動的に取得して実行します。

## vLLM { #vllm }

[vLLM](https://github.com/vllm-project/vllm) は、高いスループットと優れたメモリ効率を備えた LLM 推論バックエンドです。本番環境での LLM 実行に広く利用されています。

vLLM は、次のような最新のオープンソースモデルの多くをサポートしています。

- Transformer 型 LLM（例: `Qwen3`）
- Mixture-of-Experts 型 LLM（例: `DeepSeek V3/R1`）
- マルチモーダル LLM（例: `Qwen3-VL`）
- 埋め込みモデル（例: `Qwen3-Embedding`）
- リランカーモデル（例: `Qwen3-Reranker`）

デフォルトでは、GPUStack はモデルのメタデータに基づいてモデルインスタンスに必要な VRAM を推定します。

必要に応じてパラメーターを調整できます。次の vLLM パラメーターが役立ちます。

- `--gpu-memory-utilization`（デフォルト: 0.9）: モデルインスタンスが使用する GPU メモリの割合です。
- `--max-model-len`: モデルのコンテキスト長です。コンテキスト長の大きいモデルでは、特にリソースが限られた環境でデプロイを容易にするため、GPUStack が自動的に `8192` に設定します。必要に応じて変更できます。
- `--tensor-parallel-size`: テンソル並列レプリカの数です。デフォルトでは、GPUStack が利用可能な GPU リソースとモデルに必要なメモリの推定値に基づいて設定します。必要に応じて変更できます。

詳細は [vLLM CLI リファレンス](https://docs.vllm.ai/en/stable/cli/serve/)を参照してください。

### サポート対象モデル { #supported-models }

サポート対象モデルについては、vLLM の[ドキュメント](https://docs.vllm.ai/en/stable/models/supported_models.html)を参照してください。

### サポート対象機能 { #supported-features }

#### オムニモーダルのサポート { #omni-modal-support }

[vLLM-Omni](https://github.com/vllm-project/vllm-omni) は、オムニモーダル推論向けに設計された vLLM の拡張機能で、次をサポートします。

- **オムニモデル**: テキスト、音声、画像、動画の同時処理（例: `Qwen3-Omni`）
- **画像タスク**: 画像の生成と編集（例: `Z-Image-Turbo`）
- **動画タスク**: 動画の生成と編集（例: `Wan2.2`）
- **音声タスク**: 音声合成、音声クローンなど（例: `Qwen3-TTS`）

GPUStack は vLLM-Omni と連携し、オムニモーダルモデルのデプロイと管理を円滑に行えるようにします。vLLM バックエンドでモデルをデプロイすると、GPUStack はメタデータからオムニモーダルモデルかどうかを自動検出し、vLLM-Omni に必要なパラメーターを設定します。

#### ワーカー間の分散推論（実験的機能） { #distributed-inference-across-workers-experimental }

1 台のワーカーにモデルをホストするための VRAM や計算能力が不足している場合、vLLM は 1 つのモデルインスタンスを複数のワーカーに分割し、協調して推論を実行できます。GPUStack はこれらのノード間プロセスを 1 つのモデルインスタンスとして管理し、外部には引き続き 1 つの OpenAI 互換エンドポイントを公開します。

モデルのデプロイ時に `Allow Distributed Inference Across Workers` を選択すると有効になります。

GPUStack は vLLM の分散推論に 2 つの実行方式を提供し、vLLM の `--distributed-executor-backend` パラメーターで選択します。通常は、コンテナーイメージと vLLM のバージョンに基づいて GPUStack が自動選択するため、手動で設定する必要はありません。

##### Ray バックエンド { #ray-backend }

- **使用する場合**: GPUStack の公式 runner イメージでのデフォルトの方式です。runner イメージには、vLLM のバージョンにかかわらず Ray が同梱されています。また、バックエンドパラメーターで `--distributed-executor-backend` を明示的に `ray`、または `mp` 以外の値（例: `external_launcher`）に設定した場合も使用します。
- **仕組み**: GPUStack は参加する各ノードで [Ray](https://ray.io) のサイドカープロセスを起動してクラスターを構成します。リーダーノードがメインの vLLM プロセスを実行して API サーバーを公開し、Ray がノード間のタスクスケジューリングを調整します。

##### MultiProcessing バックエンド { #multiprocessing-backend }

- **使用する場合**: バックエンドパラメーターで `--distributed-executor-backend=mp` を明示的に設定した場合、または vLLM ≥ `0.18.0` でカスタムイメージ（バックエンドのバージョンに `-custom` 接尾辞が付くもの）を実行する場合です。vLLM は `0.18.0` でデフォルトの依存関係から Ray を削除したため、カスタムイメージには Ray が含まれないことがあります。GPUStack は起動の失敗を避けるために `mp` にフォールバックします。
- **仕組み**: Ray のサイドカーを使わず、vLLM のマルチプロセスベースのマルチノード実行を直接利用します。GPUStack はクラスター構成に基づいて `--nnodes`、`--node-rank`、`--master-addr`、`--master-port` などのトポロジー引数を注入します。
- **ヘッドレスのフォロワー**: フォロワーとしてスケジュールされたインスタンスには、自動的に `--headless` が付与されます。**そのため、分散計算にのみ参加し、OpenAI API サーバーは起動しません**。HTTP エンドポイントを公開するのはリーダーだけです。外部からのすべてのリクエストはリーダーを経由し、フォロワーに振り分けられます。フォロワーインスタンスのログに `--headless` が表示されるのは想定どおりの動作であり、フォロワーが API ポートを使用しない理由でもあります。
- **トポロジーの構成**: GPUStack は並列処理の設定（TP / PP / DP）に基づいて、`dp_only`、`mp_only`、`nested` のいずれかのパラメーター構成を自動選択します。

!!! warning "既知の制限"

    1. 参加するすべてのワーカーで、同じパスからモデルファイルにアクセスできる必要があります。共有ファイルシステムを使用するか、すべてのワーカーの同じパスにモデルファイルをダウンロードしてください。
    2. Ray モードでは、各ワーカーを同時に割り当てられる分散 vLLM モデルインスタンスは 1 つだけです。

次の条件で自動スケジューリングをサポートしています。

- 参加するワーカーの GPU 数が同じであること。
- ワーカー内のすべての GPU が gpu_memory_utilization（デフォルトは 0.9）の要件を満たすこと。
- テンソル並列処理では、アテンションヘッド数に基づいて GPU が分割されること。
- VRAM の合計割り当て量が、推定 VRAM 割り当て量を上回ること。

これらの条件を満たさない場合、モデルインスタンスは自動でスケジュールされません。ただし、モデル設定で目的のワーカー/GPU を選択すると、手動でスケジュールできます。

### パラメーターリファレンス { #parameters-reference }

vLLM がサポートするパラメーターの全一覧は[こちら](https://docs.vllm.ai/en/stable/cli/serve/)を参照してください。

## SGLang { #sglang }

[SGLang](https://github.com/sgl-project/sglang) は、大規模言語モデルと視覚言語モデル向けの高性能なサービングフレームワークです。

単一 GPU から大規模な分散クラスターまで、幅広い構成で低遅延かつ高スループットの推論を提供するよう設計されています。

デフォルトでは、GPUStack はモデルのメタデータに基づいてモデルインスタンスに必要な VRAM を推定します。

必要に応じて、GPUStack はコンテキスト長の大きいモデルの一部のパラメーターも自動設定します。主な SGLang パラメーターは次のとおりです。

- `--mem-fraction-static`（デフォルト: `0.9`）: GPU ごとに割り当て可能な VRAM の割合です。スケジューラーはこの値をリソースの照合と候補選択に使用します。モデルの `backend_parameters` で上書きできます。
- `--context-length`: モデルのコンテキスト長です。コンテキスト長の大きいモデルで、自動推定されたコンテキスト長がデバイスの能力を超える場合、GPUStack はリソースの限られた環境でのデプロイを容易にするために `8192` に設定します。必要に応じて変更できます。
- `--tp-size`: テンソル並列サイズです。明示的に指定されていない場合、GPUStack は選択した GPU に基づいて推定し、設定します。
- `--pp-size`: パイプライン並列サイズです。マルチノードのデプロイでは、GPUStack がモデルとクラスター設定に応じて `--tp-size` と `--pp-size` の組み合わせを決定します。
- マルチノード引数: `--nnodes`、`--node-rank`、`--dist-init-addr`。分散推論を有効にすると、GPUStack はマルチノード通信を初期化するためにこれらの引数を注入します。

詳細は [SGLang のドキュメント](https://docs.sglang.ai/index.html)を参照してください。

### サポート対象モデル { #supported-models_1 }

サポート対象モデルについては、SGLang の[ドキュメント](https://docs.sglang.ai/supported_models/generative_models.html)を参照してください。

SGLang は画像モデルもサポートしています。検証済みのモデルには Qwen-Image、Flux 1 Dev があります。詳細は [SGLang Diffusion のドキュメント](https://lmsys.org/blog/2025-11-07-sglang-diffusion/)を参照してください。

### サポート対象機能 { #supported-features_1 }

#### ワーカー間の分散推論（実験的機能） { #distributed-inference-across-workers-experimental_1 }

GPUStack では、複数のワーカー間での SGLang 分散推論を有効にできます。

!!! warning "既知の制限"

    1. 参加するすべてのノードで Linux を実行し、イメージ/環境の Python バージョンと通信ライブラリー（NCCL など）に互換性がある必要があります。
    2. すべてのノードで、同じパスからモデルファイルにアクセスできる必要があります。共有ファイルシステムを使用するか、各ノードの同じパスにモデルをダウンロードしてください。

自動スケジューリングの候補選択では、次の条件が考慮されます。

- 参加する GPU が `--mem-fraction-static` の要件（デフォルトは 0.9）を満たすこと。
- 単一ノードの複数 GPU、または複数ノードの複数 GPU の構成で、実効的に割り当て可能な VRAM の合計がモデルの要件を満たすこと。
- モデル並列処理の要件を満たすこと（例: アテンションヘッドの総数がテンソル並列サイズで割り切れること）。満たさない候補は除外されます。

これらの条件を満たさない場合でも、設定でワーカー/GPU を選択してモデルインスタンスを手動スケジュールできます。ただし、過剰割り当てのリスクが表示されることがあります。

#### 拡散モデル（実験的機能） { #diffusion-models-experimental }

SGLang Diffusion は、画像と動画の生成を高速化するために拡散モデル専用に設計された、高性能な推論エンジンです。
SGLang の並列処理技術と最適化されたカーネルにより、主要なベースライン（Hugging Face Diffusers など）と比較して最大 1.2 倍の生成速度を実現します。

詳細は [SGLang Diffusion](https://lmsys.org/blog/2025-11-07-sglang-diffusion/) を参照してください。

#### その他の高度な機能 { #other-advanced-features }

次のような高度な機能も利用できます。

- [投機的デコード](https://docs.sglang.ai/advanced_features/speculative_decoding.html)
- [階層型 KV キャッシュ](https://docs.sglang.ai/advanced_features/hicache.html)

使用方法は公式ドキュメントを参照してください。

### パラメーターリファレンス { #parameters-reference_1 }

SGLang がサポートするパラメーターの全一覧は[こちら](https://docs.sglang.ai/advanced_features/server_arguments.html)を参照してください。

## MindIE { #mindie }

[MindIE](https://www.hiascend.com/en/software/mindie) は、[Ascend ハードウェア](https://www.hiascend.com/en/hardware/product)上で動作する高性能な推論サービスです。

### サポート対象モデル { #supported-models_2 }

MindIE は[こちら](https://www.hiascend.com/software/mindie/modellist)に掲載されている各種モデルをサポートしています。

GPUStack では、[大規模言語モデル（LLM）](https://www.hiascend.com/software/mindie/modellist)と[マルチモーダル言語モデル（VLM）](https://www.hiascend.com/software/mindie/modellist)をサポートしています。

ただし、_埋め込みモデル_と_マルチモーダル生成モデル_はまだサポートしていません。

### サポート対象機能 { #supported-features_2 }

MindIE は[こちら](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0001.html)に示すさまざまな機能を備えています。

現在、GPUStack はそれらのうち次の機能をサポートしています。
[量子化](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0279.html)、
[コンテキストサイズの拡張](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0295.html)、
[分散推論](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0296.html)、
[Mixture of Experts（MoE）](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0297.html)、
[Split Fuse](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0300.html)、
[投機的デコード](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0301.html)、
[マルチトークン予測](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0500.html)、
[プレフィックスキャッシュ](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0302.html)、
[関数呼び出し](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0303.html)、
[マルチモーダル理解](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0304.html)、
[Multi-head Latent Attention（MLA）](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0305.html)、
[テンソル並列処理](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0511.html)、
[コンテキスト並列処理](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0508.html)、
[シーケンス並列処理](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0517.html)、
[エキスパート並列処理](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0509.html)、
[データ並列処理](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0424.html)、
[レスポンスのバッファリング（Ascend MindIE 2.0.RC1 以降）](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0425.html)。
[KV Pool（Ascend MindIE 2.2.RC1 以降）](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0538.html)。

!!! note

    1. 量子化には専用の重みが必要で、モデルの `config.json` も調整する必要があります。[リファレンス（ガイド）](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0279.html)に従って適切な重みを準備してください。

    2. 一部の機能は併用できません。使用時には注意してください。たとえば、プレフィックスキャッシュを有効にした場合、コンテキストサイズの拡張機能は使用できません。

### パラメーターリファレンス { #parameters-reference_2 }

MindIE には設定可能な[パラメーター](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_service0285.html)と[環境変数](https://www.hiascend.com/document/detail/zh/mindie/22RC1/mindiellm/llmdev/mindie_llm0416.html)があります。

JSON を直接編集せずに設定できるように、GPUStack は次のコマンドラインパラメーターを提供しています。

| パラメーター | デフォルト | 範囲 | 適用範囲 | 説明 |
|---|---|---|---|---|
| `--log-level` | Info |  | ログ設定 | MindIE のログレベルです。選択肢: `Verbose`、`Info`、`Warning`、`Warn`、`Error`、`Debug`。 |
| `--max-link-num` | 1000 | [1, 1000] | サーバー設定 | 並列リクエストの最大数です。 |
| `--token-timeout` | 60 | [1, 3600] | サーバー設定 | 1 トークンの生成のタイムアウト（秒）です。 |
| `--e2e-timeout` | 60 | [1, 3600] | サーバー設定 | E2E（リクエスト受け付けから推論終了まで）のタイムアウト（秒）です。 |
| `--openai-support` | vllm |  | サーバー設定 | OpenAI API の互換モードです。 |
| `--kv-pool-config` |  |  | バックエンド設定 | JSON 形式の KV プール設定です。例: `{"backend":"<KV pool backend name>", "configPath":"/path/to/your/config/file"}`。 |
| `--max-seq-len` | 8192 | (0, \*) | モデルデプロイ設定 | モデルのコンテキスト長です。指定しない場合、モデル設定から取得します。 |
| `--max-input-token-len` |  | (0, `--max-seq-len`] | モデルデプロイ設定 | 入力トークン長の上限です。指定しない場合、`min(--max-seq-len, --max-prefill-tokens)` から取得します。 |
| `--truncation` |  |  | モデルデプロイ設定 | 入力トークン長が `--max-input-token-len` と `--max-seq-len` - 1 の小さい方を超えた場合、切り詰めます。 |
| `--cpu-mem-size` | 0 | [0, \*) | モデル設定 | CPU スワップ領域のサイズ（GiB）です。`--max-preempt-count` を指定した場合に有効になります。 |
| `--npu-memory-fraction` (env: `NPU_MEMORY_FRACTION`) | 0.9 | (0, 1] | モデル設定 | モデル実行に使用する NPU メモリの割合（0～1）です。たとえば、`0.5` は 50% のメモリ使用率を意味します。 |
| `--trust-remote-code` |  |  | モデル設定 | リモートコードを信頼します（モデルの読み込み用）。 |
| `--models` |  |  | モデル設定 | JSON 形式のモデル設定です。エキスパート並列処理の実装方式や、テンソル並列処理の LM ヘッダー/出力アテンション分割など、特定の設定に使用します。 |
| `--async-scheduler-wait-time` | 120 | [1, 3600] | モデル設定 | 非同期スケジューラーが起動するまでの待機時間（秒）です。 |
| `--cache-block-size` | 128 |  | スケジュール設定 | KV キャッシュのブロックサイズです。2 の累乗である必要があります。 |
| `--max-prefill-batch-size` | 50 | [1, `--max-batch-size`] | スケジュール設定 | プリフィル段階でバッチ化するリクエストの最大数です。`--max-batch-size` 未満である必要があります。 |
| `--max-prefill-tokens  ` |  | [1, `--max-seq-len`] | スケジュール設定 | 各プリフィルで、現在のバッチに含まれる全入力トークンの合計は `--max-prefill-tokens` を超えられません。デフォルトは `--max-seq-len` と同じです。 |
| `--prefill-time-ms-per-req` | 150 | [0, 1000] | スケジュール設定 | リクエストごとの推定プリフィル時間（ms）です。プリフィル段階とデコード段階のどちらを実行するかの判断に使用します。 |
| `--prefill-policy-type` | 0 |  | スケジュール設定 | プリフィル段階の戦略です。<br> `0`: FCFS（先着順）。<br> `1`: STATE（FCFS と同じ）。<br> `2`: PRIORITY（優先度キュー）。<br> `3`: MLFQ（多段フィードバックキュー）。 |
| `--max-batch-size` | 200 | [1, 5000] | スケジュール設定 | デコード段階でバッチ化するリクエストの最大数です。 |
| `--max-iter-times` |  | [1, `--max-seq-len`] | スケジュール設定 | デコード段階の最大反復回数です。デフォルトは `--max-seq-len` と同じです。 |
| `--decode-time-ms-per-req` | 50 | [0, 1000] | スケジュール設定 | リクエストごとの推定デコード時間（ms）です。バッチ選択に `--prefill-time-ms-per-req` と併せて使用します。 |
| `--decode-policy-type` | 0 |  | スケジュール設定 | デコード段階の戦略です。<br> `0`: FCFS <br> `1`: STATE（プリエンプトまたはスワップされたリクエストを優先）<br> `2`: PRIORITY <br> `3`: MLFQ |
| `--max-preempt-count` | 0 | [0, `--max-batch-size`] | スケジュール設定 | デコード中に許容するプリエンプト済みリクエストの最大数です。`--max-batch-size` 未満である必要があります。 |
| `--support-select-batch` |  |  | スケジュール設定 | バッチ選択を有効にします。`--prefill-time-ms-per-req` と `--decode-time-ms-per-req` に基づいて実行優先順位を決定します。明示的に無効にするには `--no-support-select-batch` を使用します。 |
| `--max-queue-delay-microseconds` | 5000 | [500, 1000000] | スケジュール設定 | キューの最大待機時間（マイクロ秒）です。 |
| `--max-first-token-wait-time` | 2500 | [0, 3600000] | スケジュール設定 | 最初のトークン生成を待つ最大時間（ミリ秒）です。 |
| `--override-generation-config` |  |  |  | JSON 形式で生成設定を上書きまたは設定します。例: `{"temperature": 0.5}`。モデル構造内の `generation_config.json` にマージされます。 |
| `--enforce-eager` |  |  |  | Eager モードで演算子を発行します。 |
| `--no-metrics` |  |  |  | `/metrics` エンドポイントでのメトリクス公開を無効にします。 |
| `--dtype` | auto |  |  | モデルの重みとアクティベーションのデータ型です。<br> `auto`: モデル設定のデフォルトデータ型を使用。<br> `half`/`float16`: FP16。<br> `bfloat16`: BF16。<br> `float`/`float32`: FP32。 |
| `--rope-scaling` |  |  | コンテキストサイズの拡張 | JSON 形式の RoPE スケーリング設定です。例: `{"type":"yarn","factor":4.0,"original_max_position_embeddings":32768}`。モデル構造内の `config.json` にマージされます。 |
| `--rope-theta` |  |  | コンテキストサイズの拡張 | RoPE theta の設定です。モデル構造内の `config.json` にマージされます。 |
| `--enable-split` |  |  | Split Fuse | チャンク化プリフィルに似た Split Fuse を有効にします。明示的に無効にするには `--no-enable-split` を使用します。 |
| `--policy-type` | 0 |  | Split Fuse | Split Fuse の戦略です。<br> `0`: FCFS、先着順。<br> `4`: SJF、最短ジョブ優先。<br> `5`: LJF、最長ジョブ優先。<br> `6`: Skip-Join MLFQ、Skip-Join 多段フィードバックキュー。<br> `7`: SJF-MLFQ、最短ジョブ優先と多段フィードバックキュー。 |
| `--split-chunk-tokens` | 512 | [1, 8192] | Split Fuse | Split Fuse でバッチ化するトークン数です。16 の倍数です。 |
| `--split-start-batch-size` | 16 | [0, `--max-batch-size`] | Split Fuse | Split Fuse による分割を開始するバッチサイズです。 |
| `--enable-memory-decoding` |  |  | 投機的デコード / メモリデコード | メモリデコードによる投機処理を有効にします。明示的に無効にするには `--no-enable-memory-decoding` を使用します。 |
| `--memory-decoding-length` | 16 | [1, 16] | 投機的デコード / メモリデコード | メモリデコードによる投機処理の長さです。 |
| `--memory-decoding-dynamic-algo` |  |  | 投機的デコード / メモリデコード | メモリデコードによる投機処理の動的アルゴリズムを有効にします。 |
| `--enable-lookahead` |  |  | 投機的デコード / 先読み | 先読み投機処理を有効にします。明示的に無効にするには `--no-enable-lookahead` を使用します。 |
| `--lookahead-level` | 4 | [3, 16] | 投機的デコード / 先読み | 先読み投機処理のレベルです。 |
| `--lookahead-window` | 5 | [1, 16] | 投機的デコード / 先読み | 先読み投機処理のウィンドウサイズです。 |
| `--lookahead-guess-set-size` | 5 | [1, 16] | 投機的デコード / 先読み | 先読み投機処理の推測セットサイズです。 |
| `--enable-multi-token-prediction` |  |  | マルチトークン予測 | マルチトークン予測を有効にします。明示的に無効にするには `--no-enable-multi-token-prediction` を使用します。 |
| `--multi-token-prediction-tokens` | 1 | (0, \*) | マルチトークン予測 | マルチトークン予測のトークン数です。`--enable-multi-token-prediction` を有効にした場合のみ有効です。 |
| `--enable-prefix-caching` |  |  | プレフィックスキャッシュ | プレフィックスキャッシュを有効にします。明示的に無効にするには `--no-enable-prefix-caching` を使用します。 |
| `--pipeline-parallel-size`, `-pp` | 1 | (0, \*) | 並列処理 | パイプライン並列グループの数です。 |
| `--data-parallel-size`, `-dp` | -1 |  | 並列処理 | Attention 層のデータ並列グループの数です。`-1` はデータ並列処理の無効化を意味します。それ以外は 2 の累乗である必要があります。 |
| `--context-parallel-size`, `-cp` | -1 |  | 並列処理 | Attention 層のコンテキスト並列グループの数です。`-1` はコンテキスト並列処理の無効化を意味します。それ以外は 2 の累乗である必要があります。 |
| `--tensor-parallel-size`, `-tp` | -1 |  | 並列処理 | Attention 層のテンソル並列グループの数です。`-1` はワールドサイズをテンソル並列サイズに使用することを意味します。それ以外は 2 の累乗である必要があります。 |
| `--sequence-parallel-size`, `-sp` | -1 | `--tensor-parallel-size` | 並列処理 | MLP 層のシーケンス並列グループの数です。`-1` はシーケンス並列処理の無効化を意味します。それ以外は 2 の累乗である必要があります。 |
| `--moe-expert-parallel-size`, `-moe-ep` | -1 |  | 並列処理 | エキスパート並列グループの数です。`-1` は MoE エキスパート並列処理の無効化を意味します。それ以外は 2 の累乗である必要があります。 |
| `--moe-tensor-parallel-size`, `-moe-tp` | -1 |  | 並列処理 | MoE MLP 層のテンソル並列グループの数です。`-1` はワールドサイズを MoE テンソル並列サイズに使用することを意味します。それ以外は 2 の累乗である必要があります。 |
| `--enable-buffer-response` |  |  | レスポンスのバッファリング | レスポンスのバッファリングを有効にします。明示的に無効にするには `--no-enable-buffer-response` を使用します。 |
| `--prefill-expected-time-ms` |  |  | レスポンスのバッファリング | 最初のトークンまでの時間（TTFT）の目標遅延（SLO、ミリ秒）です。 |
| `--decode-expected-time-ms` |  |  | レスポンスのバッファリング | 出力トークンあたりの時間（TPOT）の目標遅延（SLO、ミリ秒）です。 |


!!! note

    GPUStack では、モデルのデプロイ時にカスタム環境変数を注入できます。ただし、一部の変数は GPUStack の管理と競合する場合があります。

    そのため、GPUStack はそれらの変数を上書きまたは抑止します。モデルインスタンスのログ出力が想定どおりかを確認してください。

## VoxBox { #voxbox }

[VoxBox](https://github.com/gpustack/vox-box) は、テキスト読み上げモデルと音声認識モデルのデプロイ向けに設計された推論エンジンです。OpenAI 音声 API と完全互換の API も提供します。

### サポート対象モデル { #supported-models_3 }

| モデル                          | 種類           | リンク                                                                                                                                                                | サポート対象プラットフォーム |
| ------------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| Faster-whisper-large-v3         | 音声認識 | [Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v3), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-large-v3)                 | AMD64,ARM64         |
| Faster-whisper-large-v2         | 音声認識 | [Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v2), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-large-v2)                 | AMD64,ARM64         |
| Faster-whisper-large-v1         | 音声認識 | [Hugging Face](https://huggingface.co/Systran/faster-whisper-large-v1), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-large-v1)                 | AMD64,ARM64         |
| Faster-whisper-medium           | 音声認識 | [Hugging Face](https://huggingface.co/Systran/faster-whisper-medium), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-medium)                     | AMD64,ARM64         |
| Faster-whisper-medium.en        | 音声認識 | [Hugging Face](https://huggingface.co/Systran/faster-whisper-medium.en), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-medium.en)               | AMD64,ARM64         |
| Faster-whisper-small            | 音声認識 | [Hugging Face](https://huggingface.co/Systran/faster-whisper-small), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-small)                       | AMD64,ARM64         |
| Faster-whisper-small.en         | 音声認識 | [Hugging Face](https://huggingface.co/Systran/faster-whisper-small.en), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-small.en)                 | AMD64,ARM64         |
| Faster-distil-whisper-large-v3  | 音声認識 | [Hugging Face](https://huggingface.co/Systran/faster-distil-whisper-large-v3), [ModelScope](https://modelscope.cn/models/gpustack/faster-distil-whisper-large-v3)   | AMD64,ARM64         |
| Faster-distil-whisper-large-v2  | 音声認識 | [Hugging Face](https://huggingface.co/Systran/faster-distil-whisper-large-v2), [ModelScope](https://modelscope.cn/models/gpustack/faster-distil-whisper-large-v2)   | AMD64,ARM64         |
| Faster-distil-whisper-medium.en | 音声認識 | [Hugging Face](https://huggingface.co/Systran/faster-distil-whisper-medium.en), [ModelScope](https://modelscope.cn/models/gpustack/faster-distil-whisper-medium.en) | AMD64,ARM64         |
| Faster-whisper-tiny             | 音声認識 | [Hugging Face](https://huggingface.co/Systran/faster-whisper-tiny), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-tiny)                         | AMD64,ARM64         |
| Faster-whisper-tiny.en          | 音声認識 | [Hugging Face](https://huggingface.co/Systran/faster-whisper-tiny.en), [ModelScope](https://modelscope.cn/models/gpustack/faster-whisper-tiny.en)                   | AMD64,ARM64         |
| CosyVoice-300M-Instruct         | テキスト読み上げ | [Hugging Face](https://huggingface.co/gpustack/CosyVoice-300M-Instruct), [ModelScope](https://modelscope.cn/models/gpustack/CosyVoice-300M-Instruct)                | AMD64               |
| CosyVoice-300M-SFT              | テキスト読み上げ | [Hugging Face](https://huggingface.co/gpustack/CosyVoice-300M-SFT), [ModelScope](https://modelscope.cn/models/iic/CosyVoice-300M-SFT)                               | AMD64               |
| CosyVoice-300M                  | テキスト読み上げ | [Hugging Face](https://huggingface.co/gpustack/CosyVoice-300M), [ModelScope](https://modelscope.cn/models/gpustack/CosyVoice-300M)                                  | AMD64               |
| CosyVoice-300M-25Hz             | テキスト読み上げ | [ModelScope](https://modelscope.cn/models/iic/CosyVoice-300M-25Hz)                                                                                                  | AMD64               |
| CosyVoice2-0.5B                 | テキスト読み上げ | [Hugging Face](https://huggingface.co/gpustack/CosyVoice2-0.5B), [ModelScope](https://modelscope.cn/models/iic/CosyVoice2-0.5B)                                     | AMD64               |
| Dia-1.6B                        | テキスト読み上げ | [Hugging Face](https://huggingface.co/nari-labs/Dia-1.6B), [ModelScope](https://modelscope.cn/models/nari-labs/Dia-1.6B)                                            | AMD64               |


### サポート対象機能 { #supported-features_3 }

#### GPU/CPU オフロードの許可 { #allow-gpucpu-offloading }

VoxBox は NVIDIA GPU へのモデルのデプロイをサポートしています。GPU リソースが不足している場合は、自動的に CPU にモデルをデプロイします。

!!! note "ストリーミングは未対応"

    VoxBox は、テキスト読み上げ（`/v1/audio/speech`）および音声認識（`/v1/audio/transcriptions`）のストリーミングをサポートしていません。ストリーミング音声 API を使用するには、vLLM 互換の音声モデルを `vLLM` バックエンドにデプロイしてください。
