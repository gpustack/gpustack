---
translation_status: translated
---

# 負荷分散ポリシーによるルーティング { #routing-with-lb-policies }

このチュートリアルでは、GPUStack のルート単位の負荷分散ポリシーを実践します。負荷を分散しつつセッションを固定し、判断サービスでリクエストごとに小型・大型モデルを選びます。設定のリファレンスは、[モデルルート管理](../user-guide/model-route-management.md#load-balancing)をご覧ください。

## 前提条件 { #prerequisites }

- GPUStack が稼働していること。ポリシーを適用する AI ゲートウェイはデフォルトでデプロイされるため、標準構成に追加設定は不要です。
- ルートの対象として使う 2 つのデプロイ済みモデル（以下では `qwen3-0.6b` と `qwen3-32b`）。
- 判断サービスの例では、`Model` - `Provider` ページに `TypeSafe Decision Service (Jev)` プロバイダーとして Jev を登録します（[モデルプロバイダー管理](../user-guide/model-provider-management.md)を参照）。

## 手順 1：ルートの作成 { #step-1-create-the-route }

1. `Routes` ページに移動します。
2. `Add Route` をクリックし、`qwen3` などの名前を付けます。
3. `Route Targets` に `qwen3-0.6b` と `qwen3-32b` を選択し、重みはすべて未設定にします。
4. `Save` をクリックします。

重みとポリシーが未設定なら、`Route By` は round-robin となり、2 モデル間の単純なラウンドロビンです。

## 手順 2：セッションアフィニティと最少処理中リクエストの相互作用 { #step-2-understand-how-session-affinity-and-least-inflight-request-interact }

ポリシープラグインを有効にすると、各プラグインがすべての対象候補を採点し、最後の処理が重み付き合計の最も良い対象を選択します。

```
total(candidate) = Σ over plugins ( plugin weight × normalized score )
```

セッションアフィニティはセッションの固定先に、最少処理中リクエストは最も空いている対象に最高点を与えます。後者の票差が前者の票差を超えた場合だけ、セッションが固定先から移動します。

両方がデフォルトの重み（1）の場合、固定先の処理中要求が `x`、ほかの対象はすべてアイドルという、最も移動しやすい条件での結果は次のとおりです。

| 対象数（N） | x=0 | x=1 | x=2 | x=3 | x=4 | 移動するか？ |
| ----------- | --- | --- | --- | --- | --- | ----------- |
| 2 | 維持 | 同点 ⚠️ | **移動** | **移動** | **移動** | はい |
| 3 | 維持 | 維持 | 同点 ⚠️ | **移動** | **移動** | はい |
| 4 | 維持 | 維持 | 維持 | 維持 | 維持 | x ≈ 5.3 の場合のみ |
| ≥5 | 維持 | 維持 | 維持 | 維持 | 維持 | **いいえ** |

- **移動**：次の候補へセッションが移動します。
- **同点 ⚠️**：票差が完全に一致し、最後の処理がランダムに決めるため、同じ要求でも移動する場合としない場合があります。
- **N ≥ 5**：最少処理中リクエストの票差の上限は `1/(N−1) ≤ 0.25` で、アフィニティの票差を下回ります。どれほど負荷が偏っても、それだけでは固定を解除できません。

対象数の多いルートで負荷を考慮するには重みを調整します。比率を変えた場合の閾値は、[重みの仕組み](#how-weights-work)をご覧ください。

## 手順 3：セッションアフィニティと最少処理中リクエストの有効化 { #step-3-enable-session-affinity-and-least-inflight-request }

1. 手順 1 のルートを編集します。
2. `Session Affinity` を有効化します。デフォルトのセッションキーチェーン（`session-id` ヘッダー、`x-client-request-id` ヘッダー、本文の `prompt_cache_key` の順）を使うか、独自に指定します。
3. `Least Inflight Request` をデフォルトの重みで有効化します。
4. 保存し、`Route By` が `Policy` であることを確認します。

同じセッションキーの要求は同じ対象に固定され、プロンプトキャッシュを維持します。ただし固定先の負荷が高くなると、上の表に従ってアイドルの対象へ移動します。

キーの適用範囲に注意してください。ヘッダーのキー（`session-id` など）はすべてのパスで使えますが、本文の `prompt_cache_key` などは `/responses` と `/messages` だけに適用されます。標準のセッション本文フィールドがない `/chat/completions` では、チェーン内のヘッダーキーだけを使います。チャットにはヘッダーキーを推奨します。

## 手順 4：判断サービスルーティングの追加 { #step-4-add-decision-service-routing }

次は、開始時のモデルに常に固定する代わりに、各要求に適切なモデルを判断エンジンへ問い合わせます。

1. ルートを編集し、`Decision Service Routing` を有効化します。
    - `Decision Service`：登録したプロバイダーを選択。
    - `Decision Model`：例：`jev-latest`。
    - `Instructions`：「要求を十分に処理できる最小のモデルを選んでください。単純・短い・事実に関する要求は高速モデルへ、複雑な推論・数学・コードの要求は大型モデルへ送ってください。」
    - `Model Criteria`：対象ごとに入力（または `Generate from targets` をクリックして調整）。
        - `qwen3-0.6b`：「非常に高速で低コストの 0.6B モデル。挨拶、単純な事実質問、書式整形、短い雑談に対応します。複数段階の推論、数学、コードには不向きです。」
        - `qwen3-32b`：「高性能な 32B 推論モデル。複雑な推論、数学、コード生成、長いコンテキストの分析に対応します。要求ごとの速度は遅く、費用は高くなります。」

2. 保存後、`qwen3` ルートへ認証付きの 2 要求を送ります（`$GPUSTACK_API_KEY` は `API Keys` ページで作成したキーに置き換えます）。

```bash
curl http://my-gpustack/v1/chat/completions \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -H "Content-Type: application/json" \
  -H "session-id: demo-session" \
  -d '{"model": "qwen3", "messages": [{"role": "user", "content": "Write a one-line greeting card."}]}'
```

```bash
curl http://my-gpustack/v1/chat/completions \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -H "Content-Type: application/json" \
  -H "session-id: demo-session" \
  -d '{"model": "qwen3", "messages": [{"role": "user", "content": "Prove that there are infinitely many primes, then implement the proof sketch in Python."}]}'
```

最初の要求は `qwen3-0.6b`、2 番目は `qwen3-32b` に振り分けられます。

## 重みの仕組み { #how-weights-work }

重みは 2 か所にあり、それぞれ意味が異なります。

### 対象の重み（トラフィックの割合） { #target-weight-traffic-share }

`Target Weight` モードでは、重みはインスタンス数に関係なく、対象**全体**のトラフィックの割合を示します。GPUStack はインスタンスごとの重みを調整し、常に設定した割合にします。

- 対象 A（重み 100、1 インスタンス）と B（重み 100、2 インスタンス）は、対象間で 50/50 に分配します。各インスタンスが 1/3 ずつではありません。
- 対象を 1 から 2 インスタンスに増やしても割合は変わらず、各インスタンスがその半分を受け取ります。

たとえば `qwen3-0.6b` の重みを 100、`qwen3-32b` を 300 にすると、レプリカ数に関係なく、`qwen3-0.6b` に 25%、`qwen3-32b` に 75% を送ります。

### ポリシーの重み（合計への影響度） { #policy-weight-influence-in-the-sum }

各プラグインの `Weight` は、`Policy` モードの重み付き合計への影響を調整する正の数で、小数も有効です。`Weight` が未設定の場合は各プラグインの組み込みデフォルト（Session Affinity と Least Inflight Request は 1、Decision Service Routing は 10）を使用します。

固定先が処理中要求 `x` を持ち、ほかの対象がアイドルの場合、セッションアフィニティと最少処理中リクエストの関係は次のようになります。

- セッションアフィニティの票差：`affDiff = 0.5 / (2·(1 − 2⁻ᴺ))`
- 最少処理中リクエストの票差：`loadDiff = x / ((1+x)·(N−1) + 1)`
- `w_leastInflight × loadDiff > w_affinity × affDiff` のときセッションが移動します。

重みの比率によって移動閾値が変わります。次の表は、比率（Least Inflight Request : Session Affinity）と対象数 N ごとの、移動に必要な固定先の最小処理中要求数 `x` です。

| N | 1:1（デフォルト） | 1.5:1 | 2:1 |
| - | ------------- | ----- | --- |
| 2 | 2 | 1 | 1 |
| 3 | 3 | 1 | 1 |
| 4 | 6 | 2 | 1 |
| 5 | なし | 3 | 2 |
| 6 | なし | 7 | 3 |
| 7 | なし | なし | 4 |
| 8 | なし | なし | 9 |

表の読み方：

- デフォルトの 1:1 では、対象が 5 つ以上の場合は移動しません。最少処理中リクエストの票差の上限が、アフィニティを下回るためです。
- Session Affinity の `Weight` を半分にすると（2:1）、対象数 8 までのルートで `x` = 2～4 から負荷の影響が現れます。比率だけが重要なので、Least Inflight Request の `Weight` を 2 倍にしても同じです。
- 完全な同点（例：1:1、N=2/x=1）は、最後の処理がランダムに決定します。

## ポリシー間の役割分担 { #how-the-policies-divide-the-work }

有効なすべてのポリシーが同じ重み付き合計に寄与し、それぞれ 1 つの役割を担います。

- **Decision Service Routing** は要求に適した*モデル*を選びます（デフォルトの `Weight` 10 は、通常は最も強い影響を持つことを示します）。
- **Session Affinity** はセッションを現在のモデルに固定し、プロンプトキャッシュのローカリティを保ちます。
- **Least Inflight Request** は、選ばれたモデルのインスタンス間で要求を分散します。

モデルに複数のインスタンスがある場合、判断サービスがモデルを絞り、Least Inflight Request がその中のインスタンスを選択します。
