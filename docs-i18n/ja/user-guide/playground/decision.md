---
translation_status: translated
---

# 判定 Playground { #decision-playground }

判定 Playground では、与えられた状況に対して型付きの質問に答える判定モデルをテストできます。state と questions を編集し、デプロイ済みの判定モデルに対して実行し、型付きの回答を確認します。API 連携用のコード例も確認できます。

## State { #state }

`State` エディターは、判定の対象となる状況を保持します。プレーンテキスト、またはチャットメッセージのような JSON オブジェクト/配列です。エディターは入力中に JSON を検証します。実行には state が必須です。

## Questions { #questions }

`Questions` エディターは、state に対して行う質問を保持します。質問 ID を質問定義に対応付ける JSON オブジェクトです。`+ choice`、`+ score`、`+ noul` ボタンで、そのまま実行できるテンプレートを挿入できます。

- `choice`——`criteria`（選択肢名 → 説明）から 1 つの選択肢を選びます。
- `score`——`criteria` 内の順序付きレベルの一覧に基づいて state を評価します。
- `noul`——二値（はい/いいえ）の質問です。

エディターは JSON を検証し、スキーマのヒントを提供しますが、質問セットはオープンのままです。エンジン固有のフィールドはそのまま渡されます。

## Run { #run }

判定モデルを選択し、`Run` ボタンをクリックするか `Ctrl`/`Cmd` + `Enter` を押して、state と questions を `/v1/systemone` に送信します。実行中のリクエストは `Stop` でキャンセルできます。

![判定 Playground](../../assets/playground/decision.png)

## Answers { #answers }

各回答はその質問 ID の下に返され、カードとして表示されます。判定結果（`Choice`、`Score`、または `noul` のはい/いいえタグ）、`Confidence`、選択肢ごとの確率です。カードの上の要約行には、応答したモデルとトークン使用量が表示されます。認識できない回答の形状は、生の JSON ビューにフォールバックします。

## モデルの選択 { #select-model }

Playground UI の右上にあるモデルのドロップダウンをクリックして、GPUStack で利用可能な判定モデルを選択します。リストに表示されるのは `decision` カテゴリのモデルだけです。モデルが表示されない場合は、実行中で正しくカテゴリ設定されていることを確認し、必要であればモデルの設定でカテゴリを調整してください。

## コードの表示 { #view-code }

state と questions を編集した後、`View Code` ボタンをクリックすると、同じ入力で API を呼び出す方法が表示されます。`curl`、`Python`、`Node.js` のコード例が提供されます。
