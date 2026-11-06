## チャット Playground { #chat-playground }

チャット補完 API と対話できます。次のスクリーンショットは、その例です。

![Playground のスクリーンショット](../../assets/quick-start/quick-chat.png)

### プロンプト { #prompts }

Playground の左側でプロンプトメッセージを調整できます。プロンプトメッセージには、system、user、assistant の 3 種類のロールがあります。

- **System**：通常、コンテキストを設定し、動作を定義し、またはモデルの応答生成に制約を設けるための、事前に定義した指示やガイダンスです。
- **User**：ユーザー（LLM と対話する人）が指定する入力や質問です。
- **Assistant**：LLM が生成した応答です。

#### システムメッセージの編集 { #edit-system-message }

Playground の上部で、システムメッセージを追加または編集できます。

#### ユーザーおよびアシスタントのメッセージの編集 { #edit-user-and-assistant-messages }

ユーザーまたはアシスタントのメッセージを追加するには、`Add` ボタンをクリックします。

ユーザーまたはアシスタントのメッセージを削除するには、そのメッセージの右端にあるマイナスボタンをクリックします。

メッセージのロールを変更するには、メッセージの先頭の `User` または `Assistant` をクリックします。

### 画像のアップロード { #upload-image }

`Upload Image` ボタンをクリックして、プロンプトに画像を追加できます。

### 音声のアップロード { #upload-audio }

`Upload Audio` ボタンをクリックして、プロンプトに音声ファイルを追加できます。

### プロンプトのクリア { #clear-prompts }

`Clear` ボタンをクリックすると、すべてのプロンプトを消去します。

### モデルの選択 { #select-model }

Playground の右上にあるモデルのドロップダウンをクリックして、GPUStack で利用可能なモデルを選択できます。モデルのデプロイ方法は、[モデルデプロイの管理](../model-deployment-management.md)を参照してください。

### パラメーターの調整 { #customize-parameters }

`Parameters` 欄で、補完のパラメーターを調整できます。

### 補完の実行 { #do-completion }

`Submit` ボタンをクリックするか、Enter キーを押すと、補完を実行できます。

### コードの表示 { #view-code }

プロンプトとパラメーターを試した後、`View Code` ボタンをクリックすると、同じ入力で API を呼び出すコードを確認できます。`curl`、`Python`、`Node.js` のコード例を提供しています。

## 比較 Playground { #compare-playground }

Playground では複数のモデルを比較できます。次のスクリーンショットは、その例です。

![比較 Playground のスクリーンショット](../../assets/playground/compare-playground.png)

### 比較モード { #comparison-mode }

比較ビューのボタンをクリックすると、比較するモデル数を 2、3、4、6 から選択できます。

### プロンプト { #prompts_1 }

チャット Playground と同じように、プロンプトメッセージを調整できます。

### 画像のアップロード { #upload-image_1 }

`Upload Image` ボタンをクリックして、プロンプトに画像を追加できます。

### 音声のアップロード { #upload-audio_1 }

`Upload Audio` ボタンをクリックして、プロンプトに音声ファイルを追加できます。

### プロンプトのクリア { #clear-prompts_1 }

`Clear` ボタンをクリックすると、すべてのプロンプトを消去します。

### モデルの選択 { #select-model_1 }

各モデルパネルの左上にあるモデルのドロップダウンをクリックして、GPUStack で利用可能なモデルを選択できます。

### パラメーターの調整 { #customize-parameters_1 }

各モデルの設定ボタンをクリックすると、補完のパラメーターを調整できます。
