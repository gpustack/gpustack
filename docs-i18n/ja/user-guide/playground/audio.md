# 音声 Playground { #audio-playground }

音声 Playground は、GPUStack の Text-to-Speech（TTS）と Speech-to-Text（STT）API をテストして試すための専用の場所です。テキストから音声、音声からテキストへの変換を対話的に試し、パラメーターを調整し、API 連携用のコード例を確認できます。

## テキストから音声への変換 { #text-to-speech }

「Text to Speech」タブに切り替えて、TTS モデルをテストします。

### テキスト入力 { #text-input }

変換するテキストを入力し、`Submit` ボタンをクリックして、対応する音声を生成します。

![テキストから音声への変換](../../assets/playground/text-to-speech.png)

### モデルの選択 { #select-model }

Playground UI の右上にあるモデルのドロップダウンをクリックして、GPUStack で利用可能な TTS モデルを選択します。

### パラメーターの調整 { #customize-parameters }

出力する音声の声と形式を調整します。

!!! tip

    対応する声はモデルによって異なる場合があります。

### コードの表示 { #view-code }

入力テキストとパラメーターを試した後、`View Code` ボタンをクリックすると、同じ入力で API を呼び出す方法を確認できます。`curl`、`Python`、`Node.js` のコード例を提供しています。

## 音声からテキストへの変換 { #speech-to-text }

「Speech to Text」タブに切り替えて、STT モデルをテストします。

### 音声ファイルの指定 { #provide-audio-file }

文字起こし用の音声は、次の 2 つの方法で指定できます。

1. **音声ファイルのアップロード**
2. **ブラウザーでの録音**

!!! note

    ブラウザーで録音できない場合は、次の理由が考えられます。

    1. HTTPS または `http://localhost` でアクセスする場合、ブラウザーでマイクの権限を有効にする必要があります。
    2. `http://{host IP}` でアクセスする場合、URL をブラウザーの信頼済み一覧に追加する必要があります。

          **例：**
          Chrome で `chrome://flags/` を開き、GPUStack の URL を「Insecure origins treated as secure」に追加して、このオプションを有効にします。

![音声の権限](../../assets/playground/audio-permission.png)

![音声からテキストへの変換](../../assets/playground/speech-to-text.png)

### モデルの選択 { #select-model_1 }

Playground UI の右上にあるモデルのドロップダウンをクリックして、GPUStack で利用可能な STT モデルを選択します。

### テキストのコピー { #copy-text }

モデルが生成した文字起こし結果をコピーします。

### パラメーターの調整 { #customize-parameters_1 }

文字起こしの精度を高めるため、音声ファイルに合う言語を選択します。

### コードの表示 { #view-code_1 }

音声ファイルとパラメーターを試した後、`View Code` ボタンをクリックすると、同じ入力で API を呼び出す方法を確認できます。`curl`、`Python`、`Node.js` のコード例を提供しています。
