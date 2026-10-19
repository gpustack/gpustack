---
translation_status: translated
---

# よくある質問 { #faq }

## サポート状況 { #support-matrix }

### 異種混在クラスターへの対応 { #hybrid-cluster-support }

GPUStack は NVIDIA、AMD、Ascend NPU、Hygon DCU、Moore Threads、Iluvatar、MetaX、Cambricon、T-head PPU が混在するクラスターに対応し、AMD64 と ARM64 の両アーキテクチャで動作します。

### 分散推論への対応 { #distributed-inference-support }

**単一ノード・複数 GPU**

- [x] vLLM
- [x] SGLang
- [x] MindIE

**複数ノード・複数 GPU**

- [x] vLLM
- [x] SGLang
- [x] MindIE

!!! tip

    関連ドキュメント：

    **vLLM**：[分散推論とサービング](https://docs.vllm.ai/en/latest/serving/distributed_serving.html)

    **SGLang**：[複数ノードへのデプロイ](https://docs.sglang.io/references/multi_node_deployment/multi_node_index.html)

    **MindIE**：[複数ノード推論](https://www.hiascend.com/document/detail/zh/mindie/20RC2/envdeployment/instg/mindie_instg_0027.html)

## インストール { #installation }

### 登録済みワーカーの名前を変更するには？ { #how-can-i-change-the-registered-worker-name }

GPUStack の実行時に `--worker-name` フラグでカスタム名を指定できます。

```diff
sudo docker run -d --name gpustack \
    ...
    gpustack/gpustack \
+    --worker-name New-Name
```

### 登録済みワーカーの IP を変更するには？ { #how-can-i-change-the-registered-worker-ip }

GPUStack の実行時に `--worker-ip` フラグでカスタム IP を指定できます。

```diff
sudo docker run -d --name gpustack \
    ...
    gpustack/gpustack \
+    --worker-ip xx.xx.xx.xx
```

### GPUStack のデータはどこに保存されますか？ { #where-are-gpustacks-data-stored }

GPUStack コンテナーの実行時は、`--volume/-v` パラメーターで Docker ボリュームをマウントします。デフォルトのデータは Docker データディレクトリの volumes サブディレクトリにあり、デフォルトパスは次のとおりです。

```bash
/var/lib/docker/volumes/gpustack-data/_data
```

次の方法で確認できます。

```bash
docker volume ls
docker volume inspect gpustack-data
```

カスタムパスに変更するには、コンテナー実行時のマウント設定を変更します。たとえば、ホストの `/data/gpustack` をマウントするには次を使用します。

```diff
sudo docker run -d --name gpustack \
    ...
    --volume /var/run/docker.sock:/var/run/docker.sock \
-    --volume gpustack-data:/var/lib/gpustack \
+    --volume /data/gpustack:/var/lib/gpustack \
    ...
    gpustack/gpustack
```

### モデルファイルはどこに保存されますか？ { #where-are-model-files-stored }

GPUStack コンテナーの実行時は、`--volume/-v` パラメーターで Docker ボリュームをマウントします。デフォルトのキャッシュは Docker データディレクトリの volumes サブディレクトリにあり、デフォルトパスは次のとおりです。

```bash
/var/lib/docker/volumes/gpustack-data/_data/cache
```

次の方法で確認できます。

```bash
docker volume ls
docker volume inspect gpustack-data
```

カスタムパスに変更するには、コンテナー実行時のマウント設定を変更します。

たとえば、ホストの `/data/model-cache` をマウントするには次を使用します。

```diff
sudo docker run -d --name gpustack \
    ...
    --volume gpustack-data:/var/lib/gpustack \
+    --volume /data/model-cache:/var/lib/gpustack/cache \
    ...
    gpustack/gpustack
```

---

## モデルの管理 { #managing-models }

### Hugging Face からモデルをデプロイするには？ { #how-can-i-deploy-the-model-from-hugging-face }

Hugging Face からモデルをデプロイするには、サーバーノードとモデルインスタンスが配置されるワーカーノードから Hugging Face にアクセスできる必要があります。ミラーを使用することもできます。

たとえば、`hf-mirror.com` ミラーは次のように設定します。

```diff
sudo docker run -d --name gpustack \
+    -e HF_ENDPOINT=https://hf-mirror.com \
    ...
    gpustack/gpustack
```

---

### ローカルパスからモデルをデプロイするには？ { #how-can-i-deploy-the-model-from-local-path }

ローカルパスからデプロイする場合、対象のワーカーからモデルパスにアクセスできることを確認します。

モデルが特定のワーカーにある場合、ワーカーセレクターでそのワーカーにモデルをデプロイできます。

複数ノードに共有ストレージをマウントする方法もあります。

モデルファイルをコンテナー内にもマウントし、ホストのディレクトリとコンテナー内のマウントパスを同一にする必要があります。

Safetensors モデルをローカルパスからデプロイする場合、パスは **`*.safetensors`、`config.json` などのファイルが含まれるモデルディレクトリの絶対パス**でなければなりません。

GGUF モデルをローカルパスからデプロイする場合、パスは **`.gguf` ファイルの絶対パス**でなければなりません。分割ファイルの場合は、最初の `.gguf` ファイル（00001）の絶対パスを使用します。

---

### モデルが `Pending` のままになる場合は？ { #what-should-i-do-if-the-model-is-stuck-in-pending-state }

`Pending` は、現在モデルの要件を満たすワーカーがないことを示します。`Pending` 状態にポインターを合わせると理由を確認できます。

まず `Workers` セクションで、ワーカーの状態が Ready であることを確認します。

その後の処理はバックエンドによって異なります。たとえば vLLM の場合は次のとおりです。

vLLM はデフォルトで、各 GPU のメモリーの 90% 超が使用可能である必要があります（`--gpu-memory-utilization` パラメーターで制御）。割り当て可能な GPU メモリーが 90% を超えていることを確認してください。ほかのモデルが `Error` や `Downloading` の状態でも、その GPU メモリーはすでに割り当て済みであることに注意してください。

すべての GPU で 90% 超のメモリーが使用可能なのに `Pending` が続く場合、メモリーが不足しています。`safetensors` モデルの必要な GPU メモリー（GB）は、次のように推定できます。

```
GPU Memory (GB) = Model weight size (GB) * 1.2 + 2
```

割り当て可能な GPU メモリーが 90% 未満でも、より少ない割り当てで実行できることがわかっている場合は、`--gpu-memory-utilization` を調整できます。たとえば `Edit Model` → `Advanced` → `Backend Parameters` に `--gpu-memory-utilization=0.5` を追加すると、GPU メモリーの 50% を割り当てます。

**注意：** 実行後にエラーになり、ログに `CUDA: out of memory` がある場合は、割り当てた GPU メモリーが不足しています。`--gpu-memory-utilization` をさらに調整する、リソースを追加する、より小さいモデルをデプロイするなどの対応が必要です。

モデルのコンテキストサイズも、必要な GPU メモリーに影響します。`--max-model-len` を調整してコンテキストを小さくできます。GPUStack で指定しない場合のデフォルトは 8192 です。バックエンドパラメーターに指定した場合は、その設定が適用されます。

必要に応じて、`--max-model-len=2048` のように小さくできます。ただし、各推論リクエストの最大トークン数は `--max-model-len` を超えられません。小さすぎるコンテキストは、推論の打ち切りを招くことがあります。

`--enforce-eager` も GPU メモリーの使用量削減に役立ちます。ただし vLLM のこのパラメーターは eager 実行を強制し、グラフ実行のように最適化のために遅延させず、呼び出された演算をすぐに実行します。デバッグは容易になりますが、グラフ実行の最適化がなくなり、実行速度や性能が低下する可能性があります。

**SGLang** と **MindIE** も同様の流れですが、パラメーターが異なります。詳細は、[組み込み推論バックエンド](user-guide/built-in-inference-backends.md)をご覧ください。

---

### モデルが `Scheduled` のままになる場合は？ { #what-should-i-do-if-the-model-is-stuck-in-scheduled-state }

モデルが配置された GPUStack コンテナーを再起動してみてください。問題が続く場合は、[ワーカーログ](troubleshooting.md#view-gpustack-logs)で原因を調べてください。

### モデルが `Error` のままになる場合は？ { #what-should-i-do-if-the-model-is-stuck-in-error-state }

`Error` 状態にポインターを合わせて理由を確認します。`View More` ボタンがある場合はクリックし、モデルログ内のエラーメッセージから原因を調べます。

### 公式 vLLM イメージを基にしたカスタムバックエンドで `PYPI_PACKAGES_INSTALL` を使うと、なぜモデルが起動しないのですか？ { #why-does-the-model-fail-to-start-when-using-a-custom-backend-version-based-on-the-official-vllm-image-with-pypi_packages_install }

公式 vLLM イメージを基にしたカスタム vLLM バックエンドバージョンで、`PYPI_PACKAGES_INSTALL` によって Python パッケージを追加すると、モデルが起動しない場合があります。

次のようなエラーが表示される場合があります。

```bash
/gpustack-command-xxxxxxxx: 40: /path/to/your_model: Permission denied
```

これは、イメージのデフォルトのエントリーポイントではなく、カスタムコマンドでコンテナーを起動することが原因です。

`Inference Backends` → `vLLM` → `Edit` → `Versions Config` で対象のバージョンを編集し、Override Image Entrypoint を次のように設定します。

```bash
vllm serve
```

### モデルを削除してもディスク容量が空かないのはなぜですか？ { #why-doesnt-deleting-a-model-free-up-disk-space }

再デプロイ時のモデル再ダウンロードを避けるためです。`Model Files` で手動で削除する必要があります。

---

## モデルの利用 { #using-models }

### 「At most 1 image(s) may be provided in one request」というエラーを解消するには？ { #how-can-i-resolve-the-error-at-most-1-images-may-be-provided-in-one-request }

これは vLLM の制限です。必要に応じて `Edit Model → Advanced → Backend Parameters` で `--limit-mm-per-prompt` を調整できます。たとえば `--limit-mm-per-prompt={"image": 4}` は、1 回の推論リクエストで最大 4 枚の画像をサポートする設定です。詳細は[こちら](https://docs.vllm.ai/en/latest/configuration/engine_args/#-limit-mm-per-prompt)をご覧ください。

---

## GPUStack の管理 { #managing-gpustack }

### プロキシ経由で GPUStack を使用するには？ { #how-do-i-use-gpustack-behind-a-proxy }

GPUStack の実行時に標準のプロキシ環境変数を渡すことを推奨します。

次の例では、NO_PROXY 環境変数に指定したアドレス以外のすべてのリクエストを、指定したプロキシに転送するよう GPUStack を設定します。

```bash
docker run -d --name gpustack \
    -e HTTPS_PROXY="http://proxy-server:port" \
    -e HTTP_PROXY="http://proxy-server:port" \
    -e NO_PROXY="127.0.0.1,10.0.0.0/8,192.168.0.0/16,172.16.0.0/16,localhost,cluster.local" \
    ...
```

!!! note

    - IP アドレスとプロキシアドレスを環境に合わせて置き換えてください。
    - 認証が必要なプロキシには、`http://username:password@proxy-server:port` 形式を使用します。パスワードの特殊文字には URL エンコードが必要な場合があります。
