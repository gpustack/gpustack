---
translation_status: translated
---

# インストール要件 { #installation-requirements }

このページでは、GPUStack を実行するノードのソフトウェアとネットワークの要件を説明します。

## オペレーティングシステムの要件 { #operating-system-requirements }

GPUStack は、**AMD64** および **ARM64** アーキテクチャの主要な最新 Linux ディストリビューションに対応しています。

!!! note

    - PyPI 経由での直接インストールには対応していません。互換性を確保するため、提供されている Docker イメージを使用してください。
    - ノード間の状態同期を一貫させるため、Network Time Protocol（NTP）パッケージが必要です。

## データベースの要件 { #database-requirements }

GPUStack サーバーは状態を SQL データベースに保存します。サーバーのコンテナーイメージには組み込み PostgreSQL が含まれ、設定なしでそのまま動作します。

`--database-url` 引数（または `GPUSTACK_DATABASE_URL` 環境変数）で外部データベースも使用できます。次のデータベースに対応しています。

| データベース | 検証済みバージョン | データベース URL の形式 |
| ---------- | --------------------- | ------------------------------------------------ |
| PostgreSQL | `13.0+`               | `postgresql://username:password@host:port/dbname` |
| MySQL      | `8.0.36+`             | `mysql://username:password@host:port/dbname`      |
| openGauss  | `v6.0.5`, `v7.0.0-RC3` | `postgresql://username:password@host:port/dbname` |
| OceanBase  | `CE-v4.3.5`           | `mysql://username:password@host:port/dbname`      |

!!! note

    openGauss は PostgreSQL、OceanBase は MySQL と互換性があるため、URL スキームにはそれぞれ `postgresql://` と `mysql://` を使用します。

設定例は、[外部データベースの使用](installation.md#using-an-external-database)をご覧ください。

## アクセラレーターランタイムの要件 { #accelerator-runtime-requirements }

GPUStack は、推論バックエンド用に次の汎用アクセラレーターに対応しています。

- [x] NVIDIA GPU
- [x] AMD GPU
- [x] Ascend NPU
- [x] Hygon DCU
- [x] MetaX GPU
- [x] MThreads GPU（実験的）
- [x] Iluvatar GPU（実験的）
- [x] Cambricon MLU（実験的）
- [x] T-Head PPU（実験的）

GPUStack を実行する前に、必要なドライバーとツールキットがすべてインストールされていることを確認してください。

### NVIDIA GPU { #nvidia-gpu }

#### 要件 { #requirements }

- NVIDIA CUDA 12.8 以降に対応する [NVIDIA GPU ドライバー](https://www.nvidia.com/en-us/drivers/)。
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit)

!!! note "Kubernetes の GPU ノード"

    Kubernetes ノードの GPU を使用する場合は、上記のとおり各ノードにドライバーとツールキットをインストールするか、[NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/overview.html) で管理します。NVIDIA GPU Operator がすでにインストールされている場合は、GPUStack のデバイス管理と競合する device plugin、GPU Feature Discovery、Node Feature Discovery、MIG Manager、CDI コンポーネントを無効にします。詳細は[ベンダー別の前提条件](https://docs.gpustack.ai/gpustack-operator/main/docs/getting-started/vendor-prerequisites/#nvidia)をご覧ください。

次のコマンドで確認します。

```bash
sudo nvidia-smi

# If using Docker
sudo docker info 2>/dev/null | grep -q "nvidia" \
    && echo "NVIDIA Container Toolkit OK" \
    || (echo "NVIDIA Container Toolkit not configured"; exit 1)
```

#### 対応推論バックエンド { #supported-inference-backends }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] [SGLang](https://github.com/sgl-project/sglang)
- [x] [VoxBox](https://github.com/gpustack/vox-box)
- [x] カスタム

### AMD GPU { #amd-gpu }

#### 要件 { #requirements_1 }

- AMD ROCm 7.0 以降に対応する [AMD GPU ドライバー](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)。
- [AMD Container Runtime](https://instinct.docs.amd.com/projects/container-toolkit/en/latest/container-runtime/overview.html)

!!! note "Kubernetes の GPU ノード"

    Kubernetes ノードの GPU を使用する場合は、上記のとおり各ノードにドライバーをインストールするか、[AMD GPU Operator](https://instinct.docs.amd.com/projects/gpu-operator/en/latest/index.html) で管理します。AMD GPU Operator がすでにインストールされている場合は、GPUStack のデバイス管理と競合する device plugin、node labeller、同梱の Node Feature Discovery、Kernel Module Management コンポーネントを無効にします。詳細は[ベンダー別の前提条件](https://docs.gpustack.ai/gpustack-operator/main/docs/getting-started/vendor-prerequisites/#amd)をご覧ください。

次のコマンドで確認します。

```bash
sudo amd-smi static

# If using Docker
sudo docker info 2>/dev/null | grep -q "amd" \
    && echo "AMD Container Toolkit OK" \
    || (echo "AMD Container Toolkit not configured"; exit 1)
```

#### 対応推論バックエンド { #supported-inference-backends_1 }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] [SGLang](https://github.com/sgl-project/sglang)（gfx9 シリーズの GPU が必要）
- [x] カスタム

### Ascend NPU { #ascend-npu }

#### 要件 { #requirements_2 }

- [Ascend NPU ドライバー](https://www.hiascend.com/hardware/firmware-drivers/community)
- [Ascend Docker Runtime](https://www.hiascend.com/document/detail/zh/mindcluster/730/clustersched/dlug/dlug_installation_017.html)

次のコマンドで確認します。

```bash
sudo npu-smi info

# If using Docker
sudo docker info 2>/dev/null | grep -q "ascend" \
    && echo "Ascend Container Toolkit OK" \
    || (echo "Ascend Container Toolkit not configured"; exit 1)
```

#### 対応デバイス { #supported-devices }

- [x] Ascend NPU 910C シリーズ
- [x] Ascend NPU 910B シリーズ（910B1～910B4）
- [x] Ascend NPU 310P3

#### 対応推論バックエンド { #supported-inference-backends_2 }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] [SGLang](https://github.com/sgl-project/sglang)
- [x] [MindIE](https://www.hiascend.com/document/detail/zh/mindie/21RC2/index/index.html)
- [x] カスタム

### Hygon DCU { #hygon-dcu }

#### 要件 { #requirements_3 }

- [Hygon DCU ドライバー](https://developer.sourcefind.cn/tool/)
- [Hygon DTK ツールキット](https://developer.sourcefind.cn/tool/)

!!! note "Kubernetes の GPU ノード"

    Kubernetes ノードの DCU を使用する場合は、上記のとおり各ノードにドライバーとツールキットをインストールするか、[Hygon DCU Operator](https://developer.sourcefind.cn/document/7541efc4-54b1-11f1-8265-0242ac150003) で管理します。Hygon DCU Operator がすでにインストールされている場合は、GPUStack のデバイス管理と競合する device plugin と node labeller コンポーネントを無効にします。詳細は[ベンダー別の前提条件](https://docs.gpustack.ai/gpustack-operator/main/docs/getting-started/vendor-prerequisites/#hygon)をご覧ください。

次のコマンドで確認します。

```bash
sudo hy-smi
```

#### 対応デバイス { #supported-devices_1 }

- [x] Hygon DCU

#### 対応推論バックエンド { #supported-inference-backends_3 }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] カスタム

### MetaX GPU { #metax-gpu }

#### 要件 { #requirements_4 }

- [MetaX GPU ドライバー](https://developer.metax-tech.com/softnova/download?package_kind=Driver)
- [MetaX MACA SDK](https://developer.metax-tech.com/softnova/download?package_kind=SDK)

!!! note "Kubernetes の GPU ノード"

    Kubernetes ノードの GPU を使用する場合は、上記のとおり各ノードにドライバーと SDK をインストールするか、[MetaX GPU Operator](https://developer.metax-tech.com/api/client/document/preview/1411/k8s/00_overview.html) で管理します。MetaX GPU Operator の device plugin は無効化できません。Operator がすでにある場合は、ノード上のドライバーと SDK を残してアンインストールしてから、ノードを GPUStack に追加してください。詳細は[ベンダー別の前提条件](https://docs.gpustack.ai/gpustack-operator/main/docs/getting-started/vendor-prerequisites/#metax)をご覧ください。

次のコマンドで確認します。

```bash
sudo mx-smi
```

#### 対応推論バックエンド { #supported-inference-backends_4 }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] [SGLang](https://github.com/sgl-project/sglang)
- [x] カスタム

### MThreads GPU { #mthreads-gpu }

#### 要件 { #requirements_5 }

- [MThreads GPU ドライバー](https://docs.mthreads.com/)
- [MThreads Container Toolkit](https://developer.mthreads.com/sdk/download/CloudNative)

!!! note "Kubernetes の GPU ノード"

    Kubernetes ノードの GPU を使用する場合は、上記のとおり各ノードにドライバーとツールキットをインストールするか、[MT GPU Operator](https://docs.mthreads.com/cloud-native/cloud-native-doc-online/introduction/) で管理します。MT GPU Operator の device plugin はデフォルトの `full` モードでは無効化できません。すでにインストールされている場合は、device plugin のない `core` モードに切り替えるか、ノード上のドライバーとツールキットを残してアンインストールしてから、ノードを GPUStack に追加してください。詳細は[ベンダー別の前提条件](https://docs.gpustack.ai/gpustack-operator/main/docs/getting-started/vendor-prerequisites/#moore-threads)をご覧ください。

次のコマンドで確認します。

```bash
sudo mthreads-gmi

# If using Docker
sudo docker info 2>/dev/null | grep -q "mthreads" \
    && echo "MThreads Container Toolkit OK" \
    || (echo "MThreads Container Toolkit not configured"; exit 1)
```

#### 対応推論バックエンド { #supported-inference-backends_5 }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] [SGLang](https://github.com/sgl-project/sglang)
- [x] カスタム

### Iluvatar GPU { #iluvatar-gpu }

#### 要件 { #requirements_6 }

- [Iluvatar GPU ドライバー](https://support.iluvatar.com/#/login)
- [Iluvatar Container Toolkit](https://github.com/Deep-Spark/ix-container-toolkit)

!!! note "Kubernetes の GPU ノード"

    Kubernetes ノードの GPU を使用する場合は、上記のとおり各ノードにドライバーとツールキットをインストールするか、[ix-GPU-Operator](https://developer.iluvatar.com/docs/generaldocs_ix_gpu_operator) で管理します。ix-GPU-Operator がすでにインストールされている場合は、GPUStack のデバイス管理と競合する device plugin と feature discovery コンポーネントを無効にします。詳細は[ベンダー別の前提条件](https://docs.gpustack.ai/gpustack-operator/main/docs/getting-started/vendor-prerequisites/#iluvatar)をご覧ください。

次のコマンドで確認します。

```bash
sudo ixsmi

# If using Docker
sudo docker info 2>/dev/null | grep -q "iluvatar" \
    && echo "Iluvatar Container Toolkit OK" \
    || (echo "Iluvatar Container Toolkit not configured"; exit 1)
```

#### 対応推論バックエンド { #supported-inference-backends_6 }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] カスタム

### Cambricon MLU { #cambricon-mlu }

#### 要件 { #requirements_7 }

- Cambricon MLU ドライバー
- Cambricon NeuWare ツールキット

次のコマンドで確認します。

```bash
sudo cnmon
```

#### 対応推論バックエンド { #supported-inference-backends_7 }

- [x] カスタム

### T-Head PPU { #t-head-ppu }

#### 要件 { #requirements_8 }

- T-Head PPU ドライバー
- T-Head PPU SDK

次のコマンドで確認します。

```bash
sudo ppu-smi
```

#### 対応推論バックエンド { #supported-inference-backends_8 }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] [SGLang](https://github.com/sgl-project/sglang)
- [x] カスタム

## ネットワークの要件 { #networking-requirements }

### 接続要件 { #connectivity-requirements }

GPUStack が正常に動作するには、次のネットワーク接続が必要です。

**サーバーからワーカー：** 推論リクエストをプロキシするため、サーバーからワーカーへの接続が必要です。

**ワーカーからサーバー：** 登録と状態更新を送信するため、ワーカーからサーバーへの接続が必要です。

**ワーカー間：** 複数ワーカーにまたがる分散推論で必要です。

### ポート要件 { #port-requirements }

GPUStack は通信に次のポートを使用します。

#### サーバーのポート { #server-ports }

| ポート | 説明 |
| --------- | ------------------------------------------------------------ |
| TCP 80    | GPUStack UI と API エンドポイントのデフォルトポート |
| TCP 443   | GPUStack UI と API エンドポイントのデフォルトポート（TLS 有効時） |
| TCP 10161 | サーバーメトリクスのエンドポイントのデフォルトポート |
| TCP 30080 | GPUStack サーバー内部 API のデフォルトポート |
| TCP 5432  | 組み込み Postgres データベースのデフォルトポート |

#### ワーカーのポート { #worker-ports }

| ポート | 説明 |
| --------------- | -------------------------------------------------------------- |
| TCP 10150       | GPUStack ワーカーのデフォルトポート |
| TCP 10151       | ワーカーメトリクスのエンドポイントのデフォルトポート |
| TCP 40000-40063 | 推論サービスのポート範囲 |
| TCP 41000-41999 | Ray サービスのポート範囲（vLLM 分散デプロイで使用） |

##### Ray を使用した分散 vLLM のポート { #distributed-vllm-with-ray-ports }

分散 vLLM を使用する場合、GPUStack は上記の Ray サービス用ポート範囲を解析し、
次の順序で割り当てます。

1. GCS サーバーポート（範囲の最初のポート）
2. Client Server ポート（互換性のために予約、現在は未使用。https://github.com/gpustack/gpustack/issues/4171 を参照）
3. Dashboard ポート
4. Dashboard gRPC ポート（Ray 2.45.0 以降は未使用、後方互換性のために維持）
5. Dashboard agent gRPC ポート
6. Dashboard agent 待受ポート
7. メトリクスエクスポート用ポート
8. Node Manager ポート
9. Object Manager ポート
10. Raylet runtime env agent ポート
11. ワーカー用の最小ポート番号
12. ワーカー用の最大ポート番号（範囲の最後のポート）

Ray のポートの詳細は、[Ray ドキュメント](https://docs.ray.io/en/latest/ray-core/configure.html#ports-configurations)をご覧ください。

#### 組み込みゲートウェイのポート { #embedded-gateway-ports }

サーバーとワーカーの組み込みゲートウェイは、内部通信に次のポートを使用します。

| ポート | ホスト | 説明 |
| --------- | --------- | ---------------------------------------------------- |
| TCP 18443 | 127.0.0.1 | ファイルベースの APIServer が HTTPS で配信するポート |
| TCP 15000 | 127.0.0.1 | Envoy ゲートウェイの管理ポート |
| TCP 15021 | 0.0.0.0   | Envoy ゲートウェイのヘルスチェックポート |
| TCP 15090 | 0.0.0.0   | Envoy ゲートウェイのメトリクスポート |
| TCP 9876  | 127.0.0.1 | Pilot-discovery のイントロスペクションポート |
| TCP 15010 | 127.0.0.1 | Pilot-discovery が HTTP/gRPC で XDS を配信するポート |
| TCP 15012 | 127.0.0.1 | Pilot-discovery がセキュア gRPC で XDS を配信するポート |
| TCP 15020 | 0.0.0.0   | Pilot-agent のメトリクスポート |
| TCP 8888  | 127.0.0.1 | Controller が HTTP で XDS を配信するポート |
| TCP 15051 | 127.0.0.1 | Controller が gRPC で XDS を配信するポート |
