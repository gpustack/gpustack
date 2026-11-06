---
translation_status: translated
---

# Kurulum Gereksinimleri { #installation-requirements }

Bu sayfada GPUStack çalıştıran düğümler için yazılım ve ağ gereksinimleri özetlenmektedir.

## İşletim Sistemi Gereksinimleri { #operating-system-requirements }

GPUStack, **AMD64** ve **ARM64** mimarilerindeki modern Linux dağıtımlarının çoğunu destekler.

!!! note

    - GPUStack'in doğrudan PyPI aracılığıyla kurulması desteklenmez. En iyi uyumluluk için sağlanan Docker görüntülerini kullanın.
    - Düğümler arasında tutarlı durum senkronizasyonu sağlamak için Ağ Zaman Protokolü (NTP) paketi kurulmalıdır.

## Veritabanı Gereksinimleri { #database-requirements }

GPUStack sunucusu durumunu bir SQL veritabanında saklar. Sunucu kapsayıcı görüntüsü, herhangi bir yapılandırma gerektirmeden kutudan çıktığı gibi çalışan yerleşik bir PostgreSQL veritabanını bir araya getirir.

Alternatif olarak, `--database-url` bağımsız değişkenini (veya `GPUSTACK_DATABASE_URL` ortam değişkenini) ayarlayarak harici bir veritabanı kullanabilirsiniz. Aşağıdaki veritabanları uyumludur:

| Veritabanı | Doğrulanmış Sürümler | Veritabanı URL Formatı |
| ---------- | --------------------- | ------------------------------------------------ |
| PostgreSQL | `13.0+`               | `postgresql://username:password@host:port/dbname` |
| MySQL | `8.0.36+`             | `mysql://username:password@host:port/dbname`      |
| openGauss | `v6.0.5`, `v7.0.0-RC3` | `postgresql://username:password@host:port/dbname` |
| OceanBase | `CE-v4.3.5`           | `mysql://username:password@host:port/dbname`      |

!!! note

    openGauss PostgreSQL uyumludur ve OceanBase MySQL uyumludur, dolayısıyla sırasıyla `postgresql://` ve `mysql://` URL şemalarını kullanırlar.

Yapılandırma örnekleri için [Harici Veritabanı Kullanma](installation.md#using-an-external-database) konusuna bakın.

## Hızlandırıcı Çalışma Zamanı Gereksinimleri { #accelerator-runtime-requirements }

GPUStack, çıkarım arka uçları olarak çeşitli Genel Amaçlı Hızlandırıcıları destekler:

- [x] NVIDIA GPU
- [x] AMD GPU
- [x] NPU'ya yüksel
- [x] Hygon DCU
- [x] MetaX GPU
- [x] MThreads GPU (Deneysel)
- [x] Iluvatar GPU (Deneysel)
- [x] Cambricon MLU (Deneysel)
- [x] T-Head PPU (Deneysel)

GPUStack'i çalıştırmadan önce gerekli tüm sürücülerin ve araç setlerinin kurulu olduğundan emin olun.

### NVIDIA GPU { #nvidia-gpu }

#### Gereksinimler { #requirements }

- [NVIDIA GPU Sürücüsü](https://www.nvidia.com/en-us/drivers/), NVIDIA CUDA 12.8 veya üstünü destekler.
- [NVIDIA Kapsayıcı Araç Seti](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit)

!!! note "Kubernetes GPU Düğümleri"

    GPU'lar Kubernetes düğümleri tarafından sağlandığında, sürücüyü ve araç setini her düğüme yukarıdaki gibi yükleyin veya [NVIDIA GPU Operatörü](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/overview.html)'nin bunları yönetmesine izin verin. NVIDIA GPU Operatörü zaten kuruluysa, GPUStack'in cihaz yönetimiyle çakışan cihaz eklentisini, GPU Özellik Keşfi, Düğüm Özellik Keşfi, MIG Yöneticisi ve CDI bileşenlerini devre dışı bırakın. Ayrıntılar için [Satıcı Önkoşulları](https://docs.gpustack.ai/gpustack-operator/main/docs/getting-started/vendor-prerequisites/#nvidia)'e bakın.

Doğrulamak için aşağıdaki komutları çalıştırın:

```bash
sudo nvidia-smi

# If using Docker
sudo docker info 2>/dev/null | grep -q "nvidia" \
    && echo "NVIDIA Container Toolkit OK" \
    || (echo "NVIDIA Container Toolkit not configured"; exit 1)
```

#### Desteklenen Çıkarım Arka Uçları { #supported-inference-backends }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] [SGLang](https://github.com/sgl-project/sglang)
- [x] [VoxBox](https://github.com/gpustack/vox-box)
- [x] Özel

### AMD GPU { #amd-gpu }

#### Gereksinimler { #requirements_1 }

- [AMD GPU Sürücüsü](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/), AMD ROCm 7.0 veya üstünü destekler.
- [AMD Konteyner Çalışma Zamanı](https://instinct.docs.amd.com/projects/container-toolkit/en/latest/container-runtime/overview.html)

!!! note "Kubernetes GPU Düğümleri"

    GPU'lar Kubernetes düğümleri tarafından sağlandığında, sürücüyü yukarıdaki gibi her düğüme yükleyin veya [AMD GPU Operatörü](https://instinct.docs.amd.com/projects/gpu-operator/en/latest/index.html)'nin yönetmesine izin verin. AMD GPU Operatörü zaten kuruluysa, GPUStack'in cihaz yönetimiyle çakışan cihaz eklentisini, düğüm etiketleyicisini, birlikte verilen Node Özellik Keşfi ve Çekirdek Modül Yönetimi bileşenlerini devre dışı bırakın. Ayrıntılar için [Satıcı Önkoşulları](https://docs.gpustack.ai/gpustack-operator/main/docs/getting-started/vendor-prerequisites/#amd)'e bakın.

Doğrulamak için aşağıdaki komutları çalıştırın:

```bash
sudo amd-smi static

# If using Docker
sudo docker info 2>/dev/null | grep -q "amd" \
    && echo "AMD Container Toolkit OK" \
    || (echo "AMD Container Toolkit not configured"; exit 1)
```

#### Desteklenen Çıkarım Arka Uçları { #supported-inference-backends_1 }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] [SGLang](https://github.com/sgl-project/sglang) (gfx9 serisi GPU'lar gerektirir)
- [x] Özel

### Ascend NPU { #ascend-npu }

#### Gereksinimler { #requirements_2 }

- [Ascend NPU Sürücüsü](https://www.hiascend.com/hardware/firmware-drivers/community)
- [Docker Çalışma Zamanını Yükselt](https://www.hiascend.com/document/detail/zh/mindcluster/730/clustersched/dlug/dlug_installation_017.html)

Doğrulamak için aşağıdaki komutları çalıştırın:

```bash
sudo npu-smi info

# If using Docker
sudo docker info 2>/dev/null | grep -q "ascend" \
    && echo "Ascend Container Toolkit OK" \
    || (echo "Ascend Container Toolkit not configured"; exit 1)
```

#### Desteklenen Cihazlar { #supported-devices }

- [x] Yükselen NPU 910C serisi
- [x] Yükselen NPU 910B serisi (910B1 ~ 910B4)
- [x] NPU 310P3'e yüksel

#### Desteklenen Çıkarım Arka Uçları { #supported-inference-backends_2 }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] [SGLang](https://github.com/sgl-project/sglang)
- [x] [MindIE](https://www.hiascend.com/document/detail/zh/mindie/21RC2/index/index.html)
- [x] Özel

### Hygon DCU { #hygon-dcu }

#### Gereksinimler { #requirements_3 }

- [Hygon DCU Sürücüsü](https://developer.sourcefind.cn/tool/)
- [Hygon DTK Araç Seti](https://developer.sourcefind.cn/tool/)

!!! note "Kubernetes GPU Düğümleri"

    DCU'lar Kubernetes düğümleri tarafından sağlandığında, sürücüyü ve araç kitini her düğüme yukarıdaki gibi yükleyin veya [Hygon DCU Operatörü](https://developer.sourcefind.cn/document/7541efc4-54b1-11f1-8265-0242ac150003)'nin bunları yönetmesine izin verin. Hygon DCU Operatörü zaten kuruluysa, GPUStack'in cihaz yönetimiyle çakışan cihaz eklentisini ve düğüm etiketleyici bileşenlerini devre dışı bırakın. Ayrıntılar için [Satıcı Önkoşulları](https://docs.gpustack.ai/gpustack-operator/main/docs/getting-started/vendor-prerequisites/#hygon)'e bakın.

Doğrulamak için aşağıdaki komutları çalıştırın:

```bash
sudo hy-smi
```

#### Desteklenen Cihazlar { #supported-devices_1 }

- [x] Hygon DCU'lar

#### Desteklenen Çıkarım Arka Uçları { #supported-inference-backends_3 }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] Özel

### MetaX GPU { #metax-gpu }

#### Gereksinimler { #requirements_4 }

- [MetaX GPU Sürücüsü](https://developer.metax-tech.com/softnova/download?package_kind=Driver)
- [MetaX MACA SDK](https://developer.metax-tech.com/softnova/download?package_kind=SDK)

!!! note "Kubernetes GPU Düğümleri"

    GPU'lar Kubernetes düğümleri tarafından sağlandığında her düğüme yukarıdaki gibi sürücüyü ve SDK'yı yükleyin veya bunları [MetaX GPU Operator](https://developer.metax-tech.com/api/client/document/preview/1411/k8s/00_overview.html) yönetsin. MetaX GPU Operator'ın cihaz eklentisi devre dışı bırakılamaz. Bu nedenle operatör zaten kuruluysa, düğümdeki sürücü ve SDK'yı koruyarak operatörü kaldırın; ardından düğümleri GPUStack'e ekleyin. Ayrıntılar için [Üretici Önkoşulları](https://docs.gpustack.ai/gpustack-operator/main/docs/getting-started/vendor-prerequisites/#metax) bölümüne bakın.

Doğrulamak için aşağıdaki komutları çalıştırın:

```bash
sudo mx-smi
```

#### Desteklenen Çıkarım Arka Uçları { #supported-inference-backends_4 }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] [SGLang](https://github.com/sgl-project/sglang)
- [x] Özel

### MThreads GPU'su { #mthreads-gpu }

#### Gereksinimler { #requirements_5 }

- [MThreads GPU Sürücüsü](https://docs.mthreads.com/)
- [MThreads Konteyner Araç Seti](https://developer.mthreads.com/sdk/download/CloudNative)

!!! note "Kubernetes GPU Düğümleri"

    GPU'lar Kubernetes düğümleri tarafından sağlandığında, sürücüyü ve araç setini her düğüme yukarıdaki gibi yükleyin veya bunları [MT GPU Operatörünün](https://docs.mthreads.com/cloud-native/cloud-native-doc-online/introduction/) yönetmesine izin verin. MT GPU Operatörünün cihaz eklentisi varsayılan `full` modunda devre dışı bırakılamaz, bu nedenle operatör zaten yüklüyse, düğümleri GPUStack'e eklemeden önce cihaz eklentisi olmadan `core` moduna geçin veya sürücüyü ve araç setini düğümde tutarak operatörü kaldırın. Ayrıntılar için [Satıcı Önkoşulları](https://docs.gpustack.ai/gpustack-operator/main/docs/getting-started/vendor-prerequisites/#moore-threads)'e bakın.

Doğrulamak için aşağıdaki komutları çalıştırın:

```bash
sudo mthreads-gmi

# If using Docker
sudo docker info 2>/dev/null | grep -q "mthreads" \
    && echo "MThreads Container Toolkit OK" \
    || (echo "MThreads Container Toolkit not configured"; exit 1)
```

#### Desteklenen Çıkarım Arka Uçları { #supported-inference-backends_5 }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] [SGLang](https://github.com/sgl-project/sglang)
- [x] Özel

### Iluvatar GPU { #iluvatar-gpu }

#### Gereksinimler { #requirements_6 }

- [Iluvatar GPU Sürücüsü](https://support.iluvatar.com/#/login)
- [Iluvatar Konteyner Araç Takımı](https://github.com/Deep-Spark/ix-container-toolkit)

!!! note "Kubernetes GPU Düğümleri"

    GPU'lar Kubernetes düğümleri tarafından sağlandığında, sürücüyü ve araç setini her düğüme yukarıdaki gibi yükleyin veya [ix-GPU-Operator](https://developer.iluvatar.com/docs/generaldocs_ix_gpu_operator)'in bunları yönetmesine izin verin. ix-GPU-Operator zaten kuruluysa, GPUStack'in cihaz yönetimiyle çakışan cihaz eklentisini ve özellik bulma bileşenlerini devre dışı bırakın. Ayrıntılar için [Satıcı Önkoşulları](https://docs.gpustack.ai/gpustack-operator/main/docs/getting-started/vendor-prerequisites/#iluvatar)'e bakın.

Doğrulamak için aşağıdaki komutları çalıştırın:

```bash
sudo ixsmi

# If using Docker
sudo docker info 2>/dev/null | grep -q "iluvatar" \
    && echo "Iluvatar Container Toolkit OK" \
    || (echo "Iluvatar Container Toolkit not configured"; exit 1)
```

#### Desteklenen Çıkarım Arka Uçları { #supported-inference-backends_6 }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] Özel

### Cambricon MLU { #cambricon-mlu }

#### Gereksinimler { #requirements_7 }

- Cambricon MLU Sürücüsü
- Cambricon NeuWare Araç Takımı

Doğrulamak için aşağıdaki komutları çalıştırın:

```bash
sudo cnmon
```

#### Desteklenen Çıkarım Arka Uçları { #supported-inference-backends_7 }

- [x] Özel

### T-Head PPU { #t-head-ppu }

#### Gereksinimler { #requirements_8 }

- T-Head PPU Sürücüsü
- T-Head PPU SDK'sı

Doğrulamak için aşağıdaki komutları çalıştırın:

```bash
sudo ppu-smi
```

#### Desteklenen Çıkarım Arka Uçları { #supported-inference-backends_8 }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] [SGLang](https://github.com/sgl-project/sglang)
- [x] Özel

## Ağ Gereksinimleri { #networking-requirements }

### Bağlantı Gereksinimleri { #connectivity-requirements }

GPUStack'in düzgün çalışması için aşağıdaki ağ bağlantısı gereklidir:

**Sunucudan Çalışana:** Sunucu, proxy çıkarım istekleri için çalışanlara ulaşabilmelidir.

**İşçiden Sunucuya:** Çalışanların kaydolmak ve güncellemeleri göndermek için sunucuya ulaşabilmeleri gerekir.

**İşçiden İşçiye:** Birden fazla çalışan arasında dağıtılmış çıkarım için gereklidir.

### Bağlantı Noktası Gereksinimleri { #port-requirements }

GPUStack iletişim için bu bağlantı noktalarını kullanır:

#### Sunucu Bağlantı Noktaları { #server-ports }

| Bağlantı Noktası | Açıklama |
| --------- | ------------------------------------------------------------ |
| TCP 80 | GPUStack UI ve API uç noktaları için varsayılan bağlantı noktası |
| TCP 443 | GPUStack UI ve API uç noktaları için varsayılan bağlantı noktası (TLS etkin) |
| TCP 10161 | Sunucu ölçümleri uç noktası için varsayılan bağlantı noktası |
| TCP 30080 | GPUStack sunucu dahili API'si için varsayılan bağlantı noktası |
| TCP 5432 | Gömülü Postgres Veritabanı için varsayılan bağlantı noktası |

#### Worker Bağlantı Noktaları { #worker-ports }

| Bağlantı Noktası | Açıklama |
| --------------- | -------------------------------------------------------------- |
| TCP 10150 | GPUStack çalışan için varsayılan bağlantı noktası |
| TCP 10151 | Çalışan ölçümleri uç noktası için varsayılan bağlantı noktası |
| TCP 40000-40063 | Çıkarım hizmetleri için bağlantı noktası aralığı |
| TCP 41000-41999 | Ray hizmetleri için bağlantı noktası aralığı (vLLM kullanılarak dağıtılmış dağıtım) |

##### Ray Bağlantı Noktalarıyla Dağıtılmış vLLM { #distributed-vllm-with-ray-ports }

Dağıtılmış vLLM kullanıldığında GPUStack, Ray hizmetleri için yukarıdaki bağlantı noktası aralığını ayrıştıracak ve bunları aşağıdaki sırayla atayacaktır:

1. GCS sunucu bağlantı noktası (aralığın ilk bağlantı noktası)
2. İstemci Sunucu bağlantı noktası (uyumluluk için ayrılmıştır, artık kullanılmamaktadır, bkz. https://github.com/gpustack/gpustack/issues/4171)
3. Kontrol paneli bağlantı noktası
4. Kontrol paneli gRPC bağlantı noktası (Ray 2.45.0'dan beri artık kullanılmıyor, geriye dönük uyumluluk için saklanıyor)
5. Kontrol paneli aracısı gRPC bağlantı noktası
6. Kontrol paneli aracısı dinleme bağlantı noktası
7. Metrik dışa aktarma bağlantı noktası
8. Düğüm Yöneticisi bağlantı noktası
9. Nesne Yöneticisi bağlantı noktası
10. Raylet çalışma zamanı env aracısı bağlantı noktası
11. Çalışan için minimum port numarası
12. Çalışan için maksimum port numarası (aralığın son portu)

Ray bağlantı noktaları hakkında daha fazla ayrıntı için [Ray belgelerine](https://docs.ray.io/en/latest/ray-core/configure.html#ports-configurations) bakın.

#### Gömülü Ağ Geçidi Bağlantı Noktaları { #embedded-gateway-ports }

Hem sunucu hem de çalışan için yerleşik ağ geçidi, dahili iletişim için aşağıdaki bağlantı noktalarını kullanır.

| Bağlantı Noktası | Sunucu | Açıklama |
| --------- | --------- | ---------------------------------------------------- |
| TCP 18443 | 127.0.0.1 | HTTPS aracılığıyla hizmet veren dosya tabanlı APIServer'ın bağlantı noktası |
| TCP 15000 | 127.0.0.1 | Envoy ağ geçidi için yönetim bağlantı noktası |
| TCP 15021 | 0.0.0.0   | Envoy ağ geçidi için sağlık kontrolü bağlantı noktası |
| TCP 15090 | 0.0.0.0   | Envoy ağ geçidi için ölçüm bağlantı noktası |
| TCP 9876 | 127.0.0.1 | Pilot keşif için iç gözlem portu |
| TCP 15010 | 127.0.0.1 | HTTP/gRPC aracılığıyla XDS'ye hizmet veren Pilot keşif bağlantı noktası |
| TCP 15012 | 127.0.0.1 | Güvenli gRPC aracılığıyla XDS'ye hizmet veren Pilot keşif bağlantı noktası |
| TCP 15020 | 0.0.0.0   | Pilot-acente için ölçüm portu |
| TCP 8888 | 127.0.0.1 | XDS'ye HTTP üzerinden hizmet veren Denetleyici Bağlantı Noktası |
| TCP 15051 | 127.0.0.1 | gRPC aracılığıyla XDS'ye hizmet veren Denetleyici Bağlantı Noktası |
