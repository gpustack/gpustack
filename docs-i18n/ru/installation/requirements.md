---
translation_status: translated
---

# Требования к установке { #installation-requirements }

На этой странице описаны требования к программному обеспечению и сети для узлов, на которых работает GPUStack.

## Требования к операционной системе { #operating-system-requirements }

GPUStack поддерживает большинство современных дистрибутивов Linux на архитектурах **AMD64** и **ARM64**.

!!! note

    - Установка GPUStack напрямую через PyPI не поддерживается. Для лучшей совместимости используйте предоставленные образы Docker.
    - Пакет Network Time Protocol (NTP) должен быть установлен для обеспечения согласованной синхронизации состояний между узлами.

## Требования к базе данных { #database-requirements }

Сервер GPUStack сохраняет свое состояние в базе данных SQL. Образ серверного контейнера включает в себя встроенную базу данных PostgreSQL, которая работает «из коробки» без какой-либо настройки.

Альтернативно вы можете использовать внешнюю базу данных, установив аргумент `--database-url` (или переменную среды `GPUSTACK_DATABASE_URL`). Совместимы следующие базы данных:

| База данных | Проверенные версии | Формат URL-адреса базы данных |
| ---------- | --------------------- | ------------------------------------------------ |
| PostgreSQL | `13.0+`               | `postgresql://username:password@host:port/dbname` |
| MySQL | `8.0.36+`             | `mysql://username:password@host:port/dbname`      |
| openGauss | `v6.0.5`, `v7.0.0-RC3` | `postgresql://username:password@host:port/dbname` |
| OceanBase | `CE-v4.3.5`           | `mysql://username:password@host:port/dbname`      |

!!! note

    openGauss совместим с PostgreSQL, а OceanBase совместим с MySQL, поэтому они используют схемы URL-адресов `postgresql://` и `mysql://` соответственно.

Примеры конфигурации см. в разделе [Использование внешней базы данных](installation.md#using-an-external-database).

## Требования к среде выполнения ускорителя { #accelerator-runtime-requirements }

GPUStack поддерживает различные ускорители общего назначения в качестве механизмов вывода, в том числе:

- [x] Графический процессор NVIDIA
- [x] Графический процессор AMD
- [x] Ascend NPU
- [x] Хайгон DCU
- [x] Графический процессор MetaX
- [x] Графический процессор MThreads (экспериментальный)
- [x] Графический процессор Илуватара (экспериментальный)
- [x] Cambricon MLU (экспериментальный)
- [x] T-Head PPU (экспериментальный)

Перед запуском GPUStack убедитесь, что все необходимые драйверы и наборы инструментов установлены.

### Графический процессор NVIDIA { #nvidia-gpu }

#### Требования { #requirements }

- [Драйвер графического процессора NVIDIA](https://www.nvidia.com/en-us/drivers/), поддерживающий NVIDIA CUDA 12.8 или выше.
- [Набор инструментов для контейнеров NVIDIA](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit)

!!! note "Узлы графического процессора Kubernetes"

    Если графические процессоры предоставляются узлами Kubernetes, установите драйвер и набор инструментов на каждый узел, как указано выше, или позвольте [Оператору NVIDIA GPU](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/overview.html) управлять ими. Если оператор NVIDIA GPU уже установлен, отключите его плагин устройства, обнаружение функций графического процессора, обнаружение функций узла, диспетчер MIG и компоненты CDI, которые конфликтуют с управлением устройствами GPUStack. Подробности см. в разделе [Требования к поставщику](https://docs.gpustack.ai/gpustack-operator/main/docs/getting-started/vendor-prerequisites/#nvidia).

Для проверки выполните следующие команды:

```bash
sudo nvidia-smi

# If using Docker
sudo docker info 2>/dev/null | grep -q "nvidia" \
    && echo "NVIDIA Container Toolkit OK" \
    || (echo "NVIDIA Container Toolkit not configured"; exit 1)
```

#### Поддерживаемые серверы вывода { #supported-inference-backends }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] [SGLang](https://github.com/sgl-project/sglang)
- [x] [VoxBox](https://github.com/gpustack/vox-box)
- [x] Пользовательский

### Графический процессор AMD { #amd-gpu }

#### Требования { #requirements_1 }

- [Драйвер графического процессора AMD](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/), поддерживающий AMD ROCm 7.0 или выше.
- [Среда выполнения контейнера AMD](https://instinct.docs.amd.com/projects/container-toolkit/en/latest/container-runtime/overview.html)

!!! note "Узлы графического процессора Kubernetes"

    Если графические процессоры предоставляются узлами Kubernetes, установите драйвер на каждый узел, как указано выше, или позвольте [оператору графического процессора AMD](https://instinct.docs.amd.com/projects/gpu-operator/en/latest/index.html) управлять им. Если AMD GPU Оператор уже установлен, отключите его плагин устройства, метку узлов, встроенные компоненты Node Feature Discovery и Kernel Module Management, которые конфликтуют с управлением устройствами GPUStack. Подробности см. в разделе [Требования к поставщику](https://docs.gpustack.ai/gpustack-operator/main/docs/getting-started/vendor-prerequisites/#amd).

Для проверки выполните следующие команды:

```bash
sudo amd-smi static

# If using Docker
sudo docker info 2>/dev/null | grep -q "amd" \
    && echo "AMD Container Toolkit OK" \
    || (echo "AMD Container Toolkit not configured"; exit 1)
```

#### Поддерживаемые серверы вывода { #supported-inference-backends_1 }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] [SGLang](https://github.com/sgl-project/sglang) (требуются графические процессоры серии gfx9)
- [x] Пользовательский

### Ascend NPU { #ascend-npu }

#### Требования { #requirements_2 }

- [Драйвер Ascend NPU](https://www.hiascend.com/hardware/firmware-drivers/community)
- [Ascend Docker Runtime](https://www.hiascend.com/document/detail/zh/mindcluster/730/clustersched/dlug/dlug_installation_017.html)

Для проверки выполните следующие команды:

```bash
sudo npu-smi info

# If using Docker
sudo docker info 2>/dev/null | grep -q "ascend" \
    && echo "Ascend Container Toolkit OK" \
    || (echo "Ascend Container Toolkit not configured"; exit 1)
```

#### Поддерживаемые устройства { #supported-devices }

- [x] Серия Ascend NPU 910C
- [x] Серия Ascend NPU 910B (910B1 ~ 910B4)
- [x] Восхождение NPU 310P3

#### Поддерживаемые серверы вывода { #supported-inference-backends_2 }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] [SGLang](https://github.com/sgl-project/sglang)
- [x] [MindIE](https://www.hiascend.com/document/detail/zh/mindie/21RC2/index/index.html)
- [x] Пользовательский

### Хайгон DCU { #hygon-dcu }

#### Требования { #requirements_3 }

- [Драйвер Hygon DCU](https://developer.sourcefind.cn/tool/)
- [Набор инструментов Hygon DTK](https://developer.sourcefind.cn/tool/)

!!! note "Узлы графического процессора Kubernetes"

    Если DCU предоставляются узлами Kubernetes, установите драйвер и набор инструментов на каждом узле, как указано выше, или позвольте [Оператору Hygon DCU](https://developer.sourcefind.cn/document/7541efc4-54b1-11f1-8265-0242ac150003) управлять ими. Если оператор Hygon DCU уже установлен, отключите его плагин устройства и компоненты метки узлов, которые конфликтуют с управлением устройствами GPUStack. Подробности см. в разделе [Требования к поставщику](https://docs.gpustack.ai/gpustack-operator/main/docs/getting-started/vendor-prerequisites/#hygon).

Для проверки выполните следующие команды:

```bash
sudo hy-smi
```

#### Поддерживаемые устройства { #supported-devices_1 }

- [x] Блоки DCU Hygon

#### Поддерживаемые серверы вывода { #supported-inference-backends_3 }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] Пользовательский

### Графический процессор MetaX { #metax-gpu }

#### Требования { #requirements_4 }

- [Драйвер графического процессора MetaX](https://developer.metax-tech.com/softnova/download?package_kind=Driver)
- [MetaX MACA SDK](https://developer.metax-tech.com/softnova/download?package_kind=SDK)

!!! note "Узлы графического процессора Kubernetes"

    Если графические процессоры предоставляются узлами Kubernetes, установите драйвер и SDK на каждый узел, как указано выше, или позвольте [Оператору графического процессора MetaX](https://developer.metax-tech.com/api/client/document/preview/1411/k8s/00_overview.html) управлять ими. Плагин устройства оператора MetaX GPU нельзя отключить, поэтому, если оператор уже установлен, удалите его, оставив драйвер и SDK на узле, прежде чем добавлять узлы в GPUStack. Подробности см. в разделе [Требования к поставщику](https://docs.gpustack.ai/gpustack-operator/main/docs/getting-started/vendor-prerequisites/#metax).

Для проверки выполните следующие команды:

```bash
sudo mx-smi
```

#### Поддерживаемые серверы вывода { #supported-inference-backends_4 }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] [SGLang](https://github.com/sgl-project/sglang)
- [x] Пользовательский

### Графический процессор MThreads { #mthreads-gpu }

#### Требования { #requirements_5 }

- [Драйвер графического процессора MThreads](https://docs.mthreads.com/)
- [Набор инструментов для контейнеров MThreads](https://developer.mthreads.com/sdk/download/CloudNative)

!!! note "Узлы графического процессора Kubernetes"

    Если графические процессоры предоставляются узлами Kubernetes, установите драйвер и набор инструментов на каждый узел, как указано выше, или позвольте [MT GPU Оператор](https://docs.mthreads.com/cloud-native/cloud-native-doc-online/introduction/) управлять ими. Плагин устройства оператора MT GPU нельзя отключить в режиме `full` по умолчанию, поэтому, если оператор уже установлен, переключитесь в режим `core` без плагина устройства или удалите оператор, оставив драйвер и набор инструментов на узле, прежде чем добавлять узлы в GPUStack. Подробности см. в разделе [Требования к поставщику](https://docs.gpustack.ai/gpustack-operator/main/docs/getting-started/vendor-prerequisites/#moore-threads).

Для проверки выполните следующие команды:

```bash
sudo mthreads-gmi

# If using Docker
sudo docker info 2>/dev/null | grep -q "mthreads" \
    && echo "MThreads Container Toolkit OK" \
    || (echo "MThreads Container Toolkit not configured"; exit 1)
```

#### Поддерживаемые серверы вывода { #supported-inference-backends_5 }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] [SGLang](https://github.com/sgl-project/sglang)
- [x] Пользовательский

### Графический процессор Илуватара { #iluvatar-gpu }

#### Требования { #requirements_6 }

- [Драйвер графического процессора Iluvatar](https://support.iluvatar.com/#/login)
- [Набор инструментов для контейнеров Илуватара](https://github.com/Deep-Spark/ix-container-toolkit)

!!! note "Узлы графического процессора Kubernetes"

    Если графические процессоры предоставляются узлами Kubernetes, установите драйвер и набор инструментов на каждый узел, как указано выше, или позвольте [ix-GPU-Operator](https://developer.iluvatar.com/docs/generaldocs_ix_gpu_operator) управлять ими. Если ix-GPU-Operator уже установлен, отключите его плагин устройства и компоненты обнаружения функций, которые конфликтуют с управлением устройствами GPUStack. Подробности см. в разделе [Требования к поставщику](https://docs.gpustack.ai/gpustack-operator/main/docs/getting-started/vendor-prerequisites/#iluvatar).

Для проверки выполните следующие команды:

```bash
sudo ixsmi

# If using Docker
sudo docker info 2>/dev/null | grep -q "iluvatar" \
    && echo "Iluvatar Container Toolkit OK" \
    || (echo "Iluvatar Container Toolkit not configured"; exit 1)
```

#### Поддерживаемые серверы вывода { #supported-inference-backends_6 }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] Пользовательский

### Камбрикон MLU { #cambricon-mlu }

#### Требования { #requirements_7 }

- Драйвер Cambricon MLU
- Набор инструментов Cambricon NeuWare

Для проверки выполните следующие команды:

```bash
sudo cnmon
```

#### Поддерживаемые серверы вывода { #supported-inference-backends_7 }

- [x] Пользовательский

### T-Head PPU { #t-head-ppu }

#### Требования { #requirements_8 }

- Драйвер T-Head PPU
- SDK T-Head PPU

Для проверки выполните следующие команды:

```bash
sudo ppu-smi
```

#### Поддерживаемые серверы вывода { #supported-inference-backends_8 }

- [x] [vLLM](https://github.com/vllm-project/vllm)
- [x] [SGLang](https://github.com/sgl-project/sglang)
- [x] Пользовательский

## Требования к сети { #networking-requirements }

### Требования к подключению { #connectivity-requirements }

Для правильной работы GPUStack требуется следующее сетевое подключение:

**Сервер-рабочий:** Сервер должен иметь возможность обращаться к работникам для выполнения запросов на вывод прокси.

**От рабочего к серверу:** Рабочие должны иметь возможность подключаться к серверу для регистрации и отправки обновлений.

**От рабочего к рабочему:** Требуется для распределенного вывода между несколькими рабочими процессами.

### Требования к порту { #port-requirements }

GPUStack использует эти порты для связи:

#### Серверные порты { #server-ports }

| Порт | Описание |
| --------- | ------------------------------------------------------------ |
| TCP 80 | Порт по умолчанию для GPUStack конечных точек пользовательского интерфейса и API. |
| TCP 443 | Порт по умолчанию для GPUStack конечных точек пользовательского интерфейса и API (TLS включен) |
| TCP 10161 | Порт по умолчанию для конечной точки метрик сервера. |
| TCP 30080 | Порт по умолчанию для внутреннего API сервера GPUStack. |
| TCP 5432 | Порт по умолчанию для встроенной базы данных Postgres. |

#### Рабочие порты { #worker-ports }

| Порт | Описание |
| --------------- | -------------------------------------------------------------- |
| TCP 10150 | Порт по умолчанию для работника GPUStack |
| TCP 10151 | Порт по умолчанию для конечной точки рабочих метрик. |
| TCP 40000-40063 | Диапазон портов для служб вывода |
| TCP 41000-41999 | Диапазон портов для служб Ray (vLLM с использованием распределенного развертывания) |

##### Распределенная vLLM с портами Ray { #distributed-vllm-with-ray-ports }

При использовании распределенной vLLM GPUStack анализирует указанный выше диапазон портов для служб Ray и назначает их в следующем порядке:

1. Порт сервера GCS (первый порт диапазона)
2. Порт клиента-сервера (зарезервирован для совместимости, больше не используется, см. https://github.com/gpustack/gpustack/issues/4171)
3. Порт приборной панели
4. Порт gRPC Dashboard (больше не используется, начиная с Ray 2.45.0, сохранен для обратной совместимости)
5. Порт gRPC агента информационной панели
6. Порт прослушивания агента информационной панели
7. Порт экспорта метрик
8. Порт диспетчера узлов
9. Порт диспетчера объектов
10. Порт агента среды выполнения Raylet
11. Минимальный номер порта для работника.
12. Максимальный номер порта для работника (последний порт диапазона)

Более подробную информацию о портах Ray см. в [документации Ray](https://docs.ray.io/en/latest/ray-core/configure.html#ports-configurations).

#### Порты встроенного шлюза { #embedded-gateway-ports }

Встроенный шлюз для сервера и работника использует следующие порты для внутренней связи.

| Порт | Хост | Описание |
| --------- | --------- | ---------------------------------------------------- |
| TCP 18443 | 127.0.0.1 | Порт для файлового API-сервера, обслуживающего через HTTPS. |
| TCP 15000 | 127.0.0.1 | Порт управления шлюзом Envoy. |
| TCP 15021 | 0.0.0.0   | Порт проверки работоспособности шлюза Envoy. |
| TCP 15090 | 0.0.0.0   | Порт метрики для шлюза Envoy. |
| TCP 9876 | 127.0.0.1 | Порт интроспекции для Пилота-дискавери |
| TCP 15010 | 127.0.0.1 | Порт для пилотного обнаружения, обслуживающего XDS через HTTP/gRPC. |
| TCP 15012 | 127.0.0.1 | Порт для Pilot-discovery, обслуживающего XDS через безопасный gRPC. |
| TCP 15020 | 0.0.0.0   | Порт метрики для Pilot-агента |
| TCP 8888 | 127.0.0.1 | Порт для контроллера, обслуживающего XDS через HTTP. |
| TCP 15051 | 127.0.0.1 | Порт для контроллера, обслуживающего XDS через gRPC. |
