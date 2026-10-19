---
translation_status: translated
template: overview.html
---

<br>

<p align="center">
    <img alt="GPUStack" src="https://raw.githubusercontent.com/gpustack/gpustack/main/docs/assets/gpustack-logo.png" width="300px"/>
</p>

<br>

<p align="center">
  <a href="https://github.com/gpustack/gpustack/blob/main/LICENSE" target="_blank">
    <img alt="License" src="https://img.shields.io/github/license/gpustack/gpustack?logo=github&logoColor=white&label=License&color=blue">
  </a>
  <a href="https://discord.gg/VXYJzuaqwD" target="_blank">
    <img alt="Discord" src="https://img.shields.io/badge/Discord-GPUStack-blue?logo=discord&logoColor=white">
  </a>
</p>

<!-- github-buttons -->

# Обзор { #overview }

GPUStack — это менеджер кластеров графических процессоров с открытым исходным кодом, предназначенный для эффективного развертывания моделей искусственного интеллекта. Он настраивает и координирует механизмы вывода — vLLM, SGLang, TensorRT-LLM или ваши собственные — для оптимизации производительности кластеров графических процессоров.

<div class="grid cards" markdown>

-   :material-server-network:{ .lg .middle .icon-blue } __Мультикластерное управление графическим процессором__

    ---

    Управляет кластерами графических процессоров в нескольких средах. Сюда входят локальные серверы, кластеры Kubernetes и облачные провайдеры.

-   :material-engine-outline:{ .lg .middle .icon-green } __Подключаемые механизмы вывода__

    ---

    Автоматически настраивает высокопроизводительные механизмы вывода, такие как vLLM, SGLang и TensorRT-LLM. При необходимости вы также можете добавить собственные механизмы вывода.

-   :material-rocket-launch-outline:{ .lg .middle .icon-orange } __Поддержка модели 0-го дня__

    ---

    Подключаемая архитектура ядра GPUStack позволяет развертывать новые модели в день их выпуска.

-   :material-speedometer:{ .lg .middle .icon-red } __Оптимизация производительности__

    ---

    Предлагает предварительно настроенные режимы для низкой задержки или высокой пропускной способности. Поддерживает расширенный KV-кеш (LMCache, HiCache) и спекулятивное декодирование (EAGLE3, MTP).

-   :material-shield-check-outline:{ .lg .middle .icon-purple } __Операции корпоративного уровня__

    ---

    Предлагает поддержку автоматического восстановления после сбоев, балансировки нагрузки, мониторинга, аутентификации и контроля доступа.

</div>

## Архитектура { #architecture }

GPUStack позволяет группам разработчиков, ИТ-организациям и поставщикам услуг предоставлять модель как услугу в любом масштабе. Он поддерживает стандартные API для моделей LLM, голоса, изображений и видео. Платформа включает в себя встроенную систему аутентификации пользователей и контроля доступа, мониторинг производительности и использования графического процессора в реальном времени, а также детальный учет использования токенов и частоты запросов API.

На рисунке ниже показано, как один сервер GPUStack может управлять несколькими кластерами графических процессоров как в локальной, так и в облачной среде. Планировщик GPUStack выделяет графические процессоры для максимального использования ресурсов и выбирает подходящие механизмы вывода для оптимальной производительности. Администраторы также получают полную информацию о состоянии и показателях системы через интегрированные информационные панели Grafana и Prometheus.

![gpustack-v2-архитектура](assets/gpustack-v2-architecture.png)

## Обзор кластера с первого взгляда { #cluster-visibility-at-a-glance }

Представление топологии кластера графических процессоров в [GPUStack Enterprise](https://gpustack.ai/enterprise/) обеспечивает в реальном времени общий вид всего вашего парка — каждый работник, графический процессор и модель развертываются в одном месте, с данными об использовании, распределении и состоянии работоспособности в реальном времени.

![gpustack-кластер-топология](assets/cluster-topology.png)

## Оптимизированная производительность вывода { #optimized-inference-performance }

Автоматический выбор механизма и оптимизация параметров GPUStack обеспечивают высокую производительность вывода без дополнительной настройки. На следующем рисунке показано улучшение пропускной способности по сравнению с конфигурациями vLLM по умолчанию:

![h200-сравнение-пропускной способности](assets/h200-throughput-comparison.png)

Подробные методы и результаты сравнительного анализа можно найти в нашей [Лаборатории производительности вывода](performance-lab/overview.md).

## Поддерживаемые ускорители { #supported-accelerators }

GPUStack поддерживает широкий спектр ускорителей для вывода ИИ:

<div class="logo-tile-grid">
    <div class="logo-tile" data-tooltip="NVIDIA GPU">
        <img src="../assets/logos/nvidia.png" alt="NVIDIA" class="off-glb">
    </div>
    <div class="logo-tile" data-tooltip="AMD GPU">
        <img src="../assets/logos/amd.png" alt="AMD" class="off-glb">
    </div>
    <div class="logo-tile" data-tooltip="Ascend NPU">
        <img src="../assets/logos/ascend.png" alt="Ascend" class="off-glb">
    </div>
    <div class="logo-tile" data-tooltip="Hygon DCU">
        <img src="../assets/logos/hygon.png" alt="Hygon" class="off-glb">
    </div>
    <div class="logo-tile" data-tooltip="MThreads GPU">
        <img src="../assets/logos/mthreads.png" alt="MThreads" class="off-glb">
    </div>
    <div class="logo-tile" data-tooltip="Iluvatar GPU">
        <img src="../assets/logos/iluvatar.png" alt="Iluvatar" class="off-glb">
    </div>
    <div class="logo-tile" data-tooltip="MetaX GPU">
        <img src="../assets/logos/metax.png" alt="MetaX" class="off-glb">
    </div>
    <div class="logo-tile" data-tooltip="Cambricon MLU">
        <img src="../assets/logos/cambricon.png" alt="Cambricon" class="off-glb">
    </div>
    <div class="logo-tile" data-tooltip="T-Head PPU">
        <img src="../assets/logos/thead.png" alt="T-Head" class="off-glb">
    </div>
</div>

Подробные требования и инструкции по настройке см. в документации [Требования к установке](installation/requirements.md).
