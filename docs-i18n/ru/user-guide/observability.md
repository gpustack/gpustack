---
translation_status: translated
---

# Наблюдательность { #observability }

В этом документе описывается, как отслеживать метрики времени выполнения службы GPUStack Server/Worker/LLM с помощью Prometheus и Grafana.

## Обзор { #overview }

GPUStack предоставляет полный набор показателей для обслуживания моделей и управления ресурсами графического процессора. Благодаря интеграции Prometheus и Grafana пользователи могут собирать, хранить и визуализировать эти показатели в режиме реального времени, обеспечивая эффективный мониторинг и устранение неполадок.

## Встроенная возможность наблюдения (по умолчанию) { #built-in-observability-default }

По умолчанию GPUStack запускается со встроенным Prometheus и Grafana. Вы можете получить к ним доступ через:

- **Prometheus**: `http://your_gpustack_server_host_ip/prometheus`
- **Графана**: `http://your_gpustack_server_host_ip/grafana`

Встроенный Grafana настроен на анонимный доступ к средству просмотра и имеет отключенную форму входа. По умолчанию учетные данные администратора остаются `admin`/`grafana`.

## Внешняя наблюдаемость (необязательно) { #external-observability-optional }

Если вам нужен внешний стек Prometheus/Grafana, мы рекомендуем использовать предоставленные файлы Docker Compose:

Выполните следующие команды, чтобы клонировать последнюю стабильную версию:

```bash
LATEST_TAG=$(
    curl -s "https://api.github.com/repos/gpustack/gpustack/releases" \
    | grep '"tag_name"' \
    | sed -E 's/.*"tag_name": "([^"]+)".*/\1/' \
    | grep -Ev 'rc|beta|alpha|preview' \
    | head -1
)
echo "Latest stable release: $LATEST_TAG"
git clone -b "$LATEST_TAG" https://github.com/gpustack/gpustack.git
cd gpustack/docker-compose
```

Перед началом установите `GPUSTACK_GRAFANA_URL` для доступного через браузер URL-адреса Grafana (а не имени хоста только для контейнера, такого как `grafana`).

Запустите внешний Prometheus/Grafana (это отключает встроенный стек):

```bash
sudo docker compose -f docker-compose.external-observability.yaml up -d
```

Если у вас уже есть внешний стек Prometheus/Grafana, вы можете настроить его вручную:

1. **Настроить Prometheus для очистки метрик GPUStack**
   Добавьте цели для конечной точки метрик GPUStack (по умолчанию `:10161`) и конечной точки обнаружения воркеров. Пример:

   ```yaml
   scrape_configs:
     - job_name: gpustack-worker-discovery
       scrape_interval: 5s
       http_sd_configs:
         - url: "http://<gpustack_server_host>:10161/metrics/targets"
           refresh_interval: 1m
     - job_name: gpustack-server
       scrape_interval: 10s
       static_configs:
         - targets: ["<gpustack_server_host>:10161"]
   ```
2. **Импортировать информационные панели GPUStack в Grafana**
   Используйте информационные панели, представленные в каталоге `docker-compose/grafana/grafana_dashboards/`, в качестве отправной точки.
3. **Укажите GPUStack на свой Grafana**
   Установите `GPUSTACK_GRAFANA_URL` для внешнего URL-адреса Grafana, чтобы перенаправление панели управления работало. Это должен быть URL-адрес, доступный браузеру.

## Доступ к метрикам { #accessing-metrics }

- **Конечная точка показателей GPUStack**:
  Доступ к метрикам по адресу `http://<gpustack_server_host>:10161/metrics`
- **Целевые показатели рабочих метрик GPUStack**:
  Доступ к метрикам по адресу `http://<gpustack_server_host>:10161/metrics/targets`
- **Интерфейс Прометея**:
  Получите доступ к Prometheus через `http://<host>:19090` по умолчанию или через порт, настроенный `--builtin-prometheus-port` / `GPUSTACK_BUILTIN_PROMETHEUS_PORT`.
- **Grafana UI**:
  Получите доступ к Grafana через `http://<host>:13000` по умолчанию или через порт, настроенный `--builtin-grafana-port` / `GPUSTACK_BUILTIN_GRAFANA_PORT`. Встроенный Grafana настроен на анонимный доступ к программе просмотра с отключенной формой входа. По умолчанию учетные данные администратора остаются `admin`/`grafana`.

## Миграция со старых настроек создания сообщений { #migration-from-older-compose-setups }

Если вы ранее использовали Docker Compose для запуска Prometheus/Grafana вместе с GPUStack:

- **Сохранять внешнюю наблюдаемость (рекомендуется для непрерывности)**:
  Оставьте существующие контейнеры Prometheus/Grafana работающими. Обновите цели очистки Prometheus до новой конечной точки метрик GPUStack и установите `GPUSTACK_GRAFANA_URL` для существующей Grafana.

- **Переключиться на встроенную наблюдаемость**:
  Остановите старые контейнеры Prometheus/Grafana, затем используйте последнюю версию `docker-compose.server.yaml` (только GPUStack). Встроенная Grafana/Prometheus возьмет верх. Исторические метрики из старого Prometheus не будут перенесены, если вы не сохраните старый стек только для чтения.

## Настройка сопоставления метрик { #customizing-metrics-mapping }

GPUStack поддерживает динамическую настройку сопоставления метрик через свой API конфигурации. Это позволяет вам обновить способ сопоставления показателей механизма выполнения с метриками GPUStack без перезапуска службы. Конфигурация управляется централизованно на сервере, и к ней можно получить доступ или изменить ее через HTTP API.

### Конечные точки API { #api-endpoints }

- **Получить текущую конфигурацию метрик**

  - ПОЛУЧИТЬ `http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/config`
  - Возвращает текущую конфигурацию сопоставления метрик в формате JSON.

- **Обновить конфигурацию метрик**

  - ПОСТ `http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/config`
  - Принимает полезную нагрузку JSON для обновления конфигурации сопоставления метрик. Изменения вступают в силу немедленно для всех воркеров.

- **Получить конфигурацию метрик по умолчанию**
  - ПОЛУЧИТЬ `http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/default-config`
  - Возвращает конфигурацию сопоставления метрик по умолчанию в формате JSON, что полезно для справки или сброса.

### Пример использования { #example-usage }

**Получить текущую конфигурацию:**

```bash
curl http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/config
```

**Обновить конфигурацию:**

```bash
curl -X POST http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/config \
     -H "Content-Type: application/json" \
     -d @custom_metrics_config.json
```

_(где `custom_metrics_config.json` — ваш новый файл конфигурации)_

**Получить конфигурацию по умолчанию:**

```bash
curl http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/default-config
```

> **Примечание**: Конфигурация должна быть предоставлена в допустимом формате JSON.

## Метрики, предоставляемые GPUStack { #metrics-exposed-by-gpustack }

Следующие метрики предоставляются GPUStack и могут собираться Prometheus. Каждая метрика включает иерархические метки для идентификации кластера, воркера, модели и экземпляра.

### LLM обслуживает показатели времени выполнения { #llm-serving-runtime-metrics }

| Имя метрики | Тип | Описание |
| -------------------------------------- | --------- | --------------------------------------------------------------------------- |
| gpustack:num_requests_running          | Gauge     | Количество запросов, обрабатываемых в данный момент.                               |
| gpustack:num_requests_waiting          | Gauge     | Количество запросов, ожидающих в очереди.                                    |
| gpustack:num_requests_swapped          | Gauge     | Количество запросов, переданных в ЦП.                                      |
| gpustack:prefix_cache_hit_rate         | Gauge     | Частота попадания в кеш префикса.                                                      |
| gpustack:kv_cache_usage_ratio          | Gauge     | Коэффициент использования KV-кэша. 1.0 означает полностью использованный.                                 |
| gpustack:prefix_cache_queries          | Counter   | Количество запросов к префиксному кэшу (общее количество токенов).                              |
| gpustack:prefix_cache_hits             | Counter   | Количество попаданий в кэш префиксов (всего токенов).                                 |
| gpustack:prompt_tokens                 | Counter   | Общее количество обработанных токенов предварительного заполнения.                                   |
| gpustack:generation_tokens             | Counter   | Общее количество сгенерированных токенов.                                           |
| gpustack:request_prompt_tokens         | Histogram | Количество токенов предварительного заполнения, обработанных за один запрос.                             |
| gpustack:request_generation_tokens     | Histogram | Количество токенов генерации, обработанных за запрос.                          |
| gpustack:time_to_first_token_seconds   | Histogram | Время создания первого токена.                                               |
| gpustack:inter_token_latency_seconds   | Histogram | Время для генерации следующего токена после создания предыдущего токена. |
| gpustack:time_per_output_token_seconds | Histogram | Время на сгенерированный токен.                                                   |
| gpustack:e2e_request_latency_seconds   | Histogram | Задержка сквозного запроса.                                                 |
| gpustack:request_success               | Counter   | Общее количество успешных запросов.                                        |
| gpustack:request_preemptions           | Counter   | Общее количество прерываний со стороны движка.                                |

Эти метрики сопоставляются с различными механизмами выполнения (vLLM, SGLang, MindIE), как определено в metrics_config.yaml.

### Метрики воркеров { #worker-metrics }

| Имя метрики | Тип | Описание |
| ------------------------------------------------ | ----- | ------------------------------------------------ |
| gpustack:worker_status                           | Gauge | Статус воркера (с меткой состояния).                |
| gpustack:worker_node_os                          | Info  | Информация об операционной системе узла-воркера. |
| gpustack:worker_node_kernel                      | Info  | Информация о ядре узла-воркера.           |
| gpustack:worker_node_uptime_seconds              | Gauge | Время безотказной работы узла-воркера в секундах.            |
| gpustack:worker_node_cpu_cores                   | Gauge | Общее количество ядер ЦП узла-воркера.              |
| gpustack:worker_node_cpu_utilization_rate        | Gauge | Коэффициент использования ЦП узла-воркера.         |
| gpustack:worker_node_memory_total_bytes          | Gauge | Общий объем памяти в байтах узла-воркера.        |
| gpustack:worker_node_memory_used_bytes           | Gauge | Используемая память в байтах узла-воркера.         |
| gpustack:worker_node_memory_utilization_rate     | Gauge | Коэффициент использования памяти узла-воркера.      |
| gpustack:worker_node_gpu                         | Info  | Информация о графическом процессоре узла-воркера.              |
| gpustack:worker_node_gpu_cores                   | Gauge | Общее количество ядер графического процессора узла-воркера.              |
| gpustack:worker_node_gpu_utilization_rate        | Gauge | Коэффициент использования графического процессора узла-воркера.         |
| gpustack:worker_node_gpu_temperature_celsius     | Gauge | температуры графического процессора в градусах Цельсия.                      |
| gpustack:worker_node_gpu_power_limit_watts       | Gauge | Предел управления питанием графического процессора в ваттах.             |
| gpustack:worker_node_gpu_power_used_watts        | Gauge | Потребляемая мощность графического процессора в ваттах.                        |
| gpustack:worker_node_gram_total_bytes            | Gauge | Общий объем оперативной памяти графического процессора в байтах.                          |
| gpustack:worker_node_gram_allocated_bytes        | Gauge | Выделенная оперативная память графического процессора в байтах.                      |
| gpustack:worker_node_gram_used_bytes             | Gauge | Используемая оперативная память графического процессора в байтах.                           |
| gpustack:worker_node_gram_utilization_rate       | Gauge | Коэффициент использования оперативной памяти графического процессора.                        |
| gpustack:worker_node_filesystem_total_bytes      | Gauge | Общий размер файловой системы в байтах.                  |
| gpustack:worker_node_filesystem_used_bytes       | Gauge | Размер используемой файловой системы в байтах.                   |
| gpustack:worker_node_filesystem_utilization_rate | Gauge | Коэффициент использования файловой системы.                     |

### Метрики сервера { #server-metrics }

| Имя метрики | Тип | Описание |
| ------------------------------------------- | ----- | ------------------------------------------------------------- |
| gpustack:cluster                            | Info  | Информация о кластере (идентификатор, имя, поставщик).                     |
| gpustack:cluster_status                     | Gauge | Статус кластера (с меткой состояния).                            |
| gpustack:model                              | Info  | Информация о модели (идентификатор, имя, время выполнения, источник, категория).      |
| gpustack:model_desired_instances            | Gauge | Желаемое количество экземпляров модели.                            |
| gpustack:model_running_instances            | Gauge | Количество запущенных экземпляров модели.                            |
| gpustack:model_instance_status              | Gauge | Статус каждого экземпляра модели (с меткой состояния).             |
| gpustack:model_instance_restart_count       | Gauge | Счётчик перезапуска экземпляра модели.                                 |
| gpustack:model_instance_latest_restart_time | Gauge | Моделируйте время последнего перезапуска экземпляра в секундах с меткой времени Unix. |

### Метрики шины событий { #event-bus-metrics }

| Имя метрики | Тип | Описание |
| ----------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------- |
| gpustack:bus_subscribers            | Gauge   | Активные подписчики шины на `topic` + `source` (несколько могут использовать общий `source=streaming`, по одному на каждый открытый поток просмотра). |
| gpustack:bus_queue_depth            | Gauge   | Максимальная глубина очереди между общими подписчиками `topic`+`source`.                                                     |
| gpustack:bus_queue_capacity         | Gauge   | Максимальный размер очереди для каждого подписчика (см. ручку env ниже).                                                               |
| gpustack:bus_queue_full             | Gauge   | 1, если у какого-либо общего подписчика `topic`+`source` есть полная очередь.                                                   |
| gpustack:bus_queue_saturation_ratio | Gauge   | Макс. `qsize / maxsize` в `[0, 1]`. Устойчиво > 0.8 ⇒ медленное потребление.                                              |
| gpustack:bus_subscriber_latest_keys | Gauge   | Максимальное количество идентификаторов, ожидающих объединенной ОБНОВЛЕННОЙ доставки (размер `latest_by_key`).                                            |
| gpustack:bus_events_total           | Counter | Совокупное количество событий, суммированное между подписчиками, совместно использующими `topic`+`source`. Дополнительные метки: `kind`, `event_type`.  |

> **Примечание**: Все метрики помечены соответствующими идентификаторами (кластер, рабочий, модель, экземпляр, пользователь) для детального мониторинга и фильтрации.
