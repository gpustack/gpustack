---
translation_status: translated
---

# Использование пользовательских механизмов вывода { #using-custom-inference-backends }

В этом руководстве объясняется, как добавлять собственные серверные части вывода в GPUStack, в том числе использовать проверенные конфигурации сообщества и создавать свои собственные с нуля.

Описание параметров см. в [Руководстве пользователя](../user-guide/inference-backend-management.md).

## Типы бэкэнда { #backend-types }

GPUStack поддерживает три типа бэкендов вывода:

- **Встроенный**: предварительно настроенные серверные части (vLLM, MindIE, VoxBox, SGLang...), поддерживаемые GPUStack, автоматически оптимизированные для различного оборудования.
- **Сообщество**: предварительно проверенные пользовательские конфигурации серверной части. По сути, это CustomBackends, помеченные как «сообщество» для упрощения ручной настройки.
- **Пользовательский**: серверные части, которые вы настраиваете самостоятельно с помощью пользовательских образов и команд Docker.

## Использование серверов сообщества { #using-community-backends }

Серверные программы сообщества предоставляют самый быстрый способ добавления популярных механизмов вывода.

**Шаги:**

1. Перейдите на страницу серверной части вывода → нажмите «Добавить серверную часть».
2. Выберите опцию «Сообщество».
3. Просмотрите «Backend Marketplace сообщества» и включите нужные вам бэкэнды.

## Создание пользовательских бэкэндов { #creating-custom-backends }

### Основные шаги { #core-steps }
1. Подготовьте образ Docker для требуемого бэкэнда вывода.
2. Изучите ENTRYPOINT или CMD образа, чтобы определить команду запуска.
3. Добавьте конфигурацию на странице Inference Backend.
4. Разверните модели и выберите вновь добавленный бэкэнд.

### Пример: TensorRT-LLM. { #example-tensorrt-llm }
Ниже в качестве примера используется TensorRT-LLM, иллюстрирующий, как добавлять и использовать серверную часть вывода.
> Эти примеры представляют собой функциональные демонстрации, а не конфигурации, оптимизированные для производительности. Для повышения производительности обратитесь к официальной документации каждого бэкэнда для настройки.

1. Найдите необходимое образ на [странице выпуска](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release), связанной с документацией TensorRT-LLM.
2. Образы TensorRT-LLM должны запускать службу вывода с помощью `trtllm-serve`; в противном случае они запускают интерактивный сеанс оболочки. `run_command` поддерживает такие заполнители, как `{{model_path}}` и `{{port}}` (и дополнительно `{{model_name}}`, `{{worker_ip}}`, `{{gpu_count}}`, `{{gpu_ids}}`), которые автоматически заменяются фактическими значениями, когда развертывание запланировано для работника. `{{gpu_count}}` и `{{gpu_ids}}` отражают графические процессоры, назначенные запланированному рабочему процессу (настраиваемые серверные части выполняются на одном рабочем процессе).
3. Добавьте конфигурацию на странице Inference Backend; Поддерживается импорт YAML. Пример:
```yaml
backend_name: TensorRT-LLM-custom
default_version: 1.2.0rc0
version_configs:
  1.2.0rc0:
    image_name: nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc0
    run_command: 'trtllm-serve {{model_path}} --host 0.0.0.0 --port {{port}}'
    custom_framework: cuda
```

4. На странице «Развертывания» выберите недавно добавленный серверный компонент и разверните модель.
![image.png](../assets/tutorials/using-custom-backend/deploy-by-custom-backend.png)

**Результат**

После запуска механизма вывода статус model_instance меняется на RUNNING.
![image.png](../assets/tutorials/using-custom-backend/custom-backend-running.png)
Вы можете участвовать в разговорах в Песочнице.
![image.png](../assets/tutorials/using-custom-backend/use-custom-backend-in-playground.png)

## Расширенная конфигурация { #advanced-configuration }

### Использование переменных среды { #using-environment-variables }

Переменные среды обеспечивают гибкую настройку без жесткого кодирования значений в командах:

```yaml
backend_name: advanced-backend-custom
default_env:
  CACHE_DIR: /models/cache
  LOG_LEVEL: info
version_configs:
  v1:
    image_name: my-backend:v1
    custom_framework: cuda
    run_command: 'serve {{model_path}} --cache {{CACHE_DIR}} --log-level {{LOG_LEVEL}} --port {{port}}'
    env:
      LOG_LEVEL: debug  # Override for this version
```

В этом примере:
- `CACHE_DIR` и `LOG_LEVEL` определены на уровне механизма вывода
- Версия `v1` переопределяет `LOG_LEVEL` на `debug`
- Обе переменные упоминаются в команде с использованием синтаксиса `{{VAR_NAME}}`.

### Пользовательская точка входа. { #custom-entrypoint }

Переопределить точку входа контейнера по умолчанию, если образ требует пользовательской инициализации. Вы можете установить точки входа как на уровне серверной части, так и на уровне версии:

```yaml
backend_name: custom-entry-backend-custom
default_entrypoint: /usr/local/bin/default-init
version_configs:
  v1:
    image_name: my-backend:v1
    custom_framework: cuda
    run_command: 'serve {{model_path}} --port {{port}}'
  v2:
    image_name: my-backend:v2
    custom_framework: cuda
    entrypoint: /usr/local/bin/v2-init  # Version-specific entrypoint overrides default
    run_command: 'serve {{model_path}} --port {{port}}'
```
