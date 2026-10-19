---
translation_status: translated
---

# Каталог моделей { #model-catalog }

Каталог моделей — это указатель моделей, настроенных с помощью GPUStack.

## Обзор моделей { #browse-models }

Вы можете просмотреть каталог моделей, перейдя на страницу `Catalog`. Вы можете фильтровать модели по названию и категории. На следующем снимке экрана показана страница каталога моделей:

![Каталог моделей](../assets/model-catalog.png)

## Развертывание модели из каталога { #deploy-a-model-from-the-catalog }

Вы можете развернуть модель из Каталога моделей, щелкнув карточку модели. Появится страница конфигурации развертывания модели. Вы можете просмотреть и настроить конфигурацию развертывания и нажать кнопку `Save`, чтобы развернуть модель.

## Настройка каталога моделей { #customize-model-catalog }

Вы можете настроить каталог моделей, предоставив файл YAML через конфигурацию сервера GPUStack, используя флаг `--model-catalog-file`. Он принимает либо локальный путь к файлу, либо URL-адрес. Схему можно найти во встроенном файле каталога модели [здесь](https://github.com/gpustack/gpustack/blob/main/gpustack/assets/model-catalog.yaml).

Чтобы настроить каталог без перезапуска сервера, вместо этого настройте источник каталога — ваш документ заменяет упакованный каталог, и ящик передает вам этот каталог, с которого можно начать. См. [Обновления контента OTA](ota-content-updates.md).

Ниже приведен пример YAML-файла пользовательского каталога моделей:

```yaml
draft_models:
- name: Qwen3-8B-EAGLE3
  algorithm: eagle3
  source: huggingface
  huggingface_repo_id: Tengyunw/qwen3_8b_eagle3
model_sets:
- name: Deepseek R1 0528
  description: DeepSeek-R1-0528 is a minor version of the DeepSeek R1 model that features enhanced reasoning depth and inference capabilities. These improvements are achieved through increased computational resources and algorithmic optimizations applied during post-training. The model delivers strong performance across a range of benchmark evaluations, including mathematics, programming, and general logic, with overall capabilities approaching those of leading models such as O3 and Gemini 2.5 Pro.
  home: https://www.deepseek.com
  icon: /static/catalog_icons/deepseek.png
  categories:
    - llm
  capabilities:
    - context/128K
  size: 671
  licenses:
    - mit
  release_date: "2025-05-28"
  specs:
    - mode: throughput
      quantization: FP8
      gpu_filters:
        vendor: nvidia
        compute_capability: ">=9.0" # Hopper or later
      source: huggingface
      huggingface_repo_id: deepseek-ai/DeepSeek-R1-0528
      backend: SGLang
      backend_parameters:
        - --enable-dp-attention
        - --context-length=32768
    - mode: standard
      quantization: FP8
      source: huggingface
      huggingface_repo_id: deepseek-ai/DeepSeek-R1-0528
      backend: vLLM
      backend_parameters:
        - --max-model-len=32768
```

### Использование каталога моделей в средах с воздушным зазором { #using-model-catalog-in-air-gapped-environments }

Встроенный каталог моделей использует модели Hugging Face или ModelScope. Если вы используете GPUStack в изолированной среде без доступа к Интернету, вы можете настроить каталог моделей для использования источника модели с локальным путем. Вот пример:

```yaml
model_sets:
- name: Deepseek R1 0528
  description: DeepSeek-R1-0528 is a minor version of the DeepSeek R1 model that features enhanced reasoning depth and inference capabilities. These improvements are achieved through increased computational resources and algorithmic optimizations applied during post-training. The model delivers strong performance across a range of benchmark evaluations, including mathematics, programming, and general logic, with overall capabilities approaching those of leading models such as O3 and Gemini 2.5 Pro.
  home: https://www.deepseek.com
  icon: /static/catalog_icons/deepseek.png
  categories:
    - llm
  capabilities:
    - context/128K
  size: 671
  licenses:
    - mit
  release_date: "2025-05-28"
  specs:
    - mode: throughput
      quantization: FP8
      gpu_filters:
        vendor: nvidia
        compute_capability: ">=9.0" # Hopper or later
      source: local_path
      local_path: /path/to/DeepSeek-R1-0528
      backend: SGLang
      backend_parameters:
        - --enable-dp-attention
        - --context-length=32768
    - mode: standard
      quantization: FP8
      source: local_path
      # assuming you have /path/to/DeepSeek-R1-0528 directory
      local_path: /path/to/DeepSeek-R1-0528
      backend: vLLM
      backend_parameters:
        - --max-model-len=32768
```

### Схема каталога моделей { #model-catalog-schema }

YAML-файл каталога моделей содержит два основных раздела: `draft_models` и `model_sets`.

- `draft_models`: Список черновых моделей для спекулятивного декодирования.
- `model_sets`: список протестированных и оптимизированных наборов моделей.

Каждая черновая модель имеет следующие поля:

| Поле | Тип | Описание |
|---------------------|---------------|-------------------------------------------------------------------------------------------------|
| name                | string        | Имя черновой модели.                                                                    |
| algorithm           | string        | Алгоритм спекулятивного декодирования модели. В настоящее время поддерживается только eagle3.                                                    |
| source              | string        | Источник модели (например,huggingface, model_scope).                                               |
| huggingface_repo_id | string        | Идентификатор репозитория Hugging Face модели (если источником является HuggingFace).                            |
| model_scope_model_id  | string        | Идентификатор репозитория ModelScope модели (если источником является model_scope).                              |

Каждый набор моделей имеет следующие поля:

| Поле | Тип | Описание |
|---------------------|---------------|-------------------------------------------------------------------------------------------------|
| name                | string        | Имя модели.                                                                      |
| description         | string        | Краткое описание модели.                                                           |
| home                | string        | URL-адрес домашней страницы модели.                                                              |
| icon                | string        | URL-адрес значка модели.                                                                  |
| categories          | list of str   | Список категорий, к которым принадлежит модель.                                                  |
| capabilities        | list of str   | Список возможностей модели.                                                       |
| size                | int           | Размер модели в миллиардах параметров.                                                                 |
| licenses            | list of str   | список лицензий модели.                                                              |
| release_date        | string        | Дата выпуска модели в формате ГГГГ-ММ-ДД.                                         |
| specs               | list of spec  | Список спецификаций развертывания модели. |


Каждая спецификация развертывания имеет следующие поля:

| Поле | Тип | Описание |
|---------------------|---------------|-------------------------------------------------------------------------------------------------|
| mode                | string        | GPUStack предоставляет как обычный, так и оптимизированный режимы для различных вариантов использования, включая пропускную способность, задержку и стандартные сценарии. Пользователи также могут определять собственные режимы по мере необходимости. |
| quantization        | string        | Тип квантования (например, FP16, FP8, INT8).                                               |
| gpu_filters         | dict          | Фильтры графического процессора для указания совместимых графических процессоров.                                                  |

Другие поля в спецификации развертывания аналогичны полям API модели. Более подробную информацию см. в документации [API Reference](../api-reference.md).
