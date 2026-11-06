---
translation_status: translated
---

# Выполнение вывода на процессорах { #running-inference-on-cpus }

GPUStack поддерживает выполнение инференса на CPU. Это полезно при нехватке GPU или когда размер модели превышает доступную видеопамять. Доступны следующие режимы выполнения на CPU:

- **Гибридный инференс CPU+GPU**: частичное ускорение с переносом части большой модели на CPU при нехватке VRAM.
- **Полный вывод ЦП**: работает полностью на ЦП, когда ресурсы графического процессора недоступны.

!!! note

    Доступно только для пользовательских механизмов вывода.

    Если разгрузка ЦП включена, GPUStack выделит память ЦП, если ресурсов ГП недостаточно. Необходимо правильно настроить серверную часть вывода для использования гибридного вывода ЦП+ГП или полного вывода ЦП.

    Настоятельно рекомендуется использовать вывод ЦП только для рабочих ЦП.

Например, чтобы развернуть модель с выводом ЦП с помощью [Text Embeddings Inference](https://huggingface.co/docs/text-embeddings-inference/index), следуйте приведенной ниже конфигурации:

Источник: `HuggingFace`

Идентификатор репо: `BAAI/bge-large-en-v1.5`

Бэкэнд: `Custom`

Имя образа: `ghcr.io/huggingface/text-embeddings-inference:cpu-1.8`

Команда выполнения: `--model-id BAAI/bge-large-en-v1.5 --huggingface-hub-cache /var/lib/gpustack/cache/huggingface --port {{port}}`

!!! note

    `TEI (Text Embeddings Inference)` поддерживает развертывание только моделей из `HuggingFace`.

    `ghcr.io/huggingface/text-embeddings-inference:cpu-1.8` — это образ вывода ЦП для TEI. См.: [Оборудование, поддерживаемое TEI](https://huggingface.co/docs/text-embeddings-inference/supported_models#supported-hardware).

    `--huggingface-hub-cache /var/lib/gpustack/cache/huggingface` устанавливает расположение кэша HuggingFace Hub для TEI по пути, по которому GPUStack хранит загруженные модели HuggingFace. Путь по умолчанию — `/var/lib/gpustack/cache/huggingface`. См.: [Аргументы TEI CLI](https://huggingface.co/docs/text-embeddings-inference/cli_arguments).

    `{{port}}` — это заполнитель, который представляет порт, автоматически назначенный GPUStack.

![TEI Вывод ЦП](../assets/tutorials/inference-on-cpus/tei-cpu-inference.png)

И в дополнительных настройках отметьте `Allow CPU Offloading`:

![Разрешить разгрузку процессора](../assets/tutorials/inference-on-cpus/allow-cpu-offloading.png)

Если вам нужен доступ к API, не совместимым с OpenAI, вы также можете отметить `Enable Generic Proxy`.

Более подробную информацию см. в разделе [Включить универсальный прокси](../user-guide/model-deployment-management.md#enable-generic-proxy).
