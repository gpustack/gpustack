---
translation_status: translated
---

# Использование моделей реранкера { #using-reranker-models }

**Модели переранжирования** — это специализированные модели, предназначенные для улучшения ранжирования списка элементов на основе релевантности данному запросу. Они обычно используются в системах поиска и поиска информации для уточнения первоначальных результатов поиска, определения приоритета элементов, которые с большей вероятностью будут соответствовать намерениям пользователя. Модели Reranker берут исходный список документов и меняют порядок элементов для повышения точности в таких приложениях, как поисковые системы, системы рекомендаций и задачи ответа на вопросы.

В этом руководстве мы покажем, как развертывать и использовать модели реранкера в GPUStack.

## Предварительные условия { #prerequisites }

Прежде чем начать, убедитесь, что у вас есть следующее:

- GPUStack установлен и работает. Если нет, обратитесь к [Краткому руководству](../quickstart.md).
- Доступ к Hugging Face для загрузки файлов модели.

## Шаг 1. Развертывание модели { #step-1-deploy-the-model }

Выполните следующие шаги, чтобы развернуть модель из каталога:

1. Перейдите на страницу `Catalog` в пользовательском интерфейсе GPUStack.
2. На странице списка моделей используйте раскрывающийся список для фильтрации с помощью `Reranker`.
3. Просмотрите описание модели, максимальную длину контекста и поддерживаемые размеры.

![Развернуть модель](../assets/using-models/using-reranker-models/deploy-model.png)

После развертывания вы можете отслеживать состояние развертывания модели на странице `Deployments`.

![Список моделей](../assets/using-models/using-reranker-models/model-list.png)

## Шаг 2. Создайте ключ API. { #step-2-generate-an-api-key }

Для взаимодействия с моделью мы будем использовать API GPUStack. Для этого вам необходимо сгенерировать API-ключ:

1. Перейдите на страницу `Access Control` > `API Keys`.
2. Нажмите кнопку `New API Key`.
3. Введите имя ключа API и нажмите кнопку `Save`.
4. Скопируйте сгенерированный ключ API. Вы можете просмотреть ключ API только один раз, поэтому обязательно сохраните его.

## Шаг 3. Переранжирование { #step-3-reranking }

Имея развернутую модель и ключ API, вы можете изменить ранжирование списка документов через API GPUStack. Вот пример сценария, использующего `curl`:

```bash
export SERVER_URL=<your-server-url>
export GPUSTACK_API_KEY=<your-api-key>
curl $SERVER_URL/v1/rerank \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $GPUSTACK_API_KEY" \
    -d '{
        "model": "qwen3-reranker-4b",
        "query": "What is a panda?",
        "top_n": 3,
        "documents": [
            "hi",
            "it is a bear",
            "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."
        ]
    }' | jq
```

Замените `<your-server-url>` URL-адресом вашего сервера GPUStack и `<your-api-key>` ключом API, который вы сгенерировали на предыдущем шаге.

Пример ответа:

```json
{
  "model": "qwen3-reranker-4b",
  "object": "list",
  "results": [
    {
      "index": 0,
      "document": {
        "text": "hi",
        "multi_modal": null
      },
      "relevance_score": 0.9996911287307739
    },
    {
      "index": 2,
      "document": {
        "text": "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.",
        "multi_modal": null
      },
      "relevance_score": 0.8206241726875305
    },
    {
      "index": 1,
      "document": {
        "text": "it is a bear",
        "multi_modal": null
      },
      "relevance_score": 0.7244728803634644
    }
  ],
  "usage": {
    "total_tokens": 51
  }
}
```
