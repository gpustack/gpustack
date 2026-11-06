---
translation_status: translated
---

# API вывода { #inference-apis }

## API-интерфейсы, совместимые с OpenAI { #openai-compatible-apis }

GPUStack предоставляет [OpenAI-совместимые API](https://platform.openai.com/docs/api-reference) в конечной точке `/v1`.

Вы можете интегрировать и использовать модели, развернутые на GPUStack, с любым приложением или платформой, поддерживающей OpenAI-совместимый API, просто указав его на OpenAI-совместимую конечную точку GPUStack.

### Поддерживаемые конечные точки { #supported-endpoints }

Поддерживаются следующие конечные точки API:

- [x] [Список моделей](https://platform.openai.com/docs/api-reference/models/list)
- [x] [Создать завершение](https://platform.openai.com/docs/api-reference/completions/create)
- [x] [Завершение создания чата](https://platform.openai.com/docs/api-reference/chat/create)
- [x] [Создать ответ](https://platform.openai.com/docs/api-reference/responses/create)
- [x] [Создать эмбеддинги](https://platform.openai.com/docs/api-reference/embeddings/create)
- [x] [Создать изображение](https://platform.openai.com/docs/api-reference/images/create)
- [x] [Создать редактирование изображения](https://platform.openai.com/docs/api-reference/images/createEdit)
- [x] [Создать речь](https://platform.openai.com/docs/api-reference/audio/createSpeech)
- [x] [Создать транскрипцию](https://platform.openai.com/docs/api-reference/audio/createTranscription)

### Использование { #usage }

#### Пример cURL { #curl-example }

```bash
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
    "model": "qwen3",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ],
    "stream": true
  }'
```

## API, совместимые с Anthropic { #anthropic-compatible-apis }

GPUStack предоставляет Anthropic-совместимый API [`/v1/messages` API](https://platform.claude.com/docs/en/api/messages/create).

### Использование { #usage_1 }

```bash
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/messages \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
    "model": "qwen3",
    "messages": [
      {
        "role": "user",
        "content": "Hello!"
      }
    ],
    "max_tokens": 1024
  }'
```

## Jina-совместимый API реранга { #jina-compatible-rerank-api }

В контексте поисково-дополненной генерации (RAG) под реранжированием понимается процесс выбора наиболее актуальной информации из извлеченных документов или источников знаний перед ее представлением пользователю или использованием для генерации ответов.

Обратите внимание, что OpenAI-совместимые API **не предоставляют** конечную точку `rerank`. Чтобы восполнить этот пробел, GPUStack предоставляет [Jina-совместимый API Rerank](https://jina.ai/reranker/) по пути `/v1/rerank`.

### Использование { #usage_2 }

#### Пример cURL { #curl-example_1 }

```bash
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/rerank \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $GPUSTACK_API_KEY" \
    -d '{
        "model": "bge-reranker-v2-m3",
        "query": "What is a panda?",
        "top_n": 3,
        "documents": [
            "hi",
            "it is a bear",
            "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."
        ]
    }' | jq
```

Пример вывода:

```json
{
  "model": "bge-reranker-v2-m3",
  "object": "list",
  "results": [
    {
      "document": {
        "text": "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."
      },
      "index": 2,
      "relevance_score": 1.951932668685913
    },
    {
      "document": {
        "text": "it is a bear"
      },
      "index": 1,
      "relevance_score": -3.7347371578216553
    },
    {
      "document": {
        "text": "hi"
      },
      "index": 0,
      "relevance_score": -6.157620906829834
    }
  ],
  "usage": {
    "prompt_tokens": 69,
    "total_tokens": 69
  }
}
```

## System One API принятия решений { #system-one-decision-api }

Модели принятия решений — развертывания в категории `decision` — отвечают по контракту System One (Jev), а не генерируют текст: получив state и набор questions, они возвращают типизированные ответы, а не дополнения. GPUStack предоставляет этот контракт по пути `/v1/systemone`.

Поле `model` обязательно и принимает имя маршрута модели, точно как в `/v1/chat/completions`. GPUStack разрешает модель и передает остальную часть тела запроса и весь ответ движку решений без изменений.

### Использование { #usage_3 }

#### Пример cURL { #curl-example_2 }

```bash
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/systemone \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
      "model": "qwen3-decision-8b",
      "state": {
          "messages": [
              { "role": "user", "content": "Write a 200-word poem in Hungarian." }
          ]
      },
      "questions": {
          "model_selection": {
              "type": "choice",
              "instructions": "Pick the model best suited.",
              "criteria": {
                  "qwen3-8b": "Strong multilingual generation.",
                  "llama3-8b": "Faster, weaker multilingual."
              }
          }
      }
  }' | jq
```

Пример вывода:

```json
{
  "model": "qwen3-decision-8b",
  "usage": {
    "input_tokens": 189,
    "output_tokens": 12
  },
  "answers": {
    "model_selection": {
      "type": "choice",
      "choice": "qwen3-8b",
      "confidence": 0.93,
      "probabilities": {
        "qwen3-8b": 0.93,
        "llama3-8b": 0.07
      }
    }
  }
}
```

`state` — ситуация, относительно которой принимается решение: простой текст или структурированный JSON, например сообщения чата. Каждая запись в `questions` сопоставляет идентификатор вопроса с его определением; распространены три типа:

| Тип | Значение | `criteria` | Поле ответа |
| --- | --- | --- | --- |
| `choice` | Выбрать один вариант | Имя варианта → описание | `choice` — имя выбранного варианта |
| `score` | Оценить state по упорядоченным уровням | Упорядоченный список описаний уровней | `score` — индекс уровня |
| `noul` | Бинарный вопрос (да/нет) | Необязательные описания сторон `true` / `false` | `noul` — логическое значение |

Ответ на вопрос возвращается в `answers` под тем же идентификатором, вместе с вердиктом несутся `confidence` и `probabilities`. Набор вопросов в контракте открыт — движки могут принимать дополнительные поля — а GPUStack проверяет только `model`, остальное пересылает без изменений. О развертывании моделей принятия решений см. [Использование моделей принятия решений](../using-models/using-decision-models.md).

## Другие API { #other-apis }

Для других типов API GPUStack позволяет включить функцию **Generic Proxy** при развертывании модели.

После включения существует два способа обращения к целевой модели:

**Форма на основе пути (рекомендуется)** — закодируйте идентификатор маршрута модели непосредственно в URL-адресе как `/model/proxy/<model_route_id>/<upstream-path>`. Дополнительный заголовок не требуется:

```bash
export GPUSTACK_API_KEY=your_api_key
# Assume the model route id is 42.
curl http://your_gpustack_server_url/model/proxy/42/embed \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{"inputs": ["What is Deep Learning?", "Deep Learning is not..."]}'
```

**Форма на основе заголовка (устарела)** — отправьте на `/model/proxy/<upstream-path>` и укажите целевую модель через заголовок `X-GPUStack-Model` или поле `model` в теле JSON:

```bash
curl http://your_gpustack_server_url/model/proxy/embed \
  -X POST \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -H "X-GPUStack-Model: bge-m3" \
  -d '{"inputs": ["What is Deep Learning?", "Deep Learning is not..."]}'
```

Форма на основе заголовка сохранена для обратной совместимости и будет удалена в будущем выпуске; новые интеграции должны использовать форму на основе пути.

Более подробную информацию см. в разделе [Включить универсальный прокси](../user-guide/model-deployment-management.md#enable-generic-proxy).
