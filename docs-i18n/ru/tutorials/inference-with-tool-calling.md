---
translation_status: translated
---

# Вывод с помощью вызова инструмента { #inference-with-tool-calling }

Вызов инструментов позволяет подключать модели к внешним инструментам и системам. Это полезно для многих вещей, таких как расширение возможностей помощников ИИ или создание глубокой интеграции между вашими приложениями и моделями.

В этом уроке вы узнаете, как настроить и использовать вызов инструментов в GPUStack для расширения возможностей вашего ИИ.

!!! note

    1. Вызов инструментов поддерживается в модулях вывода [vLLM](../user-guide/built-in-inference-backends.md#vllm) и [SGLang](../user-guide/built-in-inference-backends.md#sglang).
    2. Вызов инструментов основан на промпт-инжиниринге: для этой возможности модели должны быть обучены соответствующим шаблонам. Поэтому не все LLM поддерживают вызов инструментов.

## Предварительные условия { #prerequisites }

Прежде чем продолжить, убедитесь в следующем:

- GPUStack установлен и работает.
- Доступен рабочий узел Linux с графическим процессором. Мы будем использовать [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) в качестве модели для этого урока. Для модели требуется графический процессор с объемом видеопамяти не менее 18 ГБ.
- Доступ к Hugging Face для загрузки файлов модели.

## Шаг 1. Развертывание модели { #step-1-deploy-the-model }

### Развертывание из каталога { #deploy-from-catalog }

LLM, поддерживающие вызов инструментов, помечены в каталоге возможностью `tools`. При выборе такой модели из каталога вызов инструмента включается по умолчанию.

### Пример пользовательского развертывания с использованием llama-box { #example-of-custom-deployment-using-llama-box }

При развертывании моделей GGUF с помощью llama-box вызов инструмента включен по умолчанию для моделей, которые его поддерживают.

1. Перейдите на страницу `Deployments` в пользовательском интерфейсе GPUStack и нажмите кнопку `Deploy Model`. В раскрывающемся списке выберите `Hugging Face` в качестве источника вашей модели.
2. Установите флажок `GGUF`, чтобы фильтровать модели по формату GGUF.
3. Используйте строку поиска, чтобы найти модель `Qwen/Qwen2.5-7B-Instruct-GGUF`.
4. Нажмите кнопку `Save`, чтобы развернуть модель.

![Развернуть модель GGUF](../assets/tutorials/inference-with-tool-calling/deploy-model-gguf.png)

### Пример пользовательского развертывания с использованием vLLM { #example-of-custom-deployment-using-vllm }

При развертывании моделей с помощью vLLM необходимо включить вызов инструмента с дополнительными параметрами.

1. Перейдите на страницу `Deployments` в пользовательском интерфейсе GPUStack и нажмите кнопку `Deploy Model`. В раскрывающемся списке выберите `Hugging Face` в качестве источника вашей модели.
2. Используйте строку поиска, чтобы найти модель `Qwen/Qwen2.5-7B-Instruct`.
3. Разверните раздел `Advanced` в конфигурациях и прокрутите вниз до раздела `Backend Parameters`.
4. Нажмите кнопку `Add Parameter` и добавьте следующие параметры:

- `--enable-auto-tool-choice`
- `--tool-call-parser=hermes`

5. Нажмите кнопку `Save`, чтобы развернуть модель.

![Развернуть модель](../assets/tutorials/inference-with-tool-calling/deploy-model.png)

После развертывания вы можете отслеживать состояние модели на странице `Deployments`.

## Шаг 2. Создайте ключ API. { #step-2-generate-an-api-key }

Для взаимодействия с моделью мы будем использовать API GPUStack. Для этого вам необходимо сгенерировать API-ключ:

1. Перейдите на страницу `Access Control` > `API Keys`.
2. Нажмите кнопку `New API Key`.
3. Введите имя ключа API и нажмите кнопку `Save`.
4. Скопируйте сгенерированный ключ API для дальнейшего использования.

## Шаг 3: Сделайте вывод { #step-3-do-inference }

После развертывания модели и ключа API вы можете вызвать модель через API GPUStack. Вот пример сценария, использующего `curl` (замените `<your-server-url>` на URL-адрес вашего сервера GPUStack и `<your-api-key>` на ключ API, сгенерированный на предыдущем шаге):

```bash
export GPUSTACK_SERVER_URL=<your-server-url>
export GPUSTACK_API_KEY=<your-api-key>
curl $GPUSTACK_SERVER_URL/v1-openai/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $GPUSTACK_API_KEY" \
-d '{
  "model": "qwen2.5-7b-instruct",
  "messages": [
    {
      "role": "user",
      "content": "What'\''s the weather like in Boston today?"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"]
            }
          },
          "required": ["location"]
        }
      }
    }
  ],
  "tool_choice": "auto"
}'
```

Пример ответа:

```json
{
  "model": "qwen2.5-7b-instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "chatcmpl-tool-b99d32848b324eaea4bac5a5830d00b8",
            "type": "function",
            "function": {
              "name": "get_current_weather",
              "arguments": "{\"location\": \"Boston, MA\", \"unit\": \"fahrenheit\"}"
            }
          }
        ]
      },
      "finish_reason": "tool_calls"
    }
  ],
  "usage": {
    "prompt_tokens": 212,
    "total_tokens": 242,
    "completion_tokens": 30
  }
}
```
