---
translation_status: translated
---

# Интеграция с LiteLLM { #integrate-with-litellm }

LiteLLM может интегрироваться с GPUStack для объединения локально развернутых LLM, эмбеддингов, переранжирования, функций преобразования речи в текст и преобразования текста в речь в единый OpenAI-совместимый/шлюз Anthropic API для сотрудников предприятия.

## Развертывание моделей в GPUStack { #deploying-models-in-gpustack }

1. В пользовательском интерфейсе GPUStack перейдите на страницу `Deployments` и нажмите `Deploy Model`, чтобы развернуть нужные вам модели. Вот несколько примеров моделей:

- qwen3-8b
- qwen2.5-vl-3b-instruct
- bge-m3
- bge-reranker-v2-m3

![gpustack-models](../assets/integrations/integration-gpustack-models.png)

2. В разделе «Операции» модели откройте `API Access Info`, чтобы узнать, как интегрироваться с этой моделью.

![gpustack-api-access-info](../assets/integrations/integration-gpustack-api-access-info.png)

## Создайте ключ API в GPUStack. { #create-an-api-key-in-gpustack }

1. Перейдите на страницу `Access Control` > `API Keys` в GPUStack, затем нажмите `New API Key`.

2. Введите имя, затем нажмите `Save`.

3. Скопируйте ключ API и сохраните его для дальнейшего использования.

## Интеграция GPUStack в LiteLLM { #integrating-gpustack-into-litellm }

1. Откройте интерфейс управления LiteLLM.

```
http://your-litellm-url:4000/ui
```

![litellm-dashboard](../assets/integrations/litellm-1.png)

2. откройте меню модели+конечные точки -> добавить модель

![litellm-add-model1](../assets/integrations/litellm-2.png)
![litellm-add-model2](../assets/integrations/litellm-3.png)

3. тест в лителлме -> Песочница

![litellm-chat1](../assets/integrations/litellm-4.png)
