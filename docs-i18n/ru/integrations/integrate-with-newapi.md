---
translation_status: translated
---

# Интеграция с NewAPI { #integrate-with-newapi }

NewAPI может интегрироваться с GPUStack, объединяя локально развернутые LLM, модели эмбеддингов и переранжирования, а также преобразование речи в текст и текста в речь в единый API-шлюз, совместимый с OpenAI, для сотрудников организации.

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

## Интеграция GPUStack в NewAPI. { #integrating-gpustack-into-newapi }

1. Откройте консоль управления NewAPI, перейдите к `Channels` на левой боковой панели и нажмите `Add Channel`.

2. Настройте параметры канала следующим образом:

- **Тип**: выберите `OpenAI` или `Custom` (GPUStack предоставляет стандартные OpenAI-совместимые и конечные точки Anthropic).

- **Имя**: введите имя канала, часто такое же, как и идентификатор модели, например, `qwen3-8b`.

- **Базовый URL-адрес (Базовый URL-адрес)**: `http://your-gpustack-url`, URL-адрес должен указывать на URL-адрес доступа к GPUStack или имя домена, как OpenAI-совместимое, так и Anthropic (удалите v1, потому что newapi добавит его автоматически). Не используйте `localhost`, если NewAPI работает в контейнере Docker, поскольку он относится к внутренней сети контейнера. также можно использовать https, если сервер gpustack запускается с tls

- **Ключ (密钥)**: введите ключ API GPUStack, который вы скопировали на предыдущем шаге.

- **Модели (模型)**: введите или выберите точные имена моделей, развернутых в GPUStack (например, `qwen3-8b`, `bge-m3`, `bge-reranker-v2-m3`). Убедитесь, что имена моделей соответствуют именам развертывания GPUStack.

![newapi-add-channel](../assets/integrations/newapi1.png)
![newapi-add-channel](../assets/integrations/newapi2.png)
![newapi-add-channel](../assets/integrations/newapi3.png)
![newapi-add-channel](../assets/integrations/newapi4.png)

3. Настройка цен на модели.

![newapi-add-channel](../assets/integrations/newapi5.png)
![newapi-add-channel](../assets/integrations/newapi6.png)
![newapi-add-channel](../assets/integrations/newapi7.png)

4. Вы можете нажать `Test` в списке каналов, чтобы проверить соединение между NewAPI и GPUStack.

![newapi-add-channel](../assets/integrations/newapi8.png)

---

## Проверка и использование интегрированных моделей в NewAPI { #verifying-and-using-the-integrated-models-in-newapi }

1. Перейдите на страницу `api keys` в NewAPI, нажмите `create api key`.

* примечание: ключ API должен выбирать группу, аналогичную группе конфигурации каналов, иначе вы не сможете получить доступ к модели.

2. Скопируйте вновь созданный ключ NewAPI (начинается с `sk-`).

3. Теперь вы можете получить доступ ко всем моделям, поддерживаемым GPUStack, через единую конечную точку NewAPI:

- **Единый базовый URL**: `http://your-newapi-url/v1`
- **Единый ключ API**: `sk-xxxxxx` (Ваш новый ключ API)

Пример запроса с использованием `curl`:

```bash
curl http://your-newapi-url/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your-newapi-token" \
  -d '{
    "model": "qwen3-8b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```
