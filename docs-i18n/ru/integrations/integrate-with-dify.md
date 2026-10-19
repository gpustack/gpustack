---
translation_status: translated
---

# Интеграция с Dify { #integrate-with-dify }

Dify может интегрироваться с GPUStack для использования локально развернутых LLM, эмбеддингов, переранжирования, генерации изображений, возможностей преобразования речи в текст и текста в речь.

## Развертывание моделей { #deploying-models }

1. В пользовательском интерфейсе GPUStack перейдите на страницу `Deployments` и нажмите `Deploy Model`, чтобы развернуть нужные вам модели. Вот несколько примеров моделей:

- qwen3-8b
- qwen2.5-vl-3b-instruct
- bge-m3
- bge-reranker-v2-m3

![gpustack-models](../assets/integrations/integration-gpustack-models.png)

2. В разделе «Операции» модели откройте `API Access Info`, чтобы узнать, как интегрироваться с этой моделью.

![gpustack-api-access-info](../assets/integrations/integration-gpustack-api-access-info.png)

## Создайте ключ API. { #create-an-api-key }

1. Перейдите на страницу `Access Control` > `API Keys`, затем нажмите `New API Key`.

2. Введите имя, затем нажмите `Save`.

3. Скопируйте ключ API и сохраните его для дальнейшего использования.

## Интеграция GPUStack в Dify { #integrating-gpustack-into-dify }

1. Откройте пользовательский интерфейс Dify, перейдите в верхний правый угол и нажмите `PLUGINS`, выберите `Install from Marketplace`, найдите плагин GPUStack и выберите его установку.

![dify-install-gpustack-plugin](../assets/integrations/integration-dify-install-gpustack-plugin.png)

2. После установки перейдите к `Settings > Model Provider > GPUStack`, затем выберите `Add Model` и заполните:

- Тип модели: выберите тип модели на основе модели.

- Имя модели: имя должно соответствовать названию модели, развернутой в GPUStack.

- URL-адрес сервера: `http://your-gpustack-url`, не используйте `localhost`, так как он относится к внутренней сети контейнера. Если вы используете собственный порт, обязательно включите его. Также убедитесь, что URL-адрес доступен изнутри контейнера Dify (вы можете проверить это с помощью `curl`).

- Ключ API: введите ключ API, который вы скопировали на предыдущих шагах.

Нажмите `Save`, чтобы добавить модель:

![dify-add-model](../assets/integrations/integration-dify-add-model.png)

При необходимости добавьте другие модели, затем выберите добавленные модели в поле `System Model Settings` и сохраните:

![dify-system-model-settings](../assets/integrations/integration-dify-system-model-settings.png)

Теперь вы можете использовать модели `Studio` и `Knowledge`, вот простой случай:

1. Перейдите к `Knowledge`, чтобы создать знания, и загрузите свои документы:

![обозначить-создать-знание](../assets/integrations/integration-dify-create-knowledge.png)

2. Настройте разбиение на фрагменты и параметры поиска. Используйте модель эмбеддингов для создания векторных представлений документов, а модель переранжирования — для ранжирования результатов поиска.

![dify-set-embedding-and-rerank-model](../assets/integrations/integration-dify-set-embedding-and-rerank-model.png)

3. После успешного импорта документов создайте приложение в `Studio`, добавьте ранее созданные знания, выберите модель чата и взаимодействуйте с ним:

![dify-чат-с-моделью](../assets/integrations/integration-dify-chat-with-model.png)

4. Переключите модель на `qwen2.5-vl-3b-instruct`, удалите ранее добавленную базу знаний, включите `Vision` и загрузите изображение в чат для активации мультимодального ввода:

![dify-chat-with-vlm](../assets/integrations/integration-dify-chat-with-vlm.png)
