---
translation_status: translated
---

# Интеграция с RAGFlow { #integrate-with-ragflow }

RAGFlow может интегрироваться с GPUStack для использования локально развернутых LLM, эмбеддингов, переранжирования, возможностей преобразования речи в текст и текста в речь.

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

## Интеграция GPUStack в RAGFlow. { #integrating-gpustack-into-ragflow }

1. Откройте пользовательский интерфейс RAGFlow, перейдите в правый верхний угол и щелкните аватар, выберите `Model Providers > GPUStack`, затем выберите `Add the model` и заполните:

- Тип модели: выберите тип модели в зависимости от модели.

- Имя модели: имя должно соответствовать имени модели, развернутой в GPUStack.

- Базовый URL-адрес: `http://your-gpustack-url/v1`, URL-адрес не должен включать путь и не использовать `localhost`, поскольку он относится к внутренней сети контейнера. Если вы используете собственный порт, обязательно включите его. Также убедитесь, что URL-адрес доступен изнутри контейнера RAGFlow (вы можете проверить это с помощью `curl`).

- API-ключ: введите ключ API, который вы скопировали на предыдущих шагах.

- Максимальное количество токенов: введите максимальное количество токенов, поддерживаемое текущей конфигурацией модели.

Нажмите `OK`, чтобы добавить модель:

![ragflow-add-model](../assets/integrations/integration-ragflow-add-model.png)

2. При необходимости добавьте другие модели, затем выберите добавленные модели в поле `Set default models` и сохраните:

![ragflow-set-default-models](../assets/integrations/integration-ragflow-set-default-models.png)

Теперь вы можете использовать модели `Chat` и `Knowledge Base`, вот простой случай:

1. Перейдите к `Knowledge base`, чтобы создать новую базу знаний и добавить свой файл:

![ragflow-создать-базу-знаний](../assets/integrations/integration-ragflow-create-knowledge-base.png)

2. Перейдите к `Retrieval testing` и установите модель переоценки на `bge-reranker-v2-m3`:

![ragflow-set-rerank-model](../assets/integrations/integration-ragflow-set-rerank-model.png)

3. В `Chat` создайте помощника, привяжите ранее созданную базу знаний и выберите модель чата:

![помощник по созданию ragflow](../assets/integrations/integration-ragflow-create-assistant.png)

4. Создайте сеанс чата — теперь вы можете взаимодействовать с моделью и запрашивать базу знаний:

![ragflow-чат-с-моделью](../assets/integrations/integration-ragflow-chat-with-model.png)

5. Отредактируйте помощника и переключите модель на `qwen2.5-vl-3b-instruct`. После сохранения создайте новый чат и загрузите изображение, чтобы включить мультимодальный ввод:

![ragflow-чат-с-vlm](../assets/integrations/integration-ragflow-chat-with-vlm.png)
