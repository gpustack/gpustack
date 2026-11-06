---
translation_status: translated
---

# Интеграция с RAGFlow { #integrate-with-ragflow }

RAGFlow может интегрироваться с GPUStack для использования локально развернутых LLM, моделей зрения, эмбеддингов, переранжирования, преобразования речи в текст и текста в речь. RAGFlow сам обнаруживает модели, запущенные в GPUStack, поэтому достаточно один раз подключиться и выбрать модели для добавления.

!!! note

    Для этого руководства нужна версия RAGFlow новее v1.0.0-rc3, в которой появилась карточка настройки GPUStack.

## Развертывание моделей { #deploying-models }

1. В пользовательском интерфейсе GPUStack перейдите на страницу `Deployments` и нажмите `Deploy Model`, чтобы развернуть нужные модели. Оставьте параметр `Enable Model Route` включённым, поскольку RAGFlow обнаруживает только запущенные модели с маршрутом. Примеры моделей:

    | Модель | Тип в RAGFlow |
    | --- | --- |
    | `qwen3-4b` | Chat |
    | `qwen2.5-vl-3b-instruct` | Chat + Vision |
    | `bge-m3` | Embedding |
    | `bge-reranker-v2-m3` | Rerank |
    | `faster-whisper-small` | Speech to text |
    | `cosyvoice2-0.5b` | Text to speech |

2. Дождитесь готовности реплик каждой модели:

    ![Список развертываний GPUStack](../assets/integrations/integration-ragflow-gpustack-deployments.png)

## Создание API-ключа { #create-an-api-key }

1. Перейдите на страницу `Access Control` > `API Keys` и нажмите `Add API Key`.

2. Введите имя и в разделе `Access Permissions` оставьте включённым `Model Access` со значением `All models`. По умолчанию `Expiration` равно `1 month`, поэтому выберите больший срок, если RAGFlow будет использовать этот ключ постоянно. Затем нажмите `Save`.

    ![API-ключ GPUStack](../assets/integrations/integration-ragflow-gpustack-api-key.png)

3. Скопируйте API-ключ и сохраните его. Позже на странице `API Keys` отображается только маскированное значение.

## Интеграция GPUStack в RAGFlow { #integrating-gpustack-into-ragflow }

1. Откройте интерфейс RAGFlow, нажмите на аватар в правом верхнем углу, чтобы открыть `Model providers`, затем выберите `GPUStack` в разделе `Available models`.

2. Заполните следующие поля и нажмите `Connect`:

    - Instance name: любое имя для этого сервера GPUStack в RAGFlow, например `gpustack`.
    - GPUStack server URL: `http://your-gpustack-url`, без пути `/v1`. Если используется нестандартный порт, укажите его. Не используйте `localhost`, так как он указывает на контейнер RAGFlow. Если GPUStack работает на одном хосте с RAGFlow, можно использовать `http://host.docker.internal:<port>`.
    - API key: API-ключ, скопированный на предыдущих шагах.

3. RAGFlow показывает запущенные модели, сгруппированные по типам, с длиной контекста. Снимите флажки с моделей, которые не нужно добавлять. GPUStack не сообщает, принимает ли модель изображения, поэтому для моделей зрения, таких как `qwen2.5-vl-3b-instruct`, выберите `Chat + Vision`:

    ![Подключение GPUStack в RAGFlow](../assets/integrations/integration-ragflow-connect-gpustack.png)

    Если подключиться не удалось, RAGFlow показывает причину под формой:

    - `GPUStack rejected this API key.`: проверьте API-ключ в разделе `Access Control` > `API Keys`.
    - `Cannot reach this server from RAGFlow.`: проверьте URL и порт изнутри контейнера RAGFlow. Например, выполните следующую команду в каталоге `docker` RAGFlow, она должна вернуть `"ok"`:

        ```bash
        docker compose exec ragflow-cpu curl http://your-gpustack-url/healthz
        ```

4. Нажмите `Add 6 models` внизу списка (число зависит от выбранных моделей). Добавленные модели появятся в экземпляре:

    ![Добавление моделей в RAGFlow](../assets/integrations/integration-ragflow-add-model.png)

    Если позже вы развернёте в GPUStack другие модели, нажмите `Refresh` в экземпляре, чтобы обнаружить и добавить их.

5. Нажмите `Set default models` и выберите модель для каждого типа. Каждый выбор сохраняется сразу:

    ![Модели по умолчанию в RAGFlow](../assets/integrations/integration-ragflow-set-default-models.png)

## Использование моделей { #using-the-models }

Теперь модели можно использовать в `Dataset` и `Chat`. Простой пример:

1. Перейдите в `Dataset` и создайте набор данных. Модель эмбеддингов по умолчанию уже выбрана:

    ![Создание набора данных в RAGFlow](../assets/integrations/integration-ragflow-create-dataset.png)

2. Нажмите `Add file` > `Upload file`, включите `Parse on creation`, выберите файл и нажмите `Save`. Когда разбор завершится, у файла появится количество фрагментов:

    ![Загрузка файла в RAGFlow](../assets/integrations/integration-ragflow-upload-file.png)

3. Перейдите в `Retrieval testing`, установите для `Rerank model` значение `bge-reranker-v2-m3`, введите вопрос и нажмите `Run`:

    ![Настройка модели переранжирования в RAGFlow](../assets/integrations/integration-ragflow-set-rerank-model.png)

4. В `Chat` создайте чат и откройте его. Нажмите значок настроек в правом верхнем углу, выберите набор данных в `Datasets`, проверьте модель чата в `Model` и нажмите `Save`:

    ![Настройка чата в RAGFlow](../assets/integrations/integration-ragflow-create-assistant.png)

    Модель переранжирования, выбранная в `Retrieval testing`, к чатам не применяется. Чтобы переранжировать результаты поиска в чате, в тех же настройках разверните `Advanced settings` > `Naive`, установите для `Rerank model` значение `bge-reranker-v2-m3` и нажмите `Save`.

5. Отправьте сообщение. Теперь можно взаимодействовать с моделью и выполнять запросы к набору данных:

    ![Чат с моделью в RAGFlow](../assets/integrations/integration-ragflow-chat-with-model.png)

6. Чтобы ответы зачитывались вслух, включите `Text to speech` в `Advanced settings` и нажмите `Save`. Наведите указатель на ответ и нажмите значок динамика. Во время воспроизведения он сменится значком паузы:

    ![Чтение ответа вслух в RAGFlow](../assets/integrations/integration-ragflow-read-aloud.png)

7. Для голосового ввода нажмите значок микрофона в поле сообщения, произнесите текст и нажмите значок ещё раз, чтобы остановить запись. RAGFlow преобразует запись в текст с помощью модели Speech to text по умолчанию и отправит его как ваше сообщение.

8. Чтобы обсудить изображение, в настройках чата переключите `Model` на `qwen2.5-vl-3b-instruct` и очистите `Datasets`, чтобы в ответы не попадали ссылки на посторонние документы. Нажмите `Save`, начните новый диалог, прикрепите изображение с помощью значка скрепки и отправьте вопрос:

    ![Чат с моделью зрения в RAGFlow](../assets/integrations/integration-ragflow-chat-with-vlm.png)
