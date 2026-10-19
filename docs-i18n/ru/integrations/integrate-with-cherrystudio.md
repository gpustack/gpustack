---
translation_status: translated
---

# Интеграция с Cherry Studio { #integrate-with-cherry-studio }

Cherry Studio интегрируется с GPUStack для использования локально размещенных LLM, эмбеддингов и возможностей переранжирования.

## Развертывание моделей { #deploying-models }

1. В пользовательском интерфейсе GPUStack перейдите на страницу `Deployments` и нажмите `Deploy Model`, чтобы развернуть нужные вам модели. Вот несколько примеров моделей:

    - qwen3-instruct-2507
    - qwen2.5-vl-7b
    - bge-m3
    - bge-reranker-v2-m3

   ![Модели GPUStack](../assets/integrations/integration-gpustack-models2.png)

2. В разделе «Операции» модели откройте `API Access Info`, чтобы увидеть, как интегрироваться с этой моделью:

   ![Информация о доступе к API](../assets/integrations/integration-gpustack-api-access-info2.png)

## Создайте ключ API. { #create-an-api-key }

1. Перейдите на страницу `Access Control` > `API Keys`, затем нажмите `New API Key`.

2. Введите имя, затем нажмите `Save`.

3. Скопируйте ключ API и сохраните его для дальнейшего использования.

## Интеграция GPUStack в Cherry Studio { #integrating-gpustack-into-cherry-studio }

1. Откройте Cherry Studio, перейдите в `Settings` → `Model Provider`, найдите GPUStack, включите его и настройте, как показано:

    - `API Key`: введите ключ API, который вы скопировали на предыдущих шагах.

    - `API Host`: `Access URL` на панели `API Access Info`.

   ![Настройка стека Cherry Studio GPU](../assets/integrations/integration-cherrystudio-01.png)

2. В конфигурации провайдера GPUStack нажмите «Управление» и включите нужные вам модели:

   ![Управление моделями](../assets/integrations/integration-cherrystudio-02.png)

   ![Включить модели](../assets/integrations/integration-cherrystudio-03.png)

3. (Необязательно) Проверьте API:

   ![Тестовый API](../assets/integrations/integration-cherrystudio-04.png)

После настройки вернитесь на домашнюю страницу Cherry Studio и начните использовать свои модели.

## Использование LLM { #using-llms }

![Используйте LLM](../assets/integrations/integration-cherrystudio-05.png)

## Использование мультимодальных моделей { #using-multimodal-models }

1. Выберите мультимодальную модель:

   ![Выбрать мультимодальную модель](../assets/integrations/integration-cherrystudio-13.png)

2. Задайте мультимодальные вопросы:

   ![Мультимодальные вопросы и ответы](../assets/integrations/integration-cherrystudio-14.png)

   ![Мультимодальные вопросы и ответы](../assets/integrations/integration-cherrystudio-15.png)

## Использование эмбеддингов и переранжирования для улучшения базы знаний. Вопросы и ответы. { #use-embeddings-and-reranking-to-improve-knowledge-base-qa }

1. Откройте страницу конфигурации Базы знаний:

   ![Настройки КБ](../assets/integrations/integration-cherrystudio-06.png)

2. Добавить базу знаний:

   ![Добавить КБ](../assets/integrations/integration-cherrystudio-08.png)

3. Добавьте контент в базу знаний (на примере «Заметок»):

   ![Добавить содержимое базы знаний](../assets/integrations/integration-cherrystudio-09.png)

   ![КБ подробностей о содержимом](../assets/integrations/integration-cherrystudio-10.png)

4. Вернитесь на домашнюю страницу и воспользуйтесь базой знаний «Вопросы и ответы»:

   ![КБ Вопросы и ответы](../assets/integrations/integration-cherrystudio-11.png)

   ![КБ Пример вопросов и ответов](../assets/integrations/integration-cherrystudio-12.png)
