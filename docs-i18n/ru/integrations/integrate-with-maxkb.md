---
translation_status: translated
---

# Интеграция с MaxKB { #integrate-with-maxkb }

MaxKB может интегрироваться с GPUStack, используя локально развернутые **LLM, модели эмбеддингов и модели переранжирования** для создания ИИ-помощников на основе базы знаний.

## Развертывание моделей { #deploying-models }

1. В пользовательском интерфейсе GPUStack перейдите на страницу `Deployments` и нажмите `Deploy Model`, чтобы развернуть нужные вам модели. Вот несколько примеров моделей:

* `qwen3.5-35b-a3b`

  ![](../assets/integrations/maxkb/01.png)
  ![](../assets/integrations/maxkb/02.png)

* `qwen3-embedding-4b`

  ![](../assets/integrations/maxkb/03.png)
  ![](../assets/integrations/maxkb/04.png)

* `qwen3-reranker-4b`

  ![](../assets/integrations/maxkb/05.png)
  ![](../assets/integrations/maxkb/06.png)

2. После развертывания вы можете протестировать модель в **Playground**.

![](../assets/integrations/maxkb/07.png)
![](../assets/integrations/maxkb/09.png)
![](../assets/integrations/maxkb/08.png)

## Получение информации о доступе к модели. { #obtain-model-access-information }

1. На боковой панели GPUStack откройте страницу **Routes**.

2. Нажмите меню **Дополнительные действия** рядом с маршрутом и выберите **Информация о доступе к API**.

![](../assets/integrations/maxkb/10.png)
![](../assets/integrations/maxkb/11.png)

Запишите следующую информацию:

```
Base URL
Model Name
API Key
```

Пример:

```
Base URL: http://your-gpustack-url/v1

Model Name:
qwen3.5-35b-a3b
qwen3-embedding-4b
qwen3-reranker-4b

API Key:
gpustack_xxxxxxxxxxxxx
```

!!! note

    Вы можете создать ключ API, следуя инструкциям в пользовательском интерфейсе.

## Развертывание MaxKB { #deploy-maxkb }

MaxKB можно развернуть с помощью Docker:

```bash
docker run -d \
  --name maxkb \
  --restart always \
  -p 8080:8080 \
  -v ~/.maxkb:/opt/maxkb \
  1panel/maxkb
```

Учетные данные по умолчанию:

```
admin / MaxKB@123..
```

![](../assets/integrations/maxkb/50.png)

После первого входа в систему следуйте инструкциям, чтобы изменить пароль.

## Интеграция GPUStack в MaxKB { #integrating-gpustack-into-maxkb }

1. В пользовательском интерфейсе MaxKB перейдите к **Model** на верхней панели навигации.

![](../assets/integrations/maxkb/51.png)

2. Нажмите **Добавить модель** и настройте модель.

![](../assets/integrations/maxkb/52.png)
![](../assets/integrations/maxkb/53.png)
![](../assets/integrations/maxkb/54.png)

При настройке модели:

* **Базовая модель**: должно соответствовать имени модели, развернутому в GPUStack.
* **URL API**: `http://your-gpustack-url/v1`
* **API Key**: ключ API, созданный в GPUStack.

!!! note

    `API URL` и `API Key` появятся **после ввода базовой модели и нажатия Enter**.

3. Добавьте модели эмбеддингов и переранжирования, используя тот же метод:

* `qwen3-embedding-4b`

  ![](../assets/integrations/maxkb/56.png)

* `qwen3-reranker-4b`

  Для **qwen3-reranker-4b** включите **Generic Proxy**.

  ![](../assets/integrations/maxkb/55.png)

  Это необходимо, поскольку MaxKB использует следующую конечную точку:

  ```
  /v2/rerank
  ```

  ![](../assets/integrations/maxkb/57.png)

После настройки модели должны появиться в списке моделей.

![](../assets/integrations/maxkb/58.png)

## Создайте базу знаний { #create-a-knowledge-base }

1. Перейдите на страницу **Знания**.

2. Нажмите **Создать** и выберите **Web Knowledge**.

![](../assets/integrations/maxkb/59.png)

3. Введите URL-адрес документации или другой источник данных. MaxKB автоматически просканирует и проанализирует контент.

![](../assets/integrations/maxkb/60.png)

После завершения сканирования:

![](../assets/integrations/maxkb/61.png)

## Создайте ИИ-агента. { #create-an-ai-agent }

1. Перейдите на страницу **Агент**.

2. Нажмите **Создать**, чтобы создать нового агента.

![](../assets/integrations/maxkb/62.png)
![](../assets/integrations/maxkb/63.png)

3. Настройте агент с помощью:

* Модель чата
* База знаний
* Настройки получения

4. Нажмите **Опубликовать**, чтобы активировать агент.

![](../assets/integrations/maxkb/64.png)

## Чат с базой знаний { #chat-with-the-knowledge-base }

Откройте интерфейс чата, чтобы начать взаимодействовать с помощником.

![](../assets/integrations/maxkb/65.png)
![](../assets/integrations/maxkb/66.png)

Теперь помощник может отвечать на вопросы на основе подключенной базы знаний и моделей, развернутых в GPUStack.
