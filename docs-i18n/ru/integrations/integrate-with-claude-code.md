---
translation_status: translated
---

# Интеграция с Claude Code { #integrate-with-claude-code }

Claude Code — это инструмент агентного кодирования от Anthropic. Поскольку развертывание моделей в GPUStack совместимо с Anthropic API, вы можете легко подключить Claude Code к развертыванию GPUStack и использовать его для задач генерации кода. В этом руководстве мы рассмотрим шаги по интеграции Claude Code с GPUStack и протестируем интеграцию, попросив Клода создать игру Flappy Bird.

## Предварительные условия { #prerequisites }

- Один или несколько графических процессоров с общей видеопамятью не менее 100 ГБ.
- GPUStack установлен и запущен.
- Доступ к Hugging Face или ModelScope для загрузки файлов моделей.

!!! note

    В этом руководстве один графический процессор NVIDIA H200 используется для обслуживания модели `Qwen3-Coder-Next` с квантованием FP8.

## Развертывание модели { #deploy-the-model }

1. В пользовательском интерфейсе GPUStack перейдите на страницу **Каталог моделей**.

2. Найдите `Qwen3-Coder-Next` и разверните модель, используя конфигурацию по умолчанию.

   ![Каталог моделей](../assets/integrations/claude-code/catalog.png)

3. После развертывания модель готова для вывода с использованием API, совместимого с Anthropic.

   ![Модель готова](../assets/integrations/claude-code/model-ready.png)

## Создайте ключ API. { #create-an-api-key }

1. Перейдите на страницу **Контроль доступа** > **Ключи API**, затем нажмите **Добавить ключ API**.

2. Введите имя, затем нажмите **Сохранить**.

3. Скопируйте ключ API и сохраните его для дальнейшего использования.

   ![Ключ API](../assets/integrations/claude-code/api-key.png)

## Установка Claude Code { #install-claude-code }

Следуйте инструкциям [документации по Claude Code](https://github.com/anthropics/claude-code), чтобы установить Claude Code в вашу систему.

## Установите CC-Switch (дополнительно) { #install-cc-switch-optional }

Чтобы легко переключаться между разными поставщиками моделей, вы можете использовать CC-Switch или аналогичные инструменты. Этот шаг не является обязательным — вы также можете настроить Claude Code вручную или использовать другой инструмент по вашему выбору.

Установите [CC-Switch](https://github.com/farion1231/cc-switch) согласно документации.

## Настройка Claude Code для GPUStack { #configure-claude-code-with-gpustack }

1. Откройте CC-Switch и добавьте собственного провайдера со следующими настройками:

   - **Имя поставщика**: `GPUStack`
   - **Конечная точка API**: URL-адрес вашего сервера GPUStack.
   - **Ключ API**: ключ API, который вы создали ранее.

2. Настройте все модели на использование `qwen3-coder-next`.

   ![Настройка переключателя CC](../assets/integrations/claude-code/cc-switch.png)

## Проверьте интеграцию { #test-the-integration }

1. Создайте новый каталог и запустите Claude Code:

   ```bash
   mkdir flappybird && cd flappybird
   claude
   ```

2. Убедитесь, что конфигурация правильна, используя команду `/status`.

3. Попробуйте выполнить типовое задание, например, попросите Claude Code создать игру Flappy Bird:

   ```
   Write a Flappy Bird game.
   ```

   ![Результат кода Клода](../assets/integrations/claude-code/claude.png)

4. После завершения работы Claude Code откройте сгенерированный HTML-файл в браузере, чтобы проверить результат.

   ![Игра Flappy Bird](../assets/integrations/claude-code/flappybird.gif)

## Заключение { #conclusion }

В этом руководстве мы успешно интегрировали Claude Code с GPUStack и использовали его для создания игры Flappy Bird. Теперь вы можете решать более сложные задачи кодирования с помощью Claude Code и использовать возможности GPUStack для эффективного обслуживания моделей.
