---
translation_status: translated
---

# Интеграция с OpenClaw { #integrating-with-openclaw }

[OpenClaw](https://openclaw.ai/) — это персональный ИИ-помощник, который работает на локальных устройствах и поддерживает взаимодействие через распространенные каналы, такие как Telegram, WhatsApp, Slack и Discord.

## Развертывание модели { #deploy-the-model }

Чтобы завершить развертывание модели, обратитесь к разделу [**Развертывание модели**](../user-guide/model-deployment-management.md#deploy-model) в документации GPUStack.

## Информация о доступе к API { #api-access-info }

1. Войдите в веб-интерфейс GPUStack.
2. Перейдите на страницу **Маршруты**.
3. В меню справа от целевой модели выберите **Информация о доступе к API**.

![](../assets/integrations/openclaw-01.png)

![](../assets/integrations/openclaw-02.png)

Запишите следующую информацию (если ключ API еще не создан, следуйте инструкциям на странице, чтобы его создать):

* **URL-адрес доступа**
* **Название модели**
* **Ключ API**

## Установите OpenClaw { #install-openclaw }

Следуйте официальной документации OpenClaw для завершения установки: [https://docs.openclaw.ai/install](https://docs.openclaw.ai/install)

## Настройка GPUStack в OpenClaw. { #configure-gpustack-in-openclaw }

1. Запустите мастер интерактивной настройки:

   ```bash
   openclaw onboard --install-daemon
   ```

2. В разделе **Модель/Поставщик аутентификации** выберите **Пользовательский поставщик**

   ![](../assets/integrations/openclaw-03.png)

3. Заполните информацию, предоставленную GPUStack, согласно запросу:

    * **Базовый URL-адрес API**: URL-адрес доступа.
    * **Ключ API**: Ключ API
    * **Идентификатор модели**: Название модели.

   ![](../assets/integrations/openclaw-04.png)

После выполнения этих шагов OpenClaw будет использовать GPUStack для вызова соответствующей модели для вывода.

## Настройка каналов { #configure-channels }

Следуйте официальной документации OpenClaw, чтобы настроить желаемый канал связи: [https://docs.openclaw.ai/channels](https://docs.openclaw.ai/channels)
