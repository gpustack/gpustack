---
translation_status: translated
---

# Краткое руководство { #quickstart }

В этом руководстве рассказывается, как запустить GPUStack на локальных серверах с графическим процессором. Чтобы использовать [облачные графические процессоры](./tutorials/adding-gpucluster-using-digitalocean.md) или интегрироваться с [существующим кластером Kubernetes](./tutorials/adding-gpucluster-using-kubernetes.md), см. соответствующие руководства.

!!! info "Предварительные условия"

    1. Узел как минимум с одним графическим процессором NVIDIA. Для других типов графических процессоров ознакомьтесь с рекомендациями в пользовательском интерфейсе GPUStack при добавлении работника или обратитесь к [Документации по установке](./installation/requirements.md) для получения более подробной информации.
    2. Убедитесь, что драйвер NVIDIA, [Docker](https://docs.docker.com/engine/install/) и [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) установлены на рабочем узле.
    3. **(Необязательно)** Узел ЦП для размещения сервера GPUStack. Сервер GPUStack не требует графического процессора и может работать на машине только с процессором. [Docker](https://docs.docker.com/engine/install/) должен быть установлен. Также поддерживается Docker Desktop (для Windows и macOS). Если выделенный узел ЦП недоступен, сервер GPUStack можно установить на том же компьютере, что и рабочий узел ГП.
    4. Для рабочих узлов GPUStack поддерживается только Linux. Если вы используете Windows, рассмотрите возможность использования WSL2 и избегайте использования Docker Desktop. macOS не поддерживается для рабочих узлов GPUStack.

## Установите GPUStack. { #install-gpustack }

Выполните следующую команду, чтобы установить и запустить сервер GPUStack с помощью [Docker](https://docs.docker.com/engine/install/):

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    -p 80:80 \
    --volume gpustack-data:/var/lib/gpustack \
    gpustack/gpustack
```

??? Note "Альтернатива: использовать зеркало реестра контейнеров Quay."

    Если вы не можете получить образы из Docker Hub или загрузка происходит очень медленно, вы можете использовать наше зеркало реестра контейнеров Quay, указав свой реестр на `quay.io`:

    ```bash
    sudo docker run -d --name gpustack \
        --restart unless-stopped \
        -p 80:80 \
        --volume gpustack-data:/var/lib/gpustack \
        quay.io/gpustack/gpustack \
        --system-default-container-registry quay.io
    ```

Проверьте журналы запуска GPUStack:

```bash
sudo docker logs -f gpustack
```

После запуска GPUStack выполните следующую команду, чтобы получить пароль администратора по умолчанию:

```bash
sudo docker exec gpustack cat /var/lib/gpustack/initial_admin_password
```

Откройте браузер и перейдите к `http://your_host_ip`, чтобы получить доступ к пользовательскому интерфейсу GPUStack. Для входа используйте имя пользователя по умолчанию `admin` и пароль, который вы получили выше.

## Настройка кластера графических процессоров { #set-up-a-gpu-cluster }

1. В пользовательском интерфейсе GPUStack перейдите на страницу `Clusters`.

2. Нажмите кнопку `Add Cluster`.

3. Выберите `Docker` в качестве поставщика кластера.

4. Заполните поля `Name` и `Description` для нового кластера, затем нажмите кнопку `Save`.

5. Следуйте рекомендациям пользовательского интерфейса, чтобы настроить новый рабочий узел. Вам нужно будет запустить команду Docker на рабочем узле, чтобы подключить его к серверу GPUStack. Команда будет выглядеть примерно так:

```bash
sudo docker run -d --name gpustack-worker \
      --restart=unless-stopped \
      --privileged \
      --network=host \
      --volume /var/run/docker.sock:/var/run/docker.sock \
      --volume gpustack-data:/var/lib/gpustack \
      --runtime nvidia \
      gpustack/gpustack \
      --server-url http://your_gpustack_server_url \
      --token your_worker_token \
      --advertise-address 192.168.1.2
```

6. Выполните команду на рабочем узле, чтобы подключить его к серверу GPUStack.

7. После успешного подключения рабочего узла он появится на странице `Workers` в пользовательском интерфейсе GPUStack.

## Развертывание модели { #deploy-a-model }

1. Перейдите на страницу `Catalog` в пользовательском интерфейсе GPUStack.

2. Выберите модель `Qwen3.5-0.8B` из списка доступных моделей.

3. После прохождения проверок совместимости развертывания нажмите кнопку `Save`, чтобы развернуть модель.

![развернуть qwen3 из каталога](assets/quick-start/quick-start-qwen3.png)

4. GPUStack начнет загрузку файлов модели и ее развертывание. Когда статус развертывания показывает `Running`, модель развернута успешно.

!!! note

    GPUStack использует контейнеры для запуска моделей. При первом развертывании модели может потребоваться некоторое время для загрузки файлов модели и образов контейнеров. Вы можете нажать `View Logs` в пользовательском интерфейсе, чтобы отслеживать ход развертывания.

![модель работает](assets/quick-start/model-running.png)

5. Нажмите `Playground - Chat` в меню навигации и убедитесь, что модель `qwen3.5-0.8b` выбрана в правом верхнем углу раскрывающегося списка `Model`. Теперь вы можете общаться с моделью в Песочнице пользовательского интерфейса.

![быстрый чат](assets/quick-start/quick-chat.png)

## Используйте модель через API. { #use-the-model-via-api }

1. Перейдите на страницу `Access Control` > `API Keys`, затем нажмите кнопку `New API Key`.

2. Заполните `Name` и нажмите кнопку `Save`.

3. Скопируйте сгенерированный ключ API и сохраните его в безопасном месте. Обратите внимание, что вы можете увидеть его только один раз при создании.

4. Теперь вы можете использовать ключ API для доступа к конечным точкам API, совместимым с OpenAI, предоставляемым GPUStack. Например, используйте Curl следующим образом:

```bash
# Replace `your_api_key` and `your_gpustack_server_url`
# with your actual API key and GPUStack server URL.
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
    "model": "qwen3.5-0.8b",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Tell me a joke."
      }
    ],
    "stream": true
  }'
```

## Очистка { #cleanup }

После завершения использования развернутой модели вы можете перейти на страницу `Deployments` в пользовательском интерфейсе GPUStack и удалить модель, чтобы освободить ресурсы.
