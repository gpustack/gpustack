---
translation_status: translated
---

# Установка через Docker { #installation-via-docker }

## Предварительные условия { #prerequisites }

**GPUStack-сервер:**

- [Docker](https://docs.docker.com/engine/install/) должен быть установлен. Также поддерживается Docker Desktop (Windows и macOS).

**GPUStack работники:**

- [Docker](https://docs.docker.com/engine/install/) должен быть установлен. Docker Desktop **не поддерживается**.
- Для рабочих узлов GPUStack поддерживается только Linux. Если вы используете Windows, рассмотрите возможность использования WSL2 и избегайте использования Docker Desktop. macOS не поддерживается для рабочих узлов GPUStack.
- Убедитесь, что для вашего оборудования установлены соответствующие драйверы графического процессора и наборы инструментов для контейнеров. Подробности см. в разделе [Требования к установке](./requirements.md).

## Установите сервер GPUStack. { #install-gpustack-server }

Запустите следующую команду, чтобы установить и запустить сервер GPUStack с помощью Docker:

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    -p 80:80 \
    --volume gpustack-data:/var/lib/gpustack \
    gpustack/gpustack
```

!!! note

    GPUStack v2 использует единый образ контейнера для всех типов устройств с графическим процессором.

## Запуск { #startup }

Проверьте журналы контейнера GPUStack:

```bash
sudo docker logs -f gpustack
```

Как только сервер заработает, откройте `http://your_host_ip` в браузере, чтобы получить доступ к пользовательскому интерфейсу GPUStack.

Войдите в систему, используя имя пользователя `admin` и пароль по умолчанию. Получите первоначальный пароль с помощью:

```bash
sudo docker exec -it gpustack \
    cat /var/lib/gpustack/initial_admin_password
```

## Добавление кластеров графических процессоров и рабочих узлов. { #add-gpu-clusters-and-worker-nodes }

Следуйте инструкциям пользовательского интерфейса на страницах `Clusters` и `Workers`, чтобы добавить кластеры графических процессоров и рабочие узлы.

## Пользовательская конфигурация { #custom-configuration }

В следующих разделах описаны примеры пользовательских параметров конфигурации при запуске серверного контейнера GPUStack. Полный список доступных опций см. в [Справочнике по CLI](../cli-reference/start.md).

### Включить HTTPS с помощью специального сертификата. { #enable-https-with-custom-certificate }


```diff
 sudo docker run -d --name gpustack \
     ...
     -p 80:80 \
+    -p 443:443 \
     --volume gpustack-data:/var/lib/gpustack \
+    --volume /path/to/cert_files:/path/to/cert_files:ro \
+    -e GPUSTACK_SSL_KEYFILE=/path/to/cert_files/your_domain.key \
+    -e GPUSTACK_SSL_CERTFILE=/path/to/cert_files/your_domain.crt \
     gpustack/gpustack
     ...
```

`GPUSTACK_SSL_CERTFILE` должен содержать сертификат сервера, за которым следуют все промежуточные сертификаты. Когда сертификат сервера подписан частным центром сертификации, добавьте также его пакет PEM CA, чтобы рабочие контейнеры и контейнеры бенчмарка могли автоматически проверять сервер:

```diff
 sudo docker run -d --name gpustack \
     ...
+    -e GPUSTACK_SSL_CA_CERTFILE=/path/to/cert_files/ca-bundle.crt \
     gpustack/gpustack
     ...
```

Для самозаверяющего сертификата сервера используется сам файл сертификата, если отдельный пакет CA не настроен. Сертификаты, выданные общедоступными центрами сертификации, не требуют дополнительной настройки центра сертификации.

Команда регистрации включает контрольную сумму SHA-256 пакета начальной загрузки CA. Worker проверяет эту контрольную сумму перед установкой пакета, загруженного без проверки TLS. Это не закрепление сертификата: если сервер уже проходит обычную проверку TLS, исполнитель использует существующее хранилище доверенных сертификатов без загрузки и проверки пакета начальной загрузки.

Работник сначала пытается установить проверенный центр сертификации в системное хранилище доверенных сертификатов. Если это хранилище невозможно обновить или клиент TLS работника использует другой пакет, работник использует временный объединенный пакет через `SSL_CERT_FILE` для своего процесса и дочерних процессов. Существующие сертификаты, предоставленные оператором, сохраняются; замена внедренного загрузочного центра сертификации не сохраняет ранее внедренные пакеты.

В крайнем случае, если сертификат сервера вообще не может быть проверен на работнике, запустите этот работник с `--insecure-tls` (или `GPUSTACK_INSECURE_TLS=true`), чтобы пропустить проверку сертификата при его подключении к серверу. Трафик остается зашифрованным, но больше не защищен от перехвата, поэтому используйте его только в доверенных сетях. См. [Справочник по CLI](../cli-reference/start.md).

### Использование внешней базы данных { #using-an-external-database }

По умолчанию GPUStack использует встроенную базу данных PostgreSQL. Чтобы использовать внешнюю базу данных, такую ​​как PostgreSQL или MySQL, установите переменную среды `GPUSTACK_DATABASE_URL` или используйте аргумент `--database-url` при запуске контейнера GPUStack. См. [Требования к базам данных](requirements.md#database-requirements) для получения списка совместимых баз данных и проверенных версий.

```diff
 sudo docker run -d --name gpustack \
     ...
     --volume gpustack-data:/var/lib/gpustack \
+    -e GPUSTACK_DATABASE_URL="postgresql://username:password@host:port/dbname" \
     gpustack/gpustack
     ...
```

### Настройка URL-адреса внешнего сервера. { #configure-external-server-url }

Если вы используете облачного провайдера для подготовки работников, установите URL-адрес внешнего сервера для регистрации работников, чтобы гарантировать, что работники могут правильно подключиться к серверу.

```diff
sudo docker run -d --name gpustack \
    ...
+   -e GPUSTACK_SERVER_EXTERNAL_URL="https://your_external_server_url" \
    gpustack/gpustack
    ...
```

### Дополнительные доверенные центры сертификации { #additional-trusted-cas }

Если GPUStack необходимо взаимодействовать со службами, которые используют сертификаты, выданные частным или корпоративным центром сертификации (например, самостоятельно размещённым поставщиком удостоверений, зеркалом Hugging Face или внутренней конечной точкой API), смонтируйте сертификат CA в контейнер под `/usr/local/share/ca-certificates/`. GPUStack автоматически импортирует смонтированные сертификаты CA во время запуска и добавит их в хранилище доверенных сертификатов системы.

```diff
 sudo docker run -d --name gpustack \
     ...
     --volume gpustack-data:/var/lib/gpustack \
+    --volume /path/to/custom-root-ca.crt:/usr/local/share/ca-certificates/custom-root-ca.crt:ro \
     gpustack/gpustack
     ...
```

!!! note

    Сертификат CA должен быть закодирован в формате PEM с расширением `.crt`. Вы можете подключить несколько сертификатов CA, добавив дополнительные флаги `--volume`.

## Установка через Docker Compose. { #installation-via-docker-compose }

### Предварительные условия { #prerequisites_1 }

- [Docker Compose](https://docs.docker.com/compose/install/) должен быть установлен.
- [Необходимые порты](./requirements.md#port-requirements) должны быть доступны.

### Развертывание { #deployment }

Файлы Docker Compose и файлы конфигурации хранятся в репозитории [GPUStack](https://github.com/gpustack/gpustack/tree/main/docker-compose).

Выполните следующие команды, чтобы клонировать последнюю стабильную версию:

```bash
LATEST_TAG=$(
    curl -s "https://api.github.com/repos/gpustack/gpustack/releases" \
    | grep '"tag_name"' \
    | sed -E 's/.*"tag_name": "([^"]+)".*/\1/' \
    | grep -Ev 'rc|beta|alpha|preview' \
    | head -1
)
echo "Latest stable release: $LATEST_TAG"
git clone -b "$LATEST_TAG" https://github.com/gpustack/gpustack.git
cd gpustack/docker-compose
```

Запустите сервер GPUStack:

```bash
sudo docker compose -f docker-compose.server.yaml up -d
```

Как только сервер заработает, откройте `http://your_host_ip` в браузере, чтобы получить доступ к пользовательскому интерфейсу GPUStack.

Войдите в систему, используя имя пользователя `admin` и пароль по умолчанию. Получите первоначальный пароль с помощью:

```bash
sudo docker exec -it gpustack-server cat /var/lib/gpustack/initial_admin_password
```

Информацию о встроенных и внешних параметрах наблюдения см. в разделе [Наблюдаемость](../user-guide/observability.md).
