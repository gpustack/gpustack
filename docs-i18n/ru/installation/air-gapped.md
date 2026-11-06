---
translation_status: translated
---

# Установка в изолированной среде { #air-gapped-installation }

GPUStack можно установить в изолированной (автономной) среде без доступа к Интернету.

После запуска он сам предоставляет эту документацию — см. [Документация](#documentation) ниже.

## Предварительные условия { #prerequisites }

### Драйвер { #driver }

Убедитесь, что в вашей системе установлены соответствующие драйверы графического процессора для вашего оборудования.

Подробную информацию о совместимости драйверов см. в разделе [Требования к установке](./requirements.md).

### Среда работы контейнера { #container-running-environment }

[Docker](https://docs.docker.com/engine/install/) должен быть установлен.

Если ваша система поддерживает набор инструментов для контейнеров, установите и настройте его по мере необходимости (например, NVIDIA Container Toolkit, AMD ROCm Container Toolkit и т. д.).

### Образы контейнеров { #container-images }

GPUStack предлагает [Селектор образов](../image-selector/index.html), который помогает пользователям легко выбирать образы, которые они хотят загрузить. Он полностью работает в браузере с данными, поставляемыми с этими документами, поэтому работает без доступа в Интернет; его список версий охватывает только выпуски до той, с которой он поставляется. Для более продвинутой или автоматической синхронизации GPUStack также предоставляет команды управления образами:

- `gpustack copy-images`: синхронизировать образы из одного реестра в другой.
- `gpustack save-images`: Загрузите образы и сохраните их локально.
- `gpustack load-images`: импортировать образы из локальных пакетов.
- `gpustack list-images`: показать манифест образов для текущей версии.

Ниже приведены подробные сведения о том, как использовать эти команды CLI.

- **Копировать образы**

GPUStack предоставляет различные образы контейнеров для разных компонентов и серверов вывода, доступные на [Docker Hub](https://hub.docker.com/u/gpustack) и [Quay.io](https://quay.io/user/gpustack/).

Чтобы перенести необходимые образы контейнеров во внутренний реестр с компьютера с доступом в Интернет, используйте команду GPUStack `copy-images`:

```bash
sudo docker run --rm -it --entrypoint "" gpustack/gpustack \
    gpustack copy-images \
    --destination <your_internal_registry> \
    --destination-username <your_username> \
    --destination-password <your_password>
```

Если вы не можете получить образы из `Docker Hub` или загрузка происходит очень медленно, вы можете использовать наше зеркало `Quay.io`, указав исходный реестр на `quay.io`:

```bash hl_lines="3"
sudo docker run --rm -it --entrypoint "" gpustack/gpustack \
    gpustack copy-images \
    --source quay.io \
    --destination <your_internal_registry> \
    --destination-username <your_username> \
    --destination-password <your_password>
```

!!! note

    По умолчанию используется последняя версия. Чтобы настроить таргетинг на конкретную версию, используйте полный тег образа, например gpustack/gpustack:vx.y.z.

Более подробную информацию о `copy-images` см. в [Справочнике по CLI](../cli-reference/copy-images.md).

- **Список образов**

Если вы не можете напрямую получить доступ к своему внутреннему реестру, вы можете сначала получить образ `gpustack/gpustack`, а затем использовать команду `list-images`, чтобы увидеть, какие образы необходимо загрузить:

```bash
sudo docker run --rm -it --entrypoint "" \
    gpustack/gpustack \
    gpustack list-images
```

!!! note

    По умолчанию используется последняя версия. Чтобы настроить таргетинг на конкретную версию, используйте полный тег образа, например gpustack/gpustack:vx.y.z.

Отображаемый список образов включает все поддерживаемые ускорители, механизмы вывода, версии и архитектуры. Если вам нужен только подмножество, см. [Справочник по CLI](../cli-reference/list-images.md), где описаны параметры фильтрации.

- **Копировать и загружать образы**

Если ваша целевая среда изолирована или не имеет доступа к Интернету, вы можете сначала загрузить необходимые образы на компьютер с подключением к Интернету, а затем перенести и загрузить их в автономную среду.

GPUStack предоставляет команды `save-images` и `load-images` для этого рабочего процесса.

**Копировать образы**

Запустите следующую команду на компьютере, имеющем доступ к Интернету, чтобы загрузить и упаковать необходимые образы:

```bash
sudo docker run --rm -it --entrypoint "" \
 -v ./gpustack-air-gapped:/gpustack-air-gapped \
 --workdir /gpustack-air-gapped \
 gpustack/gpustack \
 gpustack save-images \
 --platform linux/amd64 \
 --backend cuda \
 --backend-version 12.9 \
 --service vllm \
 --service-version 0.15.1 \
 --max-workers 3 \
 --max-retries 3
```

Эта команда загружает все необходимые образы контейнеров на основе указанной платформы, серверной части и конфигурации службы и сохраняет их как локальные пакеты в каталоге `gpustack-air-gapped`. В примере команды показано, как применить параметры фильтрации для загрузки только подмножества образов.

!!! note

    По умолчанию используется последняя версия. Чтобы настроить таргетинг на конкретную версию, используйте полный тег образа, например gpustack/gpustack:vx.y.z.

Вы можете настроить фильтры, чтобы загружать только те образы, которые вам нужны. См. [Справочник по CLI](../cli-reference/save-images.md) для получения информации обо всех доступных параметрах.

После завершения загрузки перенесите созданный каталог в изолированную среду, используя предпочитаемый вами метод (например, USB-накопитель или внутреннюю передачу файлов).

**Загрузить образы**

На целевом компьютере загрузите сохраненные образы в среду выполнения локального контейнера:

```bash
sudo docker run --rm -it --entrypoint "" \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume ./gpustack-air-gapped:/gpustack-air-gapped \
    --workdir /gpustack-air-gapped \
    gpustack/gpustack \
    gpustack load-images \
    --platform linux/amd64 \
    --max-workers 3 \
    --max-retries 3 \
    /gpustack-air-gapped
```

Эта команда импортирует все пакеты образов из указанного каталога в локальный демон Docker, делая их доступными для GPUStack.

!!! note

    По умолчанию используется последняя версия. Чтобы настроить таргетинг на конкретную версию, используйте полный тег образа, например gpustack/gpustack:vx.y.z.

Более подробную информацию о `load-images` см. в [Справочнике по CLI](../cli-reference/load-images.md).

## Установка { #installation }

После подготовки внутреннего реестра контейнеров с необходимыми образами вы можете установить GPUStack в изолированной среде.

```diff
 sudo docker run -d --name gpustack \
     --restart unless-stopped \
     -p 80:80 \
     --volume gpustack-data:/var/lib/gpustack \
-    gpustack/gpustack
+    <your_internal_registry>/gpustack/gpustack \
+    --system-default-container-registry <your_internal_registry>

```

### Извлечение образов серверной части вывода из защищенного реестра. { #pulling-inference-backend-images-from-a-secure-registry }

Если ваш внутренний реестр контейнеров требует аутентификации,
установите следующие переменные среды при запуске работника GPUStack, чтобы он мог извлекать образ бегуна.

```diff
 sudo docker run -d --name gpustack \
     ...
+    --env GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY_USERNAME=<your_internal_registry_username> \
+    --env GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY_PASSWORD=<your_internal_registry_password> \
     <your_internal_registry>/gpustack/gpustack \
     --system-default-container-registry <your_internal_registry>

```

### Извлечение серверных образов вывода из пространства имен, отличного от стандартного. { #pulling-inference-backend-images-from-non-default-namespace }

Если ваш внутренний реестр контейнеров использует пространство имен, отличное от стандартного `gpustack`,
установите следующую переменную среды при запуске работника GPUStack, чтобы он мог извлекать образ бегуна.

```diff
 sudo docker run -d --name gpustack \
     ...
+    --env GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_NAMESPACE=<your_namespace> \
     <your_internal_registry>/gpustack/gpustack \
     --system-default-container-registry <your_internal_registry>
```

## Документация { #documentation }

Сервер хранит эту документацию и обслуживает ее по адресу `/help/`, поэтому в изолированной среде не требуется доступ к онлайн-сайту:

```
http://<your_gpustack_server>/help/
```

Это та же документация, что и онлайн-сайт, в той версии, из которой был создан образ сервера, и ее поиск работает в автономном режиме.
