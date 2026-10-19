---
translation_status: translated
---

# Обновление через Docker { #upgrade-via-docker }

Вы можете обновить установку GPUStack на основе Docker, извлекая новый образ и воссоздавая контейнеры.

Следующие инструкции по обновлению применимы только к GPUStack v2.0 и более поздним версиям.

Для установки до версии 0.7 обратитесь к [руководству по миграции](../migration.md).

!!! note

    1. При обновлении сначала обновите сервер GPUStack, а затем обновите рабочие процессы.

    2. Пожалуйста, **НЕ** обновляйтесь с/до основной версии (dev) или версии-кандидата (rc), поскольку они могут содержать критические изменения. Используйте новую установку, если хотите попробовать основную версию или версию rc.

!!! warning

    **Сначала сделайте резервную копию:** Прежде чем приступить к обновлению, настоятельно рекомендуется создать резервную копию базы данных.

    Для установок по умолчанию, в которых используется встроенная база данных PostgreSQL, остановите сервер GPUStack и создайте резервную копию каталога базы данных PostgreSQL, расположенного внутри контейнера по адресу:

    ```
    /var/lib/gpustack/postgresql/data
    ```

    Если вы используете внешнюю базу данных, вместо этого следуйте процедуре резервного копирования вашего поставщика базы данных.

Сначала обновите **сервер**, затем обновите **рабочие**. Сервер обновляется путем получения нового образа (либо тега конкретной версии, либо тега `latest`), удаления старого контейнера и запуска нового с обновленным образом, используя **те же аргументы и тома**, что и раньше.

## Обновите сервер { #upgrade-the-server }

```bash
docker pull gpustack/gpustack:latest  # or: docker pull gpustack/gpustack:vx.y.z

docker stop gpustack
docker rm gpustack

docker run -d --name gpustack \
  ... \
  gpustack/gpustack:latest
  ...
```

## Обновление воркеров { #upgrade-the-workers }

После запуска новой версии сервера обновите рабочие процессы в каждом кластере. См. раздел [Обновление развертывания кластера](cluster.md) для получения информации о процедурах Docker, Kubernetes и облачного кластера.

## Обновление через Docker Compose { #upgrade-via-docker-compose }

Если GPUStack развернут через Docker Compose из клонированного репозитория, переключитесь на тег нового выпуска и пересоздайте контейнеры с обновлённым образом:

```bash
cd gpustack

# Fetch the tags and check out the target stable release
git fetch --tags
git checkout <new-tag>   # e.g. vx.y.z

cd docker-compose

# Pull the new image and recreate the containers
sudo docker compose -f docker-compose.server.yaml pull
sudo docker compose -f docker-compose.server.yaml up -d
```
