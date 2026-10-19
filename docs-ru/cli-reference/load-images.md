---
hide:
  - toc
translation_status: translated
---

# gpustack load-images { #gpustack-load-images }

Загрузка образов из OCI/Docker Archive в локальное хранилище образов контейнеров с помощью https://github.com/containers/skopeo.

```bash
gpustack load-images [OPTIONS] [input]

```

## Настройки { #configurations }

| <div style="width:220px">Флаг</div>                                   | <div style="width:120px">По умолчанию</div>                          | Описание |
| --------------------------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `--repository` `REPOSITORY`                                           | (пусто) | Фильтровать образы по имени репозитория |
| `--platform` `{linux/amd64,linux/arm64}`                              | (пусто) | Фильтровать образы по платформе |
| `--deprecated`                                                        |                                                                 | Включить устаревшие образы в список |
| `--max-workers` `MAX_WORKERS`                                         | `1`                                                             | Максимальное число рабочих потоков для параллельного копирования образов |
| `--max-retries` `MAX_RETRIES`                                         | `1`                                                             | Максимальное число повторных попыток копирования образа |
| `--destination`, `--dest` `DESTINATION`                               | `docker.io` (переменная среды: `GPUSTACK_SYSTEM_DEFAULT_CONTAINER_REGISTRY`) | Переопределить целевой реестр |
| `--destination-namespace`, `--dest-namespace` `DESTINATION_NAMESPACE` | (переменная среды: `GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_NAMESPACE`)    | Переопределить пространство имён в целевом реестре |
| `--archive-format` `{oci,docker}`                                     | `oci`                                                           | Формат сохраняемого архива |
| `--storage` `{docker,podman}`                                         | `docker`                                                        | Хранилище образов контейнеров, в которое загружаются образы |
| `input`                                                               | (текущий рабочий каталог) | Входной каталог для загрузки образов |
