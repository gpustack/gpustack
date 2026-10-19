---
hide:
  - toc
translation_status: translated
---

# gpustack save-images { #gpustack-save-images }

Сохранение образов в формате OCI/Docker Archive в локальный каталог с помощью https://github.com/containers/skopeo.

```bash
gpustack save-images [OPTIONS] [output]

```

## Настройки { #configurations }

| <div style="width:220px">Флаг</div>                             | <div style="width:120px">По умолчанию</div> | Описание |
| --------------------------------------------------------------- | -------------------------------------- | ----------------------------------------------------------------------- |
| `--backend` `{cann,corex,cuda,dtk,hggc,maca,musa,neuware,rocm}` | (пусто) | Фильтровать образы gpustack/runner по имени бэкенда |
| `--backend-version` `BACKEND_VERSION`                           | (пусто) | Фильтровать образы gpustack/runner по точной версии бэкенда |
| `--backend-version-prefix` `BACKEND_VERSION_PREFIX`             | (пусто) | Фильтровать образы gpustack/runner по префиксу версии бэкенда |
| `--backend-variant` `BACKEND_VARIANT`                           | (пусто) | Фильтровать образы gpustack/runner по варианту бэкенда |
| `--service` `{voxbox,vllm,mindie,sglang}`                       | (пусто) | Фильтровать образы gpustack/runner по имени сервиса |
| `--service-version` `SERVICE_VERSION`                           | (пусто) | Фильтровать образы gpustack/runner по точной версии сервиса |
| `--service-version-prefix` `SERVICE_VERSION_PREFIX`             | (пусто) | Фильтровать образы gpustack/runner по префиксу версии сервиса |
| `--repository` `REPOSITORY`                                     | (пусто) | Фильтровать образы по имени репозитория |
| `--platform` `{linux/amd64,linux/arm64}`                        | (пусто) | Фильтровать образы по платформе |
| `--deprecated`                                                  |                                        | Включить устаревшие образы в список |
| `--max-workers` `MAX_WORKERS`                                   | `1`                                    | Максимальное число рабочих потоков для параллельного копирования образов |
| `--max-retries` `MAX_RETRIES`                                   | `1`                                    | Максимальное число повторных попыток копирования образа |
| `--source`, `--src` `SOURCE`                                    | `docker.io`                            | Исходный реестр |
| `--source-namespace`, `--src-namespace` `SOURCE_NAMESPACE`      | (пусто) | Исходное пространство имён в исходном реестре |
| `--source-username`, `--src-user` `SOURCE_USERNAME`             | (переменная среды: `SOURCE_USERNAME`)               | Имя пользователя для аутентификации в исходном реестре |
| `--source-password`, `--src-passwd` `SOURCE_PASSWORD`           | (переменная среды: `SOURCE_PASSWORD`)               | Пароль/токен для аутентификации в исходном реестре |
| `--archive-format` `{oci,docker}`                               | `oci`                                  | Формат сохраняемого архива |
| `output`                                                        | (текущий рабочий каталог) | Выходной каталог для сохранения образов |
