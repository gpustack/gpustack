---
hide:
  - toc
translation_status: translated
---

# gpustack list-images { #gpustack-list-images }

Вывод списка образов.

```bash
gpustack list-images [OPTIONS]

```

## Настройки { #configurations }

| <div style="width:220px">Флаг</div>                             | <div style="width:120px">По умолчанию</div> | Описание |
|-----------------------------------------------------------------|----------------------------------------|------------------------------------------------------------------------------------|
| `--backend` `{cann,corex,cuda,dtk,hggc,maca,musa,neuware,rocm}` | (пусто) | Фильтровать образы gpustack/runner по имени бэкенда |
| `--backend-version` `BACKEND_VERSION`                           | (пусто) | Фильтровать образы gpustack/runner по точной версии бэкенда |
| `--backend-version-prefix` `BACKEND_VERSION_PREFIX`             | (пусто) | Фильтровать образы gpustack/runner по префиксу версии бэкенда |
| `--backend-variant` `BACKEND_VARIANT`                           | (пусто) | Фильтровать образы gpustack/runner по варианту бэкенда |
| `--service` `{voxbox,vllm,mindie,sglang}`                       | (пусто) | Фильтровать образы gpustack/runner по имени сервиса |
| `--service-version` `SERVICE_VERSION`                           | (пусто) | Фильтровать образы gpustack/runner по точной версии сервиса |
| `--service-version-prefix` `SERVICE_VERSION_PREFIX`             | (пусто) | Фильтровать образы gpustack/runner по префиксу версии сервиса |
| `--repository` `REPOSITORY`                                     | (пусто) | Фильтровать образы по имени репозитория |
| `--platform` `{linux/amd64,linux/arm64}`                        | (пусто) | Фильтровать образы по платформе |
| `--format {text,json}`                                          | `text`                                 | Формат вывода: `text` — текст для чтения человеком, `json` — массив JSON |
