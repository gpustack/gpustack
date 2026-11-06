---
translation_status: translated
---

# Удаление { #uninstallation }

GPUStack обычно устанавливается с использованием контейнеризации, поэтому удаление в основном включает в себя удаление контейнера и всех связанных с ним томов данных.

Например, если GPUStack работает в контейнере Docker с именем `gpustack`, запустите:

```bash
docker rm -f gpustack

```

Чтобы при необходимости удалить связанные тома данных, используйте:

```bash
docker volume rm <data_volume_name>

```
