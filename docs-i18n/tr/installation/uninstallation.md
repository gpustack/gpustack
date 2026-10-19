---
translation_status: translated
---

# Kaldırma { #uninstallation }

GPUStack genellikle konteynerleştirme kullanılarak kurulur, bu nedenle kaldırma işlemi esas olarak konteynerin ve ilgili veri birimlerinin kaldırılmasını içerir.

Örneğin, GPUStack `gpustack` adlı bir Docker kapsayıcısında çalışıyorsa şunu çalıştırın:

```bash
docker rm -f gpustack

```

İsteğe bağlı olarak ilişkili veri birimlerini kaldırmak için şunu kullanın:

```bash
docker volume rm <data_volume_name>

```
