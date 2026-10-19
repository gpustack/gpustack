---
hide:
  - toc
translation_status: translated
---

# gpustack list-images { #gpustack-list-images }

İmajları listeler.

```bash
gpustack list-images [OPTIONS]

```

## Yapılandırma { #configurations }

| <div style="width:220px">Bayrak</div>                             | <div style="width:120px">Varsayılan</div> | Açıklama |
|-----------------------------------------------------------------|----------------------------------------|------------------------------------------------------------------------------------|
| `--backend` `{cann,corex,cuda,dtk,hggc,maca,musa,neuware,rocm}` | (boş) | gpustack/runner imajlarını arka uç adına göre filtreler |
| `--backend-version` `BACKEND_VERSION`                           | (boş) | gpustack/runner imajlarını tam arka uç sürümüne göre filtreler |
| `--backend-version-prefix` `BACKEND_VERSION_PREFIX`             | (boş) | gpustack/runner imajlarını arka uç sürümünün önekine göre filtreler |
| `--backend-variant` `BACKEND_VARIANT`                           | (boş) | gpustack/runner imajlarını arka uç çeşidine göre filtreler |
| `--service` `{voxbox,vllm,mindie,sglang}`                       | (boş) | gpustack/runner imajlarını servis adına göre filtreler |
| `--service-version` `SERVICE_VERSION`                           | (boş) | gpustack/runner imajlarını tam servis sürümüne göre filtreler |
| `--service-version-prefix` `SERVICE_VERSION_PREFIX`             | (boş) | gpustack/runner imajlarını servis sürümünün önekine göre filtreler |
| `--repository` `REPOSITORY`                                     | (boş) | İmajları depo adına göre filtreler |
| `--platform` `{linux/amd64,linux/arm64}`                        | (boş) | İmajları platforma göre filtreler |
| `--format {text,json}`                                          | `text`                                 | Çıktı biçimi. `text` okunabilir metin, `json` ise JSON dizisi biçimidir |
