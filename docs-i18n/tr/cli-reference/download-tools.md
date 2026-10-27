---
hide:
  - toc
translation_status: translated
---

# gpustack download-tools { #gpustack-download-tools }

gguf-parser ve fastfetch dahil bağımlılık araçlarını indirir.

```bash
gpustack download-tools [OPTIONS]
```

## Yapılandırma { #configurations }

| <div style="width:180px">Bayrak</div> | <div style="width:100px">Varsayılan</div> | Açıklama |
| ----------------------------------- | -------------------------------------- | ----------------------------------------------------------------------------- |
| `--tools-download-base-url` değer | (boş) | Bağımlılık araçlarının indirileceği temel URL. |
| `--save-archive` değer              | (boş) | İndirilen araçların tar arşivi olarak kaydedileceği yol. |
| `--load-archive` değer              | (boş) | Araçları indirmek yerine önceden indirilmiş araçları yüklemek için kullanılacak tar arşivinin yolu. |
| `--system` değer                    | Varsayılan, geçerli işletim sistemidir. | Araçların indirileceği işletim sistemi. Seçenekler: `linux`, `windows`, `macos`. |
| `--arch` değer                      | Varsayılan, geçerli mimaridir. | Araçların indirileceği mimari. Seçenekler: `amd64`, `arm64`. |
