---
hide:
  - toc
translation_status: translated
---

# gpustack save-images { #gpustack-save-images }

https://github.com/containers/skopeo kullanarak imajları OCI/Docker Archive biçiminde yerel bir yola kaydeder.

```bash
gpustack save-images [OPTIONS] [output]

```

## Yapılandırma { #configurations }

| <div style="width:220px">Bayrak</div>                             | <div style="width:120px">Varsayılan</div> | Açıklama |
| --------------------------------------------------------------- | -------------------------------------- | ----------------------------------------------------------------------- |
| `--backend` `{cann,corex,cuda,dtk,hggc,maca,musa,neuware,rocm}` | (boş) | gpustack/runner imajlarını arka uç adına göre filtreler |
| `--backend-version` `BACKEND_VERSION`                           | (boş) | gpustack/runner imajlarını tam arka uç sürümüne göre filtreler |
| `--backend-version-prefix` `BACKEND_VERSION_PREFIX`             | (boş) | gpustack/runner imajlarını arka uç sürümünün önekine göre filtreler |
| `--backend-variant` `BACKEND_VARIANT`                           | (boş) | gpustack/runner imajlarını arka uç çeşidine göre filtreler |
| `--service` `{voxbox,vllm,mindie,sglang}`                       | (boş) | gpustack/runner imajlarını servis adına göre filtreler |
| `--service-version` `SERVICE_VERSION`                           | (boş) | gpustack/runner imajlarını tam servis sürümüne göre filtreler |
| `--service-version-prefix` `SERVICE_VERSION_PREFIX`             | (boş) | gpustack/runner imajlarını servis sürümünün önekine göre filtreler |
| `--repository` `REPOSITORY`                                     | (boş) | İmajları depo adına göre filtreler |
| `--platform` `{linux/amd64,linux/arm64}`                        | (boş) | İmajları platforma göre filtreler |
| `--deprecated`                                                  |                                        | Kullanımdan kaldırılmış imajları listeye dahil eder |
| `--max-workers` `MAX_WORKERS`                                   | `1`                                    | İmajları eş zamanlı kopyalamak için kullanılacak en fazla iş parçacığı sayısı |
| `--max-retries` `MAX_RETRIES`                                   | `1`                                    | Bir imajı kopyalamak için en fazla yeniden deneme sayısı |
| `--source`, `--src` `SOURCE`                                    | `docker.io`                            | Kaynak kayıt deposu |
| `--source-namespace`, `--src-namespace` `SOURCE_NAMESPACE`      | (boş) | Kaynak kayıt deposundaki kaynak ad alanı |
| `--source-username`, `--src-user` `SOURCE_USERNAME`             | (ortam değişkeni: `SOURCE_USERNAME`)               | Kaynak kayıt deposunda kimlik doğrulama için kullanıcı adı |
| `--source-password`, `--src-passwd` `SOURCE_PASSWORD`           | (ortam değişkeni: `SOURCE_PASSWORD`)               | Kaynak kayıt deposunda kimlik doğrulama için parola/token |
| `--archive-format` `{oci,docker}`                               | `oci`                                  | Kaydedilecek arşiv biçimi |
| `output`                                                        | (geçerli çalışma dizini) | İmajların kaydedileceği çıktı dizini |
