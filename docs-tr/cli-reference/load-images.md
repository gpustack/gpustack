---
hide:
  - toc
translation_status: translated
---

# gpustack load-images { #gpustack-load-images }

https://github.com/containers/skopeo kullanarak OCI/Docker Archive dosyalarındaki imajları yerel konteyner imajı depolamasına yükler.

```bash
gpustack load-images [OPTIONS] [input]

```

## Yapılandırma { #configurations }

| <div style="width:220px">Bayrak</div>                                   | <div style="width:120px">Varsayılan</div>                          | Açıklama |
| --------------------------------------------------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------------------- |
| `--repository` `REPOSITORY`                                           | (boş) | İmajları depo adına göre filtreler |
| `--platform` `{linux/amd64,linux/arm64}`                              | (boş) | İmajları platforma göre filtreler |
| `--deprecated`                                                        |                                                                 | Kullanımdan kaldırılmış imajları listeye dahil eder |
| `--max-workers` `MAX_WORKERS`                                         | `1`                                                             | İmajları eş zamanlı kopyalamak için kullanılacak en fazla iş parçacığı sayısı |
| `--max-retries` `MAX_RETRIES`                                         | `1`                                                             | Bir imajı kopyalamak için en fazla yeniden deneme sayısı |
| `--destination`, `--dest` `DESTINATION`                               | `docker.io` (ortam değişkeni: `GPUSTACK_SYSTEM_DEFAULT_CONTAINER_REGISTRY`) | Hedef kayıt deposunu geçersiz kılar |
| `--destination-namespace`, `--dest-namespace` `DESTINATION_NAMESPACE` | (ortam değişkeni: `GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_NAMESPACE`)    | Hedef kayıt deposundaki ad alanını geçersiz kılar |
| `--archive-format` `{oci,docker}`                                     | `oci`                                                           | Kaydedilecek arşiv biçimi |
| `--storage` `{docker,podman}`                                         | `docker`                                                        | İmajların yükleneceği konteyner imajı depolaması |
| `input`                                                               | (geçerli çalışma dizini) | İmajların yükleneceği kaynak dizin |
