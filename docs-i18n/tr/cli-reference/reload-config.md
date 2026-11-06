---
hide:
  - toc
translation_status: translated
---

# gpustack reload-config { #gpustack-reload-config }

Çalışma sırasında güvenle değiştirilebilen yapılandırma değerlerini yeniden yükler.

```bash
gpustack reload-config [OPTIONS]
```

## Yapılandırma { #configurations }

| <div style="width:180px">Bayrak</div> | <div style="width:100px">Varsayılan</div> | Açıklama |
| ----------------------------------- | -------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--set` değer                       | (boş) | İzin verilen tek bir alanı `key=value` biçiminde ayarlar (anahtarlar tireli yazılır). Değerler alan türüne dönüştürülür: mantıksal değerler için `true/false`, `1/0`, `yes/no`, `y/n`, `on/off`; listeler için virgülle ayrılmış dizeler; sözlükler için JSON dizeleri kullanılır. Birden çok `--set` belirtilebilir; sonraki değerler öncekileri ve `--file` değerlerini geçersiz kılar. |
| `--file` değer                      | (boş) | Yapılandırmayı YAML dosyasından yükler. Yalnızca izin verilen alanlar uygulanır. Anahtarlar snake_case biçimine dönüştürülür. `--set` ile verilen değerler dosyadaki değerleri geçersiz kılar. |
| `--list`                            | `False`                                | İzin verilen alanları ve değerleri gösterir. Belirtildiğinde diğer seçenekler yok sayılır. |
| `--api-key` değer                   | (boş) | Sunucu tarafında yönetici kullanıcı olarak kimlik doğrulamak için bir API anahtarı sağlar. |
| `--server-port` değer               | `30080`                                | Çalışma zamanı yapılandırmasını uygulamak veya listelemek için GPUStack API sunucusunun hedef portu. Belirtilmezse ayarlanmış olan `GPUSTACK_API_PORT`, aksi takdirde `30080` kullanılır. |
| `--worker-port` değer               | `10150`                                | Çalışma zamanı yapılandırmasını uygulamak veya listelemek için GPUStack worker hedef portu. Belirtilmezse ayarlanmış olan `GPUSTACK_WORKER_PORT`, aksi takdirde `10150` kullanılır. |
