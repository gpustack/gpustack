---
translation_status: translated
---

# Sorun giderme { #troubleshooting }

## GPUStack Günlüklerini Görüntüle { #view-gpustack-logs }

Varsayılan kurulum için GPUStack günlüklerini aşağıdaki komutlarla görüntüleyebilirsiniz:

```bash
docker logs -f gpustack
```

## Hata Ayıklama Modunu Etkinleştir { #enable-debug-mode }

GPUStack'i çalıştırırken `--debug` bayrağını ayarlayarak `DEBUG` modunu etkinleştirebilirsiniz:

```diff
sudo docker run -d --name gpustack \
    ...
    gpustack/gpustack \
+    --debug
    ...
```

GPUStack'in hata ayıklama modunu çalışma zamanında **sunucu kapsayıcısı** içinde aşağıdaki komutu çalıştırarak da etkinleştirebilirsiniz:

```bash
gpustack reload-config --set debug=true
```

`--debug`, Python bellek ayırma izlemeyi başlatmadan ayrıntılı günlük kaydını etkinleştirir. Bellek profillemesini etkinleştirmek için GPUStack başlamadan önce `GPUSTACK_DEBUG_ENABLE_TRACEMALLOC=true` ayarlayın. Varsayılan değer `false` olur; bu ayar `--debug` seçeneğinden bağımsızdır ve yeniden başlatma gerektirir. İzleme CPU ve bellek yükünü artırdığından, bellek kullanımını araştırırken etkinleştirin.

## Günlük Düzeyini Yapılandır { #configure-log-level }

Günlük düzeyindeki uç noktalar kimlik doğrulama gerektirir.

**sunucu kapsayıcısı** içinde aşağıdaki komutu çalıştırarak GPUStack sunucusunun günlük düzeyini çalışma zamanında yapılandırabilirsiniz. Yönetici API anahtarıyla kimlik doğrulaması yapın:

```bash
curl -X PUT http://localhost/debug/log_level \
    -H "Authorization: Bearer <YOUR_API_KEY>" \
    -d "debug"
```

Aynı durum GPUStack çalışanları için de geçerlidir. Yerel çalışan jetonuyla kimlik doğrulama yapın (varsayılanı `/var/lib/gpustack/worker_token`'dır):

```bash
curl -X PUT http://localhost:10150/debug/log_level \
    -H "Authorization: Bearer $(cat /var/lib/gpustack/worker_token)" \
    -d "debug"
```

Mevcut günlük seviyeleri şunlardır: `trace`, `debug`, `info`, `warning`, `error`, `critical`.

## Yönetici Şifresini Sıfırla { #reset-admin-password }

Yönetici şifresini unutmanız durumunda, **sunucu kapsayıcısı** içinde aşağıdaki komutu çalıştırarak şifreyi sıfırlayabilirsiniz:

```bash
gpustack reset-admin-password
```

GPUStack'i başlatırken `--port` kullanarak varsayılan bağlantı noktasını değiştirdiyseniz, `--server-url` parametresini kullanarak GPUStack URL'sini belirtin. Sunucuda yerel olarak çalıştırılmalı ve `localhost` üzerinden erişilmelidir:

```bash
gpustack reset-admin-password --server-url http://localhost:9090
```

## Hızlandırıcı Tespit Teşhisine Yardımcı Olun { #assist-in-accelerators-detection-diagnosis }

GPUStack Worker'ı [kurulum kılavuzunda](./installation/requirements.md) açıklandığı gibi başarıyla dağıttıktan sonra,
Çalışan herhangi bir cihazı tespit edemezse,
lütfen ilgili Worker kapsayıcısını girin, aşağıdaki komutu çalıştırın ve sonuçları [GPUStack](https://github.com/gpustack/gpustack/issues)'e bildirin.

```bash
time GPUSTACK_RUNTIME_LOG_LEVEL=debug GPUSTACK_RUNTIME_LOG_EXCEPTION=1 gpustack-runtime detect --format json
```

## Model Dağıtım Teşhisine Yardımcı Olun { #assist-in-model-deployment-diagnosis }

Bir modeli dağıttıktan sonra sorunlarla karşılaşırsanız lütfen ilgili Worker kapsayıcısına girin, aşağıdaki komutu çalıştırın ve sonuçları [GPUStack](https://github.com/gpustack/gpustack/issues)'e bildirin.

```bash
gpustack-runtime inspect <model instance name>
```
