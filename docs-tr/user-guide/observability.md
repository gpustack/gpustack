---
translation_status: translated
---

# Gözlemlenebilirlik { #observability }

Bu belge, Prometheus ve Grafana kullanılarak GPUStack Sunucusu/Çalışan/LLM sunan çalışma zamanı ölçümlerinin nasıl izleneceğini açıklamaktadır.

## Genel Bakış { #overview }

GPUStack, model sunumu ve GPU kaynak yönetimi için kapsamlı bir ölçüm seti sağlar. Kullanıcılar, Prometheus ve Grafana'yı entegre ederek bu ölçümleri gerçek zamanlı olarak toplayabilir, saklayabilir ve görselleştirebilir, böylece etkili izleme ve sorun giderme işlemleri yapılabilir.

## Dahili Gözlemlenebilirlik (Varsayılan) { #built-in-observability-default }

Varsayılan olarak GPUStack gömülü Prometheus ve Grafana ile başlar. Bunlara şu yollarla erişebilirsiniz:

- **Prometheus**: `http://your_gpustack_server_host_ip/prometheus`
- **Grafana**: `http://your_gpustack_server_host_ip/grafana`

Yerleşik Grafana, anonim Görüntüleyici erişimi için yapılandırılmıştır ve oturum açma formu devre dışı bırakılmıştır. Yönetici kimlik bilgileri varsayılan olarak `admin` / `grafana` olarak kalır.

## Harici Gözlemlenebilirlik (Opsiyonel) { #external-observability-optional }

Harici bir Prometheus/Grafana yığını istiyorsanız sağlanan Docker Compose dosyalarını kullanmanızı öneririz:

En son kararlı sürümü kopyalamak için aşağıdaki komutları çalıştırın:

```bash
LATEST_TAG=$(
    curl -s "https://api.github.com/repos/gpustack/gpustack/releases" \
    | grep '"tag_name"' \
    | sed -E 's/.*"tag_name": "([^"]+)".*/\1/' \
    | grep -Ev 'rc|beta|alpha|preview' \
    | head -1
)
echo "Latest stable release: $LATEST_TAG"
git clone -b "$LATEST_TAG" https://github.com/gpustack/gpustack.git
cd gpustack/docker-compose
```

Başlamadan önce, `GPUSTACK_GRAFANA_URL`'i tarayıcı tarafından erişilebilen bir Grafana URL'sine ayarlayın (`grafana` gibi yalnızca kapsayıcı ana bilgisayar adı değil).

Harici Prometheus/Grafana'yı başlatın (bu, yerleşik yığını devre dışı bırakır):

```bash
sudo docker compose -f docker-compose.external-observability.yaml up -d
```

Zaten harici bir Prometheus/Grafana yığınınız varsa, bunu manuel olarak yapılandırabilirsiniz:

1. **Prometheus'u GPUStack ölçümlerini çıkaracak şekilde yapılandırın**
   GPUStack ölçüm uç noktası (varsayılan `:10161`) ve çalışan keşif uç noktası için hedefler ekleyin. Örnek:

   ```yaml
   scrape_configs:
     - job_name: gpustack-worker-discovery
       scrape_interval: 5s
       http_sd_configs:
         - url: "http://<gpustack_server_host>:10161/metrics/targets"
           refresh_interval: 1m
     - job_name: gpustack-server
       scrape_interval: 10s
       static_configs:
         - targets: ["<gpustack_server_host>:10161"]
   ```
2. **GPUStack kontrol panellerini Grafana'ya aktarın**
   Başlangıç noktası olarak `docker-compose/grafana/grafana_dashboards/` dizininde sağlanan kontrol panellerini kullanın.
3. **GPUStack'i Grafana'nıza yönlendirin**
   Kontrol paneli yönlendirmelerinin çalışması için `GPUSTACK_GRAFANA_URL`'i harici olarak erişilebilen Grafana URL'sine ayarlayın. Bu, tarayıcı tarafından erişilebilen bir URL olmalıdır.

## Metriklere Erişim { #accessing-metrics }

- **GPUStack Metrikleri Uç Noktası**:
  `http://<gpustack_server_host>:10161/metrics`'deki ölçümlere erişim
- **GPUStack Çalışan Metrik Hedefleri**:
  `http://<gpustack_server_host>:10161/metrics/targets`'deki ölçümlere erişim
- **Prometheus Kullanıcı Arayüzü**:
  Prometheus'a varsayılan olarak `http://<host>:19090` adresinden veya `--builtin-prometheus-port` / `GPUSTACK_BUILTIN_PROMETHEUS_PORT` ile yapılandırılan bağlantı noktasından erişin.
- **Grafana kullanıcı arayüzü**:
  Grafana'ya varsayılan olarak `http://<host>:13000` adresinden veya `--builtin-grafana-port` / `GPUSTACK_BUILTIN_GRAFANA_PORT` ile yapılandırılan bağlantı noktasından erişin. Yerleşik Grafana, oturum açma formu devre dışı bırakılarak anonim Görüntüleyici erişimi için yapılandırılmıştır. Yönetici kimlik bilgileri varsayılan olarak `admin` / `grafana` olarak kalır.

## Eski Oluşturma Kurulumlarından Geçiş { #migration-from-older-compose-setups }

Daha önce Prometheus/Grafana'yı GPUStack ile birlikte çalıştırmak için Docker Compose'u kullandıysanız:

- **Harici gözlemlenebilirliği koruyun (süreklilik için önerilir)**:
  Mevcut Prometheus/Grafana konteynerlerinizi çalışır durumda bırakın. Prometheus kazıma hedeflerini yeni GPUStack ölçüm uç noktasına güncelleyin ve `GPUSTACK_GRAFANA_URL`'ı mevcut Grafana'nıza ayarlayın.

- **Yerleşik gözlemlenebilirliğe geçin**:
  Eski Prometheus/Grafana kapsayıcılarını durdurun, ardından en yeni `docker-compose.server.yaml`'yi (yalnızca GPUStack) kullanın. Yerleşik Grafana/Prometheus görevi devralacak. Eski yığını salt okunur olarak tutmadığınız sürece, eski Prometheus'un geçmiş ölçümleri taşınmayacaktır.

## Metrik Eşlemeyi Özelleştirme { #customizing-metrics-mapping }

GPUStack, yapılandırma API'si aracılığıyla metrik eşlemesinin dinamik olarak özelleştirilmesini destekler. Bu, hizmeti yeniden başlatmadan çalışma zamanı motor ölçümlerinin GPUStack ölçümleriyle nasıl eşlendiğini güncellemenize olanak tanır. Yapılandırma sunucuda merkezi olarak yönetilir ve HTTP API aracılığıyla erişilebilir veya değiştirilebilir.

### API Uç Noktaları { #api-endpoints }

- **Mevcut Metrik Yapılandırmasını Al**

  - AL `http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/config`
  - Geçerli metrik eşleme yapılandırmasını JSON formatında döndürür.

- **Metrik Yapılandırmasını Güncelle**

  - POST `http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/config`
  - Metrik eşleme yapılandırmasını güncellemek için bir JSON yükünü kabul eder. Değişiklikler tüm çalışanlar için anında geçerli olur.

- **Varsayılan Metrik Yapılandırmasını Al**
  - AL `http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/default-config`
  - Referans veya sıfırlama için yararlı olan varsayılan metrik eşleme yapılandırmasını JSON formatında döndürür.

### Örnek Kullanım { #example-usage }

**Mevcut yapılandırmayı al:**

```bash
curl http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/config
```

**Yapılandırmayı güncelle:**

```bash
curl -X POST http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/config \
     -H "Content-Type: application/json" \
     -d @custom_metrics_config.json
```

_(burada `custom_metrics_config.json` yeni yapılandırma dosyanızdır)_

**Varsayılan yapılandırmayı al:**

```bash
curl http://<gpustack_server_host>:<gpustack_server_port>/v2/metrics/default-config
```

> **Not**: Yapılandırma geçerli JSON formatında sağlanmalıdır.

## GPUStack Tarafından Ortaya Çıkan Metrikler { #metrics-exposed-by-gpustack }

Aşağıdaki metrikler GPUStack tarafından ortaya çıkarılmıştır ve Prometheus tarafından kazınabilir. Her metrik, küme, çalışan, model ve örnek tanımlamaya yönelik hiyerarşik etiketleri içerir.

### LLM Hizmet Çalışma Zamanı Metrikleri { #llm-serving-runtime-metrics }

| Metrik Adı | Tür | Açıklama |
| -------------------------------------- | --------- | --------------------------------------------------------------------------- |
| gpustack:num_requests_running          | Gauge     | Şu anda işlenmekte olan isteklerin sayısı.                               |
| gpustack:num_requests_waiting          | Gauge     | Kuyrukta bekleyen isteklerin sayısı.                                    |
| gpustack:num_requests_swapped          | Gauge     | CPU'ya aktarılan isteklerin sayısı.                                      |
| gpustack:prefix_cache_hit_rate         | Gauge     | Önek önbellek isabet oranı.                                                      |
| gpustack:kv_cache_usage_ratio          | Gauge     | KV-önbellek kullanım oranı. 1.0 tamamen kullanılmış anlamına gelir.                                 |
| gpustack:prefix_cache_queries          | Counter   | Önek önbellek sorgularının sayısı (toplam token'lar).                              |
| gpustack:prefix_cache_hits             | Counter   | Önek önbellek isabetlerinin sayısı (toplam token'lar).                                 |
| gpustack:prompt_tokens                 | Counter   | İşlenen önceden doldurma token'larının toplam sayısı.                                   |
| gpustack:generation_tokens             | Counter   | Oluşturulan token'ların toplam sayısı.                                           |
| gpustack:request_prompt_tokens         | Histogram | İstek başına işlenen önceden doldurma token'larinin sayısı.                             |
| gpustack:request_generation_tokens     | Histogram | İstek başına işlenen oluşturma token'larinin sayısı.                          |
| gpustack:time_to_first_token_seconds   | Histogram | İlk token'ı oluşturma zamanı.                                               |
| gpustack:inter_token_latency_seconds   | Histogram | Önceki token üretildikten sonra sonraki token'ı oluşturma zamanı. |
| gpustack:time_per_output_token_seconds | Histogram | Oluşturulan token başına süre.                                                   |
| gpustack:e2e_request_latency_seconds   | Histogram | Uçtan uca istek gecikmesi.                                                 |
| gpustack:request_success               | Counter   | Başarılı isteklerin toplam sayısı.                                        |
| gpustack:request_preemptions           | Counter   | Motordan gelen toplam önleme sayısı.                                |

Bu ölçümler, metrics_config.yaml'da tanımlandığı gibi çeşitli çalışma zamanı motorlarından (vLLM, SGLang, MindIE) eşlenir.

### Çalışan Metrikleri { #worker-metrics }

| Metrik Adı | Tür | Açıklama |
| ------------------------------------------------ | ----- | ------------------------------------------------ |
| gpustack:worker_status                           | Gauge | Çalışan durumu (durum etiketiyle).                |
| gpustack:worker_node_os                          | Info  | Çalışan düğümün işletim sistemi bilgisi. |
| gpustack:worker_node_kernel                      | Info  | Çalışan düğümün çekirdek bilgisi.           |
| gpustack:worker_node_uptime_seconds              | Gauge | Çalışan düğümün saniye cinsinden çalışma süresi.            |
| gpustack:worker_node_cpu_cores                   | Gauge | Çalışan düğümün toplam CPU çekirdeği.              |
| gpustack:worker_node_cpu_utilization_rate        | Gauge | Çalışan düğümün CPU kullanım oranı.         |
| gpustack:worker_node_memory_total_bytes          | Gauge | Çalışan düğümün bayt cinsinden toplam belleği.        |
| gpustack:worker_node_memory_used_bytes           | Gauge | Çalışan düğümün bayt cinsinden kullanılan bellek.         |
| gpustack:worker_node_memory_utilization_rate     | Gauge | Çalışan düğümün bellek kullanım oranı.      |
| gpustack:worker_node_gpu                         | Info  | Çalışan düğümün GPU bilgisi.              |
| gpustack:worker_node_gpu_cores                   | Gauge | Çalışan düğümün toplam GPU çekirdekleri.              |
| gpustack:worker_node_gpu_utilization_rate        | Gauge | Çalışan düğümün GPU kullanım oranı.         |
| gpustack:worker_node_gpu_temperature_celsius     | Gauge | Santigrat cinsinden GPU sıcaklığı.                      |
| gpustack:worker_node_gpu_power_limit_watts       | Gauge | Watt cinsinden GPU güç yönetimi sınırı.             |
| gpustack:worker_node_gpu_power_used_watts        | Gauge | Watt cinsinden GPU güç kullanımı.                        |
| gpustack:worker_node_gram_total_bytes            | Gauge | Bayt cinsinden toplam GPU RAM.                          |
| gpustack:worker_node_gram_allocated_bytes        | Gauge | Bayt cinsinden tahsis edilen GPU RAM.                      |
| gpustack:worker_node_gram_used_bytes             | Gauge | Bayt cinsinden kullanılan GPU RAM.                           |
| gpustack:worker_node_gram_utilization_rate       | Gauge | GPU RAM kullanım oranı.                        |
| gpustack:worker_node_filesystem_total_bytes      | Gauge | Bayt cinsinden toplam dosya sistemi boyutu.                  |
| gpustack:worker_node_filesystem_used_bytes       | Gauge | Bayt cinsinden kullanılan dosya sistemi boyutu.                   |
| gpustack:worker_node_filesystem_utilization_rate | Gauge | Dosya sistemi kullanım oranı.                     |

### Sunucu Metrikleri { #server-metrics }

| Metrik Adı | Tür | Açıklama |
| ------------------------------------------- | ----- | ------------------------------------------------------------- |
| gpustack:cluster                            | Info  | Küme bilgisi (kimlik, ad, sağlayıcı).                     |
| gpustack:cluster_status                     | Gauge | Küme durumu (durum etiketiyle).                            |
| gpustack:model                              | Info  | Model bilgisi (kimlik, ad, çalışma zamanı, kaynak, kategori).      |
| gpustack:model_desired_instances            | Gauge | İstenilen model örneği sayısı.                            |
| gpustack:model_running_instances            | Gauge | Çalışan model örneklerinin sayısı.                            |
| gpustack:model_instance_status              | Gauge | Her model örneğinin durumu (durum etiketiyle).             |
| gpustack:model_instance_restart_count       | Gauge | Model örneği yeniden başlatma sayısı.                                 |
| gpustack:model_instance_latest_restart_time | Gauge | Unix zaman damgası saniyesi olarak model örneği en son yeniden başlatma süresi. |

### Olay Veri Yolu Metrikleri { #event-bus-metrics }

| Metrik Adı | Tür | Açıklama |
| ----------------------------------- | ------- | ---------------------------------------------------------------------------------------------------------------- |
| gpustack:bus_subscribers            | Gauge   | `topic` + `source` başına aktif veri yolu aboneleri (birkaç kişi `source=streaming`'i paylaşabilir, açık izleme akışı başına bir tane). |
| gpustack:bus_queue_depth            | Gauge   | `topic`+`source` paylaşan aboneler arasında maksimum kuyruk derinliği.                                                     |
| gpustack:bus_queue_capacity         | Gauge   | Abone başına kuyruk maksimum boyutu (aşağıdaki env düğmesine bakın).                                                               |
| gpustack:bus_queue_full             | Gauge   | 1, eğer `topic`+`source` paylaşan herhangi bir abonenin kuyruğu doluysa.                                                   |
| gpustack:bus_queue_saturation_ratio | Gauge   | Maksimum `qsize / maxsize` `[0, 1]`'de. Sürekli > 0.8 ⇒ yavaş tüketici.                                              |
| gpustack:bus_subscriber_latest_keys | Gauge   | Birleştirilmiş GÜNCELLENMİŞ teslimatı bekleyen maksimum kimlikler (boyut `latest_by_key`).                                            |
| gpustack:bus_events_total           | Counter | `topic`+`source` paylaşımı yapan aboneler arasında toplanan kümülatif olay sayıları. Ekstra etiketler: `kind`, `event_type`.  |

> **Not**: Ayrıntılı izleme ve filtreleme için tüm ölçümler ilgili tanımlayıcılarla (küme, çalışan, model, örnek, kullanıcı) etiketlenmiştir.
