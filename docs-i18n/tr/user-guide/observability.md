---
translation_status: translated
---

# Gözlemlenebilirlik { #observability }

GPUStack metriklerini Prometheus ve Grafana ile izleyin ve tek tek istekleri incelemek için çıkarım günlüklerini ve izlerini kullanın.

## Genel Bakış { #overview }

GPUStack, model sunumu ve GPU kaynak yönetimi için kapsamlı bir ölçüm seti sağlar. Kullanıcılar, Prometheus ve Grafana'yı entegre ederek bu ölçümleri gerçek zamanlı olarak toplayabilir, saklayabilir ve görselleştirebilir, böylece etkili izleme ve sorun giderme işlemleri yapılabilir.

## Yerel model izleme { #native-model-monitoring }

**Model Service → Deployments** altında dağıtım adını açın. Overview, Monitoring, Instances, GPU Resources ve Configuration sekmeleri bulunur. İzleme grupları ayrı ayrı açılabilir; zaman, örnek ve PD rolü filtrelerini paylaşır.

- Grafikler çıkarım motoru metriklerini kullanır. Eksik motor örnekleri ağ geçidi değerleriyle doldurulmaz. PD gecikmesi kaynak rolüyle gösterilir ve yerel alt isteği ölçer; otomatik olarak tüm istemci isteğini temsil etmez.
- Desteklenen gecikme istatistikleri Avg, P50, P95 ve P99'dur. İstatistik satırı seçilen aralığın tamamını, grafik noktaları sayfada belirtilen daha kısa hız aralığını kapsar. Eksik gözlemler sıfır yapılmaz; desteklenmeyen, uygulanamayan, başarısız ve eski veriler ayrılır.
- GPU ve sunucu geçmişi sahip bilgisi içeren kararlı kimlikleri kullanır. Yeni geçmiş serileri için Worker'ları güncelleyin. Eski Worker'ların son bildirdiği değerler gösterilebilir; doğrulanamayan geçmiş GPU indekslerinden tahmin edilmez. Fiziksel değerler paylaşılan cihazdaki veya sunucudaki diğer iş yüklerini de içerir.
- Motor geçmişi mevcut örnekleri ve gözlemlenen kaynak nesillerini; donanım geçmişi şu anda ilişkili cihazları ve kaydedilen sahipliği kapsar. Tam bir geçmiş yerleşim denetimi değildir.

Model izleme, model panosuna yönlendirme ve kıyaslama izleme için kuruluşun Owner rolü veya platform yöneticisi erişimi gerekir. GPUStack ayrıca istenen dağıtımın veya kıyaslamanın ve ilişkili kaynakların görünürlüğünü denetler.

Worker ve GPU izleme ve panolarına yönlendirme, kaynak görünürlüğüne göre yetkilendirilir. Kuruluş üyeleri bu uçları görüntüleyebildikleri kaynaklar için kullanabilir. Grafana ve Prometheus proxy uçları kendi erişim izinlerini uygular.

API uçları `GET /v2/models/{id}/metrics` ve `GET /v2/models/{id}/resource-metrics`'tir. Yalnızca kontrollü filtreleri kabul ederler; istemci PromQL'i kabul edilmez. Grafikler, tablolar ve CSV aynı yetkilendirilmiş sonuçları kullanır.

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

   `tunnel` proxy modundaki worker'lara yalnızca sunucunun ileri vekil sunucusu (varsayılan port `30079`) üzerinden erişilebilir ve bu vekil, aktardığı her isteği doğrular. Bu worker'ların ölçümlerini kazanmak için, vekil atlama kimlik bilgisi olarak bir GPUStack API anahtarı (arayüzde oluşturun ve değiştirmeden kopyalayın — üretilen anahtar `gpustack_<access_key>_<secret_key>` biçimindedir) sunan bir iş ekleyin:

   ```yaml
     - job_name: gpustack-proxy-worker-discovery
       scrape_interval: 5s
       proxy_url: "http://<gpustack_server_host>:30079"
       authorization:
         credentials: "<gpustack_api_key>"
       http_sd_configs:
         - url: "http://<gpustack_server_host>:10161/metrics/proxy-targets"
           refresh_interval: 1m
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

## Çıkarım Günlükleri ve İzleri { #inference-logs-and-traces }

Çıkarım günlüklerini ve izlerini bulmak için `X-GPUStack-Request-Id` yanıt başlığını kullanın. Günlük veya izleme yapılandırıldığında paketle gelen ağ geçidi bu başlığı ekler. Docker sunucuları varsayılan olarak izleri gömülü Tempo'ya aktarır. Helm sunucu sürümleri varsayılan olarak ayrı bir Tempo iş yükü dağıtır. Günlük aktarımı ayrı yapılandırılır ve bir günlük uç noktası gerektirir.

| İz hedefi veya ayarı | Docker | Helm |
| --- | --- | --- |
| Yerleşik Tempo | Sunucuda varsayılan | Sunucu sürümünde varsayılan |
| Harici OTLP/gRPC alıcısı | `GPUSTACK_TRACE_OTLP_ENDPOINT` ayarlayın | `tracing.endpoint` ayarlayın |
| Yerleşik Tempo bloklarının saklama süresi (varsayılan `168h`) | `GPUSTACK_BUILTIN_TEMPO_RETENTION` ayarlayın | `tracing.tempo.retention` ayarlayın |
| Devre dışı | `GPUSTACK_DISABLE_TRACING=true` ayarlayın | `tracing.enabled=false` ayarlayın |

Tempo, saklama süresi dolan blokları düzenli temizlik sırasında siler. Saklama süresi, blokların yaşını belirler. Toplam disk kullanımı, alınan izlerin hacmine ve önden yazma günlüğü (WAL), blok birleştirme ve silinmeyi bekleyen bloklar için gereken alana bağlıdır.

Saklanan blokların kaplayacağı alanı hesaplamak için saniyede alınan iz sayısını, iz başına diskteki ortalama blok baytı ve saniye cinsinden saklama süresiyle çarpın. Örnek hesapta saniyede 10 istek, %100 örnekleme, örneklenen istek başına bir iz ve iz başına diskte 1.000 blok baytı varsayın. Bu, günde yaklaşık 864 MB ve yedi günde yaklaşık 6,05 GB blok alanı demektir. Örnek olarak %50 ek alan ayrıldığında toplam yaklaşık 9,1 GB olur. Gerekli birim kapasitesini ve ek alanı belirlemek için temsili bir iş yükünde iz başına blok baytını ve WAL, blok birleştirme ve silinmeyi bekleyen blokların en yüksek alan kullanımını ölçün.

Dağıtım ayrıntılarındaki `Request Tracing` sekmesi başlangıçta son 24 saati arar. İzleri görüntülemek için kuruluş sahibi rolü veya platform yöneticisi yetkisi gerekir. GPUStack, dağıtım metrikleriyle aynı model görünürlüğü kurallarını uygular. Sonuçlar arasında gezinin veya İstek Kimliğini ya da İz Kimliğini GPUStack veya Tempo'da göründüğü biçimde yapıştırın. Günlük aktarımı yapılandırılmışsa istek ayrıntılarındaki İstek Kimliğini kopyalayıp günlük arka ucunuzda arayın. Harici Tempo için sorgu uç noktasını `GPUSTACK_TRACE_QUERY_URL` veya `tracing.queryURL` ile ayarlayın.

Günlük aktarımı etkin olduğunda Sunucu ve Worker'lar kapsayıcı konsoluna tek bir JSON akışı yazar. Alloy bu akışı desteklenen motor günlüğü biçimleriyle birlikte okur. Atılan kayıtları kontrol etmek için Alloy'un `otelcol_processor_filter_logs_filtered_total` metriğine `engine_format` veya `higress_format` bileşeniyle bakın. Motor filtresi, tanınan bir günlük öneki olmayan bağımsız düz metin satırlarını eler.

İlgili aktarım etkinse ve motor ayarları izin veriyorsa GPUStack, vLLM `0.32.0` veya SGLang `0.5.22` kullanan uyumlu Runner imajlarına motor gözlemlenebilirlik ayarlarını otomatik ekler. Sürüm kontrolleri otomatik yapılandırma için geçerlidir. Diğer imajlar veya sürümler için telemetriyi motorun kendi ayarlarıyla yapılandırın.

Her izleme bileşeni span'leri doğrudan seçilen alıcıya gönderir. Günlük aktaran bileşenler günlük uç noktasına ulaşabilmelidir. İz aktaran Sunucu, Worker'lar, ağ geçitleri ve çıkarım kapsayıcıları iz uç noktasına ulaşabilmelidir. Alım bağlantısı şifresiz gRPC kullanır, bu nedenle güvenilen bir ağdaki adresi kullanın. Harici bir alıcıya geçmek veya izlemeyi devre dışı bırakmak yerleşik Tempo'yu durdurur.

### Docker'ı Yapılandırma { #configure-docker }

1. Günlük aktarımını etkinleştirmek için sunucuda ve günlük aktaran her ayrı çalışanda `GPUSTACK_LOG_OTLP_ENDPOINT=http://<collector-host>:4318/v1/logs` değerini ayarlayın. `/var/run/docker.sock:/var/run/docker.sock` soketini bağlayın ve `docker logs` komutunu destekleyen bir günlük sürücüsü kullanın.
2. İzleri harici bir alıcıya aktarmak için sunucuda ve her ayrı çalışanda `GPUSTACK_TRACE_OTLP_ENDPOINT=<collector-host>:4317` değerini ayarlayın. Uzak çalışanlarla yerleşik Tempo'yu kullanmak için [Mevcut Docker Çalışanlarında İzlemeyi Etkinleştirme](../upgrade/cluster.md#enable-tracing-on-existing-docker-workers) bölümüne bakın.
3. `<collector-host>` yerine bu konteynerlerin erişebildiği bir ad veya adres yazın. Değişkenleri `docker run -e` ile ya da Compose dosyanızdaki `environment` altında iletin, ardından konteynerleri yeniden oluşturun.

Docker, Tempo ve Alloy'u GPUStack kapsayıcısında çalıştırır. Tempo, WAL ve blokları `<data_dir>/tempo` altında saklar ve sorguları yalnızca loopback üzerinden kabul eder. İz dosyaları sunucu veri biriminin yaşam döngüsünü izler. Blok saklama süresini değiştirmek için Sunucu'da `GPUSTACK_BUILTIN_TEMPO_RETENTION` değerini ayarlayıp kapsayıcıyı yeniden oluşturun.

### Helm'i Yapılandırma { #configure-helm }

1. Her GPUStack sürümünün Helm values dosyasında alıcı uç noktalarını yapılandırın. Günlük aktarımını etkinleştirmek için `logging.endpoint`, harici iz alıcısı kullanmak için `tracing.endpoint` değerini ayarlayın. Yalnızca çalışan içeren sürümlerde `tracing.endpoint` değerini ayarlayarak sunucunun alıcısını kullanabilirsiniz.

   ```yaml
   logging:
     endpoint: "http://<collector-host>:4318/v1/logs"
   tracing:
     endpoint: "<collector-host>:4317"
   ```

2. `<collector-host>` yerine alıcı adresini yazın ve Helm sürümünü yükseltin. Collector başka bir ad alanındaysa ad alanını içeren bir Service DNS adı kullanın.

Yerleşik Higress ağ geçidi varsayılan olarak JSON biçiminde erişim günlüğü yazar. Biçimi değiştirmek için Helm values dosyasında `higress-core.meshConfig.accessLogEncoding` ve `higress-core.meshConfig.accessLogFormat` değerlerini ayarlayın.

Tempo ve Alloy imajları, varsayılan değeri `docker.io` olan `global.hub` ayarını kullanır. Özel bir kayıt defterine yansıtırken `grafana/tempo` ve `grafana/alloy` depo yollarını koruyun.

Chart, Tempo'yu Sunucu Pod'larından ayrı çalıştırır ve varsayılan olarak `/var/tempo` yoluna bir `emptyDir` bağlar. Tempo Pod'u değiştirildiğinde bu birimde saklanan izler kaybolur. Pod değişimlerinde izleri korumak için `tracing.tempo.existingClaim` değerini sürümün ad alanındaki mevcut bir PVC'ye ayarlayın. Birim, `10001` UID'si veya grubu tarafından yazılabilir olmalıdır. Blok saklama süresini değiştirmek için `tracing.tempo.retention` değerini ayarlayın ve sürümü yükseltin. Saklama süresi boş veya geçersizse `168h` kullanılır.

Tek Sunuculu sürümler dahil tüm Sunucu Pod'ları, karşılıklı TLS ile `https://<release>-tempo.<namespace>.svc.<cluster-domain>:3200` adresindeki yerleşik Tempo'yu sorgular. Chart varsayılan olarak sorgu için bir TLS Secret oluşturur. Worker'lar ve çıkarım motorları izleri Service'in OTLP/gRPC `4317` portuna gönderir.

GitOps dağıtımlarında, Sunucu sürümünün ad alanında bir Secret oluşturun ve adını `tracing.tempo.queryTLSSecretName` olarak ayarlayın. `ca.crt`, `server.crt`, `server.key`, `client.crt` ve `client.key` dosyalarını ekleyin. Sunucu sertifikası `<release>-tempo.<namespace>.svc.<cluster-domain>` adını kapsamalıdır. Sunucu ve istemci sertifikalarını yenilerken aynı CA'yı kullanın. Tempo ve Sunucu Pod'larını yenilenen sertifikalarla yeniden başlatmak için yeni bir Secret oluşturun ve Helm değerlerindeki adını güncelleyin.

İz aktarım uç noktasını değiştirdikten sonra çalışan model örneklerini yeniden başlatın veya yeniden dağıtın.

### Aktarılan Günlükleri Maskeleme { #redact-exported-logs }

Alloy günlükleri aktarmadan önce kayıt gövdesindeki, istisna yığın izindeki ve `path` alanındaki eşleşmeleri `***` ile değiştirebilir. Boş olmayan RE2 düzenli ifadelerinden oluşan bir liste kullanın. Boş liste maskelemeyi devre dışı bırakır. İstek, iz ve span kimlikleri kayıtları ilişkilendirmek için kullanılmaya devam edilebilir.

Docker için günlük aktaran her konteynere `GPUSTACK_LOG_REDACT_PATTERNS` değişkenini bir JSON dize dizisi olarak iletin. Bu örnek e-posta adreslerini ve Çin ana karasındaki cep telefonu numaralarını maskeler.

```bash
GPUSTACK_LOG_REDACT_PATTERNS='["(?i)[A-Z0-9._%+-]+@[A-Z0-9.-]+[.][A-Z]{2,}","\\b1[3-9][0-9]{9}\\b"]'
```

Helm için günlük aktaran her sürümde eşdeğer YAML listesini ayarlayın.

```yaml
logging:
  redactPatterns:
    - '(?i)[A-Z0-9._%+-]+@[A-Z0-9.-]+[.][A-Z]{2,}'
    - '\b1[3-9][0-9]{9}\b'
```

Kuralları uygulamak için Docker konteynerlerini yeniden oluşturun veya Helm sürümünü yükseltin. Günlük sisteminizde aktarılan kayıtları `request_id` ile kontrol edin. Kurallar GPUStack tarafından yönetilen Alloy'un yeni aktardığı kayıtlara uygulanır. Özgün kayıtlara `docker logs`, `kubectl logs` veya çalışma zamanının günlük dosyalarından erişebilirsiniz. Daha önce aktarılmış kayıtlar ve izler bu kuralların kapsamı dışındadır. Geçersiz kurallar Alloy'un günlük aktarımını durdurabilir. Bu durumda GPUStack isteklere hizmet vermeyi sürdürür.

### Bir İsteği Bulma { #look-up-a-request }

1. Çalışan bir sohbet modeline istek gönderin ve yanıt başlıklarını saklayın. `<gpustack-host>`, `<api-key>` ve `<model-name>` yerine dağıtımınızın değerlerini yazın. Çıkarım izinleri için [API Anahtarı Yönetimi](api-key-management.md) bölümüne bakın.

   ```bash
   curl --silent --show-error --no-buffer \
     --dump-header response-headers.txt \
     'http://<gpustack-host>/v1/chat/completions' \
     --header 'Authorization: Bearer <api-key>' \
     --header 'Content-Type: application/json' \
     --data '{"model":"<model-name>","messages":[{"role":"user","content":"Reply with one word."}],"max_tokens":8,"stream":true}'

   grep -i '^X-GPUStack-Request-Id:' response-headers.txt
   ```

2. Günlük sisteminizde zaman aralığını istek zamanını kapsayacak biçimde ayarlayın ve `request_id` günlük özniteliğiyle filtreleyin. `higress`, `gpustack.server` veya `gpustack.engine` gibi bir kaynak seçmek için `service.name` OTLP kaynak özniteliğini kullanın. Varsa `worker_id`, `model_instance_id` ve `role` alanlarıyla sonuçları daraltabilirsiniz.
3. Bir günlük kaydı OTLP iz kimliği içeriyorsa iz sisteminizde o kimlikle izi bulun. İzleme etkinse günlük aktarımı kapalı olsa bile platform istek kimliğini `gpustack.request_id` span özniteliğiyle arayabilirsiniz.

### Loki ve Grafana Örneği { #example-with-loki-and-grafana }

1. [OTLP alımı ve yapılandırılmış meta verileri etkin](https://grafana.com/docs/loki/latest/send-data/otel/) bir Loki dağıtımında `GPUSTACK_LOG_OTLP_ENDPOINT` veya `logging.endpoint` değerini `http://<loki-host>:3100/otlp/v1/logs` gibi tam alım URL'si olarak ayarlayın. İzler için `GPUSTACK_TRACE_OTLP_ENDPOINT` veya `tracing.endpoint` değerini span gönderen bileşenlerin erişebileceği Tempo OTLP/gRPC adresi olarak ayarlayın. Grafana veri kaynaklarını Loki ve Tempo'nun HTTP sorgu URL'leriyle yapılandırın.
2. Grafana'da `Explore` bölümünü açıp Loki'yi seçin, isteğin zaman aralığını ayarlayın ve `<request-id>` yerine `X-GPUStack-Request-Id` yanıt başlığının değerini yazın. Loki'nin varsayılan OTLP eşlemesi, `service.name` değerini `service_name` akış etiketine dönüştürür ve `request_id` değerini yapılandırılmış meta veri olarak kaydeder.

   ```logql
   {service_name=~"higress|gpustack[.](server|worker|engine)"} | request_id="<request-id>"
   ```

   Sonuçları daraltmak için bu alanları taşıyan kayıtlarda `| worker_id="<worker-id>"` veya `| model_instance_id="<instance-id>"` ifadesini ekleyin.
3. Bir sonuç `trace_id` içeriyorsa bu değeri Tempo veri kaynağının `TraceQL` sorgu alanına kopyalayın. İzleri doğrudan günlük sonuçlarından açmak için Loki'de `^trace_id$` ile eşleşen `Label` türünde bir [türetilmiş alan](https://grafana.com/docs/grafana/latest/datasources/loki/configure/#derived-fields) yapılandırın. Tempo veri kaynağına iç bağlantı ekleyin ve sorgu olarak `${__value.raw}` kullanın.

Veri saklama ve erişim ayarlarını harici platformda yönetin. Kiracı kullanıcıları verileri sorgulayabiliyorsa kiracı yalıtımını o platformda uygulayın.

!!! tip "PD motor günlüklerini bulma"

    SGLang Mooncake için her iki motorun günlüklerini `request_id` ile bulun ve gönderim denemelerini `pd_attempt_id` ile ayırt edin. vLLM NIXL için giriş kaydını `request_id` ile bulun, ardından buradaki `pd_coordinator_id` değerini iki motorun günlüklerinde arayın. Giriş kaydında bu alan yoksa isteğin zamanı ve model örneğiyle Prefill günlüklerini daraltın ve doğruladığınız kayıttaki `pd_coordinator_id` değerini kopyalayın.

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
