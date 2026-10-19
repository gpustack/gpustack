---
translation_status: translated
---

# Ortam Değişkenleri { #environment-variables }

GPUStack, yapılandırma için çeşitli ortam değişkenlerini destekler.

Çoğu komut satırı parametresi, `GPUSTACK_` öneki ve büyük harf formatındaki ortam değişkenleri aracılığıyla da ayarlanabilir (örneğin, `--data-dir`, `GPUSTACK_DATA_DIR` aracılığıyla ayarlanabilir).

Ortam değişkenleri olarak ayarlanabilen komut satırı parametrelerinin tam listesi için bkz. [CLI Referansı](cli-reference/start.md#available-environment-variables). Bu konu burada tartışılmayacaktır.

### Öncelik Sırası { #priority-order }

Yapılandırma değerleri aşağıdaki öncelik sırasına göre uygulanır (en yüksekten en düşüğe):

1. Komut satırı argümanları
2. Ortam değişkenleri
3. Yapılandırma dosyası
4. Varsayılan değerler

Bu, komut satırı argümanlarının her zaman ortam değişkenlerini geçersiz kılacağı ve ortam değişkenlerinin de yapılandırma dosyasındaki değerleri geçersiz kılacağı anlamına gelir.

## GPUStack Çekirdek Ortam Değişkenleri { #gpustack-core-environment-variables }

Bu ortam değişkenleri genellikle üçüncü taraf hizmet entegrasyonları için kullanılır.

**Applies to** sütunu, ortam değişkeninin nerede ayarlanması gerektiğini gösterir:

- **Sunucu** - GPUStack sunucusu için geçerlidir.
- **İşçi** - GPUStack çalışanları için geçerlidir.
- **Model** - Model dağıtım yapılandırmaları için geçerlidir.

### Proxy Yapılandırması { #proxy-configuration }

| <div style="width:100px">Değişken</div> | Açıklama | Varsayılan | Şunlar için geçerlidir: |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------- | --------------- |
| `HTTP_PROXY`                            | HTTP proxy URL'si. örneğin, `http://proxy-server:port` | (boş) | Sunucu ve Çalışan |
| `HTTPS_PROXY`                           | HTTPS proxy URL'si. örneğin, `https://proxy-server:port` | (boş) | Sunucu ve Çalışan |
| `NO_PROXY`                              | Hariç tutulacak ana bilgisayarların virgülle ayrılmış listesi. örneğin, `127.0.0.1,10.0.0.0/8,192.168.0.0/16,172.16.0.0/16,localhost,cluster.local` | (boş) | Sunucu ve Çalışan |

### Hugging Face Hub { #hugging-face-hub }

| <div style="width:100px">Değişken</div> | Açıklama | Varsayılan | Şunlar için geçerlidir: |
| --------------------------------------- | -------------------------------------------------------- | ------- | --------------- |
| `HF_ENDPOINT`                           | Hugging Face Hub bitiş noktası. örneğin, `https://hf-mirror.com` | (boş) | Sunucu ve Çalışan |
| `HF_TOKEN`                              | Hugging Face Hub erişim jetonu. | (boş) | Sunucu ve Çalışan |

### Veritabanı Yapılandırması { #database-configuration }

| Değişken | Açıklama | Varsayılan | Şunlar için geçerlidir: |
| -------------------------- | -------------------------------------------- | ------- | ---------- |
| `GPUSTACK_DB_ECHO`         | Veritabanı sorgu günlüğünü etkinleştirin. | `false` | Sunucu |
| `GPUSTACK_DB_POOL_SIZE`    | Veritabanı bağlantı havuzu boyutu. | `30`    | Sunucu |
| `GPUSTACK_DB_MAX_OVERFLOW` | Veritabanı bağlantı havuzu maksimum taşması. | `20`    | Sunucu |
| `GPUSTACK_DB_POOL_TIMEOUT` | Saniye cinsinden veritabanı bağlantı havuzu zaman aşımı. | `30`    | Sunucu |
| `GPUSTACK_DB_POOL_RECYCLE` | Saniye cinsinden veritabanı bağlantısı geri dönüşüm süresi. | `1800`  | Sunucu |

### Ağ Yapılandırması { #network-configuration }

| Değişken | Açıklama | Varsayılan | Şunlar için geçerlidir: |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------- | --------------- |
| `GPUSTACK_PROXY_TIMEOUT_SECONDS`               | Saniye cinsinden temsili çıkarım isteğinde mutlak tavan. | `1800`  | Sunucu ve Çalışan |
| `GPUSTACK_PROXY_TTFT_TIMEOUT_SECONDS`          | Bir akış yanıtının ilk parçası için saniyeler içinde beklemede geri durdurma. Sıralamayı ve önceden doldurmayı kapsar, böylece cömert kalır. | `300`   | Sunucu ve Çalışan |
| `GPUSTACK_PROXY_STREAM_IDLE_TIMEOUT_SECONDS`   | Bir akış yanıtı başladıktan sonra parçalar arasındaki saniye cinsinden maksimum boşluk. Gövdesi yalnızca sona ulaşan akışsız bir yanıta hiçbir zaman uygulanmaz. | `60`    | Sunucu ve Çalışan |
| `GPUSTACK_PROXY_UPSTREAM_IDLE_TIMEOUT_SECONDS` | Higress'in upstream bağlantı havuzundaki boşta kalma zaman aşımı, saniye cinsinden. İstek süresiyle ilgili değildir: boşta kalan bağlantıları kapatır, isteğin süresini sınırlamaz. | `3`     | Sunucu |
| `GPUSTACK_TCP_CONNECTOR_LIMIT`                 | HTTP istemcisi TCP bağlayıcı sınırı. | `1000`  | Sunucu ve Çalışan |

### Sunucu Önbellek Yapılandırması { #server-cache-configuration }

| Değişken | Açıklama | Varsayılan | Şunlar için geçerlidir: |
| ----------------------------------- | ---------------------------- | ------- | ---------- |
| `GPUSTACK_SERVER_CACHE_TTL_SECONDS` | Saniyeler içinde sunucu önbelleği TTL. | `600`   | Sunucu |

### Kimlik Doğrulama ve Güvenlik { #authentication-security }

| Değişken | Açıklama | Varsayılan | Şunlar için geçerlidir: |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | --------------- |
| `GPUSTACK_JWT_TOKEN_EXPIRE_MINUTES` | JWT jetonunun dakika cinsinden son kullanma süresi. | `120`   | Sunucu |
| `GPUSTACK_PROVIDER_TEST_EGRESS_ALLOWLIST` | Arayan tarafından sağlanan bir temel URL'yi (`/get-models`, `/test-model`, `/test-decision-model`) çevirerek sağlayıcı uç noktalarının ulaşabileceği CIDR'lerin, IP adreslerinin ve DNS adlarının izin verilenler listesi. Bir hedefe yalnızca ana bilgisayar adının adlardan birine eşit olması veya bu adlardan biriyle bitmesi (nokta sınırı) veya çözümlenen her adresin CIDR'lerden birinin içinde olması durumunda izin verilir; http(s) olmayan temel URL dahil olmak üzere diğer her şey reddedilir. Boş, kısıtlama olmadığı anlamına gelir. örneğin, `api.openai.com,10.0.1.0/24,vllm.internal.corp` | (boş) | Sunucu |

### Ağ Geçidi Yapılandırması { #gateway-configuration }

| Değişken | Açıklama | Varsayılan | Şunlar için geçerlidir: |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | ---------- |
| `GPUSTACK_GATEWAY_PORT_CHECK_INTERVAL`             | Saniye cinsinden aralık GPUStack Sunucunun yerleşik ağ geçidi dinleme bağlantı noktasını kontrol etmesi | `2`       | Sunucu |
| `GPUSTACK_GATEWAY_PORT_CHECK_RETRY_COUNT`          | Yeniden deneme sayısı GPUStack Sunucu, yerleşik ağ geçidi dinleme bağlantı noktasını kontrol ediyor | `300`     | Sunucu |
| `GPUSTACK_GATEWAY_AUTH_RECONCILE_INTERVAL_SECONDS` | Sunucunun, ağ geçidinin yerel olarak doğruladığı API anahtarlarını veritabanından yeniden hesaplama aralığı. ORM'yi atlayan silmeler olay üretmez; bu nedenle genel bir rotada silinmiş bir anahtarın çalışmaya devam edebileceği en uzun süre budur. | `30`      | Sunucu |
| `GPUSTACK_GATEWAY_AUTH_ALLOW_CUSTOM_KEYS`          | Özel bir API anahtarının (kullanıcının sırrını sağladığı anahtar) ağ geçidinde doğrulanıp doğrulanamayacağı. Kapalı, çalışmaya devam ediyor ancak her istekte sunucuya soruyor. Açık olduğunda, anahtar, sırrın tuzsuz hızlı karması tarafından indekslenen ağ geçidinin yapılandırmasında yayınlanır; dağıtımlar arasında aynıdır, bu nedenle zayıf bir sır, önceden hesaplanmış bir tabloya düşer. `custom` hiçbir entropi gereksinimi getirmez, bu nedenle kullanıcıların kendi anahtarlarını seçmeleri durumunda bunu kapatın; her mutabakat sırasında yeniden okunur, böylece yalnızca yeni anahtarları değil, açıkken yayınlanan özel anahtarları da geri çeker. | `true`    | Sunucu |
| `GPUSTACK_GATEWAY_AUTH_MAX_CR_BYTES`               | Sunucunun ağ geçidinin kimlik doğrulama eklentisine yazdığı anahtar tablolar ve genel rota kuralları için bayt bütçesi. Etcd'nin ~1,5 MiB nesne sınırının altında boyutlandırılmıştır; onu geçen anahtarlar, her istekte sunucu aracılığıyla kimlik doğrulaması yapar. | `1100000` | Sunucu |
| `GPUSTACK_GATEWAY_TLS_MIN_PROTOCOL_VERSION`        | Ağ geçidinin HTTPS dinleyicilerinin kabul ettiği en düşük TLS protokolü sürümü. `TLSv1.0`, `TLSv1.1`, `TLSv1.2`, `TLSv1.3`'den biri; başlangıçta başka herhangi bir şey reddedilir. Boş, TLS 1.0 ve 1.1'i kabul eden varsayılan ayarda Higress bırakır. `incluster` modunda yoksayıldı — aşağıya bakın. | (boş) | Sunucu |
| `GPUSTACK_GATEWAY_TLS_MAX_PROTOCOL_VERSION`        | Dinleyicilerin kabul ettiği en yüksek versiyon. Aynı değerler ve aynı kapsam; nadiren ihtiyaç duyulur. | (boş) | Sunucu |

TLS protokolü sürüm yapılandırması, `GPUSTACK_GATEWAY_MIRROR_INGRESS_NAME` ile adlandırılan girişe yazılır ve GPUStack'in bir LLM rotası için oluşturduğu her giriş, sınırları buradan kopyalar. `incluster` modunda, giriş Helm chart'ına aittir, bu nedenle bunun yerine `server.ingress.tls.minProtocolVersion` / `maxProtocolVersion`'i oraya ayarlayın — değişkenler göz ardı edilir ve bir ayarlandığında bir uyarı günlüğe kaydedilir. Bir değişiklik bu girişe hemen ve bir sonraki mutabakat sırasında mevcut LLM rotalarına uygulanır; Bir kerede yayılmak için sunucuyu yeniden başlatın.

Bir ağ geçidi eklentisinin **inside** yapılandırmasıyla sonuçlanan ayarlar, burası yerine `gateway_plugin` altındaki yapılandırma dosyasında ayarlanır, böylece her mekanizmanın bir etkili zaman hikayesi vardır - buradaki bir değer bir sonraki mutabakatta etkili olur, yeniden başlatma sırasında buradaki bir değişken. Aşağıdaki ikisi oraya taşındı ve kullanımdan kaldırıldı. Yapılandırma dosyası varsayılan olarak geçersiz kılabileceğinden, bunlar hâlâ dikkate alınmaktadır; bu nedenle yükseltme, onları ayarlayan dağıtımı sessizce geri döndürmez; her iki ayar da bir uyarı günlüğe kaydeder.

| Kullanımdan kaldırılan değişken | Açıklama | Tarafından değiştirildi | Varsayılan | Şunlar için geçerlidir: |
| ----------------------------------------------------- | -------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------ | ---------- |
| `GPUSTACK_HIGRESS_EXT_AUTH_TIMEOUT_MS`                | Ağ geçidi kimlik doğrulama çağrısı zaman aşımı, milisaniye cinsinden. | `gateway_plugin.gpustack-ext-auth.config.authz.timeout`    | `30000`                              | Sunucu |
| `GPUSTACK_GATEWAY_AI_STATISTICS_PLUGIN_CONTENT_TYPES` | Belirteç kullanımı için ölçülen virgülle ayrılmış içerik türleri. | `gateway_plugin.ai-statistics.config.enable_content_types` | `application/json,text/event-stream` | Sunucu |

### Kullanım Takibi Yapılandırması { #usage-tracking-configuration }

| Değişken | Açıklama | Varsayılan | Şunlar için geçerlidir: |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | ---------- |
| `GPUSTACK_TIMEZONE`                                | Tüm kullanım görünümlerini gruplandırmak ve Son Etkin / olay zamanlarını oluşturmak için platform çapında IANA zaman dilimi (ör. `Asia/Shanghai`) (aşağıdaki nota bakın). Boş ⇒ İşletim sisteminin yerel saat dilimini kullanın (`TZ` / `/etc/localtime`). Kullanımdan kaldırılan `GPUSTACK_USAGE_ROLLUP_TIMEZONE`'in yerine geçer (hala takma ad olarak kabul edilmektedir). | (boş) | Sunucu |
| `GPUSTACK_USAGE_ESTIMATED_BYTES_PER_INPUT_TOKEN`   | Bir ağ geçidi raporu `completed=false` ile geldiğinde ve `input_token` boş olduğunda istem belirteçlerini tahmin etmek için kullanılan belirteç başına bayt böleni. Varsayılanlar, İngilizce eğilimli GPT tarzı belirteçleri hedefler; CJK ağırlıklı iş yükleri için değeri düşürün (ör. `2`). | `4`          | Sunucu |
| `GPUSTACK_USAGE_ESTIMATED_TOKENS_PER_OUTPUT_CHUNK` | Parça başına jeton çarpanı, `output_token` boş olduğunda tamamlanmamış akışlar için tamamlanma jetonlarını tahmin etmek için kullanılır. | `1`          | Sunucu |
| `GPUSTACK_USAGE_DETAILS_RETENTION_MONTHS`          | `model_usage_details` için saklama penceresi (istek başına denetim tablosu). Bundan daha eski satırlar (`COALESCE(completed_at, created_at)`'e sabitlenmiştir) yalnızca lider arşivleyici tarafından `model_usage_details_archive`'e taşınır. | `13`         | Sunucu |
| `GPUSTACK_USAGE_DETAILS_ARCHIVE_CRON`              | Kullanım ayrıntıları arşivleyicisinin yinelenen taraması için Cron ifadesi (UTC). Arşivleyici ayrıca bu programdan bağımsız olarak sunucu başlatıldığında bir kez çalışır. | `0 3 * * *`  | Sunucu |
| `GPUSTACK_USAGE_DETAILS_ARCHIVE_BATCH_SIZE`        | Arşiv hareketleri için parti başına satır sayısı. Daha küçük gruplar, çoğaltma gecikmesine duyarlı ortamlarda işlemlerin kısa kalmasını sağlar; daha büyük gruplar gidiş-dönüş masraflarını azaltır. | `1000`       | Sunucu |
| `GPUSTACK_USAGE_DETAILS_BUFFER_MAX_SIZE`           | Yıkamalar arasında tutulan, istek başına bellek içi ayrıntılar arabelleğindeki sabit sınır. Temizlemeler sürekli olarak başarısız olduğunda (veritabanı çökmesi, şema kayması) bellek büyümesini sınırlar. Taşma durumunda en eski girişler `WARNING` günlüğüyle bırakılır. | `100000`     | Sunucu |
| `GPUSTACK_RESOURCE_USAGE_TICK_SECONDS`             | Örnek kullanımı toplayıcının, son işaretten bu yana herhangi bir yaşam döngüsü olayı olmayan uzun süredir çalışan GPU/CPU örnekleri için birikmiş çalışma süresini temizleyen periyodik işaret aralığı (saniye). | `300`        | Sunucu |
| `GPUSTACK_STORAGE_USAGE_TICK_SECONDS`              | Kalıcı birim (depolama) kullanımı toplayıcının periyodik tıklama aralığı (saniye). | `300`        | Sunucu |
| `GPUSTACK_METERED_USAGE_SEAL_GRACE_SECONDS`        | Bir saatlik zaman dilimi bittikten sonra kesinleştirilmeden önce beklenecek ek süre (saniye). Dilim, `now >= bucket_end + grace` olduğunda kesinleştirilir; böylece geciken olaylar ve saat farkları hesaba katılır. Bu değeri periyodik veri toplama aralığından büyük tutun. Kesinleştirilen dilimler değiştirilemez. | `900`        | Sunucu |
| `GPUSTACK_METERED_USAGE_RETENTION_MONTHS`          | `metered_usage` (saatlik kaynak kullanımı özeti) için saklama penceresi. Bundan daha eski mühürlü satırlar, yalnızca lider arşivleyici tarafından `metered_usage_archive`'ye taşınır. Koleksiyoncunun yerleşim ufkunun (saat) oldukça üzerinde kalmalıdır. | `13`         | Sunucu |
| `GPUSTACK_METERED_USAGE_ARCHIVE_CRON`              | Ölçülü kullanımlı arşivleyicinin yinelenen taraması için Cron ifadesi (UTC). Arşivleyici ayrıca bu programdan bağımsız olarak sunucu başlatıldığında bir kez çalışır. | `0 4 * * *`  | Sunucu |
| `GPUSTACK_METERED_USAGE_ARCHIVE_BATCH_SIZE`        | Ölçülü kullanımlı arşiv hareketleri için parti başına satır sayısı. Daha küçük gruplar işlemleri kısa tutar; daha büyük gruplar gidiş-dönüş masraflarını azaltır. | `5000`       | Sunucu |
| `GPUSTACK_USAGE_EVENTS_RETENTION_MONTHS`           | `resource_events` (kaynak yaşam döngüsü / denetim günlüğü) için saklama penceresi. Bundan daha eski satırlar, yalnızca lider arşivleyici tarafından `resource_events_archive`'ye taşınır. | `13`         | Sunucu |
| `GPUSTACK_USAGE_EVENTS_ARCHIVE_CRON`               | Kaynak olayları arşivleyicisinin yinelenen taraması için Cron ifadesi (UTC). Arşivleyici ayrıca bu programdan bağımsız olarak sunucu başlatıldığında bir kez çalışır. | `30 3 * * *` | Sunucu |
| `GPUSTACK_USAGE_EVENTS_ARCHIVE_BATCH_SIZE`         | Kaynak olayları arşiv hareketleri için toplu satır sayısı. | `5000`       | Sunucu |
| `GPUSTACK_USAGE_BREAKDOWN_MAX_NO_PAGINATION_ROWS`  | Maksimum paket sayfalandırılmamış (`page=-1`) bir döküm geri dönebilir; trend grafikleri ve ihracatlar tüm seriyi aynı anda getirir. Bunu aşan bir istek reddedilir (HTTP 400), sessizce kesilmez. Geniş gösterge panelleri için yükseltin, belleğin üst sınırına kadar indirin. | `50000`      | Sunucu |

!!! note "`GPUSTACK_TIMEZONE` kapsam ve DST"

    Bu tek zaman dilimi her kullanım görünümünü yönetir: günlük `model_usages` belirteç toplaması, GPU/depolama zaman dilimleri ve görüntülenen Son Aktif ve kaynak olayı süreleri. **DST** gözlemleyen bir bölge için bunun bilmeye değer iki sonucu vardır:

    - DST geçişi yakınında, bir etkinliğin görüntülenen saat zamanı, sayıldığı gruptan farklı bir takvim gününe denk gelebilir.
    - Aynı geçmiş sorgunun bir DST anahtarından sonra yeniden çalıştırılması, eski satırların bir saatliğine yeniden gruplandırılmasına neden olabilir, bu nedenle sonuç zaman içinde kayar.

    DST dışı bölgeler (UTC, `Asia/Shanghai`, …) kesindir ve etkilenmez.

### Küme Yapılandırması { #cluster-configuration }

| Değişken | Açıklama | Varsayılan | Şunlar için geçerlidir: |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ------- | ---------- |
| `GPUSTACK_DEFAULT_CLUSTER_KUBERNETES` | Otomatik olarak varsayılan bir küme oluşturulursa, bu değişken ayarlandığında `Kubernetes` sağlayıcısını kullanacaktır. | `false` | Sunucu |

### İçerik Kaynağı Yapılandırması { #content-source-configuration }

| Değişken | Açıklama | Varsayılan | Şunlar için geçerlidir: |
| ------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ---------- |
| `GPUSTACK_BOOTSTRAP_WITH_EMBEDDED_SOURCES` | Her içerik türünü (model kataloğu, topluluk arka uçları, yerleşik arka uç sürümleri, önbellek sağlayıcıları) yayınlanan belgeler yerine gömülü içeriğine getirin, böylece OTA sunucusundan hiçbir şey alınmaz. Bu sadece başlangıç ​​noktasını belirler: Daha sonra hizmete sunulan bir tür orada kalır. OTA sunucusuna ulaşılamadığında veya içeriğin kurulu sürüme sabitlenmesinde kullanışlıdır. | `false` (`true` geliştirme yapısında; boş bir değer ayarlanmamış olarak sayılır) | Sunucu |

### Zamanlayıcı Yapılandırması { #scheduler-configuration }

| Değişken | Açıklama | Varsayılan | Şunlar için geçerlidir: |
| --------------------------------------------------- | --------------------------------------------------------------------------- | ------- | ---------- |
| `GPUSTACK_SCHEDULER_SCALE_UP_PLACEMENT_MAX_SCORE`   | Zamanlayıcı yerleştirme puanlayıcısı tarafından kullanılan maksimum yerleştirme puanı. | `100`   | Sunucu |
| `GPUSTACK_SCHEDULER_SCALE_UP_LOCALITY_MAX_SCORE`    | Model dosyaları zaten mevcut olduğunda yerleştirme puanına maksimum yerellik puanı eklenir. | `5`     | Sunucu |
| `GPUSTACK_SCHEDULER_SCALE_DOWN_STATUS_MAX_SCORE`    | Durum puanlayıcı için maksimum katkının ölçeği azaltıldı (normalleştirilmiş). | `100`   | Sunucu |
| `GPUSTACK_SCHEDULER_SCALE_DOWN_OFFLOAD_MAX_SCORE`   | Yük boşaltma skorlayıcısı için ölçek küçültülmüş maksimum katkı (normalleştirilmiş). | `10`    | Sunucu |
| `GPUSTACK_SCHEDULER_SCALE_DOWN_PLACEMENT_MAX_SCORE` | Yerleştirme puanlayıcısı için ölçek küçültülmüş maksimum katkı (normalleştirilmiş). | `1`     | Sunucu |
| `GPUSTACK_SCALING_SCHEDULER_INTERVAL`               | Zamanlanmış ölçeklendirmenin her modelin kopya sayısını pencerelerinden yeniden hesapladığı saniye cinsinden aralık. Mutabakat seviye tetiklemelidir, dolayısıyla bu yalnızca bir pencere sınırının ne kadar süreyle fark edilmeden kalabileceğini sınırlar, doğruluk asla. Minimum `1` saniyeye sabitlendi. | `30`    | Sunucu |

#### Çok Rollü Grup Planlama { #multi-role-group-scheduling }

Bu ağırlıklar ve pencereler, önceden doldurma-kod çözme ayrıştırılmış grup ve yönlendiricisi gibi rolleri bildiren dağıtımlar için geçerlidir. Düz, tek rollü bir dağıtım üzerinde hiçbir etkileri yoktur. Aşağıdaki her puan, planlayıcının puan zincirine maksimum katkıdır; birini `0` olarak ayarlamak bu değerlendirmeyi tamamen ortadan kaldırır.

| Değişken | Açıklama | Varsayılan | Şunlar için geçerlidir: |
| ---------------------------------------------------- | --------------------------------------------------------------------------- | ------- | ---------- |
| `GPUSTACK_SCHEDULER_TOPOLOGY_PROXIMITY_MAX_SCORE`    | Bir rolün ölçeğini genişletirken veya yönlendiriciyi yerleştirirken, aday çalışanın grubun önceden yerleştirilmiş üyeleriyle paylaştığı topoloji başına maksimum puan. Yedek kapasitenin beyan edilen topolojinin tüm basamağını geçememesi için yerleştirme puanlayıcısının 100 puanlık dağılımının üzerine ayarlayın. `0` sonraki üyeleri yalnızca kaynağa uygun olarak yerleştirir. | `150`   | Sunucu |
| `GPUSTACK_SCHEDULER_PAIRING_AFFINITY_MAX_SCORE`      | Ayrıştırılmış bir grubun bir rolünü büyütürken aday worker üzerinde zaten bulunan her karşıt rol üyesi için en yüksek puan; KV aktarımının aynı host üzerinde kalmasını bu sağlar. Bu, kullanılan ağırlık değil, **alt sınırdır**: zamanlayıcı, bu değer ile zincirdeki diğer puanlayıcıların toplam üst sınırının bir fazlasından büyük olanı alır. Böylece hangi puanlayıcılar etkin olursa olsun «en fazla karşıt rol kardeşi olan kazanır, eşitliği kapasite bozar» kuralı geçerli olur. `0` kapatma anahtarı olarak yorumlanır ve asla alt sınıra yükseltilmez. | `200`   | Sunucu |
| `GPUSTACK_SCHEDULER_SCALE_DOWN_PAIRING_MAX_SCORE`    | Bir grubun bir rolünü küçültürken aynı worker üzerindeki karşıt rol üyelerinin bir üyeyi koruma puanına en yüksek katkısı. Tüm fazla üyelerin sağlıklı olduğu, durum ve boşaltma puanlarının eşit olduğu olağan durumda hangi üyenin silineceğini belirler. Durum puanlayıcısının «başlatılıyor» ile «çalışıyor» arasındaki farkı olan **50 değerinin altında** tutun. 50 veya üzerinde, komşuları olan başlatılmakta bir üye tek başına duran sağlıklı üyeyi geride bırakır ve zamanlayıcı artık önce bozuk üyeyi silmez. `0`, silinecek üyeyi yalnızca durum ve yerleşime göre seçer. | `20`    | Sunucu |
| `GPUSTACK_SCHEDULER_DRAIN_WINDOW_SECONDS`            | Küçültülen bir grup üyesinin yönlendiricinin kayıt defterinden çıktıktan sonra ne kadar süre çalışmaya devam edeceği. Bu, işlem sonlandırma tolerans süresi değildir: hâlâ o üyeden KV önbelleği alan decode işlemlerinin tamamlanması için gereken süredir; motor bunu kendi başına bekleyemez. `0` hemen siler; rol tanımlamayan dağıtımlarda zaten bu davranış uygulanır. | `60`    | Sunucu |
| `GPUSTACK_SCHEDULER_GATHER_BLOCKED_DWELL_SECONDS`    | Bir `MustGather` grubunun bir üyesinin dağıtım raporları `gather_blocked_scale_out` gelmeden ne kadar süre yersiz oturabileceği. Geçtiğinde hiçbir şeyin süresi dolmaz - yalnızca "zamanlayıcı henüz çalışmadı" ile "zamanlayıcı çalıştı ve toplama etki alanı içinde bunu koyacak hiçbir yer yok" ifadesini ayırır. Her ölçeklendirme bunlardan ilkini harcar, dolayısıyla bekleme olmadan işaretçi hepsinde görünür. `0`, yeri doldurulmamış bir üye bulunduğu anda reddi bildirir. | `120`   | Sunucu |

### GPU Örnek Yapılandırması { #gpu-instance-configuration }

| Değişken | Açıklama | Varsayılan | Şunlar için geçerlidir: |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- | ---------- |
| `GPUSTACK_GPU_INSTANCE_TRANSITIONING_REQUEUE_INTERVAL` | Denetleyicinin hâlâ durum değiştiren GPU örneğini bellekteki kuyruğa yeniden ekleyerek tekrar kontrol ettiği aralık, saniye cinsinden. PV/PVT sonlandırma denetleyicileri, hâlâ sonlandırılan örneği yeniden kontrol etmek için aynı aralığı kullanır. Hazır durumdaki kayıtlardaki değişiklikler ise alt katmandaki watch mekanizmasıyla izlenir. En düşük değer `1` saniyedir. | `15`    | Sunucu |
| `GPUSTACK_GPU_INSTANCE_READY_SWEEP_INTERVAL`           | Ready durumuna yerleşmiş GPU örneklerini yeniden kontrol eden isteğe bağlı, düşük sıklıklı taramanın saniye cinsinden aralığı. Watch mekanizmasının yeniden bağlanırken kaçırabileceği worker tarafındaki değişiklikleri yakalayan yedek mekanizmadır. `0` (varsayılan) taramayı kapatır; yalnızca böyle bir izleme boşluğu görülürse düşük sıklıkta etkinleştirin. | `0`     | Sunucu |

### Çalışan ve Model Yapılandırması { #worker-and-model-configuration }

| Değişken | Açıklama | Varsayılan | Şunlar için geçerlidir: |
|------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|---------|----------------|
| `GPUSTACK_WORKER_HEARTBEAT_INTERVAL`                             | Saniye cinsinden işçi kalp atışı aralığı. | `30`    | İşçi |
| `GPUSTACK_WORKER_STATUS_SYNC_INTERVAL`                           | Saniye cinsinden çalışan durumu senkronizasyon aralığı. | `30`    | İşçi |
| `GPUSTACK_WORKER_UNREACHABLE_CHECK_MODE`                         | İşçiye ulaşılamıyor kontrol modu. Seçenekler: `auto`, `enabled`, `disabled`. `auto` işçi sayısı > 50 olduğunda kontrolü devre dışı bırakır. | `auto`  | Sunucu |
| `GPUSTACK_WORKER_HEARTBEAT_GRACE_PERIOD`                         | Worker kalp atışı için saniye cinsinden tolerans süresi. | `150`   | Sunucu |
| `GPUSTACK_MODEL_INSTANCE_RESCHEDULE_GRACE_PERIOD`                | Model örneğinin yeniden zamanlanması için saniye cinsinden tolerans süresi. | `300`   | Sunucu |
| `GPUSTACK_MODEL_EVALUATION_CACHE_MAX_SIZE`                       | Model değerlendirme önbelleğinin maksimum boyutu. | `1000`  | Sunucu |
| `GPUSTACK_MODEL_EVALUATION_CACHE_TTL`                            | Saniyeler içinde model değerlendirme önbelleğinin TTL'si. | `3600`  | Sunucu |
| `GPUSTACK_WORKER_ORPHAN_WORKLOAD_CLEANUP_GRACE_PERIOD`           | Worker üzerinde sahipsiz kalan iş yüklerinin temizlenmesi için saniye cinsinden tolerans süresi. | `300`   | İşçi |
| `GPUSTACK_WORKER_ORPHAN_BENCHMARK_WORKLOAD_CLEANUP_GRACE_PERIOD` | Worker üzerinde sahipsiz kalan benchmark iş yüklerinin temizlenmesi için saniye cinsinden tolerans süresi. | `300`   | İşçi |
| `GPUSTACK_WORKER_STATUS_COLLECTION_LOG_SLOW_SECONDS`             | Saniye cinsinden bu süreyi aşarsa yavaş çalışan durumu toplama için hata ayıklama günlüğü ekleyin. | `180`   | İşçi |
| `GPUSTACK_MODEL_INSTANCE_HEALTH_CHECK_INTERVAL`                  | Saniye cinsinden örnek durum kontrolü aralığını modelleyin. | `3`     | İşçi |
| `GPUSTACK_DISABLE_OS_FILELOCK`                                   | İşletim sistemi dosya kilidini devre dışı bırakın. | `false` | İşçi |
| `GPUSTACK_ENABLE_CUDA_MINOR_VERSION_COMPATIBILITY`               | Daha düşük CUDA alt sürümüne sahip cihazların daha yüksek alt sürüm kullanan imajları çalıştırmasına izin verir. Worker genelinde veya model başına ayarlanır; model ayarı önceliklidir. | `false` | İşçi ve Model |
| `GPUSTACK_RESTART_IN_FLIGHT_LAPSE_SECONDS`                       | İlk yeniden başlatma hala yeniden oluşturulurken `POST /models/{id}/restart` HTTP 409 ile ikinci bir isteği ne kadar süreyle reddetmeye devam eder? Koruma olmadan, ikinci sökme işlemi, ilk oluşturulan değiştirmeleri siler. Koruma kendi kendine sona erer ve böylece `running`'e asla ulaşmayan bir konuşlandırma yine de yeniden başlatılabilir; görüntüyü çeken ve diskteki ağırlıkları okuyan soğuk başlatmadan daha uzun süre dayanacak şekilde boyutlandırın. | `900`   | Sunucu |
| `GPUSTACK_SERVE_LOG_MAX_BYTES`                                   | Bir model sunum günlüğünün tek yeniden başlatma için bayt sınırı. Sınır aşıldığında günlüğün başlangıcı korunur, son bölümü döndürülür ve araya bir işaret yerleştirilir. `0` sınırı kaldırır. | `67108864` | İşçi |
| `GPUSTACK_SERVE_LOG_HEAD_BYTES`                                  | Bu sınırın günlüğün başlangıcına ayrılan kısmı; bir kez yazılır ve bir daha üzerine yazılmaz. Toplam sınırın yarısını aşamaz; `0` yalnızca ilk satırı korur. | `8388608` | İşçi |

### Karşılaştırma Yapılandırması { #benchmark-configuration }

| Değişken | Açıklama | Varsayılan | Şunlar için geçerlidir: |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- | ---------- |
| `GPUSTACK_BENCHMARK_DATASET_SHAREGPT_PATH` | `dataset_name` `ShareGPT` olarak ayarlandığında kıyaslama kapsayıcısı tarafından kullanılan ShareGPT veri kümesi yolu. Varsayılan görüntü zaten bu veri kümesini içeriyor. | `/workspace/benchmark-runner/sharegpt_data/ShareGPT_V3_unfiltered_cleaned_split.json` | İşçi |
| `GPUSTACK_BENCHMARK_REQUEST_TIMEOUT`       | Her kıyaslama isteği için saniye cinsinden zaman aşımı. | `3600`                                                                                | İşçi |

### Model Dağıtım Yapılandırması { #model-deployment-configuration }

!!! note

    Bu ortam değişkenleri GPUStack başlatılırken **ayarlanmaz**. Model dağıtılırken **Gelişmiş Seçenekler > Ortam Değişkenleri** bölümünde yapılandırılmalıdır. Modelin sunum davranışını özelleştirmek için kullanılırlar.

| <div style="width:180px">Değişken</div> | Açıklama | Varsayılan | Şunlar için geçerlidir: |
|-----------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|----------------|
| `GPUSTACK_MODEL_SERVING_COMMAND_SCRIPT_DISABLED`          | Otomatik sunum komutu komut dosyası yürütmesini devre dışı bırakın. `1` veya `true` olarak ayarlandığında, paket kurulumunu ve diğer kurulum görevlerini yürüten komut dosyası çalışmayacaktır. | `0`     | Modeli |
| `PYPI_PACKAGES_INSTALL`                                   | Model hizmet ortamına kurulacak ek PyPI paketleri. Birden fazla paket boşluklarla ayrılmalıdır. Komut dosyası, varsa `uv pip install`'i, aksi takdirde `pip install`'yi kullanacaktır. | (boş) | Modeli |
| `GPUSTACK_MODEL_RAM_CLAIM`                                | Kapasite planlaması için zamanlayıcı tarafından kullanılan, model için kullanıcı tarafından beyan edilen RAM gereksinimi (Bayt cinsinden). | (boş) | Modeli |
| `GPUSTACK_MODEL_VRAM_CLAIM`                               | Kapasite planlama için zamanlayıcı tarafından kullanılan, model için kullanıcı tarafından beyan edilen VRAM gereksinimi (Bayt cinsinden). | (boş) | Modeli |
| `GPUSTACK_APPLY_QWEN3_RERANKER_TEMPLATES`                 | Qwen3 yeniden sıralama şablonlarını istek gövdesine uygulayın. https://huggingface.co/Qwen/Qwen3-Reranker-0.6B.'deki talimatlara bakın | (boş) | Modeli |
| `GPUSTACK_SKIP_MODEL_EVALUATION`                          | Dağıtım sırasında model değerlendirme veya doğrulama adımını atlar. | (boş) | Modeli |
| `GPUSTACK_SKIP_GPU_COUNT_CHECK`                           | Manuel olarak seçilen GPU sayısının paralellik dünya boyutuna (tp×pp×dp) eşit olması gerektiği doğrulamasını atlar. Yalnızca GPU'lar manuel olarak seçildiğinde etkili olur; otomatik planlama hiçbir zaman etkilenmez. Paralellik parametrelerinin kasıtlı olarak seçilen GPU sayısından farklı olduğu manuel tek düğümlü dağıtımlar için kullanın. | (boş) | Modeli |
| `GPUSTACK_DISABLE_METRICS`                                | Model için metrik açığa çıkarmayı ve toplamayı devre dışı bırakır. | (boş) | Modeli |
| `GPUSTACK_MODEL_HEALTH_CHECK_PATH`                        | Modelin sunduğu HTTP durum denetimi yolunu belirtir. | (boş) | Modeli |
| `GPUSTACK_MODEL_RUNTIME_UID`                              | Container içerisinde çalışan işlemlerin kullanıcı izinlerini kontrol edin. | (boş) | Modeli |
| `GPUSTACK_MODEL_RUNTIME_GID`                              | Container içerisinde çalışan işlemlerin grup izinlerini kontrol edin. | (boş) | Modeli |
| `GPUSTACK_MODEL_RUNTIME_SHM_SIZE_GIB`                     | GiB'deki kapsayıcı için paylaşılan bellek boyutu. | `10.0`  | Modeli |
| `GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_ENABLED`           | Bu model için çıkarım durumu denetimini etkinleştirin. Etkinleştirildiğinde, GPUStack modelin doğru yanıt verdiğini doğrulamak için periyodik olarak minimum çıkarım istekleri gönderir. | `false` | Modeli |
| `GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_INTERVAL`          | Saniye cinsinden sağlık kontrolü aralığını çıkarım (minimum `60`). Bu aralıkta yakın zamandaki başarılı çıkarım trafiği gözlemlenirse etkin denetim atlanır. | `300`   | Modeli |
| `GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_TIMEOUT`           | Her periyodik çıkarım durumu kontrolü isteği için saniye cinsinden zaman aşımı. SGLang PD motor rolleri için varsayılan değer 25, aksi takdirde 15'tir; açık bir değer önceliklidir. | `15/25` | Modeli |
| `GPUSTACK_MODEL_INFERENCE_HEALTH_CHECK_FAILURE_THRESHOLD` | Örneği sağlıksız olarak işaretlemeden önce ardışık çıkarım durumu kontrolü başarısızlıklarının sayısı. | `3`     | Modeli |
| `GPUSTACK_ENABLE_CUDA_MINOR_VERSION_COMPATIBILITY`        | Daha düşük CUDA alt sürümüne sahip cihazların daha yüksek alt sürüm kullanan imajları çalıştırmasına izin verir. Worker genelinde veya model başına ayarlanır; model ayarı önceliklidir. | `false` | İşçi ve Model |

#### Kullanım Örneği { #usage-example }

Bir modeli dağıtırken **Gelişmiş Seçenekler > Ortam Değişkenleri** seçeneğine gidin ve şunu ekleyin:

```bash
# Install additional packages before model serving starts
PYPI_PACKAGES_INSTALL=torch-audio==2.0.0 transformers==4.30.0

# Disable the serving command script entirely
GPUSTACK_MODEL_SERVING_COMMAND_SCRIPT_DISABLED=1
```

Sunum komut dosyası otomatik olarak şunları yönetir:

- `PYPI_PACKAGES_INSTALL`'de belirtilen ek PyPI paketlerini yükleme
- Paket kurulumu için hem `uv pip` hem de `pip` desteği
- Özel PyPI endekslerini `PIP_INDEX_URL` ve `PIP_EXTRA_INDEX_URL` aracılığıyla yönetme

## GPUStack Çalışma Zamanı Ortamı Değişkenleri { #gpustack-runtime-environment-variables }

Bu ortam değişkenleri GPUStack çalışma zamanı tarafından kullanılır. Genellikle Docker/Kubernetes'te çalışan çıkarım arka uçlarının davranışını ayarlamak için kullanılır.

Sadece çalışanlar arasında kullanılabilir. Düzgün bir şekilde etkili olmalarını sağlamak için lütfen ortam değişkenlerini çalışanların kapsayıcılarına ayarlayın.

### Küresel Değişkenler { #global-variables }

| Değişken | Açıklama | Varsayılan |
| -------------------------------- | -------------------------- | ------- |
| `GPUSTACK_RUNTIME_LOG_LEVEL`     | Günlük seviyesi. | `INFO`  |
| `GPUSTACK_RUNTIME_LOG_WARNING`   | Günlüğe kaydetme uyarılarını etkinleştirin. | `0`     |
| `GPUSTACK_RUNTIME_LOG_EXCEPTION` | Günlüğe kaydetme istisnalarını etkinleştirin. | `0`     |

### Dedektör Değişkenleri { #detector-variables }

| Değişken | Açıklama | Varsayılan |
| -------------------------------------------------- |--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------| --------------------------------------- |
| `GPUSTACK_RUNTIME_DETECT`                          | Kullanılacak dedektör. Seçenekler: Otomatik, AMD, ASCEND, CAMBRICON, HYGON, ILUVATAR, METAX, MTHREADS, NVIDIA. | `Auto`                                  |
| `GPUSTACK_RUNTIME_DETECT_NO_PCI_CHECK`             | Algılama sırasında PCI kontrolünün yapılmamasını etkinleştirin. WSL ortamları için kullanışlıdır. | (boş) |
| `GPUSTACK_RUNTIME_DETECT_PCI_CLASS_PREFIXES`       | Algılama sırasında kontrol edilecek PCI sınıfı önekleri. `GPUSTACK_RUNTIME_DETECT_NO_PCI_CHECK` yanlış olduğunda dedektör, cihazın PCI sınıfının verilen öneklerden herhangi biriyle başlayıp başlamadığını kontrol edecektir. | 02,03,0b,12                             |
| `GPUSTACK_RUNTIME_DETECT_BACKEND_MAP_RESOURCE_KEY` | Kaynak anahtarlarıyla algılanan arka uç eşlemesi, örneğin `{"cuda": "nvidia.com/devices", "rocm": "amd.com/devices"}`. Gpustack-runner'ın arka uç adını karşılık gelen kaynak anahtarıyla eşlemek için kullanılır. | Her satıcı tarafından adlandırılan varsayılan değerler |
| `GPUSTACK_RUNTIME_DETECT_PHYSICAL_INDEX_PRIORITY`  | Cihaz tespitinde fiziksel indeks önceliğini kullanın. | `1`                                     |

#### ROCm Dedektörüne Özel Değişkenler { #rocm-detector-specific-variables }

!!! note

    Hygon gibi ROCm tabanlı arka uçlara da uygulanabilir.

| Değişken | Açıklama | Varsayılan |
| -------------------- | ----------------------- | ----------- |
| `ROCM_SMI_LIB_PATH`  | ROCm SMI kütüphane yolu. | (boş) |
| `ROCM_HOME`          | ROCm ana dizini. | (boş) |
| `ROCM_PATH`          | ROCm yolu. | `/opt/rocm` |
| `ROCM_CORE_LIB_PATH` | ROCm çekirdek kütüphane yolu. | (boş) |

### Dağıtıcı Değişkenleri { #deployer-variables }

| Değişken | Açıklama | Varsayılan |
| ------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `GPUSTACK_RUNTIME_DEPLOY`                                          | Kullanılacak dağıtıcı. Seçenekler: Otomatik, Docker, Kubernetes, Podman(Deneysel). | `Auto`                                          |
| `GPUSTACK_RUNTIME_DEPLOY_API_CALL_ERROR_DETAIL`                    | Dağıtım sırasında API çağrı hatasının ayrıntılandırılmasını etkinleştirin. | `1`                                             |
| `GPUSTACK_RUNTIME_DEPLOY_PRINT_CONVERSION`                         | Dağıtım sırasında dönüşümün yazdırılmasını etkinleştirin. GPUStack Çalışma Zamanı, belirli Container Runtime API çağrılarına (ör. Docker SDK, Kubernetes API, Podman SDK) dönüştürülecek birleşik bir İş Yükü tanımı API'si sağlar. Bu seçeneğin etkinleştirilmesi, hata ayıklama amacıyla INFO günlüğündeki son dönüştürülen API çağrılarını yazdıracaktır. | `0`                                             |
| `GPUSTACK_RUNTIME_DEPLOY_ASYNC`                                    | Eşzamansız dağıtımı etkinleştirin. | `1`                                             |
| `GPUSTACK_RUNTIME_DEPLOY_ASYNC_THREADS`                            | İş parçacığı havuzundaki iş parçacıklarının sayısı. | (boş) |
| `GPUSTACK_RUNTIME_DEPLOY_MIRRORED_NAME`                            | Dağıtıcının adı. | (boş) |
| `GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_ENVIRONMENTS`  | Yansıtılmış dağıtım sırasında göz ardı edilecek ortam değişkeni adları. | (boş) |
| `GPUSTACK_RUNTIME_DEPLOY_MIRRORED_DEPLOYMENT_IGNORE_VOLUMES`       | Yansıtılmış dağıtım sırasında yok sayılacak birim bağlama hedefleri. | (boş) |
| `GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY_USERNAME`      | Varsayılan kapsayıcı kayıt defteri için kullanıcı adı. GPUStack koşucu görüntüleri, kimlik doğrulamalı erişim gerektiren bir kapsayıcı kayıt defterinde barındırılıyorsa, bu değişkeni, her GPUStack çalışanın (veya hepsi bir arada GPUStack sunucusunun) başlangıcındaki erişim kullanıcı adına ayarlayın. Uygulamada, erişim kullanıcı adının bunun yerine gerçek kapsayıcı çalışma zamanında yapılandırılması önerilir. | (boş) |
| `GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY_PASSWORD`      | Varsayılan kapsayıcı kaydının şifresi. GPUStack koşucu görüntüleri, kimlik doğrulamalı erişim gerektiren bir kapsayıcı kayıt defterinde barındırılıyorsa, bu değişkeni, her GPUStack çalışanın (veya hepsi bir arada GPUStack sunucusunun) başlangıcında erişim parolasına ayarlayın. Uygulamada, erişim parolasının bunun yerine gerçek kapsayıcı çalışma zamanında yapılandırılması önerilir. | (boş) |
| `GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_NAMESPACE`              | Varsayılan çalıştırıcı konteyner görüntüleri için ad alanı. GPUStack koşucu görselleri `gpustack` ad alanının dışında barındırılıyorsa, ör. `gpustack-ai`, bu değişkeni `gpustack-ai` olarak ayarlayın. Çok düzeyli bir ad alanı için, ör. `awesome.com/amazing-group/amazing-subgroup/gpustack-ai`, `--system-default-container-registry` (veya çalışan `system_default_container_registry` yapılandırma anahtarı veya `GPUSTACK_SYSTEM_DEFAULT_CONTAINER_REGISTRY` ortam değişkeni) aracılığıyla üst ad alanıyla kayıt defterini bildirin, ör. `--system-default-container-registry=awesome.com/amazing-group/amazing-subgroup`. | `gpustack` |
| `GPUSTACK_RUNTIME_DEPLOY_IMAGE_PULL_POLICY`                        | Dağıtıcı için görüntü çekme politikası (örneğin, Her Zaman, IfNotPresent, Never). | `IfNotPresent`                                  |
| `GPUSTACK_RUNTIME_DEPLOY_CDI_SPECS_DIRECTORY`                      | Dağıtım sırasında, Konteyner Cihaz Arayüzü (CDI) spesifikasyonlarını içeren dizinin yolu veya CDI spesifikasyonlarının oluşturulacağı dizin. | `/var/run/cdi`                                  |
| `GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_CDI`                     | Konteyner Çalışma Zamanına konteynere hangi cihazların enjekte edileceğini söylemek için kullanılan konteyner cihazı arayüzlerinin manuel eşlemesi, örneğin `nvidia.com/devices=nvidia.com/gpu;amd.com/devices=amd.com/gpu`. Anahtar, kaynak anahtarıdır ve değer, Konteyner Cihaz Arayüzü (CDI) anahtarıdır. | Her satıcı tarafından adlandırılan varsayılan değerler |
| `GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_RUNTIME_VISIBLE_DEVICES` | Konteyner Çalışma Zamanına hangi cihazların konteynere enjekte edileceğini söylemek için kullanılan, çalışma zamanı görünür cihazları ortam değişkenlerinin manuel eşlemesi, örneğin, `nvidia.com/devices=NVIDIA_VISIBLE_DEVICES;amd.com/devices=AMD_VISIBLE_DEVICES`. Anahtar, kaynak anahtarıdır ve değer, ortam değişkeninin adıdır. | Her satıcı tarafından adlandırılan varsayılan değerler |
| `GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_BACKEND_VISIBLE_DEVICES` | Cihaz Çalışma Zamanına (örn., ROCm, CUDA, OneAPI) kapsayıcı içinde hangi cihazların kullanılacağını söylemek için kullanılan arka uç görünür cihaz ortam değişkenlerinin manuel eşlemesi, örn., `nvidia.com/devices=CUDA_VISIBLE_DEVICES;amd.com/devices=ROCR_VISIBLE_DEVICES`. Anahtar, kaynak anahtarıdır ve değer, ortam değişkeni adlarının bir listesidir. | Her satıcı tarafından adlandırılan varsayılan değerler |
| `GPUSTACK_RUNTIME_DEPLOY_RUNTIME_VISIBLE_DEVICES_VALUE_UUID`       | Verilen çalışma zamanı görünür aygıt ortam değişkenleri için UUID'leri kullanın. | (boş) |
| `GPUSTACK_RUNTIME_DEPLOY_BACKEND_VISIBLE_DEVICES_VALUE_ALIGNMENT`  | Verilen arka uç görünür aygıt ortam değişkenleri için değer hizalamasını etkinleştirin. | `ASCEND_RT_VISIBLE_DEVICES,NPU_VISIBLE_DEVICES` |
| `GPUSTACK_RUNTIME_DEPLOY_CPU_AFFINITY`                             | Dağıtılan iş yükleri için CPU benzeşimini etkinleştirin. | `0`                                             |
| `GPUSTACK_RUNTIME_DEPLOY_NUMA_AFFINITY`                            | Dağıtılan iş yükleri için NUMA benzeşimini etkinleştirin. Etkinleştirildiğinde `GPUSTACK_RUNTIME_DEPLOY_CPU_AFFINITY` de ima edilir. | `0`                                             |

#### Docker Deployer'a Özel Değişkenler { #docker-deployer-specific-variables }

| Değişken | Açıklama | Varsayılan |
| --------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `GPUSTACK_RUNTIME_DOCKER_HOST`                      | Docker bağlantısı için ana bilgisayar. Varsayılan Docker ana bilgisayarını geçersiz kılmak için kullanılır. | `http+unix:///var/run/docker.sock` |
| `GPUSTACK_RUNTIME_DOCKER_PAUSE_IMAGE`               | Docker'deki duraklatma kapsayıcısı için kullanılan kapsayıcı resmi. | `gpustack/runtime:pause`           |
| `GPUSTACK_RUNTIME_DOCKER_UNHEALTHY_RESTART_IMAGE`   | Docker'de kapsayıcının sağlıksız yeniden başlatılması için kullanılan kapsayıcı görüntüsü. | `gpustack/runtime:health`          |
| `GPUSTACK_RUNTIME_DOCKER_RESOURCE_INJECTION_POLICY` | Docker konuşlandırıcı için kaynak ekleme politikası (ör. Env, CDI). <br/>`Env`: Kaynakları `GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_RUNTIME_VISIBLE_DEVICES`'e dayalı standart ortam değişkenini kullanarak enjekte eder. <br/>`CDI`: `GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_CDI`'e dayalı olarak CDI kullanarak kaynakları enjekte eder. | `Env`                              |
| `GPUSTACK_RUNTIME_DOCKER_CDI_SPECS_GENERATE`        | CDI kaynak ekleme politikasını kullanırken dağıtım sırasında CDI spesifikasyonları oluşturun, `GPUSTACK_RUNTIME_DEPLOY_CDI_SPECS_DIRECTORY`'in mevcut olmasını gerektirir. Yalnızca `GPUSTACK_RUNTIME_DOCKER_RESOURCE_INJECTION_POLICY` `CDI` olarak ayarlandığında çalışır. Dağıtıcı için CDI spesifikasyonlarını oluşturmak amacıyla dahili bilgiyi kullanarak, çıktı dosyası CDI spesifikasyonlarını oluşturan diğer araçlarla (örneğin, NVIDIA Container Toolkit) çakışıyorsa, lütfen bunu devre dışı bırakın ve çıktı dosyasını manuel olarak kaldırın. | `1`                                |

#### Kubernetes Dağıtıcısına Özel Değişkenler { #kubernetes-deployer-specific-variables }

| Değişken | Açıklama | Varsayılan |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- |
| `GPUSTACK_RUNTIME_KUBERNETES_NODE_NAME`                      | İş yüklerinin dağıtılacağı Kubernetes Düğümünün adı. | (boş) |
| `GPUSTACK_RUNTIME_KUBERNETES_NAMESPACE`                      | İş yüklerinin dağıtılacağı Kubernetes'in ad alanı. | `default`       |
| `GPUSTACK_RUNTIME_KUBERNETES_DOMAIN_SUFFIX`                  | Kubernetes hizmetleri için alan adı son eki. | `cluster.local` |
| `GPUSTACK_RUNTIME_KUBERNETES_SERVICE_TYPE`                   | Kubernetes hizmetler için hizmet türü. Seçenekler: ClusterIP, NodePort, LoadBalancer. | `ClusterIP`     |
| `GPUSTACK_RUNTIME_KUBERNETES_QUORUM_READ`                    | Kubernetes hizmetleri için yetersayı okumanın kullanılıp kullanılmayacağı. | `0`             |
| `GPUSTACK_RUNTIME_KUBERNETES_DELETE_PROPAGATION_POLICY`      | Kubernetes kaynakları için silme yayılım politikası. Seçenekler: Foreground, Background, Orphan. | `Foreground`    |
| `GPUSTACK_RUNTIME_KUBERNETES_RESOURCE_INJECTION_POLICY`      | Kubernetes konuşlandırıcı için kaynak ekleme politikası. Seçenekler: Otomatik, Zarf, KDP. <br/>`Auto`: Ortama göre kaynak enjeksiyon politikasını otomatik olarak seçin. <br/>`Env`: Standart ortam değişkenini kullanarak kaynakları enjekte eder, `GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_RUNTIME_VISIBLE_DEVICES`'ye dayalı olarak temel Container Toolkit'e bağlıdır. <br/>`KDP`: Kubernetes Cihaz Eklentisini kullanarak kaynakları enjekte eder. | `Auto`          |
| `GPUSTACK_RUNTIME_KUBERNETES_KDP_PER_DEVICE_MAX_ALLOCATIONS` | Kubernetes Cihaz Eklentisinde bir cihaz için maksimum tahsisler. | `10`            |
| `GPUSTACK_RUNTIME_KUBERNETES_KDP_DEVICE_ALLOCATION_POLICY`   | Kubernetes Cihaz Eklentisi için cihaz tahsis politikası. Seçenekler: Otomatik, CDI, Zarf, Opak. <br/>`Auto`: Ortama göre cihaz tahsis politikasını otomatik olarak seçin. <br/>`Env`: `GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_RUNTIME_VISIBLE_DEVICES`'ye dayalı olarak çalışma zamanında görülebilen ortam değişkenlerini kullanarak cihazları tahsis eder; Container Toolkit desteği gerektirir. <br/>`CDI`: `GPUSTACK_RUNTIME_DEPLOY_RESOURCE_KEY_MAP_CDI`'e dayalı olarak oluşturulan CDI spesifikasyonlarını kullanarak cihazları tahsis eder, hata ayıklamayı ve sorun gidermeyi kolaylaştırır; `GPUSTACK_RUNTIME_DEPLOY_CDI_SPECS_DIRECTORY`'in var olmasını gerektirir. <br/>`Opaque`: Dağıtım için uygun ancak sorun gidermesi zor olan tahsis için dahili mantığı kullanır. | `Auto`          |
| `GPUSTACK_RUNTIME_KUBERNETES_KDP_CDI_SPECS_GENERATE`         | Dağıtım sırasında CDI spesifikasyonları oluşturun, `GPUSTACK_RUNTIME_DEPLOY_CDI_SPECS_DIRECTORY`'in mevcut olmasını gerektirir. Yalnızca `GPUSTACK_RUNTIME_KUBERNETES_KDP_DEVICE_ALLOCATION_POLICY` `CDI` olarak ayarlandığında çalışır. Dağıtıcı için CDI spesifikasyonlarını oluşturmak amacıyla dahili bilgiyi kullanarak, çıktı dosyası CDI spesifikasyonlarını oluşturan diğer araçlarla (örneğin, NVIDIA Container Toolkit) çakışıyorsa, lütfen bunu devre dışı bırakın ve çıktı dosyasını manuel olarak kaldırın. | `1`             |

#### Podman Dağıtıcısına Özel Değişkenler { #podman-deployer-specific-variables }

!!! note

    Podman dağıtıcısı deneyseldir ve 4.9 veya daha yüksek bir sürüme ihtiyaç duyar.

| Değişken | Açıklama | Varsayılan |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------- |
| `GPUSTACK_RUNTIME_PODMAN_HOST`                    | Podman bağlantısı için ana bilgisayar. Varsayılan Podman ana bilgisayarını geçersiz kılmak için kullanılır. | `http+unix:///run/podman/podman.sock` |
| `GPUSTACK_RUNTIME_PODMAN_PAUSE_IMAGE`             | Podman'da duraklatma kapsayıcısı için kullanılan kapsayıcı resmi. Varsayılan `GPUSTACK_RUNTIME_DOCKER_PAUSE_IMAGE` ile aynıdır. | `gpustack/runtime:pause`              |
| `GPUSTACK_RUNTIME_PODMAN_UNHEALTHY_RESTART_IMAGE` | Podman'da kapsayıcının sağlıksız yeniden başlatılması için kullanılan kapsayıcı görüntüsü. Varsayılan `GPUSTACK_RUNTIME_DOCKER_UNHEALTHY_RESTART_IMAGE` ile aynıdır. | `gpustack/runtime:health`             |
| `GPUSTACK_RUNTIME_PODMAN_CDI_SPECS_GENERATE`      | Dağıtım sırasında CDI spesifikasyonları oluşturun, `GPUSTACK_RUNTIME_DEPLOY_CDI_SPECS_DIRECTORY`'in mevcut olmasını gerektirir. Dağıtıcı için CDI spesifikasyonlarını oluşturmak amacıyla dahili bilgiyi kullanarak, çıktı dosyası CDI spesifikasyonlarını oluşturan diğer araçlarla (örneğin, NVIDIA Container Toolkit) çakışıyorsa, lütfen bunu devre dışı bırakın ve çıktı dosyasını manuel olarak kaldırın. Varsayılan `GPUSTACK_RUNTIME_DOCKER_CDI_SPECS_GENERATE` ile aynıdır. | `1`                                   |
