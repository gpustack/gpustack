---
hide:
  - toc
translation_status: translated
---

# gpustack start { #gpustack-start }

GPUStack sunucusunu veya worker'ı başlatır.

```bash
gpustack start [OPTIONS]
```

!!! note "CLI argümanlarının konumu"

    Docker'da `docker run` tarafından yürütülen asıl komut, **ENTRYPOINT** ve **COMMAND** bileşenlerinden oluşur.

    `gpustack/gpustack` imajında bir `ENTRYPOINT` tanımlıdır. Bunu basitçe aşağıdaki komut olarak
    anlayabilir veya onunla eşdeğer kabul edebilirsiniz:

        gpustack start

    Bu nedenle, `gpustack start` için CLI argümanları imaj adından sonra,
    `docker run` komutunun sonunda belirtilmelidir; bunlar `docker run` komutunun kendi seçenekleri değildir.

    Örnek:

        docker run [docker options] gpustack/gpustack <start-args>

    Kubernetes'te bu argümanlar konteyner tanımındaki `args` alanında
    belirtilmelidir.

## Yapılandırma { #configurations }

### Ortak seçenekler { #common-options }

| <div style="width:180px">Bayrak</div>         | <div style="width:100px">Varsayılan</div> | Açıklama |
|---------------------------------------------|----------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| `--advertise-address` değer                 | (boş)                                | Dış erişim için yayımlanacak IP adresi.<br/>Ayarlanmazsa sistem uygun bir yerel IP adresini otomatik olarak algılar. |
| `--config-file` değer                       | (boş)                                | YAML yapılandırma dosyasının yolu. |
| `-d` değer, `--debug` değer                 | `False`                                | Hata ayıklama modunu etkinleştirir. Kısa -d bayrağı PowerShell tarafından CommonParameters için ayrıldığından Windows'ta desteklenmez. |
| `--data-dir` değer                          | (boş)                                | Verilerin saklanacağı dizin. Varsayılan işletim sistemine bağlıdır. |
| `--cache-dir` değer                         | (boş)                                | Önbelleğin (örneğin model dosyalarının) saklanacağı dizin. Varsayılan <data-dir>/cache dizinidir. |
| `--huggingface-token` değer                 | (boş)                                | Hugging Face Hub'da kimlik doğrulamak için kullanıcı erişim token'ı. `HF_TOKEN` ortam değişkeniyle de ayarlanabilir. |
| `--insecure-tls`                            | `False`                                | GPUStack'in kendi HTTPS bağlantılarında (worker'dan sunucuya, sunucudan harici kimlik doğrulama IdP'sine) TLS doğrulamasını atlar. Sunucu sertifikasının başka şekilde doğrulanamadığı güvenilir ağlar içindir. |
| `--bin-dir` değer                           | (boş)                                | Sürümlendirilmiş arka uç çalıştırılabilir dosyaları gibi ek ikili dosyaların saklanacağı dizin. |
| `--pipx-path` değer                         | (boş)                                | Sürümlendirilmiş arka uçları yüklemek için kullanılan pipx çalıştırılabilir dosyasının yolu. |
| `--system-default-container-registry` değer | `docker.io`                            | GPUStack'in sistem ve çıkarım imajlarını çekeceği varsayılan konteyner kayıt deposu.<br/>GPUStack'e ait tüm iş imajlarını `gpustack` ad alanında tutun; çok seviyeli bir ad alanı gerekiyorsa `gpustack` son seviye olarak kalmalıdır, örneğin `awesome.com/amazing-group/amazing-subgroup/gpustack`. Ardından bu bayrağı (veya sunucu/worker yapılandırma anahtarı `system_default_container_registry` ya da `GPUSTACK_SYSTEM_DEFAULT_CONTAINER_REGISTRY` ortam değişkenini) üst yola, örneğin `awesome.com/amazing-group/amazing-subgroup` değerine ayarlayın.<br/>İmajlar `gpustack` ad alanında kalamıyorsa (örneğin `gpustack-ai` kullanılıyorsa), `--image-repo` ve `--benchmark-image-repo` değerlerini buna göre değiştirin. |
| `--image-repo` değer                        | `gpustack/gpustack`                    | GPUStack konteyneri için varsayılan imaj deposunu geçersiz kılar. Değere etiket eklemeyin; GPUStack sürüm etiketini otomatik olarak ekler.<br/>`gpustack/gpustack:<tag>` imajı `gpustack` ad alanı dışındaysa, örneğin `awesome.com/amazing-group/amazing-subgroup/gpustack-ai/gpustack:<tag>`, önce üst ad alanını içeren kayıt deposunu `--system-default-container-registry=awesome.com/amazing-group/amazing-subgroup` ile belirtin. Ardından bu bayrağı (veya sunucu/worker yapılandırma anahtarı `image_repo` ya da `GPUSTACK_IMAGE_REPO` ortam değişkenini) `gpustack-ai/gpustack` olarak ayarlayın. |
| `--gateway-mode` değer                      | `auto`                                 | Ağ geçidinin çalışma modu. Seçenekler: embedded, in-cluster, external, disabled veya auto (varsayılan). |
| `--gateway-kubeconfig` değer                | (boş)                                | Ağ geçidi için kubeconfig dosyasının yolu. Yalnızca external gateway-mode için kullanılır. |
| `--gateway-namespace` değer                 | `higress-system`                       | Ağ geçidi bileşeninin dağıtıldığı ad alanı. |
| `--service-discovery-name` değer            | (boş)                                | DNS'teki servis keşfi servisinin adı. Yalnızca servis keşfi kullanılan Kubernetes dağıtımlarında geçerlidir. |
| `--namespace` değer                         | (boş)                                | GPUStack'in ağ geçidi yönlendirme kurallarını ve model örneklerini dağıtacağı Kubernetes ad alanı. |

### Sunucu seçenekleri { #server-options }

| <div style="width:180px">Bayrak</div>              | <div style="width:100px">Varsayılan</div>           | Açıklama |
|--------------------------------------------------|--------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--port` değer                                   | `80`                                             | Sunucunun bağlanacağı port. |
| `--tls-port` değer                               | `443`                                            | TLS sunucusunun bağlanacağı port. |
| `--api-port` değer                               | `30080`                                          | GPUStack API sunucusunun bağlanacağı port. |
| `--proxy-port` değer                             | `30079`                                          | GPUStack proxy sunucusunun bağlanacağı port. |
| `--database-port` değer                          | `5432`                                           | Yerleşik PostgresSQL veritabanının portu. |
| `--metrics-port` değer                           | `10161`                                          | Sunucu metriklerinin yayımlanacağı port. |
| `--disable-metrics`                              | `False`                                          | Sunucu metriklerini devre dışı bırakır. |
| `--disable-worker`                               | (boş)                                          | (KULLANIMDAN KALDIRILDI) GPUStack sunucusunun yerleşik worker'ını devre dışı bırakır. Yeni kurulumlarda varsayılan olarak yerleşik worker bulunmaz. Gerekirse '--enable-worker' ile etkinleştirin. İki bayrak da ayarlanmazsa geriye uyumluluk için v2.0.1 öncesindeki eski kurulumlarda yerleşik worker varsayılan olarak etkinleştirilir. |
| `--enable-worker`                                | `False`                                          | GPUStack sunucusunun yerleşik worker'ını etkinleştirir. |
| `--bootstrap-password` değer                     | Otomatik oluşturulur.                                  | Varsayılan yönetici kullanıcısının ilk parolası. |
| `--database-url` değer                           | Yerleşik PostgreSQL.                             | Veritabanının URL'si. PostgreSQL 13.0+, MySQL 8.0.36+ ve openGauss (postgresql:// şeması) ile OceanBase (mysql:// şeması) gibi uyumlu veritabanları desteklenir. Örnek: postgresql://username:password@host:port/dbname veya mysql://username:password@host:port/dbname |
| `--redis-url` değer                              | None                                             | Hızlı ve paylaşılan duruma ihtiyaç duyan özellikler (LB eklentisinin paylaşılan durum arka ucu) için harici Redis URL'si. Biçim: `redis://host[:port][/db]`. Kimlik bilgileri ağ geçidi CR'sine yazılacağı için kabul edilmez; TLS (`rediss://`) desteklenmez. |
| `--ssl-keyfile` değer                            | (boş)                                          | SSL anahtar dosyasının yolu. |
| `--ssl-certfile` değer                           | (boş)                                          | SSL sertifika dosyasının yolu. |
| `--ssl-ca-certfile` değer                        | (boş)                                          | Worker'ların GPUStack sunucusunu doğrulamak için kullandığı PEM CA paketinin yolu. Sunucu sertifikası özel bir CA tarafından imzalanmışsa gereklidir. |
| `--force-auth-localhost`                         | (ayarlanmamış)                                          | (Kullanımdan kaldırıldı, etkisiz) Localhost üzerinden gelen çağrıların kimliği her zaman doğrulanır. Geriye uyumluluk için korunmuştur ve gelecekteki bir sürümde kaldırılacaktır. |
| `--disable-update-check`                         | `False`                                          | Güncelleme denetimini devre dışı bırakır. |
| `--ota-server-url` değer                         | (boş)                                          | Resmî içerik kaynaklarının okunduğu OTA sunucusunun URL'si. İndeks ve dosyalar doğrudan bu URL altında bulunur, örneğin https://ota.example.com/gpustack. Varsayılan, genel OTA sunucusudur. Ağınızdan erişilemiyorsa kendi kopyanızı belirtin veya kurum içi dağıtımı kendi deponuzdan beslemek için bu ayarı kullanın. Üç içerik türünün tamamını kapsar; `Official` durumları ve otomatik güncellemeleri korunur. Arayüzdeki `Reset to Official Source` işlemi bu adrese döner. [OTA İçerik Güncellemeleri](../user-guide/ota-content-updates.md) bölümüne bakın. |
| `--disable-openapi-docs`                         | `False`                                          | Otomatik oluşturulan OpenAPI belge uç noktalarını devre dışı bırakır: /docs adresindeki Swagger UI, /redoc adresindeki ReDoc ve /openapi.json adresindeki ham belirtim. |
| `--model-catalog-file` değer                     | (boş)                                          | Model kataloğu dosyasının yolu veya URL'si. |
| `--enable-cors`                                  | `False`                                          | Sunucuda Cross-Origin Resource Sharing (CORS) özelliğini etkinleştirir. |
| `--allow-credentials`                            | `False`                                          | Çapraz kaynak isteklerinde çerezlere ve kimlik bilgilerine izin verir. |
| `--allow-origins` değer                          | `["*"]`                                          | Çapraz kaynak isteklerinde izin verilen kaynaklar. Birden çok kaynak için bayrağı tekrarlayın. Örnek: `--allow-origins https://example.com --allow-origins https://api.example.com` |
| `--allow-methods` değer                          | `["GET", "POST"]`                                | Çapraz kaynak isteklerinde izin verilen HTTP yöntemleri. Birden çok yöntem için bayrağı tekrarlayın. Örnek: `--allow-methods GET --allow-methods POST` |
| `--allow-headers` değer                          | `["Authorization", "Content-Type", "X-API-Key"]` | Çapraz kaynak isteklerinde izin verilen HTTP istek başlıkları. Birden çok başlık için bayrağı tekrarlayın. Örnek: `--allow-headers Authorization --allow-headers X-API-Key --allow-headers Content-Type` |
| `--oidc-issuer` değer                            | (boş)                                          | OpenID Connect veren kuruluş URL'si. |
| `--oidc-client-id` değer                         | (boş)                                          | OpenID Connect istemci kimliği. |
| `--oidc-client-secret` değer                     | (boş)                                          | OpenID Connect istemci sırrı. |
| `--oidc-redirect-uri` değer                      | (boş)                                          | OIDC uygulamanızda yapılandırılan yönlendirme URI'si. `<server-url>/auth/oidc/callback` olarak ayarlanmalıdır. |
| `--external-auth-post-logout-redirect-key` değer | (boş)                                          | Farklı IdP'lerde oturum kapatma sonrası yönlendirme için kullanılan ortak parametre adı. Hem OIDC hem SAML için uygulanır. |
| `--oidc-skip-userinfo`                           | `False`                                          | OIDC userinfo_endpoint isteğini atlar ve bilgileri doğrudan başlıktan ayrıştırmayı dener. |
| `--oidc-use-userinfo`                            | (boş)                                          | [Kullanımdan kaldırıldı] Kimlik doğrulamadan sonra kullanıcı ayrıntılarını almak için UserInfo uç noktasını kullanır. |
| `--saml-idp-server-url` değer                    | (boş)                                          | SAML kimlik sağlayıcısı sunucu URL'si. |
| `--saml-idp-entity-id` değer                     | (boş)                                          | SAML kimlik sağlayıcısı varlık kimliği. |
| `--saml-idp-x509-cert` değer                     | (boş)                                          | SAML kimlik sağlayıcısı X.509 sertifikası. |
| `--saml-sp-entity-id` değer                      | (boş)                                          | SAML servis sağlayıcısı varlık kimliği. |
| `--saml-sp-acs-url` değer                        | (boş)                                          | SAML servis sağlayıcısının Assertion Consumer Service URL'si. `<server-url>/auth/saml/callback` olarak ayarlanmalıdır. |
| `--saml-sp-x509-cert` değer                      | (boş)                                          | SAML servis sağlayıcısı X.509 sertifikası. |
| `--saml-sp-private-key` değer                    | (boş)                                          | SAML servis sağlayıcısı özel anahtarı. |
| `--saml-idp-logout-url` değer                    | (boş)                                          | SAML kimlik sağlayıcısının Single Logout uç noktası URL'si. |
| `--saml-sp-slo-url` değer                        | (boş)                                          | SAML servis sağlayıcısının Single Logout Service geri çağrı URL'si (örneğin `<server-url>/auth/saml/logout/callback`). |
| `--saml-security` değer                          | (boş)                                          | JSON biçiminde SAML güvenlik ayarları. |
| `--cas-server-url` değer                         | (boş)                                          | CAS (Central Authentication Service) sunucusunun temel URL'si, örneğin `https://cas.example.com/cas`. Ayarlanması CAS girişini etkinleştirir. |
| `--cas-callback-url` değer                       | (boş)                                          | CAS sunucusuna kaydedilmiş CAS geri çağrı URL'si. Ayarlanmazsa `<server-url>/auth/cas/callback` kullanılır. |
| `--cas-validate-endpoint` değer                  | `/p3/serviceValidate`                            | `--cas-server-url` değerine göre CAS bilet doğrulama uç noktası. CAS 3.0 öncesi sunucularda `/serviceValidate` kullanın (öznitelikler desteklenmez). |
| `--cas-username-attribute` değer                 | (boş)                                          | GPUStack kullanıcı adı olarak kullanılacak CAS XML öznitelik adı. Ayarlanmazsa CAS `cas:user` öğesi kullanılır. |
| `--cas-full-name-attribute` değer                | (boş)                                          | Kullanıcının tam adı olarak kullanılacak CAS XML öznitelik adı. |
| `--cas-avatar-attribute` değer                   | (boş)                                          | Kullanıcının avatar URL'si olarak kullanılacak CAS XML öznitelik adı. |
| `--external-auth-provider-name` değer            | (boş)                                          | Giriş sayfasındaki SSO düğmesinde gösterilecek kimlik sağlayıcısı adı, örneğin `Okta`. Ayarlanmazsa protokol adı (OIDC/SAML/CAS) kullanılır. |
| `--external-auth-name` değer                     | (boş)                                          | Harici kimlik doğrulama kullanıcı bilgisinin kullanıcı adına eşlenmesi, örneğin preferred_username. |
| `--external-auth-full-name` değer                | (boş)                                          | Harici kimlik doğrulama kullanıcı bilgisinin tam ada eşlenmesi. `name` veya `firstName+lastName` gibi birden çok öğe birleştirilebilir. |
| `--external-auth-avatar-url` değer               | (boş)                                          | Harici kimlik doğrulama kullanıcı bilgisinin avatar URL'sine eşlenmesi. |
| `--external-auth-default-inactive`               | `False`                                          | Yeni kullanıcıların varsayılan olarak devre dışı olması gerekiyorsa True. |
| `--external-auth-insecure-skip-tls-verify`       | `False`                                          | Harici kimlik doğrulama IdP el sıkışmasında (OIDC/CAS) TLS doğrulamasını atlar. Yalnızca kendinden imzalı IdP'lerle test içindir; üretimde asla kullanmayın. |
| `--server-external-url` değer                    | (boş)                                          | Worker kaydı için sunucunun harici URL'si. Bulut sağlayıcıları üzerinden worker oluştururken gereklidir ve worker'ların sunucuya doğru bağlanmasını sağlar. |
| `--trusted-hosts` değer                          | (boş)                                          | Ters proxy arkasındaki `the X-Forwarded-Host` başlığı için izin listesi. Ayarlanmazsa `--server-external-url` üzerinden türetilir. İkisi de ayarlanmazsa tüm ana makinelere güvenmek için `*` kullanılır (sunucuya yalnızca güvenilir bir proxy üzerinden erişilemiyorsa önerilmez). |
| `--saml-sp-attribute-prefix` değer               | (boş)                                          | --external-auth-\* ile belirtilen öznitelikleri almak için SAML servis sağlayıcısı öznitelik öneki. |
| `--gateway-concurrency` değer                    | `16`                                             | Yerleşik ağ geçidinin eş zamanlı bağlantı sayısı. |
| `--disable-builtin-observability`                | `False`                                          | Yerleşik Grafana ve Prometheus servislerini devre dışı bırakır. |
| `--builtin-prometheus-port` değer                | `19090`                                          | Yerleşik Prometheus servisinin portu. |
| `--builtin-grafana-port` değer                   | `13000`                                          | Yerleşik Grafana servisinin portu. |
| `--prometheus-url` değer                         | (boş)                                          | Sorgulanacak harici Prometheus'un temel URL'si. Ayarlanmazsa etkin olan yerleşik servis kullanılır. Gözlemlenebilirliği kendi yığınına devreden bir dağıtımda metriklere bağlı özellikleri (model ve PD panoları, bant genişliği öz denetimi) çalışır tutmak için ayarlayın. |
| `--grafana-url` değer                            | (boş)                                          | Pano yönlendirmeleri ve proxy işlemleri için Grafana temel URL'si. Yalnızca konteyner içinden erişilen bir ana makine adı değil, tarayıcıdan erişilebilir bir adres olmalıdır. Ayarlanırsa yerleşik Grafana ve Prometheus devre dışı bırakılır. Yalnızca harici Grafana için gereklidir. |
| `--grafana-worker-dashboard-uid` değer           | (boş)                                          | Worker panosu yönlendirmeleri için Grafana pano UID'si. |
| `--grafana-model-dashboard-uid` değer            | (boş)                                          | Model panosu yönlendirmeleri için Grafana pano UID'si. |

### Worker seçenekleri { #worker-options }

| <div style="width:180px">Bayrak</div>      | <div style="width:100px">Varsayılan</div> | Açıklama |
|------------------------------------------|----------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `-t` değer, `--token` değer              | Otomatik oluşturulur.                        | Worker kaydında kullanılan paylaşılan sır. |
| `-s` değer, `--server-url` değer         | (boş)                                | Bağlanılacak sunucu. |
| `--worker-name` değer                    | (boş)                                | Worker düğümünün adı. Varsayılan olarak ana makine adı kullanılır. |
| `--worker-ip` değer                      | (boş)                                | Worker düğümünün IP adresi. Varsayılan olarak otomatik algılanır. |
| `--disable-worker-metrics`               | `False`                                | Metrikleri devre dışı bırakır. |
| `--worker-metrics-port` değer            | `10151`                                | Metriklerin yayımlanacağı port. |
| `--worker-port` değer                    | `10150`                                | Worker'ın bağlanacağı port. Tüm worker'larda aynı değeri kullanın. |
| `--service-port-range` değer             | `40000-40063`                          | Çıkarım servislerinin port aralığı, 'N1-N2' biçiminde bir dizeyle belirtilir. Her iki uç dahildir. |
| `--ray-port-range` değer                 | `41000-41999`                          | Ray servislerinin (vLLM dağıtık dağıtımında kullanılır) port aralığı, 'N1-N2' biçiminde bir dizeyle belirtilir. Her iki uç dahildir. |
| `--log-dir` değer                        | (boş)                                | Günlüklerin saklanacağı dizin. |
| `--system-reserved` değer                | (boş)                                | Sistemin zamanlama sırasında worker için ayırdığı kaynaklar, GiB cinsinden. Varsayılan olarak kaynak ayrılmaz. Örnek: '{"ram": 2, "vram": 1}'. |
| `--enable-hf-transfer`                   | `False`                                | [Kullanımdan kaldırıldı] hf_transfer desteği huggingface_hub v1.0 sürümünde kaldırılmıştır; bu bayrak etkisizdir. Varsayılan indirme aracı artık hf_xet'tir. |
| `--enable-hf-xet`                        | `False`                                | [Kullanımdan kaldırıldı] Model dosyalarını Hugging Face Xet ile indirmeyi etkinleştirir. |
| `--worker-ifname` değer                  | (boş)                                | Worker düğümünün ağ arayüzü adı. Varsayılan olarak otomatik algılanır. |
| `--kv-transfer-ifname` değer                      | (boş)                                | Ayrıştırılmış (prefill/decode) dağıtımlarda KV önbellek trafiğini taşıyan ağ arayüzü. Ana makinede yalnızca bir yönlendirilebilir arayüz varsa otomatik algılanır. Birden çok arayüz olduğunda gereklidir; yanlış seçim, motorun karşı uçtan erişilemeyen bir adresi duyurmasına neden olur. |
| `--proxy-mode` değer                     | (boş)                                | Model örneklerine erişim için proxy modu: `direct`, `worker` veya `tunnel` (worker'ın sunucuya dışa doğru bağlantı kurduğu, güvenlik duvarı/NAT arkasındaki worker'lara yönelik mod). Varsayılan, yerleşik worker için `direct`, bağımsız worker için `worker` değeridir. |
| `--benchmark-image-repo` değer           | `gpustack/benchmark-runner`            | GPUStack benchmark konteyneri için varsayılan imaj deposunu geçersiz kılar. `--image-repo` seçeneğinden farklı olarak bu değer olduğu gibi kullanılır; etiketi ekleyin.<br/>`gpustack/benchmark-runner:<tag>` imajı `gpustack` ad alanı dışındaysa, örneğin `awesome.com/amazing-group/amazing-subgroup/gpustack-ai/benchmark-runner:<tag>`, önce üst ad alanını içeren kayıt deposunu `--system-default-container-registry=awesome.com/amazing-group/amazing-subgroup` ile belirtin. Sonra bu bayrağı (veya worker yapılandırma anahtarı `benchmark_image_repo` ya da `GPUSTACK_BENCHMARK_IMAGE_REPO` ortam değişkenini) `gpustack-ai/benchmark-runner:<tag>` olarak ayarlayın. |
| `--benchmark-dir` değer                  | `<data-dir>/benchmarks`                | Benchmark sonuçlarının saklanacağı dizin. |
| `--benchmark-max-duration-seconds` değer | (boş)                                | Zaman aşımından önceki en uzun benchmark süresi. Boşsa devre dışıdır. |

### Kullanılabilir ortam değişkenleri { #available-environment-variables }

Komut satırı parametrelerinin çoğu, `GPUSTACK_` öneki eklenerek ve adları büyük harfle
yazılarak ortam değişkenleriyle de ayarlanabilir (örneğin `--data-dir`, `GPUSTACK_DATA_DIR` ile ayarlanabilir).

Yukarıdaki komut satırı parametreleri dışında kalan ortam değişkenleri için
[ortam değişkenleri belgesine](../environment-variables.md) bakın.

## Yapılandırma dosyası { #config-file }

GPUStack sunucusu veya worker başlatılırken başlangıç seçeneklerini YAML biçimindeki bir yapılandırma dosyasında belirtebilirsiniz. Aşağıda
eksiksiz bir örnek verilmiştir:

```yaml
# Common Options
advertise_address: exposed_server_or_worker_ip
debug: false
data_dir: /path/to/data_dir
cache_dir: /path/to/cache_dir
benchmark_dir: /path/to/benchmark_dir
benchmark_image_repo: gpustack/benchmark-runner
token: your_token
huggingface_token: your_huggingface_token
insecure_tls: false

# Server Options
port: 80
tls_port: 443
api_port: 30080
metrics_port: 10161
disable_metrics: false
enable_worker: false
bootstrap_password: your_admin_password
database_url: postgresql://username:password@host:port/dbname
# database_url: mysql://username:password@host:port/dbname
ssl_keyfile: /path/to/keyfile
ssl_certfile: /path/to/certfile
ssl_ca_certfile: /path/to/ca-bundle.pem
disable_update_check: false
disable_openapi_docs: false
model_catalog_file: /path_or_url/to/model_catalog_file
enable_cors: false
allow_credentials: false
allow_origins: [ "*" ]
allow_methods: [ "GET", "POST" ]
allow_headers: [ "Authorization", "Content-Type", "X-API-Key" ]
oidc_issuer: https://your_oidc_issuer
oidc_client_id: your_oidc_client_id
oidc_client_secret: your_oidc_client_secret
oidc_redirect_uri: http://your_gpustack_server_url/auth/oidc/callback
saml_idp_server_url: https://your_saml_idp_server_url
saml_idp_entity_id: your_saml_idp_entity_id
saml_idp_x509_cert: your_saml_idp_x509_cert_pem
saml_sp_entity_id: your_saml_sp_entity_id
saml_sp_acs_url: http://your_gpustack_server_url/auth/saml/callback
saml_sp_x509_cert: your_saml_sp_x509_cert_pem
saml_sp_private_key: your_saml_sp_private_key_pem
saml_security: '{"wantAssertionsSigned": true, "wantMessagesSigned": true}'
cas_server_url: https://your_cas_server_url
cas_username_attribute: uid
cas_full_name_attribute: displayName
external_auth_provider_name: Okta
external_auth_name: email
external_auth_full_name: name
external_auth_avatar_url: picture
server_external_url: http://your_gpustack_server_url_for_external_access
trusted_hosts: [ "your_reverse_proxy_hostname" ]
disable_builtin_observability: false
builtin_prometheus_port: 19090
builtin_grafana_port: 13000
# Per-plugin overrides for the API gateway, keyed by the plugin's manifest name
# -- which for several of them is not the name of the WasmPlugin resource they
# are deployed as (`transformer` is deployed as `gpustack-header-transformer`,
# `ai-statistics` as `gpustack-ai-statistics`). Keying by the wrong one matches
# nothing, silently.
#
# `url` / `sha256` / `image_pull_policy` say where Envoy pulls the module from,
# replacing the address derived from the bundled plugin manifest. `config` is
# the plugin's own settings and is validated against what that plugin accepts,
# so a field it does not declare fails startup rather than being ignored.
gateway_plugin:
  gpustack-ext-auth:
    config:
      local_auth:
        # Whether the gateway authenticates API keys itself. Turned off, every
        # credential is forwarded to the server instead, as it was before.
        enabled: true
      authz:
        # How long the gateway waits for the server to authorize a request.
        # Sized for a loaded server rather than a healthy one: a server that is
        # down refuses the connection and is detected at once, so this governs
        # only the case of one too busy to answer yet -- where waiting is what
        # keeps a load spike from being read as an outage.
        timeout: 30000
      # Let a caller the gateway authenticated itself through while the server
      # is unreachable -- a rolling upgrade included. Authorization is skipped
      # for the duration; unknown keys and anonymous callers are still refused.
      failure_mode_allow_authenticated: true
  ai-statistics:
    config:
      enable_content_types: [ application/json, text/event-stream ]
  # Fields are the plugin's own camelCase config keys; only whitelisted ones
  # are accepted, so a field the server does not vouch for fails startup
  # instead of being forwarded.
  gpustack-ai-proxy:
    config:
      # Reject tool-call messages that arrive without the assistant message
      # that initiated them ("strict"), or the plugin's default ("off").
      toolCallValidation: strict

# Worker Options
server_url: http://your_gpustack_server_url
worker_name: your_worker_name
worker_ip: 192.168.1.101
disable_worker_metrics: false
worker_metrics_port: 10151
worker_port: 10150
service_port_range: 40000-40063
ray_port_range: 41000-41999
log_dir: /path/to/log_dir
system_reserved:
  ram: 2
  vram: 1
enable_hf_transfer: false
proxy_mode: worker
```
