---
translation_status: translated
---

# Karşılaştırma { #benchmarking }

GPUStack, çalışan model örnekleri üzerinde benchmark çalıştırabilir. Benchmark'lar çalışanlar tarafından özel bir konteyner imajında yürütülür; sonuçlar ve günlükler çalışanda saklanır.

## Önkoşullar { #prerequisites }

- Bir model örneği çalışıyor ve sağlıklı durumda.

### Rota benchmark'ı için ağ geçidi adresi { #route-benchmark-gateway-address }

Rota benchmark'ları, çalışanın erişebildiği Higress temel URL'sini (`/v1` olmadan) aşağıdaki sırayla belirler:

1. Benchmark'ı yürüten çalışanın ait olduğu kümenin `server_url` değeri.
2. Sunucunun `server_external_url` değeri.
3. Sunucunun `trusted_hosts` yapılandırmasında açıkça izin verilen oluşturma isteğinin origin adresi.

İstek origin adresine geri dönüldüğünde joker karakterler kabul edilmez. Yalnızca ana makine adı standart HTTP/HTTPS portlarına izin verir; özel portları `gateway.example:8443` gibi açıkça belirtin. URL bir yol öneki içeriyorsa öndeki proxy, isteği Higress'e iletmeden önce bu öneki kaldırmalıdır.

Hedef, benchmark oluşturulurken sabitlenir. Değiştirilen URL'yi kullanmak için benchmark'ı yeniden oluşturun. İlerleme bildirimi çalışanın yönetim adresini kullanmaya devam eder.

## Karşılaştırma Oluştur { #create-benchmark }

1. `Benchmarks` sayfasına gidin.
2. `Add Benchmark`'a tıklayın.
3. Bir örnek seçin ve yapılandırmaları doldurun.
4. `Save` düğmesine tıklayın.

### ShareGPT İş Yükü Uzunlukları { #sharegpt-workload-lengths }

Yerleşik ShareGPT veri kümesinde giriş token uzunluğu için isteğe bağlı alt sınır, üst sınır veya ikisini birden belirleyebilirsiniz. GPUStack, tokenleştiricinin saydığı token sayısı bu sınırlara uyan ilk tur istemlerini tutar. Çıkış token uzunluğu isteğe bağlıdır: ayarlanırsa seçilen her istem için aynı üst sınır istenir; ayarlanmazsa her istemin veri kümesindeki referans yanıt uzunluğu kullanılır. Aralığa uyan istem yoksa çalışma benchmark istekleri gönderilmeden başarısız olur.

## Karşılaştırma Sonuçlarını Görüntüle { #view-benchmark-results }

1. `Benchmarks` sayfasına gidin.
2. Görüntülemek istediğiniz kriteri bulun.
3. Sonuçları ve yapılandırma anlık görüntüsünü görüntülemek için kıyaslama adına tıklayın.

## Karşılaştırma Sonuçlarını Dışa Aktarma { #export-benchmark-results }

1. `Benchmarks` sayfasına gidin.
2. Dışa aktarmak istediğiniz kriteri bulun.
3. Dışa aktarmak için bir veya daha fazla kıyaslama çalıştırması seçin.
4. Sonuçları indirmek için `Export` düğmesine tıklayın.

## Karşılaştırmayı Düzenle { #edit-benchmark }

1. `Benchmarks` sayfasına gidin.
2. Düzenlemek istediğiniz kriteri bulun.
3. `Operations` sütunundaki `Edit` düğmesine tıklayın.
4. Adı ve açıklamayı gerektiği gibi değiştirin.
5. `Save` düğmesine tıklayın.

## Karşılaştırmayı Sil { #delete-benchmark }

1. `Benchmarks` sayfasına gidin.
2. Silmek istediğiniz kriteri bulun.
3. `Operations` sütunundaki `Delete` düğmesine tıklayın.
4. Silme işlemini onaylayın.

## Özel CA'lı HTTPS Sunucusunda Karşılaştırmalar { #benchmarks-on-an-https-server-with-a-private-ca }

İlerleme bildirimi ve rota çıkarımı çalışanın TLS politikasını izler. Çalışanda `GPUSTACK_INSECURE_TLS` veya `insecure_tls` yapılandırma seçeneği (`--insecure-tls`) etkin olmadıkça sertifika doğrulaması açıktır. Benchmark için ayrı bir TLS ayarı yoktur.

Girişlerden biri HTTPS kullanıyor ve doğrulama etkinse çalışan, güvendiği CA paketini dosya içeriği olarak benchmark konteynerine aktarır. Buna çalışana içe aktarılan özel CA'lar da dahildir (bkz. [Ek Güvenilir CA'lar](../installation/installation.md#additional-trusted-cas)). İlerleme istemcisi `--progress-ca-cert` kullanır; HTTPS çıkarımı, arka uç doğrulaması açıkken aktarılan paketi `SSL_CERT_FILE` üzerinden kullanır. Yönetim adresinin HTTP olması, HTTPS ağ geçidi için CA aktarılmasını engellemez.

Çalışan doğrulamayı açıkça atlıyorsa benchmark her iki kanalda da doğrulamayı atlar ve CA paketi aktarmaz. Doğrulama gerektiren HTTPS çıkarımı için paket okunamıyorsa benchmark farklı bir güven deposu kullanmak yerine başlatılmayı reddeder. Yalnızca ilerleme bildirimi için en iyi çaba yaklaşımı sürer: bildirim hataları günlüğe yazılır, ancak benchmark ölçümleri atılmaz.
