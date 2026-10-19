---
translation_status: translated
---

# Karşılaştırma { #benchmarking }

GPUStack, çalışan model örnekleri üzerinde benchmark çalıştırabilir. Benchmark'lar çalışanlar tarafından özel bir konteyner imajında yürütülür; sonuçlar ve günlükler çalışanda saklanır.

## Önkoşullar { #prerequisites }

- Bir model örneği çalışıyor ve sağlıklı durumda.

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

Benchmark konteyneri ilerlemesini sunucu URL'si üzerinden GPUStack sunucusuna bildirir; bu nedenle HTTPS dağıtımında sunucu sertifikasını doğrulamalıdır. Benchmark ayrı bir imajda çalışır ve CA'ları kendisi içe aktarmaz. Çalışan, `/usr/local/share/ca-certificates/` altına bağlanan özel CA'lar dahil kendi güvendiği CA paketini ona aktarır (bkz. [Ek Güvenilir CA'lar](../installation/installation.md#additional-trusted-cas)). CA çalışanda mevcutsa ek yapılandırma gerekmez.

Doğrulama yine başarısız olursa benchmark'ın kendisi etkilenmez; yük model örneğine düz HTTP üzerinden gönderilir. Ancak çalışma tamamlanana kadar ilerleme 0% olarak kalır ve neden çalışanın günlüğüne yazılır. Düzeltmek için sunucunun CA'sını çalışana aktarın.
