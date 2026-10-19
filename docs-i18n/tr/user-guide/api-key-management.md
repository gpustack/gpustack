---
translation_status: translated
---

# API Anahtar Yönetimi { #api-key-management }

GPUStack, API anahtarlarını kullanarak kimlik doğrulamayı destekler. Her GPUStack kullanıcısı kendi API anahtarlarını oluşturabilir ve yönetebilir.

## API Anahtarı Oluşturun { #create-api-key }

1. `Access Control` > `API Keys` sayfasına gidin.
2. `Add API Key` düğmesine tıklayın.
3. API anahtarının `Name`, `Description` alanlarını doldurun ve `Expiration`'yi seçin.
4. API anahtarının `Type`'ını seçin:

   - **Otomatik olarak oluşturulan**: GPUStack sizin için API anahtarını oluşturur.
   - **Custom**: `Key` alanına kendi anahtar değerinizi girin. **Custom** seçilirken bu alan gereklidir. Hiçbir zaman hatırlamak zorunda kalmayacağınız bir şifre gücünde bir değer seçin; aşağıdaki nota bakın.

5. `Access Permissions` bölümünde API anahtarının izinlerini seçin:

   - **Platform Yönetimi**: Platform yönetimi uç noktalarına (kullanıcılar, modeller, çalışanlar vb.) erişim sağlar.
   - **Model Erişimi**: Çıkarım API'lerine erişim izni verir. Seçildiğinde `All models` veya `Allowed models`'yi seçin ve `Allowed models`'i seçerseniz listeden bu API anahtarının hangi modellere erişebileceğini seçin.

   Hiçbir izin seçilmezse API anahtarının erişim izni yoktur.
6. `Save` düğmesine tıklayın.
7. Anahtarı kopyalayıp güvenli bir yere saklayın, ardından `Done` düğmesine tıklayın.

!!! note

    Tam API anahtarı değeri oluşturulduktan sonra yalnızca bir kez gösterilir; daha sonra yalnızca maskelenmiş bir değer görüntülenir. Özel anahtarlar istisnadır; onların değerini zaten biliyorsunuz.

!!! warning "Özel tuşlar sizin onlara verdiğiniz gücü taşır"

    GPUStack herhangi bir dizeyi özel anahtar olarak kabul eder ve `123456`'den rastgele bir dizeyi ayırt edemez. Otomatik olarak oluşturulan anahtarlar, kriptografik rastgele bir kaynaktan gelen 128 bittir ve daha zayıf hiçbir şey asla kazara yaratılmaz.

    Bu fark, bariz tahminin ötesinde önemlidir: Anahtarların sunucu yerine API ağ geçidinde doğrulanması için GPUStack, her anahtarın hızlı bir karmasını ağ geçidinin yapılandırmasına yayınlar; bu, onu orada okuyabilen herkes tarafından görülebilir ve teşhis paketleri halinde seyahat eder. Hash'in tuzsuz olduğu ve sırdan başka hiçbir şeye bağlı olmadığı özel bir anahtar için, bu nedenle her dağıtımda aynı değere sahiptir; bir kez oluşturulan önceden hesaplanmış bir tablo hepsine karşı çalışır. Rastgele bir anahtarın aranması her iki durumda da mümkün olmadığından etkilenmez, tahmin edilebilir bir anahtarın ise çevrimdışı kurtarılması ucuz hale gelir.

    Mümkün olduğunda özel anahtarları rastgele bir kaynaktan üretin. İnsanların elle seçtiği anahtarlar için `GPUSTACK_GATEWAY_AUTH_ALLOW_CUSTOM_KEYS=false` ayarlayın. Böylece özel anahtarlar eskisi gibi her istekte sunucuda doğrulanır ve karma değerleri yayımlanmaz. Ağ geçidine yalnızca bu kimlikte bir anahtarın var olduğu ve ne zaman sona ereceği bildirilir; bu bilgi, sunucu kapalıyken de anahtarı iptal edebilmeyi sağlar.

    Bu ayar yalnızca yeni anahtarlar için değil, halihazırda var olan anahtarlar için de geçerlidir: bunu kapatmak, bir sonraki mutabakat sırasında açıkken oluşturulan özel anahtarların karmalarını geri çeker.

## Erişim İzinlerini Düzenle { #edit-access-permissions }

1. `Access Control` > `API Keys` sayfasına gidin.
2. Düzenlemek istediğiniz API anahtarını bulun.
3. `Operations` sütunundaki `Edit` düğmesine tıklayın.
4. Gerekirse `Description`'ı güncelleyin.
5. `Access Permissions` bölümünde seçilen izinleri ayarlayın:

   - **Platform Yönetimi**: Platform yönetimi uç noktalarına (kullanıcılar, modeller, çalışanlar vb.) erişim sağlar.
   - **Model Erişimi**: Çıkarım API'lerine erişim izni verir. Seçildiğinde `All models` veya `Allowed models`'yi seçin ve `Allowed models`'i seçerseniz listeden bu API anahtarının hangi modellere erişebileceğini seçin.

6. `Save` düğmesine tıklayın.

!!! note

    Değişiklikler bir dakika içinde geçerli olacaktır.

## API Anahtarını Sil { #delete-api-key }

1. `Access Control` > `API Keys` sayfasına gidin.
2. Silmek istediğiniz API anahtarını bulun.
3. `Operations` sütunundaki `Delete` düğmesine tıklayın.
4. Silme işlemini onaylayın.

## Sunucu Kullanılamıyorken Anahtarı İptal Etme { #revoke-a-key-while-the-server-is-unavailable }

Normalde bir anahtarın silinmesi yeterlidir: sunucu API ağ geçidine bilgi verir ve anahtarın kabulü birkaç saniye içinde durdurulur. Bu bölüm yalnızca bir anahtarın **now** çalışmayı durdurması gerektiği ve GPUStack sunucusunun kapalı olduğu (örneğin yükseltme sırasında) durumlar içindir.

Ağ geçidi kabul edebileceği anahtarların kendi kopyasını tutar; böylece sunucu kullanılamadığında çıkarım hizmeti devam eder. Bu kopya sunucuda değil Kubernetes'te bulunduğundan sunucu olmadan da düzenlenebilir:

```bash
kubectl -n higress-system edit wasmplugin gpustack-llm-ext-auth
```

Anahtarın `spec.defaultConfig.local_auth` altındaki kaydını kaldırın. Kayıt genellikle `keys` içindedir ve anahtarın erişim anahtarıyla, yani otomatik üretilen `gpustack_<access key>_<secret>` değerinin orta bölümüyle indekslenir. Özel anahtarda erişim anahtarı bulunmaz; bunun yerine `keys` içinde sahibinin sayısal `user_id` değerini arayın veya `exp` değerini anahtarın sona erme zamanı ile eşleştirin.

`GPUSTACK_GATEWAY_AUTH_ALLOW_CUSTOM_KEYS`'in `false` olarak ayarlandığı yerde, özel anahtarlar bunun yerine `refs`'dedir ve API'de gösterildiği gibi anahtarın sayısal kimliğine göre indekslenir.

Her ikisinde de bulunmayan bir anahtar, zaten her istekte sunucuya soruyor, dolayısıyla kaldırılacak bir şey yok - sunucu geri döndüğü anda çalışmayı durduruyor ve silme işlemini gerçekleştiriyor.

Değişiklik, kaydedildikten birkaç saniye sonra ağ geçidine ulaşır; kaynakta yapılan diğer değişikliklerle aynı şekilde seyahat eder. Halihazırda yayında olan oturumlar da kapsam dahilindedir: Ağ geçidinin sunucuya sormadan kabul ettiği her şey ilk olarak bu girişlere göre yeniden kontrol edilir, dolayısıyla bir tanesinin kaldırılması onu her yerde aynı anda geçersiz kılar.

!!! warning

    Bu geçici bir önlemdir, kalıcı iptal değildir. Sunucu başladıktan birkaç saniye sonra listeyi veritabanından yeniden oluşturur; dolayısıyla **yalnızca buradan kaldırılan anahtar, sunucu geri geldiğinde yeniden geçerli olur**. Sunucu kullanılabilir olduğunda anahtarı normal yoldan da silin.

!!! note

    Kaynağı düzenlemek, ağ geçidinin çalıştığı kümeye erişim gerektirir ve yukarıdaki `higress-system` varsayılan ağ geçidi ad alanıdır; değiştirilmişse sizinkiyle değiştirin. GPUStack veritabanını doğrudan değiştirmek desteklenen bir alternatif değildir ve dağıtımın tutarsız olmasına neden olabilir.

!!! warning "Bu ad alanını kimler okuyabilir?"

    Aynı kaynak, ağ geçidinin dahili kimlik token'larını imzaladığı anahtarı da içerir. Ağ geçidinin ad alanındaki kaynakları okuyabilen biri böyle bir token üretip herhangi bir kullanıcı olarak kabul edilebilir. Bu nedenle buradaki okuma erişimini platform düzeyinde yetki olarak değerlendirin ve buna göre verin.

## API Anahtarını Kullan { #use-api-key }

GPUStack, API anahtarının taşıyıcı token olarak kullanılmasını destekler.

OpenAI uyumlu `GET /v1/models` uç noktası, yalnızca kullanıcı ve mevcut organizasyon bağlamı tarafından görülebilen hazır modelleri döndürür. API anahtarı izin verilen modellerle sınırlıysa liste, herhangi bir kuruluş öneki de dahil olmak üzere (örneğin, `team-a/qwen3`) bu model kimliklerine göre daha fazla filtrelenir. Bu aynı zamanda eski `GET /v1-openai/models` uç noktası ve yöneticiler tarafından oluşturulan anahtarlar için de geçerlidir. Ayarlanmamış veya boş bir `allowed_model_names` listesi hiçbir ek model kısıtlaması getirmez.

Aşağıda curl'un kullanıldığı bir örnek verilmiştir:

```bash
export GPUSTACK_API_KEY=your_api_key
curl http://your_gpustack_server_url/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GPUSTACK_API_KEY" \
  -d '{
    "model": "qwen3",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Hello!"
      }
    ],
    "stream": true
  }'
```
