---
translation_status: translated
---

# Prefill ve Decode Ayrıştırması (PD) { #pd-disaggregation }

Prefill/decode (PD) ayrıştırması, dağıtımı ayrı çalışan rollere ayırır: **prefill** istemin KV önbelleğini hesaplar, **decode** bu önbellekten token üretir ve önlerindeki **yönlendirici** her isteği iki rol arasında taşır. KV önbelleği yeniden hesaplanmak yerine ağ üzerinden aktarılır.

İki aşamanın kaynak ihtiyaçları farklıdır. Prefill işlem gücüyle sınırlıdır ve geniş tensör paralelliğinden yararlanır; decode bellek bant genişliğiyle sınırlıdır ve çok sayıda eşzamanlı dizi ister. Tek dağıtım iki aşama için ortak parametreler seçmek zorundadır. Ayrıştırma, her tarafın kaynaklarını ve ayarlarını yaptığı işe göre belirlemeyi sağlar. Oran iş yüküne göre değişir: istemi ağır iş yükü daha fazla prefill, uzun yanıt üreten iş yükü daha fazla decode ister.

Bunun bir maliyeti de vardır: işlenen istek ağdan geçer, bir yerine üç rolün çalışır tutulması gerekir ve grubun kullanılabilirliği en az kullanılabilir rolüyle sınırlıdır. Ayrıştırma ancak iki aşamanın ihtiyaçları gerçekten çatışıyorsa yararlıdır.

## Ne zaman yardımcı olur ve ne zaman olmaz { #when-it-helps-and-when-it-does-not }

Üç soru buna karar verir:

1. **Önceden doldurma ve kod çözme farklı parametreler mi istiyor?** Tek bir motor parametre seti her iki aşamaya da kabul edilebilir şekilde hizmet ediyorsa ayrıştırma, verimi artırmadan makine ekler.
2. **İstem-oluşturma oranı 1:1'den uzak mı?** Kazanç, iki tarafın bağımsız olarak boyutlandırılmasından gelir; bu, yalnızca farklı şekilde yüklendiklerinde işe yarar.
3. **KV aktarımı yeterince hızlı mı?** İstemin KV önbelleği her istekte ağ üzerinden geçer. Yavaş veya paylaşılan bir bağlantıda aktarım süresi, kaydettiğiniz ön doldurma süresinin yerini alır.

Tek bir GPU veya istemleri ve nesilleri kısa olan bir iş yükü, onsuz genellikle daha hızlı olacaktır.

## Ayrıştırılmış bir modeli dağıtma { #deploying-a-disaggregated-model }

1. Bir modeli her zamanki gibi dağıtın, ardından `Disaggregation` bölümünü genişletin ve açın.
2. Bir **PD modu** seçin. Mod, motor, KV konektörü ve yönlendiriciden oluşan eksiksiz bir tariftir; dolayısıyla konektör, el sıkışma değişkenleri ve bağlantı noktalarının tümü bu tek seçime göre belirlenir ve bunların hiçbiri formda görünmez. Kullanılabilir modlar arka uca bağlıdır; başka bir motor için oluşturulmuş bir mod, nedeni ile birlikte devre dışı olarak gösterilir.
3. Her rolün **kopyalarını** ayarlayın. Bu oran: 2 ön doldurma ve 1 kod çözme, 2P1D grubudur.
4. İki tarafın farklı olduğu rol başına ayarları geçersiz kılın. Miras olarak kalan her alan, dağıtım düzeyindeki değeri alır, dolayısıyla yalnızca farkların yazılması gerekir.
5. Kaydet.

### Rollere özel bağımsız arka uçlar { #independent-role-backends }

Prefill, Decode ve Router farklı arka uçlar, sürümler ve imajlar kullanabilir.
**Dağıtım varsayılanları**, dağıtım formunun üst kısmındaki arka uç ayarlarıdır.

**Ayarları devralma veya bir rolü ayrı yapılandırma**

- Dağıtımın arka ucunu, sürümünü ve imajını kullanmak için **Dağıtım ayarlarını devral** seçeneğini koruyun.
- Bir rolü ayrı yapılandırmak için o rolün ayarlarından bir arka uç seçin. Rolün arka ucu, sürümü, imajı ve komutu artık ayrı bir ayar kümesi oluşturur. Dağıtım varsayılanlarında sonradan yapılan değişiklikler bu kümeyi değiştirmez.

Örneğin, Prefill ve Decode vLLM kullanmaya devam ederken Router, yönlendirme için kaydettiğiniz bir arka ucu kullanabilir.

**Hangi ayar önceliklidir?**

Bir rol için arka uç seçtikten sonra aşağıdaki kurallar uygulanır. Öncelik listelerinde kullanılabilir ilk değer esas alınır.

| Ayar | Kural |
| --- | --- |
| Sürüm | Rol için seçilen sürüm kullanılır. **Auto**, bu rolün arka ucunun bir sürüm seçmesini sağlar; dağıtımın sürümünü kullanmaz. |
| İmaj | Rol için girilen imaj → arka uç/sürüm yapılandırmasındaki imaj → uyumlu GPUStack runner imajı. Girdiğiniz imajı kendi tag veya digest değeri tanımlar. |
| Prefill / Decode komutu | Rol için girilen komut önceliklidir. Girilmemişse GPUStack, seçilen çıkarım arka ucunun normal başlatma davranışını kullanır. |

Doğrudan bir imaj belirtmek için **Custom** seçin. Prefill ve Decode için bir çalıştırma komutu da gerekir. Router, aşağıdaki komut önceliğini kullanır.

**Router komut önceliği**

GPUStack kaynakları aşağıdaki sırayla kontrol eder ve kullanılabilir ilk komutu kullanır:

1. Router rolü için girilen çalıştırma komutu.
2. Router için seçilen sürümün yapılandırmasında tanımlanan çalıştırma komutu.
3. Router için seçilen arka ucun varsayılan çalıştırma komutu.
4. PD tarifinin ürettiği Router komutu.

Bu sıradaki ikinci ve üçüncü maddeler, Router rolünün kendisi için seçilen arka uçtan gelir. Yerleşik, topluluk tarafından sağlanan ve kullanıcı tarafından kaydedilen arka uçlarda aynı sıra geçerlidir. Yalnızca Router imajını değiştirmek bu komut önceliğini korur. Bu kaynakların hiçbiri bir komut sağlamıyorsa dağıtım kaydedilemez.

**Arka uç parametreleri ve ortam değişkenleri**

Bunlar arka uç, sürüm, imaj ve komuttan ayrı yapılandırılır:

- **Parametreler:** Prefill veya Decode rolünün kendi listesi dağıtımın listesinin yerini alır. Router kendi parametre listesini kullanır ve bunu Router komutuna ekler. Bir arka uç seçildiğinde varsayılan parametreleri forma doldurulur; başlangıçta hangi parametrelerin aktarılacağını rolün kaydedilmiş listesi belirler.
- **Ortam değişkenleri:** rolün kendi değişken haritası, dağıtımın haritasının tamamının yerini alır. Prefill ve Decode için forma getirilen arka uç varsayılanları, rolün kaydedilmiş yapılandırmasının bir parçası olur. Router, varsayılanları başlangıçta da yükler; aynı adlı değişkenler için öncelik şöyledir: rolün değerleri (rolün kendi haritası yoksa dağıtımın değerleri) → seçilen arka uç sürümü → arka uç varsayılanları → geçerli olduğu durumlarda PD tarifinin varsayılanları.

Prefill ve Decode rollerinin kullandığı arka uçlar ve sürümler, seçilen PD tarifinin gereksinimlerini karşılamalıdır. Sağladığınız her imaj gerekli başlatma programını içermeli ve eşleşen konektör protokolünü desteklemelidir.

### Router çalışma zamanı komut şablonları { #runtime-router-command-arguments }

Rol veya arka uç komutu tüm bayrakları belirler. GPUStack P/D hazır olduğunda çalışma zamanı verilerini doldurur ve şema bayraklarını eklemez. İkisi de komut sağlamıyorsa PD şemasının varsayılan komutu ve üyelik politikası kullanılır. Şablonlar `custom` PD modunda da çalışır.

- `{{worker_ip}}` bu Router’ın Worker IP adresidir; `{{port}}` ayrılan servis portudur; `{{ports.prometheus}}` tarif ayırıyorsa adlandırılmış porttur.
- `prefill` / `decode` geçerli gruptaki tüm hazır uç noktaları kararlı adres sırasıyla içerir. Her öğenin alanları: `ip`, `port`, `url`, `ports`. Adlandırılmış portlar her uç noktaya aittir; örneğin `{{p.ports.bootstrap}}`.
- `prefill_urls` / `decode_urls` şu işlemi destekleyen URL listeleridir: `join`.

İki prefill ve iki decode uç noktası varsa her döngü iki argüman grubu üretir. Bayrakları ve dinleme adresini kullandığınız programa göre değiştirebilirsiniz. vLLM NIXL örneği:

```text
vllm-router --vllm-pd-disaggregation --kv-connector nixl
  --host 0.0.0.0 --port {{port}}
  --prometheus-host 0.0.0.0 --prometheus-port {{ports.prometheus}}
  {% for p in prefill %}
    --prefill {{p.url}}
  {% endfor %}
  {% for d in decode %}
    --decode {{d.url}}
  {% endfor %}
```

Virgülle ayrılmış uç nokta listelerini kabul eden bir program için:

```text
my-router --listen {{worker_ip}} --listen-port {{port}}
  --prefill-endpoints "{{prefill_urls | join(',')}}"
  --decode-endpoints "{{decode_urls | join(',')}}"
```

Yalnızca değişken değerleri, uç nokta/URL listelerinde tek seviyeli döngüler ve sabit ayırıcıyla URL birleştirme desteklenir. Bilinmeyen adlar ve desteklenmeyen sözdizimi kayıtta reddedilir; bulunamayan adlandırılmış portlar başlatmayı durdurur. İşlev çağrıları, içe aktarma, koşullar, iç içe döngüler ve diğer filtreler desteklenmez. Yeni satırlar argümanları ayırır; bu, örtük shell çalıştırmayan bir argv şablonudur. Boşluk veya tırnak içeren değerler tek argüman olarak kalır.

Şablonlar, `{{ROUTER_CONFIG}}` gibi açıkça yapılandırılmış ortam değişkenlerine de başvurabilir. Rolün ortam haritası belirtilmişse o kullanılır, aksi halde dağıtım ortamı devralınır; katalog değerleri varsayılan olarak birleştirilir. Aynı adlı çalışma zamanı şablon değişkenleri önceliklidir.

Router üst imajı ve sürümü devralır ancak hiçbir PD modunda üst çıkarım komutunu devralmaz. Bağımsız seçilen arka uç, sürümün `run_command` veya `default_run_command` değerini normal entrypoint ve argüman anlamıyla sağlar. Rolün `run_command` değeri tam başlatma komutudur ve entrypoint değerini geçersiz kılar. `custom` PD modu dahil hiçbir komut kaynağı yoksa kayıt eksik Router komutu hatasıyla reddedilir.

Komutun sabit metni POSIX tırnak kurallarını izler. Düzenli ifade veya yollardaki ters eğik çizgileri korumak için aşağıdaki gibi tek tırnak kullanın:

```text
my-router --regex '\d+\s+' --config 'C:\models\router.json' --port {{port}}
```

Dağıtım özgün şablonu saklar; örnek yapılandırma anlık görüntüsüne açılmış komut kaydedilir. Eş verilerine başvuran komutlar, üyeler veya yeniden başlatma nesilleri değiştiğinde yeniden açılır ve Router yeniden oluşturulur. Eş verisi kullanmayan özel komutlar elle üyelik güncellemesi gerektirir; GPUStack bayraklardan üyelik API’si çıkarmaz.

Router veya dağıtım imajı açıkça belirtildiğinde tam komut imajın uygulama ENTRYPOINT ayarını geçersiz kılar; imaj içinde shell gerekmez.

## Bir grup nasıl başlar? { #how-a-group-starts }

Sıralama tesadüfi değildir ve bilmeye değer çünkü bir grup geldiğinde göreceğiniz şey şu:

1. Prefill ve decode örnekleri tek işlemde **birlikte** oluşturulur. Toplu kabul destekleniyorsa kümeye de birlikte kabul edilirler; yalnızca yarısı başlatılan grup hizmet veremez.
2. Her birinin çalışan bir üyesi olduğunda, **router** oluşturulur. En son oluşturulur çünkü komut satırı eşlerinin adreslerini içerir ve bu adresler eşler başlatılana kadar mevcut değildir.
3. Yönlendirici açıldığında grubun uç noktası kaydedilir. Yalnızca yönlendirici kayıtlıdır; önceden doldurma ve kod çözme üyeleri, kendi bağlantı noktalarında OpenAI şeklindeki bir API'ye hizmet eder, ancak hiçbiri bir isteğin tamamını tek başına yanıtlayamaz.

Bu nedenle bir grup servis yapmadan önce bir süre `Partially ready` bildirir ve bu bir hatadan ziyade normal yoldur.

## Bir grubun durumunu okuma { #reading-a-groups-state }

| Ne görüyorsun | Ne anlama geliyor |
| --- | --- |
| `Pending` | Henüz üye yok. |
| `Partially ready` | Üyeler hazır ve grup hâlâ hizmet veremiyor — hazır üyenin olmadığı bir rol veya yönlendiricinin uç noktası henüz kaydedilmemiş. |
| `Running` | Her rolün en az bir hazır üyesi vardır ve bitiş noktası kayıtlıdır. |
| `Ratio 2:1 (currently 1:1)` | Grup hizmet veriyor ancak istediğiniz oranda değil. Devletin yanında gösterilir, onun yerine değil. |
| Saat simgesi / `Config changed` | Çalışan üyeler gösterilen yapılandırmadan öncesine dayanır ve siz yeniden başlatana kadar bir rolün ölçeği artırılamaz. Bkz. [Yapılandırma değişikliği uygulama](#applying-a-configuration-change). |

İstenilen oranın altında olmak bir durum değil, bir bozulmadır: 2P1D çalıştıran bir 3P1D grubu hala hizmet vermektedir, ancak sizin ölçtüğünüz verimde değildir.

## Yapılandırma değişikliği uygulama { #applying-a-configuration-change }

Ayrıştırılmış dağıtımı düzenlemek onu **yeniden başlatmaz**. Değişiklik kaydedilir, grup eski yapılandırmayı çalıştırıyor olarak işaretlenir ve siz yeniden başlatana kadar hiçbir şey değişmez.

Bu bilinçli bir davranıştır. Üyeleri tek tek yeniden başlatmak, yeni yapılandırmalı prefill'i bir süre eski yapılandırmalı decode ile eşleştirir; motorlar bu eşleşmeyi kabul eder. Örneğin bağlam uzunlukları uyuşmasa bile el sıkışma ve KV aktarımı başarılı olur, kısa istemler doğru yanıtlanır. Yalnızca decode penceresinden uzun istem, prefill işi tamamlandıktan sonra başarısız olur. Yanlış yanıt yeniden başlatmadan daha kötü olduğundan herhangi bir üye yeniden başlamadan önce grubun tamamı durdurulur.

Değişikliği uygulamak için dağıtımda `Restart` kullanın. Grubun tamamı kaldırılıp güncel yapılandırmayla yeniden oluşturulur. Yalnızca bazı rolleri yeniden başlatma seçeneği sunulmaz; bu tam olarak yukarıdaki karma eşleşmeyi oluştururdu.

Ölçeklendirme, herhangi bir yapılandırma değişikliğinin beklemede olmaması koşuluyla, tüm grubu yeniden başlatmadan her rolün örnek sayısını değiştirir. Yönlendirici davranışı `pd-modes.yaml`'deki `router.membership_update`'a bağlıdır:

| Politika | Gönderilen modlar | Ölçeklendirme veya üye kurtarma davranışı |
| --- | --- | --- |
| `dynamic` | vLLM NIXL, SGLang Mooncake/NIXL, vLLM Ascend Mooncake | GPUStack, yönlendiriciyi üyelik API'si aracılığıyla günceller. |
| `restart` | vLLM Mooncake | GPUStack, eklenen GPU üyeleri hazır olduğunda yönlendiricinin yerini alır, başlangıç eşlerini ve önyükleme meta verilerini yeniler. Kaldırılan veya yeniden başlatılan üyeler, her iki tarafta da kullanılabilir bir eş bulunduğunda yenilemeyi tetikler. Hayatta kalan GPU örnekleri çalışmaya devam ediyor ancak hizmet kısa süreliğine kullanılamıyor ve yayın sırasındaki istekler başarısız olabilir. |
| `manual` | Özel | Operatör, yönlendirici eşlerini ve önyükleme meta verilerini günceller ve gerektiğinde yönlendiriciyi yeniden başlatır. |

Çalışma zamanı argüman yer tutucuları içermeyen statik Router komutu, üyeliği elle yönetir. Yer tutuculu komutlar yukarıdaki güncelleme kurallarını kullanır. Dağıtım formu ve replika düzenleyicisi katalogdaki politikayı gösterir. Örnek silinirken de kurtarma için yönlendirici değişimi veya operatör müdahalesi gerekiyorsa uyarı gösterilir.

Statik yönlendirici anlık görüntüleri, örnek kimliğini, yeniden başlatma oluşturmayı, HTTP adresini ve adlandırılmış bağlantı noktalarını içerir. Aynı adreste değiştirme veya yeniden başlatma bu nedenle Mooncake önyükleme meta verilerini yenilemeye devam eder. Anlık görüntüsü olmayan mevcut yönlendiriciler, eşleri hazır olduğunda bir kez değiştirilir.

**Kaydedilen değişiklik yeniden başlatmayı beklerken rolün replika sayısı artırılamaz.** Yeni üye düzenlenen yapılandırmadan oluşturulup eski yapılandırmalı üyelerle çalışarak yukarıdaki karma eşleşmeyi oluştururdu. Artırım, değişiklikle birlikte yeniden başlatma sırasında uygulanır.

**aşağı** ölçeklendirme, düzenleme beklemede olsun ya da olmasın, hemen uygulanır: bir üyeyi kaybetmek uyumsuz bir çift üretemez.

## Onunla paylaşılan bir KV önbelleği kullanma { #using-a-shared-kv-cache-with-it }

Ayrıştırılmış dağıtım ve paylaşılan KV önbelleği alternatif olmaktan çok tamamlayıcıdır ve GPUStack bunları sizin için birleştirir: her zamanki gibi bir önbellek hizmeti ekleyin ve iki konektör, motorun kabul ettiği tek bir yapılandırmaya katlanır.

Önbellek kullanan her vLLM rolü `MultiConnector[PD, Store]` oluşturur. Sarmalayıcı prefill'de `kv_producer`, decode'da `kv_consumer` kullanır. Her alt konnektör kendi aktarımının veya deposunun gerektirdiği rolü korur.

| Mod | Önbellek yok | İşlem içi önbellek | Önbellek hizmeti |
| --- | --- | --- | --- |
| `vllm-nixl` (NVIDIA varsayılanı) | `NixlConnector` | `NixlConnector` + `LMCacheConnectorV1` | Yalnızca önceden doldurma: `NixlConnector` + LMCache; Yalnızca LMCache |
| `vllm-mooncake` (NVIDIA) | `MooncakeConnector` | `MooncakeConnector` + `LMCacheConnectorV1` | `MooncakeConnector` + `MooncakeStoreConnector`; Sadece Mooncake |
| `vllm-ascend-mooncake` | Model seçilmiş Mooncake PD konnektörü | PD konnektörü + `SimpleCPUOffloadConnector` | PD konnektörü + `AscendStoreConnector`; Sadece Mooncake |

Süreç içi LMCache alt öğesi her iki rolde de `kv_both` kullanır. Paylaşılan LMCache, yalnızca NIXL için önceden doldurmada desteklenir. MP konektörü, MultiConnector isteğin yükünü NIXL'e atadığında bile önbelleğe alınmış bir öneki yükleyebilir; ikinci eşzamansız tamamlama kod çözmeyi çökertebilir ([vLLM #53049](https://github.com/vllm-project/vllm/issues/53049)). NVIDIA Mooncake Store, ön doldurmada `kv_both` ve kod çözmede `kv_consumer` kullanır. Ascend Store, ön doldurmada `kv_producer` ve kod çözmede `kv_consumer` kullanır. Bu konfigürasyonlar [vLLM 0.29 Mooncake Store](https://docs.vllm.ai/en/v0.29.0/features/mooncake_store_connector_usage/#disaggregated-prefill-decode-xpyd) ve [vLLM-Ascend 0.23 KV havuzu](https://docs.vllm.ai/projects/ascend/en/v0.23.0/user_guide/feature_guide/kv_pool.html)'u takip etmektedir.

Ascend, Qwen3, GLM-5 ve DeepSeek V3.1 dahil standart KV düzenleri için `MooncakeConnectorV1`'ı ve DeepSeek V4, V4 Flash ve V4 Pro gibi hibrit düzenler için `MooncakeHybridConnector`'i seçer. Seçim, dağıtım adı yerine sıkıştırılmış dikkat ve yinelenen katmanlar dahil olmak üzere önceden eğitilmiş yapılandırmayı okur. Her iki rol de aynı kayıtlı seçimi kullanır. Düzen meta verileri kullanılamıyorsa karma bağlayıcı, standart düzen geri dönüşünü yönetir. GPUStack, dağıtımın model girişleri için bu seçimi korur; böylece bir üye başladıktan sonra kullanılabilir hale gelen meta veriler, başka bir üyenin farklı bir bağlayıcı kullanmasını sağlayamaz.

NIXL ile önbellek birleştirme, vLLM 0.26'dan itibaren bulunan gerçek blok sözleşmesini kullanır. PD modlarının tanımları desteklenen önbellek sağlayıcılarını ve konnektör ayarlarını bildirir; önbellek hizmeti sürümlerini veya istemci paketi gereksinimlerini tanımlamaz ya da denetlemez. Hizmet ve istemci uyumluluğundan önbellek sağlayıcıları sorumludur. Runner bağımlılığı 0.5.4 olarak kalır. Her iki NVIDIA modu da paketlenen runner için vLLM 0.27.1 veya üstünü temel alır: bu CUDA runner'ları, motor imajından çalışan `vllm-router` 0.1.15'i içerir. Ascend modu aynı nedenle CANN 0.23 temelini kullanır. Bu alt sınırlar her konnektörün ilk upstream sürümünü değil, GPUStack'in paketlediği imajları tanımlar.

Paylaşılan Store kullanan NVIDIA Mooncake ayrıca 0.29 temel sürümünü gerektirir. GPUStack'in 0.29 runner'ları, reddedilen yükleme düzeltmesini içerir: Store alt konnektörü PD alt konnektörüne atanan yüklemeyi atlamalıdır. MultiConnector seçilmeyen Store'a sıfır harici token aktarır; düzeltme bekleyen yüklemeyi başlatmadan önce `can_load` kontrolü yapar. Bu, decode tarafındaki önbellek isabetleri için de geçerlidir. Standart vLLM 0.29'a bu düzeltmenin geri taşınması gerekir; upstream 0.30 düzeltmeyi içerir. Bkz. [runner düzeltmesi](https://github.com/gpustack/runner/commit/f0264ed).

Hizmeti ihtiyaç duyan her role ayrı bağlayın. Açılır menü hizmetleri moda, sağlayıcıya ve role göre filtreler; API bu uyumluluk kurallarını uygular. PD motor sürümü aralıkları tavsiye niteliğindedir: aralık dışındaki yerleşik sürüm seçildiğinde mod veya etkin önbellek için uyarı gösterilir, ancak dağıtıma ve önbellek eklenmesine izin verilir. Özel sürümlerde ve açıkça belirtilen imajlarda bu uyarılar gösterilmez. Başlatma hataları örnek durumu ve günlükleriyle bildirilir.

Otomatik seçim şu anda runner'ı arka uca, hızlandırıcı çalışma ortamına ve platforma göre belirler; sürümleri PD modu veya önbellek birleştirme aralıklarına göre filtrelemez. SGLang, LMCache veya MeshFusion'ı vLLM MultiConnector yerine bayraklar ve yapılandırma dosyasıyla bağlar.

NVIDIA Mooncake, çıkarım imajında aktarım motoru çalışma ortamını ve RDMA desteğini gerektirir. GPUStack runner imajları Mooncake aktarım/sona erme metriklerini de sunar; aktarım sayaçları prefill'den okunur. GPUStack benzersiz bir prefill bootstrap portu ayırıp yönlendiriciye iletir. vllm-router 0.1.15'te üyelik değişiklikleri bootstrap meta verilerini yenilemek için yönlendiriciyi yeniden başlatır; bu trafik akışını kısa süre kesebilir.

## Ağ gereksinimleri { #networking-requirements }

### Kubernetes'te: Pod'ların kümeden istediği şey { #on-kubernetes-what-the-pods-ask-the-cluster-for }

Bir grubun üyelerinin ana bilgisayar ağı, ana bilgisayar bağlantı noktası, ana bilgisayar IPC'si ve cihaz bağlantılarına ihtiyacı vardır — dördünün birden aynı anda olması, `baseline` Pod Güvenlik Giriş seviyesinin izin verdiğinden daha fazladır. GPUStack, sahip olduğu ad alanlarını buna göre etiketler; Kendi getirdiğiniz bir kümede bunun ne anlama geldiğini ve *kapsamadığı* konuları da görmek için [Ad Alanları ve Pod Güvenlik Girişi](cluster-management.md#namespaces-and-pod-security-admission)'ye bakın.

### KV Aktarım Arayüzü { #the-kv-transfer-interface }

KV önbelleği her istekte ağ üzerinden geçer ve geçtiği arayüz önemlidir. Kendi başına bırakıldığında, taşıma katmanı mutlu bir şekilde bir konteyner köprüsünü (`docker0`, `br-*`, `flannel.1`) alacak ve eşin ulaşamayacağı bir adresin reklamını yapacaktır; bu, bir yapılandırma hatası olarak değil, el sıkışma sırasında bir bağlayıcı hatası olarak başarısız olur.

GPUStack, çalışan başına arayüzü türetir:

- **Tek yönlendirilebilir arayüz** — kullanılır. Yapılandırılacak bir şey yok.
- **Birkaç yönlendirilebilir arayüz** — GPUStack tahmin etmeyi reddediyor. Söz konusu çalışandaki `--kv-transfer-ifname`'yi (veya `kv_transfer_ifname` çalışan yapılandırma anahtarını veya `GPUSTACK_KV_TRANSFER_IFNAME`) KV trafiğini taşıyan arayüze ayarlayın. Çalışan günlüğü bulduğu adayları listeler.

Bunu birden fazla yapıya sahip herhangi bir makinede açıkça ayarlayın — bir yönetim NIC'si ve bir RDMA NIC'si yaygın bir durumdur ve yönetim NIC'si neredeyse hiçbir zaman doğru cevap değildir.

### Bağlantı Noktaları { #ports }

Her rolün kendi HTTP bağlantı noktasına ve konektörünün gerektirdiği bağlantı noktalarına ihtiyacı vardır ve GPUStack bunları çalışanın hizmet bağlantı noktası aralığından bitişik bantlar olarak tahsis eder, böylece bir ana bilgisayardaki iki üye çarpışamaz.

Varsayılan aralık **64 bağlantı noktası** içerir; bu çoğu dağıtım için yeterlidir ancak hepsi için yeterli değildir: sıra başına bağlantı noktasına sahip bir konnektör kullanan tensör paralel-8 üyesi bunlardan dokuzunu tüketir. Ana bilgisayar başına bu türden birkaç üye çalıştırıyorsanız `--service-port-range`'yi genişletin.

!!! warning "GPUStack'in yönetemediği üç tür bağlantı noktası"

    Bazı konektörler, çalışma zamanında motor tarafından seçilen ve GPUStack'in okuyabileceği hiçbir yerde bildirilmeyen bağlantı noktalarını bağlar:

    - SGLang'ın ana bilgisayarın geçici menzilinden alınan rütbe limanı;
    - Mooncake'in kademe başına transfer motoru el sıkışma bağlantı noktaları;
    - Mooncake'in segment portları.

    Bunlarda yapılandırma dosyasındaki değer, gerçekten bağlanan port **değildir**. GPUStack bu portları ayıramaz veya ana bilgisayar portu olarak bildiremez. Bu portlar 15000–17000 ve 20000–20800 civarında gözlemlenmiştir; varsayılan hizmet aralığı (40000–40063) bunlarla çakışmaz. Çakışmayı önleyen bir rezervasyon değil, aralıkların ayrı olmasıdır.

    **Bu nedenle `--service-port-range` değerini bu aralıklara taşımayın.** Ölçülen bir durumda konnektör aralığı 20001'den başlıyordu. Üyenin ikinci rank'ı 20003'e bağlanmaya çalıştı; ancak aktarım motoru bu portu birkaç an önce kendisi için seçmişti. Aynı dağıtım varsayılan aralık kullanıldığında ilk denemede başladı.

    Bir üye tekrar tekrar yeniden başlarsa ve hiçbir zaman `Running`'a ulaşmazsa, motor günlüğünü `Address already in use` için kontrol edin.

    Port tekilleştirme aynı zamanda yalnızca bir çalışan sürecini kapsar. Aynı ana bilgisayardaki ikinci bir GPUStack çalışanı veya elle başlattığınız bir kapsayıcı onun dışındadır.

## Kısıtlamalar { #constraints }

Yanlış yanıtlar sunan bir grup oluşturacak yapılandırma, çalışma zamanından ziyade kaydetme zamanında reddedilir:

- **Ön doldurma ve kod çözme aynı bağlam uzunluğunu bildirmelidir.** Hiçbir motor bunu kontrol etmez ve uzun bir komut gelene kadar bir uyumsuzluk görünmez.
- **Kod çözmenin tensör paralelliği en azından önceden doldurmanınki olmalıdır.** Motor bunu iddia eder, ancak bunu bir yapılandırma sorunu yerine kod çözme içindeki dahili bir hata olarak bildirir.
- **Veri türü, KV önbellek veri türü, blok boyutu ve KV önbellek düzeni iki taraf arasında** aynı olmalıdır.
- **Bir rolün motoru, seçilen modun yapılandırabileceği bir motor olmalıdır.** Motorları roller arasında karıştırmak, bağlantı parametrelerinin sizin sağlayacağınız `custom` modunu gerektirir.
- **`custom` modunu kullanan ayrıştırılmış bir dağıtım aynı zamanda genişletilmiş bir KV önbelleği de kullanamaz.** Diğer modlardan her biri sizin için ikisini oluşturur (bkz. [Paylaşılan bir KV önbelleğini onunla kullanma](#using-a-shared-kv-cache-with-it)); `custom` hiçbir bağlayıcı yapılandırması eklemez, bu nedenle önbelleği oluşturacak hiçbir şey yoktur; birleştirilmiş yapılandırmayı kendiniz yazın.
- **Zamanlanmış ölçeklendirme ve ayrıştırma birleştirilemez.**

`vllm-nixl` için uyumluluk kontrolleri role özgü ortam değişkenlerini içerir. `VLLM_KV_CACHE_LAYOUT` eşdeğer takma adları kabul eder (`NHD`/`LBNHC` ve `HND`/`LBHNC`). MLA modellerinde `VLLM_MLA_DISABLE` uyumlu gizli KV temsilleri üretmelidir. NIXL'in yinelenen durum aktarımı, desteklenen çalışma zamanlarında `VLLM_SSM_CONV_STATE_LAYOUT=DS` gerektirir. Parçalanmış yerel dikkat için, `VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE` HMA bayraklarıyla birlikte işaretlenir. Açık dikkat arka uç seçenekleri de karşılaştırılır.

Ayarlanmamış bir ortama sahip bir rol, dağıtımın ortamını devralır; boş bir sözlük de dahil olmak üzere açık bir sözlük onun yerini alır. Modele bağlı kontroller, kaynak yapılandırmasını ve rol geçersiz kılmalarını kullanır. Bilinmeyen yapılandırmalar, çalışma zamanı sürümleri ve aktarım düzenleri `pairing_unverified` aracılığıyla raporlanır. Ağ adresleri, bağlantı noktaları, cihaz adları ve performans ayarları roller arasında farklılık gösterebilir. Deneysel düzen permütasyonu, motorun yönüne ve HMA kısıtlamalarına tabi olmaya devam eder.

## Sınırlamalar { #limitations }

- Yönlendirici tek bir kopya olarak çalışır.
- **`custom` yönlendirici tam olarak yazıldığı biçimde dağıtılır.** Yerleşik modlar upstream yönlendiricinin devre kesicisini varsayılan on başarısız istek eşiğine ayarlar; kendi sağladığınız yönlendiricinin kendi varsayılanları korunur.
- Paketle gelen runner imajını kullanan yönlendiriciler, `nofile` yumuşak sınırını mevcut sert sınırı aşmadan 65536'ya yükseltir. Yönlendirici rolünde açıkça belirtilen veya dağıtımdan devralınan bir `image_name`, kabuk içermeyen imajların da başlayabilmesi için doğrudan çalıştırmayı korur. Bu imajlar için `nofile` sınırını konteyner çalışma zamanı üzerinden yapılandırın.
- **vLLM modlarında, hataları döndüren bir ön doldurma üyesi rotasyonun dışına alınmaz.** Yukarı akış yönlendiricisi yalnızca bir aktarım hatasını bir ön doldurma hatası olarak ele alır, dolayısıyla ön doldurmadan bir HTTP 500 günlüğe kaydedilir ve istek kodu çözmeye devam eder ve 200 döndürür - on iki isteğin her birinde başarısız olan bir saplama ön doldurmasıyla ölçülür, hiçbiri devreyi açmaz. SGLang modları kesiciyi besler. Her iki durumda da, grubun panelindeki **PD Etkinliği** rakamı, katkıda bulunmayı bırakan bir üyeyi tespit eden rakamdır; yönlendiricinin kendi istek sayaçları bunu göstermez.
- **Grup boşta beklerken duran üye, önceden değil bir sonraki istekte fark edilir.** Tespit istek işleme yolunda gerçekleşir; trafik yoksa bu yoldan tespit de yapılmaz. Yönlendiricinin arka plan taraması dakikada bir çalışır. Boşta geçen sürenin ardından ilk istekler yeniden denenir ve ardından arızalı üye atlanarak yönlendirilir.
- Bir grubun tüm üyeleri aynı GPU türünü kullanmalıdır; grup karıştırma kartı türleri atomik olarak kabul edilemez.
- Roller önceden doldurma, kod çözme ve yönlendirmeyle sınırlıdır.
- **Öneki dikkate alan yönlendirme decode'u değil, prefill'i seçmek için kullanılır.** Paylaşılan önek prefill'de yarar sağladığından yerleşik modlar bu üyeyi yaklaşık önek eşleşmesine göre seçer. Decode, isteği tüm üretim boyunca tuttuğundan istekler dengeli ve dönüşümlü dağıtılır. `vllm-ascend-mooncake` dahil tüm yerleşik modlar açıkça `--prefill-policy cache_aware` ve `--decode-policy round_robin` ayarlar. Farklı yönlendirme politikası gerekiyorsa yönlendirici rolünün parametrelerinde bunları değiştirin.
