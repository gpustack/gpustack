---
translation_status: translated
---

# Model Rotaları Yönetme { #managing-model-routes }

Bu kılavuz, çeşitli yaygın kullanım durumlarını ve bunların yapılandırma yöntemlerini kapsayan model rotalarının nasıl kullanılacağını tanıtmaktadır.

## Modeli dağıtırken rota oluşturun { #create-route-when-deploying-model }

Modelleri dağıtırken `Enable Model Route` onay kutusu otomatik olarak etkinleştirilir. Bu dağıtım için aynı adla bir model rota oluşturacaktır. Bu, kullanıcıların dağıtımdan sonra aynı adı kullanarak modele doğrudan erişmesine olanak tanır.

## Model Takma Adıyla Model Yükseltme { #model-upgrade-with-model-alias }

Bir modelin yeni bir sürümü yayınlandığında yönetici, aynı model adını koruyarak modeli yükseltmek isteyebilir. Bu durumda yönetici, model rotası devre dışı bırakılarak modelin yeni bir sürümünü dağıtabilir ve mevcut model rotasını düzenleyerek trafiği değiştirebilir.

1. Modelin yeni sürümü için yeterli GPU kaynağını GPUStack'e ekleyin.
2. Modeli yeni sürümle dağıtın, model dağıtım çekmecesindeki `Enable Model Route` seçeneğinin işaretini kaldırın.
3. `Routes` sayfasında modelin eski sürümünü hedefleyen model rotasını bulun.
4. Bu rotayı düzenleyin ve hedefi yeni sürüm modeli dağıtımıyla değiştirin.
5. Bu model rotasına gelen yeni istekler, yeni sürüm modeline yönlendirilecektir.

## Kendi Kendine Barındırılan Modeller ve Genel MaaS Modelleri için Hizmet Modeli { #serve-model-for-self-host-models-and-public-maas-models }

Kendi kendine barındırılan bir model için istek hacmi arttığında gecikme meydana gelebilir. Ölçek büyütmek için kaynak yoksa Genel MaaS'ı kullanıma sunmak etkili bir çözümdür. Model rotasında hem dağıtım modeli hedefini hem de sağlayıcının model hedefini yapılandırarak ve ağırlıklar atayarak, mevcut modelin erişim yükünün üstesinden gelmeye yardımcı olması için Genel MaaS hizmetlerini kullanabilirsiniz.

1. `Providers` sayfasına gidin.
2. Gerektiğinde sağlayıcı ekleyin ve sağlayıcıda kullanılacak modelleri seçin.
3. `Routes` sayfasına gidin ve düzenlenecek modeli bulun.
4. Modeli düzenleyin ve bu model rotası için rota hedefleri ekleyin.
5. Hem GPUStack `Deployments`'deki modeller hem de `Providers`'deki modeller aynı rotada hedef olarak seçilebilir.
6. Her hedefin ağırlığı (varsayılan 100'dür), bu rotanın trafik yüzdesini belirler. Örneğin, A hedefinin ağırlığı 100 ve B hedefinin ağırlığı 200 ise isteklerin %33'ü A'ya ve %67'si B'ye yönlendirilecektir.

## Model Rota Geri Dönüşü { #model-route-fallback }

Bir model rotasına Genel MaaS modeli hedefi atamak uygun bir yaklaşım olsa da önemli maliyetlere de yol açabilir. Trafik dağıtım kuralları her zaman etkin olduğundan, şirket içinde barındırılan model ağır yük altında olmasa bile trafik, yapılandırmaya göre Genel MaaS'ye iletilecektir. Bu gibi durumlarda Model Rota `Fallback` özelliğini kullanmak çok etkili olabilir.

1. `Routes` sayfasına gidin ve geri dönüş ayarlamak istediğiniz model rotasını bulun.
2. Bu rotayı düzenleyin ve `Fallback Route Target`'ı yapılandırın. Diğer rota hedefleri gibi GPUStack `Deployments` veya `Providers`'den bir model olabilir.
3. Geri dönüş hedefi için, trafik dağıtım stratejisiyle birbirini dışlar, dolayısıyla bu hedef için ağırlıkları tasarım gereği yapılandıramazsınız.

## Model Rotası Aracılığıyla Proxy OpenAI Uyumlu Çıkarım Hizmeti { #proxy-openai-compatible-inference-service-via-model-route }

Çalışan bir çıkarım hizmeti (ollama veya lm-studio gibi) proxy oluşturma, erişim kontrolü ve belirteç kullanım istatistikleri için GPUStack'i kullanmak istiyorsa, barındırma için özel bir yol `OpenAI` Model Sağlayıcı oluşturabilirsiniz.

1. `Providers` sayfasına gidin ve `Add Provider` düğmesine tıklayın.
2. Tür olarak `OpenAI`'i seçin ve OpenAI uyumlu çıkarım sunucunuz için `Custom Base URL`'i `http://<ip>:<port>/v1` biçiminde ayarlayın. Adı, API anahtarını ve açıklamayı gerektiği gibi ayarlayın.
3. Bu sağlayıcı için modeller ekleyin. Çıkarım sunucusunun `/v1/models` API'yi desteklemesi durumunda mevcut modeller seçim için listelenecektir.
4. `Save` düğmesine tıklayın.
5. Bu sağlayıcı için `Operations` sütununda `Add Route` seçeneğine tıklayın.
6. Rotayı önceden yapılandırmak için sağlayıcının ilk modeli kullanılacaktır. Rota yapılandırmasını gerektiği gibi ayarlayın.
7. Model rotayı uygulamak için `Save` düğmesine tıklayın. Modeliniz artık GPUStack tarafından proxy olarak kullanılıyor.
8. `Operations` sütunundaki `Access Setting`'i kullanarak bu rotaya erişime izin verin.
