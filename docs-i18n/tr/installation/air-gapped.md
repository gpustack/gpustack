---
translation_status: translated
---

# Ağdan Yalıtılmış Kurulum { #air-gapped-installation }

GPUStack, internet erişimi olmayan, ağdan yalıtılmış (çevrimdışı) bir ortama kurulabilir.

Çalıştırıldığında bu dokümantasyonu kendisi sunar — aşağıdaki [Belgeler](#documentation)'e bakın.

## Önkoşullar { #prerequisites }

### Sürücü { #driver }

Sisteminizde donanımınız için uygun GPU sürücülerinin kurulu olduğundan emin olun.

Sürücü uyumluluğuna ilişkin ayrıntılar için [Kurulum Gereksinimleri](./requirements.md)'e bakın.

### Konteyner Çalıştırma Ortamı { #container-running-environment }

[Docker](https://docs.docker.com/engine/install/) kurulu olmalıdır.

Sisteminiz bir kapsayıcı araç setini destekliyorsa, bunu gerektiği şekilde kurun ve yapılandırın (örn. NVIDIA Container Toolkit, AMD ROCm Container Toolkit, vb.).

### Konteyner İmajları { #container-images }

GPUStack, kullanıcıların indirmek istedikleri imajları kolayca seçmelerine yardımcı olmak için bir [İmaj Seçici](../image-selector/index.html) sunar. Tamamen tarayıcıda bu belgelerle birlikte gönderilen verilere göre çalışır, dolayısıyla internet erişimi olmadan çalışır; sürüm listesi yalnızca birlikte gönderildiği sürüme kadar olan sürümleri kapsar. Daha gelişmiş veya otomatik senkronizasyon için GPUStack ayrıca imaj yönetimi komutları da sağlar:

- `gpustack copy-images`: İmajları bir kayıt defterinden diğerine senkronize edin
- `gpustack save-images`: İmajları indirin ve yerel olarak kaydedin
- `gpustack load-images`: Yerel paketlerden imajları içe aktarın
- `gpustack list-images`: Geçerli sürüm için imajların manifestosunu göster

Aşağıda bu CLI komutlarının nasıl kullanılacağına ilişkin detaylar yer almaktadır.

- **İmajları Kopyala**

GPUStack, farklı bileşenler ve çıkarım arka uçları için [Docker Hub](https://hub.docker.com/u/gpustack) ve [Quay.io](https://quay.io/user/gpustack/)'de bulunan çeşitli konteyner imajları sağlar.

Gerekli konteyner imajlarıni internet erişimi olan bir makineden dahili kayıt defterinize aktarmak için GPUStack `copy-images` komutunu kullanın:

```bash
sudo docker run --rm -it --entrypoint "" gpustack/gpustack \
    gpustack copy-images \
    --destination <your_internal_registry> \
    --destination-username <your_username> \
    --destination-password <your_password>
```

`Docker Hub`'den imaj alamıyorsanız veya indirme işlemi çok yavaşsa, kaynak kayıt defterini `quay.io`'e işaret ederek `Quay.io` aynamızı kullanabilirsiniz:

```bash hl_lines="3"
sudo docker run --rm -it --entrypoint "" gpustack/gpustack \
    gpustack copy-images \
    --source quay.io \
    --destination <your_internal_registry> \
    --destination-username <your_username> \
    --destination-password <your_password>
```

!!! note

    Bu, varsayılan olarak en son sürümü kullanır. Belirli bir sürümü hedeflemek için tam imaj etiketini kullanın (ör. gpustack/gpustack:vx.y.z).

`copy-images` hakkında daha fazla ayrıntı için [CLI Referansı](../cli-reference/copy-images.md)'ye bakın.

- **İmajları Listele**

Dahili kayıt defterinize doğrudan erişemiyorsanız, önce `gpustack/gpustack` imajını çekebilir, ardından hangi imajların indirilmesi gerektiğini görmek için `list-images` komutunu kullanabilirsiniz:

```bash
sudo docker run --rm -it --entrypoint "" \
    gpustack/gpustack \
    gpustack list-images
```

!!! note

    Bu, varsayılan olarak en son sürümü kullanır. Belirli bir sürümü hedeflemek için tam imaj etiketini kullanın (ör. gpustack/gpustack:vx.y.z).

Görüntülenen imaj listesi, desteklenen tüm hızlandırıcıları, çıkarım arka uçlarını, sürümleri ve mimarileri içerir. Yalnızca bir alt kümeye ihtiyacınız varsa filtreleme seçenekleri için [CLI Referansı](../cli-reference/list-images.md)'e bakın.

- **İmajları Kopyala ve Yükle**

Hedef ortamınız ağdan yalıtılmış ise veya internet erişimi yoksa, öncelikle gerekli imajları internet bağlantısı olan bir makineye indirebilir, daha sonra çevrimdışı ortama aktarıp yükleyebilirsiniz.

GPUStack, bu iş akışı için `save-images` ve `load-images` komutlarını sağlar.

**İmajları Kopyala**

Gerekli imajları indirip paketlemek için internete erişebilen bir makinede aşağıdaki komutu çalıştırın:

```bash
sudo docker run --rm -it --entrypoint "" \
 -v ./gpustack-air-gapped:/gpustack-air-gapped \
 --workdir /gpustack-air-gapped \
 gpustack/gpustack \
 gpustack save-images \
 --platform linux/amd64 \
 --backend cuda \
 --backend-version 12.9 \
 --service vllm \
 --service-version 0.15.1 \
 --max-workers 3 \
 --max-retries 3
```

Bu komut, belirtilen platform, arka uç ve hizmet yapılandırmasına göre gerekli tüm kapsayıcı imajlarıni indirir ve bunları `gpustack-air-gapped` dizini altında yerel paketler olarak kaydeder. Örnek komut, imajların yalnızca bir alt kümesini indirmek için filtreleme seçeneklerinin nasıl uygulanacağını gösterir.

!!! note

    Bu, varsayılan olarak en son sürümü kullanır. Belirli bir sürümü hedeflemek için tam imaj etiketini kullanın (ör. gpustack/gpustack:vx.y.z).

Filtreleri yalnızca ihtiyacınız olan imajları indirecek şekilde ayarlayabilirsiniz. Mevcut tüm seçenekler için [CLI Referansı](../cli-reference/save-images.md)'e bakın.

İndirme işlemi tamamlandıktan sonra, oluşturulan dizini tercih ettiğiniz yöntemi (örneğin USB sürücü veya dahili dosya aktarımı) kullanarak ağdan yalıtılmış ortamınıza aktarın.

**İmajları Yükle**

Hedef makinede, kaydedilen imajları yerel konteyner çalışma zamanına yükleyin:

```bash
sudo docker run --rm -it --entrypoint "" \
    --volume /var/run/docker.sock:/var/run/docker.sock \
    --volume ./gpustack-air-gapped:/gpustack-air-gapped \
    --workdir /gpustack-air-gapped \
    gpustack/gpustack \
    gpustack load-images \
    --platform linux/amd64 \
    --max-workers 3 \
    --max-retries 3 \
    /gpustack-air-gapped
```

Bu komut, tüm imaj paketlerini belirtilen dizinden yerel Docker arka plan programına aktararak onları GPUStack için kullanılabilir hale getirir.

!!! note

    Bu, varsayılan olarak en son sürümü kullanır. Belirli bir sürümü hedeflemek için tam imaj etiketini kullanın (ör. gpustack/gpustack:vx.y.z).

`load-images` hakkında daha fazla ayrıntı için [CLI Referansı](../cli-reference/load-images.md)'ye bakın.

## Kurulum { #installation }

Gerekli imajlarla dahili konteyner kayıt defterini hazırladıktan sonra ağdan yalıtılmış ortamda GPUStack kurulumunu yapabilirsiniz.

```diff
 sudo docker run -d --name gpustack \
     --restart unless-stopped \
     -p 80:80 \
     --volume gpustack-data:/var/lib/gpustack \
-    gpustack/gpustack
+    <your_internal_registry>/gpustack/gpustack \
+    --system-default-container-registry <your_internal_registry>

```

### Güvenli Kayıt Defterinden Çıkarım Arka Uç İmajlarıni Çekme { #pulling-inference-backend-images-from-a-secure-registry }

Dahili konteyner kayıt defteriniz kimlik doğrulama gerektiriyorsa,
GPUStack çalışanını çalıştırıcı imajını çekmesine izin vermek için başlatırken aşağıdaki ortam değişkenlerini ayarlayın.

```diff
 sudo docker run -d --name gpustack \
     ...
+    --env GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY_USERNAME=<your_internal_registry_username> \
+    --env GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_REGISTRY_PASSWORD=<your_internal_registry_password> \
     <your_internal_registry>/gpustack/gpustack \
     --system-default-container-registry <your_internal_registry>

```

### Varsayılan olmayan Ad Alanından Çıkarım Arka Uç İmajlarıni Çekme { #pulling-inference-backend-images-from-non-default-namespace }

Dahili kapsayıcı kayıt defteriniz varsayılan `gpustack`'den farklı bir ad alanı kullanıyorsa,
GPUStack çalışanını çalıştırırken koşucu imajını çekmesine izin vermek için aşağıdaki ortam değişkenini ayarlayın.

```diff
 sudo docker run -d --name gpustack \
     ...
+    --env GPUSTACK_RUNTIME_DEPLOY_DEFAULT_CONTAINER_NAMESPACE=<your_namespace> \
     <your_internal_registry>/gpustack/gpustack \
     --system-default-container-registry <your_internal_registry>
```

## Belgeler { #documentation }

Sunucu bu belgeleri taşır ve `/help/`'de sunar, dolayısıyla ağdan yalıtılmış bir ortamın çevrimiçi siteye erişmesine gerek yoktur:

```
http://<your_gpustack_server>/help/
```

Çevrimiçi siteyle aynı belgedir, sunucu imajının oluşturulduğu sürümdedir ve araması çevrimdışı çalışır.
