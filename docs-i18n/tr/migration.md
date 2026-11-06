---
translation_status: translated
---

# v0.7 ve Önceki Sürümlerden v2'ye Geçiş { #migrating-from-v07-and-earlier-versions-to-v2 }

!!! note

    v2.0.0'dan bu yana, GPUStack Worker resmi olarak yalnızca Linux'u desteklemektedir. Windows veya macOS kullanıyorsanız taşıma işlemini gerçekleştirmek için lütfen veri dizininizi bir Linux sistemine taşıyın.

    Windows ve macOS'ta GPUStack Sunucusu (katıştırılmış çalışan olmadan) Docker Desktop kullanılarak çalıştırılabilir.

## Geçişten Önce { #before-migration }

### Geriye Dönük Uyumluluğu Bozan Değişiklikler { #breaking-changes }

#### 1. Ollama Model Kaynağının Kaldırılması (v0.7.x'ten beri) { #1-removal-of-ollama-model-source-since-v07x }

- **Değişiklik:** Sürüm 0.7'den başlayarak, GPUStack artık model kaynağı olarak `ollama`'yi desteklememektedir.
- **Etki:** Kaynağı `ollama` olan Modeller, Model Dosyaları ve Model Örnekleri, yükseltme işlemi sırasında korunmayacaktır.
- **Gerekli Eylem:** v0.7'den önceki bir sürümden yükseltme yapıyorsanız ve halihazırda `ollama` kaynağından dağıtılan modeller varsa, yükseltmeden önce bu modelleri manuel olarak taşımalısınız.
  Etkilenen modellerin desteklenen kaynaklardan birini kullanarak yeniden dağıtılmasını öneririz:
    - Hugging Face
    - ModelScope
    - Yerel yol

    Yükseltmeyi başlatmadan önce modelleri **GPUStack UI** aracılığıyla yeniden dağıtarak bu geçişi gerçekleştirebilirsiniz.

### Verilerinizi Yedekleyin { #backup-your-data }

!!! warning

      **Önce Yedekleme:** Sunucu geçişine başlamadan önce veritabanınızı yedeklemeniz önemle tavsiye edilir.

      V0.7 veya önceki sürümlerdeki varsayılan kurulumlar için, GPUStack sunucusunu durdurun ve şu adresteki kapsayıcının içinde bulunan veri dizininin bir yedeğini oluşturun:

      ```
      /var/lib/gpustack
      ```

Lütfen taşıma işlemine başlamadan önce [Yükleme Gereksinimleri](./installation/requirements.md) bölümüne gidin.

V0.7.1'den önceki sürümlerde GPUStack **Docker** olmadan kullandıysanız (örneğin, pip kurulumu veya kurulum komut dosyası yoluyla), lütfen geçiş işlemine devam etmeden önce Docker Engine [Kurulum Kılavuzu](https://docs.docker.com/engine/install/)'i takip ederek Docker'ı yükleyin.

v0.7.1'den önce GPUStack'te çıkarım için GPU hızlandırmayı kullandıysanız, lütfen Docker'ı yükledikten sonra ilgili hızlandırıcı çalışma zamanının Container Toolkit'ini veya Container Runtime'ı yüklemeniz gerekip gerekmediğini kontrol edin. Bunları kontrol etmek ve yüklemek için **Kurulum Gereksinimleri** bölümündeki adımları takip edebilirsiniz.

## Geçiş Adımları { #migration-steps }

### Eski Veri Dizininizi Belirleyin { #identify-your-legacy-data-directory }

Önceki GPUStack kurulumunuz tarafından kullanılan veri dizinini bulun. Varsayılan yol:

```
/var/lib/gpustack
```

Diğer kurulum yöntemleri için, veri dizinini bulmak amacıyla bu [link](faq.md/#where-are-gpustacks-data-stored)'e bakın. Aşağıdaki adımlarda bu yola `${your-data-dir}` olarak başvurulmaktadır.

V2.0.0'daki mimari değişiklikler nedeniyle, `postgres` ve `s6-supervisor` gibi yeni bileşenler, root olmayan kullanıcılar kullanılarak `${your-data-dir}`'deki dosyalardan okuyabilir ve dosyalara yazabilir. `${your-data-dir}`'in tüm gerekli kullanıcılar tarafından erişilebilir olduğundan emin olun. İzinleri ayarlamak için aşağıdaki komut dosyasını kullanabilirsiniz, ancak lütfen güvenlik risklerinin farkında olun ve komut dosyasını çalıştırmadan önce tam olarak ne yaptığını anlayın.

```bash
dir="${your-data-dir}"
chmod a+rx "${dir}/log" || true
chmod a+rx $dir
while [ "$dir" != "/" ]; do
  chmod a+x "$dir"
  dir=$(dirname "$dir")
done
```

### Docker'ı Kullanarak Sunucuyu Taşıyın { #migrate-server-using-docker }

V2.0.0'dan bu yana, sunucu veya çalışan imajları için artık GPU hesaplama platformunu veya sürümünü (`-cpu`, `-cuda12.8` veya `-rocm` gibi) belirtmenize gerek yok. Bunun yerine birleştirilmiş resmi kullanın: `gpustack/gpustack:latest`.

#### Gömülü Veritabanı Geçişi (SQLite → PostgreSQL) { #embedded-database-migration-sqlite-postgresql }

v0.7 ve öncesinde GPUStack, yönetim verilerini depolamak için varsayılan olarak yerleşik bir SQLite veritabanı kullanıyordu. GPUStack, v2.0.0'dan itibaren SQLite desteğini kaldırdı ve artık gelişmiş performans ve ölçeklenebilirlik için varsayılan olarak yerleşik bir PostgreSQL veritabanı kullanıyor.

Gömülü veritabanı geçişini etkinleştirmek için GPUStack'i `GPUSTACK_DATA_MIGRATION=true` ile başlatın. `${your-data-dir}`'i orijinal SQLite veritabanını ve ilgili dosyaları içeren eski veri dizininizle değiştirin:

```bash
sudo docker run -d --name gpustack \
  --restart=unless-stopped \
  --privileged \
  --network=host \
  --env GPUSTACK_DATA_MIGRATION=true \
  --volume /var/run/docker.sock:/var/run/docker.sock \
  --volume ${your-data-dir}:/var/lib/gpustack \
  --runtime nvidia \
  gpustack/gpustack:latest
```

Hatayı docker `docker: Error response from daemon: unknown or invalid runtime name: nvidia`'dan alıyorsanız, lütfen önce [Accelerator Runtime Requirements](./installation/requirements.md#accelerator-runtime-requirements)'i kontrol edin ve `--runtime nvidia` argümanını çalışma zamanınızla değiştirmek için [Diğer GPU Mimarileri](#other-gpu-architectures) belgesini takip edin.

Veri dizininin `--data-dir` veya `GPUSTACK_DATA_DIR` aracılığıyla özelleştirilmesi de veritabanı geçişi sırasında desteklenir. Komutu aşağıdaki gibi ayarlayın:

```diff
...
-  --volume ${your-data-dir}:/var/lib/gpustack
+  --volume ${your-data-dir}:${your-data-dir}
...
+   gpustack/gpustack:latest \
+   --data-dir ${your-data-dir}
```

#### Harici Veritabanı Taşıma { #external-database-migration }

GPUStack, yönetim verilerini depolamak için harici bir veritabanı kullanılmasını destekler. Daha önce GPUStack'i harici bir veritabanıyla dağıttıysanız sunucuyu aşağıdaki komutla başlatın:

```bash
sudo docker run -d --name gpustack-server \
  --restart=unless-stopped \
  --privileged \
  --network=host \
  --volume /var/run/docker.sock:/var/run/docker.sock \
  --volume ${your-data-dir}:/var/lib/gpustack \
  --runtime nvidia \
  gpustack/gpustack \
  --database-url ${your-database-url}
```

### Docker'ı Kullanarak Çalışanları Taşıyın { #migrate-workers-using-docker }

Çalışan düğümleri için, `${your-data-dir}`'i eski çalışan veri dizini yolu ile değiştirin. Aşağıdaki komutu kullanın:

```bash
sudo docker run -d --name gpustack-worker \
  --restart=unless-stopped \
  --privileged \
  --network=host \
  --volume ${your-data-dir}:/var/lib/gpustack \
  --volume /var/run/docker.sock:/var/run/docker.sock \
  --runtime nvidia \
  gpustack/gpustack \
  --server-url ${server-url} \
  --token ${token}
```

Lütfen docker komutuna hem `--volume /var/run/docker.sock:/var/run/docker.sock` hem de `--runtime nvidia` eklendiğinden emin olun. Bunlar önceki sürüm için gerekli değildir. Farklı hızlandırıcı çalışma ortamları için, `--runtime nvidia`'den farklı bir seçenek kullanmak için [Diğer GPU Mimarileri](#other-gpu-architectures)'e bakın.

Bu, mevcut verilerinizi kullanarak GPUStack çalışanını başlatacak ve onu belirtilen sunucuya bağlayacaktır.

### Diğer GPU Mimarileri { #other-gpu-architectures }

NVIDIA dışındaki mimariler için (ör. AMD, Ascend) geçiş süreci aynı kalır. Lütfen GPU mimarinizin [Hızlandırıcı Çalışma Zamanı Gereksinimleri](./installation/requirements.md#accelerator-runtime-requirements) belgesinde desteklendiğini doğrulayın.

Örneğin, sunucuyu ve çalışanı ana bilgisayarda AMD GPU ile çalıştırmak için, docker run komutunu şununla değiştirin:

```diff
  sudo docker run -d --name gpustack \
  ...
- --runtime nvidia \
+ --runtime amd \
  ...
```

### Model Örneklerini Yeniden Oluşturma { #recreating-model-instances }

Yükseltme tamamlandıktan sonra mevcut Model Örnekleri `Starting` durumunda takılı kalabilir. Böyle bir durumda Model Örneğinin yeniden oluşturulması modelin normal şekilde çalışmasına olanak tanır.

### llama-box'tan geçiş { #migration-from-llama-box }

Önceki sürümlerde çıkarım arka ucu olarak llama-box'ı kullanıyorsanız, lütfen lllama-box'ın v2.0.0'dan itibaren desteklenmediğini unutmayın. Bunun yerine özel çıkarım arka ucu aracılığıyla llama.cpp'yi kullanın.

1. Çıkarım Arka Uç sayfasında `llama.cpp` özel bir arka uç oluşturun. Llama.cpp yapılandırması için bu [belge](./user-guide/inference-backend-management.md#enable-community-inference-backend)'ye bakın.
2. Dağıtım sayfasına gidin, orijinal olarak llama-box ile başlatılan modeli değiştirin, arka ucu llama.cpp olarak değiştirin
3. Kaydettikten sonra model örneğini yeniden oluşturun.

!!! Note

    Birden fazla çalışana dağıtılmış çıkarım şu anda özel çıkarım arka uçlarıyla desteklenmemektedir.

### Özel Arka Uç Sürümlerinden Geçiş { #migration-from-custom-backend-versions }

V2.0.0'dan önceki GPUStack sürümlerinde özel bir arka uç sürümü kullanıyorsanız lütfen bu sürümlerin, v2.0.0 itibarıyla **artık desteklenmeyen** Python sanal ortamlarına dayandığını unutmayın. Tüm çıkarım arka uçları artık kapsayıcıya alınmış ortamlarda çalışıyor.

Modellerinizi kullanmaya devam etmek için aşağıdaki yaklaşımlardan birini kullanarak onları **yeniden oluşturmanız** gerekir:

**Seçenek 1 - Yerleşik Arka Uç Sürümünü Kullan:**

GPUStack v2.0.0+, yerleşik çıkarım arka uçlarının birden fazla önceden yapılandırılmış sürümünü sağlar. Model dağıtımınızı yeniden oluşturun ve modelinizin gereksinimlerine en iyi uyan yerleşik arka uç sürümünü seçin.

**Seçenek 2 - Yerleşik Arka Uca Özel Sürüm Ekleme:**

Yerleşik sürümlerin hiçbiri ihtiyaçlarınızı karşılamıyorsa, özel bir sürüm ekleyerek yerleşik çıkarım arka ucunu genişletebilirsiniz. Ayrıntılı talimatlar için şu kılavuza bakın: [Yerleşik vLLM Çıkarımı Arka Ucuna Özel Bir Sürüm Ekleme](user-guide/inference-backend-management.md#example-add-a-custom-version-to-the-built-in-vllm-inference-backend).
