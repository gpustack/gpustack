---
translation_status: translated
---

# Docker aracılığıyla kurulum { #installation-via-docker }

## Önkoşullar { #prerequisites }

**GPUStack sunucusu:**

- [Docker](https://docs.docker.com/engine/install/) kurulu olmalıdır. Docker Desktop (Windows ve macOS) da desteklenmektedir.

**GPUStack çalışanları:**

- [Docker](https://docs.docker.com/engine/install/) kurulu olmalıdır. Docker Desktop **desteklenmez**.
- GPUStack çalışan düğümleri için yalnızca Linux desteklenir. Windows kullanıyorsanız WSL2 kullanmayı düşünün ve Docker Desktop'ı kullanmaktan kaçının. macOS, GPUStack çalışan düğümleri için desteklenmez.
- Donanımınız için uygun GPU sürücülerinin ve konteyner araç setlerinin kurulu olduğundan emin olun. Ayrıntılar için [Kurulum Gereksinimleri](./requirements.md)'e bakın.

## GPUStack Sunucusunu Kurun { #install-gpustack-server }

GPUStack sunucusunu Docker kullanarak kurmak ve başlatmak için aşağıdaki komutu çalıştırın:

```bash
sudo docker run -d --name gpustack \
    --restart unless-stopped \
    -p 80:80 \
    --volume gpustack-data:/var/lib/gpustack \
    gpustack/gpustack
```

!!! note

    GPUStack v2, tüm GPU cihaz türleri için tek bir birleştirilmiş kapsayıcı görüntüsü kullanır.

## Başlangıç { #startup }

GPUStack kapsayıcı günlüklerini kontrol edin:

```bash
sudo docker logs -f gpustack
```

Sunucu açıldığında, GPUStack kullanıcı arayüzüne erişmek için tarayıcıda `http://your_host_ip` öğesini açın.

Kullanıcı adı `admin` ve varsayılan şifreyle giriş yapın. İlk şifreyi şununla alın:

```bash
sudo docker exec -it gpustack \
    cat /var/lib/gpustack/initial_admin_password
```

## GPU Kümeleri ve Çalışan Düğümleri Ekleme { #add-gpu-clusters-and-worker-nodes }

GPU kümeleri ve çalışan düğümleri eklemek için lütfen `Clusters` ve `Workers` sayfalarındaki kullanıcı arayüzü talimatlarını izleyin.

## Özel Yapılandırma { #custom-configuration }

Aşağıdaki bölümlerde GPUStack sunucu kapsayıcısını başlatırken özel yapılandırma seçeneklerinin örnekleri açıklanmaktadır. Mevcut seçeneklerin tam listesi için [CLI Referansı](../cli-reference/start.md)'e bakın.

### HTTPS'yi Özel Sertifikayla Etkinleştirin { #enable-https-with-custom-certificate }


```diff
 sudo docker run -d --name gpustack \
     ...
     -p 80:80 \
+    -p 443:443 \
     --volume gpustack-data:/var/lib/gpustack \
+    --volume /path/to/cert_files:/path/to/cert_files:ro \
+    -e GPUSTACK_SSL_KEYFILE=/path/to/cert_files/your_domain.key \
+    -e GPUSTACK_SSL_CERTFILE=/path/to/cert_files/your_domain.crt \
     gpustack/gpustack
     ...
```

`GPUSTACK_SSL_CERTFILE` sunucu sertifikasını ve ardından herhangi bir ara sertifikayı içermelidir. Sunucu sertifikası özel bir CA tarafından imzalandığında, çalışan ve kıyaslama kapsayıcılarının sunucuyu otomatik olarak doğrulayabilmesi için PEM CA paketini de ekleyin:

```diff
 sudo docker run -d --name gpustack \
     ...
+    -e GPUSTACK_SSL_CA_CERTFILE=/path/to/cert_files/ca-bundle.crt \
     gpustack/gpustack
     ...
```

Kendinden imzalı bir sunucu sertifikası için, ayrı bir CA paketi yapılandırılmadığında sertifika dosyasının kendisi kullanılır. Genel CA'lar tarafından verilen sertifikalar ek CA yapılandırması gerektirmez.

Kayıt komutu, önyükleme CA paketinin SHA-256 sağlama toplamını içerir. Çalışan, TLS doğrulaması olmadan indirilen bir paketi kurmadan önce bu sağlama toplamını kontrol eder. Bu, sertifika sabitleme değildir: Sunucu zaten normal TLS doğrulamasını geçiyorsa, çalışan, önyükleme paketini indirmeden veya kontrol etmeden mevcut güven deposunu kullanır.

Çalışan ilk önce doğrulanmış CA'yı sistemin güven deposuna yüklemeye çalışır. Bu güven deposu güncellenemiyorsa veya çalışanın TLS istemcisi farklı bir paket kullanıyorsa çalışan, süreci ve alt süreçleri için `SSL_CERT_FILE` aracılığıyla geçici birleştirilmiş paket kullanır. Operatör tarafından sağlanan mevcut sertifikalar korunur; enjekte edilen önyükleme CA'sının değiştirilmesi, daha önce enjekte edilen paketleri korumaz.

Son çare olarak, bir çalışan üzerinde sunucu sertifikası hiçbir şekilde doğrulanamadığında, o çalışanın sunucuya bağlantısında sertifika doğrulamasını atlamak için `--insecure-tls` (veya `GPUSTACK_INSECURE_TLS=true`) ile başlatın. Trafik şifreli kalır ancak artık müdahaleye karşı korunmaz; bu nedenle, trafiği yalnızca güvenilir ağlarda kullanın. Bkz. [CLI Referansı](../cli-reference/start.md).

### Harici Veritabanı Kullanma { #using-an-external-database }

Varsayılan olarak GPUStack yerleşik bir PostgreSQL veritabanı kullanır. PostgreSQL veya MySQL gibi harici bir veritabanı kullanmak için `GPUSTACK_DATABASE_URL` ortam değişkenini ayarlayın veya GPUStack kapsayıcısını başlatırken `--database-url` bağımsız değişkenini kullanın. Uyumlu veritabanlarının ve doğrulanmış sürümlerin listesi için [Veritabanı Gereksinimleri](requirements.md#database-requirements)'e bakın.

```diff
 sudo docker run -d --name gpustack \
     ...
     --volume gpustack-data:/var/lib/gpustack \
+    -e GPUSTACK_DATABASE_URL="postgresql://username:password@host:port/dbname" \
     gpustack/gpustack
     ...
```

### Harici Sunucu URL'sini Yapılandır { #configure-external-server-url }

Çalışanları hazırlamak için bir bulut sağlayıcı kullanıyorsanız çalışanların sunucuya doğru şekilde bağlanabilmesini sağlamak amacıyla çalışan kaydı için harici sunucu URL'sini ayarlayın.

```diff
sudo docker run -d --name gpustack \
    ...
+   -e GPUSTACK_SERVER_EXTERNAL_URL="https://your_external_server_url" \
    gpustack/gpustack
    ...
```

### Ek Güvenilir CA'lar { #additional-trusted-cas }

GPUStack'in özel veya kurumsal bir CA tarafından verilen sertifikaları kullanan hizmetlerle iletişim kurması gerekiyorsa (örneğin, kendi kendine barındırılan bir Kimlik Sağlayıcı, Hugging Face aynası veya dahili bir API uç noktası), CA sertifikasını `/usr/local/share/ca-certificates/` altındaki kapsayıcıya ekleyin. GPUStack, başlangıç ​​sırasında bağlanan CA sertifikalarını otomatik olarak içe aktaracak ve bunları sistemin güven deposuna ekleyecektir.

```diff
 sudo docker run -d --name gpustack \
     ...
     --volume gpustack-data:/var/lib/gpustack \
+    --volume /path/to/custom-root-ca.crt:/usr/local/share/ca-certificates/custom-root-ca.crt:ro \
     gpustack/gpustack
     ...
```

!!! note

    CA sertifikası, `.crt` uzantılı PEM kodlu olmalıdır. Ek `--volume` bayrakları ekleyerek birden fazla CA sertifikası bağlayabilirsiniz.

## Docker Compose aracılığıyla kurulum { #installation-via-docker-compose }

### Önkoşullar { #prerequisites_1 }

- [Docker Compose](https://docs.docker.com/compose/install/) kurulu olmalıdır.
- [Gerekli bağlantı noktaları](./requirements.md#port-requirements) mevcut olmalıdır.

### Dağıtım { #deployment }

Docker Compose dosyaları ve yapılandırma dosyaları [GPUStack deposunda](https://github.com/gpustack/gpustack/tree/main/docker-compose) tutulur.

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

GPUStack sunucusunu başlatın:

```bash
sudo docker compose -f docker-compose.server.yaml up -d
```

Sunucu açıldığında, GPUStack kullanıcı arayüzüne erişmek için tarayıcıda `http://your_host_ip` öğesini açın.

Kullanıcı adı `admin` ve varsayılan şifreyle giriş yapın. İlk şifreyi şununla alın:

```bash
sudo docker exec -it gpustack-server cat /var/lib/gpustack/initial_admin_password
```

Yerleşik ve harici gözlemlenebilirlik seçenekleri için bkz. [Gözlemlenebilirlik](../user-guide/observability.md).
