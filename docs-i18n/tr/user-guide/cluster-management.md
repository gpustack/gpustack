---
translation_status: translated
---

# Küme Yönetimi { #cluster-management }

GPUStack, küme tabanlı çalışan yönetimini destekler ve birden fazla küme türü sağlar. `DigitalOcean` gibi bir `Cloud Provider` aracılığıyla bir kümenin temel hazırlığını yapabilir veya şirket içinde barındırılan bir küme oluşturabilir ve `Docker` çalıştırma komutlarını kullanarak çalışanlar ekleyebilirsiniz. Alternatif olarak, şirket içinde barındırılan bir `Kubernetes` kümesindeki tüm düğümleri GPUStack çalışanları olarak kaydedebilirsiniz.

## Küme Oluştur { #create-cluster }

1. `Clusters` sayfasına gidin.
2. `Add Cluster` düğmesine tıklayın.
3. Bir küme sağlayıcısı seçin. `Self-Host` sağlayıcısı için `Docker` ve `Kubernetes` ve `Cloud Provider` için `DigitalOcean` ve `SHUIHUA FUTURE` vardır.
4. Sağlayıcıya bağlı olarak `Base Configuration` ve `Add Worker` adımlarında farklı seçeneklerin ayarlanması gerekir.
5. Temel Yapılandırmadaki `Advanced` küme ayarları, `Worker Configuration YAML`'yi kullanarak çalışan seçeneklerini önceden yapılandırmanıza olanak tanır.

### Docker Kümesi Oluşturun { #create-docker-cluster }

1. `Basic Configuration` adımında `Name` alanı zorunludur, `Description` ise isteğe bağlıdır.
2. `Save`'a tıklayın.
3. `Add Worker` adımında, `docker run` komutu aracılığıyla bir çalışan eklemeden önce bazı seçeneklere ve doğrulamalara ihtiyaç vardır.
4. `Select the GPU vendor`. Test edilen sağlayıcılar arasında `Nvidia`, `AMD` ve `Ascend` bulunmaktadır. Deneysel satıcılar arasında `Hygon`, `Moore Threads`, `Iluvatar`, `Cambricon` ve `Metax` bulunmaktadır. Bir satıcı seçtikten sonra `Next`'a tıklayın.
5. `Check Environment`. Ortamınızın bir çalışan eklemeye hazır olduğunu doğrulamak için bir kabuk komutu sağlanır. Komut dosyasını kopyalayın ve ortamınızda çalıştırın. Komut dosyası Tamam'a döndükten sonra `Next`'e tıklayın.
6. `Specify arguments` eklenecek çalışan için. Aşağıdaki argümanları sağlayın ve `Next`'e tıklayın:
   - `Specify the worker IP` veya çalışanın `Auto-detect the Worker IP` olmasına izin verin. Çalışan IP'sine sunucudan erişilebildiğinden emin olun.
   - Çalışan konteyneri için bir `Additional Volume Mount` belirtin. Montaj yolu mevcut model dosyalarını yeniden kullanmak için kullanılabilir.

7. `Run command` çalışan konteynerini oluşturmak ve başlatmak için. Bash betiğini kopyalayın ve ortamınızda çalıştırın.

Cluster oluşturulduktan sonra çalışan de eklenebilir.

1. `Clusters` sayfasına gidin.
2. Çalışan eklemek istediğiniz kümeyi bulun.
3. İşlemler sütunundaki üç nokta düğmesine tıklayın, ardından `Add Worker`'ı seçin.
4. Çalışan ekleme seçeneklerini seçin. `Select the GPU vendor`'den `Run command`'e kadar yukarıdaki adımların aynısını takip edin.

### Kubernetes Kümesini Kaydedin { #register-kubernetes-cluster }

1. `Basic Configuration` adımında `Name` alanı zorunludur, `Description` ise isteğe bağlıdır.
2. `Save`'a tıklayın.
3. `Select the GPU vendor`. Test edilen sağlayıcılar arasında `Nvidia`, `AMD` ve `Ascend` bulunmaktadır. Deneysel satıcılar arasında `Hygon`, `Moore Threads`, `Iluvatar`, `Cambricon` ve `Metax` bulunmaktadır. `Kubernetes` kümeler için birden fazla GPU satıcısı seçebilir veya yalnızca CPU kümeleri için hiçbirini seçebilirsiniz; seçilen satıcı başına bir çalışan DaemonSet oluşturulur ve her DaemonSet'in `nodeSelector`'i, bildirim zamanında satıcının PCI varlığı etiketinden türetilir. Satıcıyı/satıcıları seçtikten sonra `Next`'e tıklayın.
4. `Check environment`. Ortamınızın bir çalışan eklemeye hazır olduğunu doğrulamak için bir kabuk komutu sağlanır. Komut dosyasını kopyalayın ve ortamınızda çalıştırın. Komut dosyası Tamam'a döndükten sonra `Next`'e tıklayın.
5. `Run command` Çalışan manifestolarını uygulamak. Bash betiğini kopyalayın ve `kubectl`'in kurulu olduğu ve `kubeconfig`'in yapılandırıldığı bir ortamda çalıştırın.

Kubernet'ler, küme oluşturulduktan sonra kaydedilebilir.

1. `Clusters` sayfasına gidin.
2. Kubernetes kümesine kayıt etmek istediğiniz kümeyi bulun.
3. İşlemler sütunundaki üç nokta düğmesine tıklayın, ardından `Register Cluster`'ı seçin.
4. Kümeyi kaydetme seçeneklerini seçin. `Select the GPU vendor`'den `Run command`'e kadar yukarıdaki adımların aynısını izleyin.

#### Kubernetes Küme Seçenekleri { #kubernetes-cluster-options }

Bir `Kubernetes` kümesi oluştururken veya düzenlerken, `Basic Configuration` adımında `Name` ve `Description`'e ek olarak aşağıdaki seçenekler mevcuttur:

- `Cluster Type` (gerekli) — kümenin nasıl kullanılacağını seçin:
    - `Model Service` — LLM çıkarımı ve API sunumu için, ör. Model API'lerini ve token tabanlı hizmetleri açığa çıkarıyor.
    - `GPU Service` — isteğe bağlı GPU hesaplaması için, ör. etkileşimli geliştirme, eğitim işleri veya özel ortamlar.

`Advanced` ayarları aşağıdaki Kubernetes dağıtım seçeneklerini ortaya çıkarır:

- `Namespace` — kümenin bildirimlerinin işlendiği Kubernetes ad alanı. `gpustack-system` kullanmak için boş bırakın.
- `Volume Mounts` — her çalışan bölmesine ekstra hacimler monte edilir. Her binek için bir `Volume Name`, `Container Path` ve `Read Only` bayrağı belirtin, ardından bir `Source Type` seçin:
    - `Host Path` — düğümde `Path Type` olan bir yol (örneğin `Directory`, `Directory (create if not exists)`, `File`, `Socket`, `Character Device`, `Block Device`).
    - `Persistent Volume Claim (PVC)` — mevcut bir `PVC Name`, isteğe bağlı olarak salt okunur.
    - `ConfigMap` — a `ConfigMap Name`, isteğe bağlı olarak isteğe bağlı olarak işaretlenir.
- `Image Credentials` — GPUStack görüntülerini özel bir kayıt defterinden çekmek için kullanılan görüntü çekme sırları. Her giriş için bir `Registry`, `Username` ve `Password` belirtin.
- `Node Selector` — bu kümenin GPUStack iş yüklerini, etiketleri eşleşen düğümlerle sınırlar. Grafiğin `global.nodeSelector`'si olarak iletilir ve her bir bileşen buna geri döner, böylece çalışan DaemonSets'e ve operatöre ve onu onurlandıran operatör sürümünden operatörün dağıttığı bileşenlere (Kueue, Node Feature Discovery'nin kontrol düzlemi, CSI denetleyicileri) ulaşır. İki tür iş yükü bilerek dışarıda bırakıldı: Hizmet verdikleri düğümleri kapsaması gerekenler (Node Feature Discovery'nin etiketleyicisi, CSI düğüm eklentileri), çünkü bunları sınırlamak diğer tüm düğümleri etiketsiz bırakacaktır - çalışanların kendilerinin seçtiği etiketler dahil - ve sürüm ortaya çıkmadan önce çalışan ve bu nedenle sürümün kendisinin uygulayacağı bir etiketi bekleyemeyen grafiğin yükleme kancaları. Çalışan bir DaemonSet, bunların yanı sıra GPU satıcı etiketini de korur; bir düğümün her ikisiyle de eşleşmesi gerekir.
- `Default Container Registry` — bu küme için GPUStack görüntülerini çözümlemek için kullanılan varsayılan kayıt defteri. Ayarlanmadığında sunucu varsayılanına geri döner (yer tutucu `docker.io`).
- `Operator Image` — GPUStack Operatörü kapsayıcı görüntüsü için geçersiz kılma. Sunucu varsayılanını kullanmak için boş bırakın.
- `GPU Service Static Access Address` — yalnızca `Cluster Type` `GPU Service` olduğunda gösterilir. Operatörün bu kümedeki GPU örneklerine erişmek için kullandığı statik adres (ör. LoadBalancer VIP). İsteğe bağlı.
- `Worker Configuration YAML` — aşağıdaki [Çalışan Yapılandırması YAML](#worker-configuration-yaml)'ye bakın.

#### Ad Alanları ve Pod Güvenlik Girişi { #namespaces-and-pod-security-admission }

GPUStack, sahip olduğu ad alanlarını oluşturur - `gpustack-system` (veya `Namespace` ne şekilde ayarlanmışsa) ve kümeyi kullanan kuruluş başına bir `gpustack-<org>` - ve bunların her birini `privileged` düzeyinde [Pod Güvenlik Girişi](https://kubernetes.io/docs/concepts/security/pod-security-admission/) için etiketler:

```yaml
pod-security.kubernetes.io/enforce: privileged
pod-security.kubernetes.io/audit: privileged
pod-security.kubernetes.io/warn: privileged
```

**Neden gereklidir.** Pod Güvenlik Girişi Kubernetes'te yerleşiktir ve bir Pod *oluşturulduğunda* uygulanır*: seviyeye uymayan bir Pod, Beklemede bırakmak yerine doğrudan reddedilir. Model, önbellek hizmeti ve kıyaslama Pod'ları, `baseline` ve `restricted` seviyelerinin yasakladığı şeylere yasal olarak ihtiyaç duyar:

| Gereksinim | Neye ihtiyaç var |
| --- | --- |
| Ana bilgisayar ağı | RDMA, GID'sini bir NIC adresine bağlar |
| hostPort | yan kanal, ayrıştırılmış bir çift, eşini verir |
| Ana Bilgisayar IPC'si | KV önbellek arabelleklerinin CUDA-IPC paylaşımı |
| Cihaz montajları | hızlandırıcıların kendileri |

Bir ön doldurma/kod çözme dağıtımı aynı anda dördüne de ihtiyaç duyar, bu nedenle etiket olmadan ilk ve en zor şekilde başarısız olan durumdur.

**Neden üç anahtar da.** Küme varsayılanında `warn` ve `audit` bırakıldığında, her Pod oluşturma işlemi hala bir uyarı döndürür ve bir denetim ek açıklaması yazar. Bu, muafiyetin beklendiği bir ad alanındaki gürültüdür ve başka yerlerde önemli olan uyarıları gizler.

**Patlama yarıçapı.** `privileged`, bu ad alanlarındaki her Pod'un Pod Güvenlik Girişinden muaf olduğu anlamına gelir. Bu, bunların kendi iş yüklerinizle paylaşılanlar yerine GPUStack'in oluşturduğu ve sahip olduğu ad alanları olmasının bir nedenidir: bunlara başka hiçbir şey planlanmamalıdır. Etiketler aynı zamanda Kyverno, Gatekeeper veya OPA hakkında da hiçbir şey yapmaz; bunlar ayrı giriş web kancalarıdır ve ana bilgisayar ağını veya cihaz montajını yasaklayan politikaları bu Pod'ları yine de reddedecektir. Kümenizde bir tane çalıştırılıyorsa GPUStack ad alanlarına da orada izin verin.

**Daha katı bir düzeyin neyi reddedeceğini kontrol etmek.** Kendi çalışan iş yüklerinize karşı, hiçbir şeyi değiştirmeden:

```bash
kubectl label --dry-run=server --overwrite ns <namespace> \
  pod-security.kubernetes.io/enforce=baseline
```

API sunucusu, ad alanının mevcut Pod'larını seviyeye göre değerlendirir ve reddedilecek her biri hakkında, ihlal ettiği kontrolleri adlandırarak uyarır.

#### Grafik Değerleri { #chart-values }

Bir `Kubernetes` kümesinin kaydedilmesi, buna GPUStack Helm grafiğini yükler ve yukarıdaki seçenekler o grafik için değerlere dönüştürülür. `helmValues` bu seçeneklerin kapsamadığı şeyler için aynı grafiğe doğrudan ulaşır: anahtarlar grafiğin kendisine aittir, kelimesi kelimesine alınır, dolayısıyla grafiğin veya alt grafiklerinin sunduğu her şey burada kendi seçeneğini beklemeden yapılandırılabilir.

Henüz kullanıcı arayüzünde bir alan yok, bu yüzden API aracılığıyla ayarlayın. `PUT` tüm kümenin yerini alır, bu nedenle önce kümeyi okuyun ve `k8sOptions` altına eklenen `helmValues` ile geri gönderin:

```bash
# Read the cluster.
curl -sS -H "Authorization: Bearer <api-key>" \
  http://<server>/v2/clusters/<id> > cluster.json

# Add the values. This one keeps the chart from installing the S3 CSI driver,
# for a cluster that already provides its own storage.
jq '.k8sOptions.helmValues = {
      "gpustack-operator": {"csi-driver-s3": {"enabled": false}}
    }' cluster.json > updated.json

# Send it back.
curl -sS -X PUT -H "Authorization: Bearer <api-key>" \
  -H "Content-Type: application/json" --data @updated.json \
  http://<server>/v2/clusters/<id>
```

Değerler, kümeden türetilen değerlerin üzerine anahtar bazında ve derinlemesine birleştirilir. Helm'deki gibi listeler genişletilmek yerine tümüyle değiştirilir. Anahtarlar chart'larda belgelenmiştir: GPUStack chart'ının `values.yaml` dosyasında ve `gpustack-operator` altındakiler için [GPUStack Operator chart'ında](https://github.com/gpustack/gpustack-operator).

Bazı yollar birleştirilmek yerine reddedilir: sunucunun bu kümenin kaydından türettiği, sürümün verildiği kümeyle eşleştiğine karar veren yollar — çalışanlarının rapor ettiği yer, hangi görüntüyü çalıştırdıkları, bu görüntünün hangi kayıt defterinden geldiği, hangi Sır'ın belirteci taşıdığı. API reddettiği yolu adlandırır ve `helmValues` alan açıklaması geçerli listeyi taşır; bu sayfa bunu tekrarlamıyor, bu yüzden ikisi sürüklenemez.

!!! note

    Kümenin kaydedilmesi Kubernetes'te tek başına hiçbir şeyi değiştirmez. `Register Cluster`'i yeniden çalıştırın ve size verdiği bildirimi uygulayın: küme içi Job, bildirimin istediklerini sürümün yüklendiği kaynakla karşılaştırır ve yalnızca bunlar farklı olduğunda yükseltilir.

!!! note

    Sürümün dağıttığı şeyi değiştiren bir yükseltme — CPU çalışan DaemonSet'in kapatılması (`helmValues`'de `worker.cpuEnabled=false`, yalnızca CPU düğümleri kontrol düzlemini taşıyan bir küme için) veya kümenin artık seçmediği bir GPU sağlayıcısının bırakılması — kendi başına oluşturmayı durdurduğu çalışan DaemonSets'i kaldırır: `kubectl apply` olmaz (bildirimin atladığı bir nesneyi hiçbir zaman silmez), dolayısıyla yükseltme İşi her şeyi siler `<release>-worker[-<vendor>]` Helm revizyonuna sahip olsun ya da olmasın, sürümün artık oluşturmadığı DaemonSet.

    Bu temizleme işleminin öncesinden kalanlar da aynı şekilde kaldırılır: geçerli bildirimi bir kez uyguladığınızda, yapılandırma o zamandan beri değişmemiş olsa bile İş onu tarar. Yalnızca İşi hiç çalıştıramayan bir kümenin manuel forma ihtiyacı vardır:

    ```bash
    kubectl delete daemonset/gpustack-worker -n gpustack-system
    kubectl delete daemonset/gpustack-worker-<vendor> -n gpustack-system
    ```

    Yalnızca sürümü yükseltmek istediğinizde `Register Cluster` parametrelerini (GPU satıcıları, seçenekler) son uygulanan bildirimle aynı tutun; bunun dışındaki her şey yükseltme değil, yapılandırma değişikliğidir.

!!! warning

    Kueue ve Node Feature Discovery zorunludur. Operatör planlama zincirini bunlara dayandırır ve başlangıçta CRD'lerini bekler. Kümenin kendisinin sağlamadığı birini kapatmak operatörün başlamasını engeller. Bu sürümün kurduğu bir bileşeni kapatmak onu kaldırır; Kueue CRD'leri kendi chart'ından gelir ve tüm `Workload` ile `ClusterQueue` nesneleri de onlarla birlikte silinir.

#### Kubernetes Kümesini Kaldırma { #uninstalling-a-kubernetes-cluster }

Kaldırma işlemi, kaydın yüklü olduğu sürüm(ler)in `helm uninstall`'i ve ayrıca Helm'in hiçbir zaman sahip olmadığı nesnelerdir. Bir hazırlık adımı, temiz bir şekilde tamamlanıp tamamlanmadığına karar verir: operatörün ağacı (Kueue, Node Feature Discovery, CSI sürücüleri, cihaz yöneticileri), özel kaynakları sonlandırıcıları taşıyan denetleyicileri dağıtır ve kaldırma işlemi bu denetleyicileri uzaklaştırır - bu nedenle önce kaynakların gitmesi gerekir, aksi takdirde içinde yaşadıkları ad alanları (`gpustack-system`, `gpustack-default`) asla `Terminating`'den ayrılmaz. Sırayla:

1. Önce kümenin çalıştırdığı öğeleri sunucudan kaldırın; kendisine dağıtılan modelleri silin (veya taşıyın), ardından küme kaydını silin — böylece sunucu, kaybolmak üzere olan çalışanlara yönelik planlamayı durdurur.
2. Denetleyicileri hala çalışırken**özel kaynakları** silin ve temizlenmelerini bekleyin. Tahmin etmek yerine bu kümenin sürümlerinin neler içerdiğini keşfedin:

    ```bash
    kubectl get crd -o json | jq -r '.items[]
      | select(((.metadata.annotations // {})["meta.helm.sh/release-name"] // "")
        | . == "gpustack" or . == "gpustack-kueue"
          or . == "gpustack-node-feature-discovery"
          or . == "gpustack-csi-driver-nfs" or . == "gpustack-csi-driver-s3"
          or . == "gpustack-operator-device-manager")
      | .metadata.name'
    ```

    Listelenen her CRD için tüm nesnelerini silin ve hiçbiri kalmayana kadar bekleyin:

    ```bash
    kubectl delete <crd-name> --all -A          # e.g. clusterqueues.kueue.x-k8s.io
    kubectl get <crd-name> -A                   # empty before you move on
    ```

3. Sürümleri kaldırın. Ana olan `gpustack`; beş `gpustack-*` uygulama sürümü, grafik nesnelerini benimsemeden önce kayıtlı kümelerde mevcuttur ve mevcut olmayan bir sürümdeki `helm uninstall` bir hatadır, bu nedenle önce şunu kontrol edin:

    ```bash
    for release in gpustack gpustack-kueue gpustack-node-feature-discovery \
                   gpustack-csi-driver-nfs gpustack-csi-driver-s3 \
                   gpustack-operator-device-manager; do
      kubectl get secret -n gpustack-system \
        --selector "owner=helm,name=${release}" >/dev/null 2>&1 || continue
      helm uninstall "${release}" -n gpustack-system --timeout 10m
    done
    ```

    Buradan `--wait` geçmeyin. `--wait` ile kaldırma, Node Feature Discovery **silme sonrası prune hook**'e ulaşır, kancanın `node-feature-discovery-prune` ServiceAccount ve ClusterRole'unu oluşturur ve ardından silme beklemesinde bloke eder - bekleme `--timeout`'den daha uzun yaşar (`helm --debug` ile doğrulanmıştır: `waiting for resources to be deleted count=1`'de oturur) zaman aşımını geçen dakikalar boyunca ve yeni oluşturulan ServiceAccount hala oradadır). Ondan kaçmak için Helm'i öldürmek, kuru erik RBAC'ı sahipsiz bırakır; bu nedenle, 4. adımın bu nesneleri elle kaldırmasının nedeni kancadır. `--wait` olmadan, kaldırma işlemi hemen geri döner, geri kalanlarla birlikte kanca kaynakları da temizlenir ve nesneler aynı şekilde gider — 4. adım, Helm'in asla sahip olmadığı başıboş olanları ortadan kaldırır.

4. `helm uninstall`'in kaldırmadığını kaldırın. Helm, CRD'leri asla silmez ve operatör, hiçbir sürümün sahip olmadığı üç tür küme kapsamlı nesneyi daha kaydeder:

    - **Operatörün kendi CRD'leri** — `devices`, `instances` ve `instancetypes`, `worker.gpustack.ai` altında. Operatör bunları grafik yerine başlangıçta oluşturur, dolayısıyla `meta.helm.sh/release-name` ek açıklaması taşımazlar ve 2. adımın keşif sorgusu bunları listelemez. `instancetype` nesnesi `gpustack.ai/controlled` sonlandırıcıyla hayatta kalır; CRD'leri silmeden önce veya silerken onu çıkarın (aşağıya bakın), aksi halde CRD silinmiş halde kalır ve grubun API'si asla kaybolmaz.
    - **Operatörün kabul web kancaları** — `gpustack-worker-mutation` (Değişme) ve `gpustack-worker-validation` (Doğrulama). Operatörün Hizmetine iletirler ve kapatılamazlar, dolayısıyla silinene kadar bir `worker.gpustack.ai` nesnesine yapılan her yazma (aşağıdaki sonlandırıcı şeridi dahil) `service "gpustack-operator-worker" not found` ile reddedilir.
    - **Operatörün APIServices** — `v1.gpustack.ai` ve `v1.worker.gpustack.ai`, aynı şekilde operatörün Hizmeti tarafından desteklenir. Bir ad alanı, bu gruplara ilişkin raporları `stale GroupVersion discovery` silinene kadar, bu gruplar kaldırılıncaya kadar siliyor ve bu işlem hiçbir zaman bitmiyor.

    ```bash
    kubectl delete mutatingwebhookconfiguration gpustack-worker-mutation --ignore-not-found
    kubectl delete validatingwebhookconfiguration gpustack-worker-validation --ignore-not-found
    kubectl delete apiservice v1.gpustack.ai v1.worker.gpustack.ai --ignore-not-found
    kubectl delete crd devices.worker.gpustack.ai instances.worker.gpustack.ai \
      instancetypes.worker.gpustack.ai --ignore-not-found
    ```

    Node Feature Discovery erik RBAC'ı da `helm uninstall`'den daha uzun süre yaşayabilir — `ClusterRole/node-feature-discovery-prune` ve onun bağlayıcısı, silme sonrası erik kancasının kaynaklarıdır (yukarıdaki `--wait` notuna bakın), Miğfer kancanın ortasında öldürüldüğünde sahipsiz kalır. Daha sonraki bir kayıt, bunları benimsemek yerine yüklemeyi reddeder, bu nedenle bunları geri kalanıyla birlikte kaldırın:

    ```bash
    kubectl delete clusterrole node-feature-discovery-prune --ignore-not-found
    kubectl delete clusterrolebinding node-feature-discovery-prune --ignore-not-found
    ```

    Sonra küme kapsamlı önyükleme RBAC'si, ardından ad alanları. `gpustack-system` aynı zamanda ConfigMap, ServiceAccount ve kayıt belirteci Secret'ı da içerir; ad alanının silinmesi bunları da beraberinde getirir:

    ```bash
    kubectl delete clusterrolebinding gpustack-bootstrap --ignore-not-found
    kubectl delete namespace gpustack-default gpustack-system
    ```

5. *(İsteğe bağlı)* Düğümlerde kalan verileri temizleyin. Çalışan Pod'ları ana bilgisayardaki `/var/lib/gpustack` dizinini bağlar; yukarıdaki adımlar indirilen model ağırlıklarını ve önbellekleri tutan bu dizine dokunmaz. Veriler yeniden kuruluma engel değildir: aynı düğümlerde yeniden kaydedilen küme dizini tekrar kullanır ve mevcut dosyaları yeniden indirmez. Yalnızca alan başka amaçla gerekiyorsa kaldırın:

    ```bash
    rm -rf /var/lib/gpustack
    ```

`kubectl get all -n gpustack-system` (`NotFound` olmalıdır), `kubectl get clusterrolebinding | grep gpustack`, `kubectl get ds -A | grep -E 'gpustack|csi'` ve — yalnızca 4. adımın kaldırdığı nesneler için — `kubectl get apiservice,mutatingwebhookconfiguration,validatingwebhookconfiguration 2>/dev/null | grep gpustack` ve `kubectl get crd | grep -E 'gpustack|worker\.gpustack'` (her ikisi de hiçbir şey yazdırmamalıdır) ile doğrulayın.

Bir ad alanı zaten `Terminating`'de takılı kalmışsa, nesneleri hâlâ denetleyicileri gitmiş olan sonlandırıcıları tutar. Bunları bulun ve sonlandırıcıları elle sıyırın; yalnızca çıkardığınız kümede güvenlidir. Önce kabul web kancalarını silin (adım 4) veya yamanın kendisi, Hizmeti artık mevcut olmayan bir web kancası tarafından ele geçirilir:

```bash
kubectl api-resources --verbs=list --namespaced -o name |
  xargs -n1 -I{} sh -c 'kubectl get {} -n gpustack-default -o name 2>/dev/null'
kubectl patch <kind>/<name> -n gpustack-default \
  -p '{"metadata":{"finalizers":null}}' --type=merge
```

Aynı şerit, küme kapsamındaki artıklar için de geçerlidir — `gpustack.ai/controlled` sonlandırıcıyı tutan bir `instancetype.worker.gpustack.ai` nesnesi, CRD'sini (ve onunla birlikte tüm `worker.gpustack.ai` grubunu) silme işleminde tutar: `kubectl patch instancetypes.worker.gpustack.ai <name> -p '{"metadata":{"finalizers":null}}' --type=merge`.

### DigitalOcean Kümesi Oluşturma { #creating-digitalocean-cluster }

1. `Basic Configuration` adımında `Name` alanı zorunlu, `Description` ise isteğe bağlıdır. DigitalOcean API ile iletişim kurmak için bir Bulut Kimlik Bilgisi oluşturun veya seçin. GPU Damlacıklarını destekleyen bir Bölge seçin. Ayrıca yeni oluşturulan DigitalOcean Damlacıklarından erişilebilecek `GPUStack Server URL`'i de yapılandırmanız gerekir.
2. `Next`'a tıklayın.
3. Bir veya daha fazla ekleme `Worker Pools`. Her havuz için `Name`, `Instance Type`, `OS Image`, `Replicas`, `Batch Size`, `Labels` ve `Volumes` belirtilebilir.
4. Çalışan havuzları yapılandırıldıktan sonra `Save` seçeneğine tıklayın.

Küme oluşturulduktan sonra ilave çalışan havuzları eklenebilir.

1. `Clusters` sayfasına gidin.
2. Çalışan havuzuna eklemek istediğiniz `DigitalOcean` kümesini bulun.
3. İşlemler sütunundaki üç nokta düğmesini tıklayın, ardından `Add Worker Pool`'ı seçin
4. Yukarıdaki 3. Adımdaki seçeneklerle yeni çalışan havuzu ekleniyor.

### SHUIHUA GELECEK Kümesini Oluşturmak { #creating-shuihua-future-cluster }

1. `Basic Configuration` adımında `Name` alanı zorunludur ve `Description` isteğe bağlıdır. Shuihua API ile iletişim kurmak için bir Bulut Kimlik Bilgisi oluşturun veya seçin. Shuihua'nın bölgesi yok, dolayısıyla seçilecek bölge yok. Ayrıca yeni oluşturulan bulut sunucularından erişilebilecek `GPUStack Server URL`'yi de yapılandırmanız gerekir.
2. Bu sağlayıcı için `Default Container Registry` zorunludur. Shuihua örnekleri Docker Hub'a erişemediğinden Docker Hub'a çözümlenen kayıt adresleri reddedilir. Alan `quay.io` ve `swr.cn-south-1.myhuaweicloud.com` önerir; kendi Harbor'ınız veya aynanız gibi erişilebilir başka bir kayıt da kullanılabilir.
3. `Next`'a tıklayın.
4. Bir veya daha fazla ekleme `Worker Pools`. Her havuz için `Name`, `Instance Type`, `OS Image`, `Replicas`, `Batch Size` ve `Labels` belirtilebilir. `Instance Type`, GPU modeli, saatlik fiyatı ve kalan stokuyla birlikte listelenen bir Shuihua spesifikasyon şablonudur; tükenen bir şablon seçilemez. Birimler sunulmuyor — Shuihua'nın blok depolama API'si yok.
5. Çalışan havuzları yapılandırıldıktan sonra `Save` seçeneğine tıklayın.

Küme oluşturulduktan sonra DigitalOcean kümesinde olduğu gibi ek çalışan havuzları eklenebilir.

Shuihua örnekleri, yalnızca 22 ve 80 numaralı bağlantı noktalarının eşlendiği NAT'ın arkasında yer alır; bu nedenle küme, varsayılan olarak `tunnel` proxy modunu kullanır ve çalışanları, sunucunun WebSocket tüneli aracılığıyla çıkarım sunar. Listelenen IP'leri, örneğin özel adresidir; Gerçekten ulaşılabilir olan uç noktayı elde etmek için çalışan üzerinde `View SSH Access` kullanın. Tam açıklama için [Shuihua Kullanarak GPU Kümesi Ekleme](../tutorials/adding-gpucluster-using-shuihua.md) konusuna bakın.

### Çalışan Havuzlarının İşletilmesi { #operating-worker-pools }

Bulut sağlayıcı kümeleri için çalışan havuzlarını `Clusters` sayfasından yönetebilirsiniz:

1. `Clusters` sayfasına gidin.
2. Yönetmek istediğiniz bulut sağlayıcı kümesini bulun ve çalışan havuzlarını görüntüleyecek şekilde genişletin.
3. Bir çalışanın kopya sayısını düzenlemek için bunu doğrudan çalışan sütununda değiştirin.
4. Çalışan havuzunu düzenlemek için `Edit` düğmesine tıklayın ve `Name`, `Replica`, `Batch Size` ve `Labels`'yi gerektiği gibi güncelleyin.
5. Bir çalışan havuzunu silmek için, çalışan havuzunun işlemler sütunundaki üç nokta düğmesini tıklayın ve ardından `Delete` seçeneğini seçin.

## Kümeyi Güncelle { #update-cluster }

1. `Clusters` sayfasına gidin.
2. Düzenlemek istediğiniz kümeyi bulun.
3. `Edit` düğmesine tıklayın.
4. `Name`, `Description` ve `Worker Configuration YAML`'yi gerektiği gibi güncelleyin.
5. `Save` düğmesine tıklayın.

## Kümeyi Sil { #delete-cluster }

1. `Clusters` sayfasına gidin.
2. Silmek istediğiniz kümeyi bulun.
3. İşlemler sütunundaki üç nokta düğmesine tıklayın, ardından `Delete`'ı seçin.
4. Silme işlemini onaylayın.
5. Bir kümede halen mevcut model veya çalışanlar varsa onu silemezsiniz.

## Çalışan Yapılandırması YAML { #worker-configuration-yaml }

Bir küme oluştururken veya güncellerken, aşağıdaki örnek YAML'yi kullanarak çalışan yapılandırmasını önceden tanımlayabilirsiniz:

```yaml
# ========= log level & tools ===========
debug: false
tools_download_base_url: https://mirror.your_company.com
# ========= directories ===========
pipx_path: "/usr/local/bin/pipx"
cache_dir: "/var/lib/gpustack/cache"
log_dir: "/var/lib/gpustack/log"
bin_dir: "/var/lib/gpustack/bin"
# ========= container & image ===========
image_name_override: "gpustack/gpustack:main"
image_repo: "gpustack/gpustack"
# ========= service & networking ===========
worker_ifname: en0
worker_port: 10150
worker_metrics_port: 10151
disable_worker_metrics: false
service_port_range: "40000-40063"
ray_port_range: "41000-41999"
proxy_mode: worker
# ========= system reserved resources ===========
system_reserved:
  ram: 0
  vram: 0
# ========= huggingface ===========
huggingface_token: xxxxxx
enable_hf_transfer: false
# ========= TLS ===========
insecure_tls: false
```

Yukarıdaki YAML, `Worker Configuration YAML` için şu anda desteklenen tüm seçenekleri listeler. Her seçeneğin anlamı için GPUStack [config dosyası belgelerinin](../cli-reference/start.md#config-file) tamamına bakın. `proxy_mode` seçeneği, sunucunun çalışana nasıl ulaştığını kontrol eder - bkz. [Çalışan Bağlantı Modları](#worker-connection-modes).

Varsayılan kapsayıcı kaydı artık burada `Docker` veya `Kubernetes` kümeler için ayarlanmamaktadır; bunun yerine `Advanced` ayarlarındaki küme düzeyi `Default Container Registry` seçeneğini kullanarak yapılandırın. `Kubernetes` kümeler için ad alanı da aynı şekilde `Namespace` seçeneği aracılığıyla yapılandırılır (bkz. [Kubernetes Küme Seçenekleri](#kubernetes-cluster-options)).

## Çalışan Bağlantı Modları { #worker-connection-modes }

Çıkarım isteklerini bir çalışandaki model örneklerine iletmek için sunucu ve API ağ geçidinin o çalışana geri dönen bir ağ yoluna ihtiyacı vardır. [Çalışan Yapılandırması YAML](#worker-configuration-yaml) veya `--proxy-mode` bayrağı aracılığıyla ayarlanan `proxy_mode` çalışan seçeneği şu yolu seçer:

- `direct` — Sunucu ve ağ geçidi doğrudan çalışanın tanıtılan adresine ve bağlantı noktasına bağlanır. En düşük ek yük, ancak çalışanın sunucudan doğrudan erişilebilir olması gerekir.
- `worker` — istekler çalışanın yerleşik HTTP ters vekilinden geçerek yerel çıkarım sürecine iletilir. Çalışana sunucudan yine erişilebilmelidir; ancak yalnızca çalışan portunun açılması yeterlidir.
- `tunnel` — çalışan sunucuya tek bir **giden** WebSocket bağlantısı tutar; sunucu çalışana yalnızca bu tünelden ulaşır. Çalışan sunucudan gelen bağlantıları kabul edemediğinde bu modu kullanın.

Varsayılan değer, sunucu içindeki yerleşik çalışan için `direct` ve bağımsız çalışanlar için `worker`'dir.

### Tünel Modu (Güvenlik Duvarı veya NAT Arkasındaki Çalışan) { #tunnel-mode-worker-behind-a-firewall-or-nat }

`direct` ve `worker` modlarında sunucu bağlantıyı başlatır, dolayısıyla çalışanın sunucudan erişilebilir olması gerekir. Bu her zaman mümkün değildir; örneğin çalışan bir güvenlik duvarı veya NAT arkasında, farklı bir ağda veya yalnızca giden trafiğe izin veren özel bir alt ağda oturduğunda.

`tunnel` modu yönü tersine çevirir: bağlantı, çalışandan sunucuya **tek yönlü** kurulur.

1. Çalışan, sunucuya - kayıt için kullandığı aynı `--server-url` uç noktaya - belirteci ile kimliği doğrulanan ve düşerse otomatik olarak yeniden bağlanan kalıcı bir giden WebSocket bağlantısı açar.
2. Sunucu, proxy bağlantı noktasında bir HTTP/HTTPS proxy çalıştırır (varsayılan `30079`, `--proxy-port` aracılığıyla ayarlanır).
3. Ağ geçidinin tünel modu çalışanındaki bir model örneğine ulaşması gerektiğinde, isteği sunucunun proxy bağlantı noktasına yönlendirir ve bu da onu mevcut tünel üzerinden çalışana iletir.

Çalışan yalnızca arama yaptığından, çalışan tarafında herhangi bir gelen bağlantı noktasının açılmasına gerek yoktur. Yalnızca sunucunun API bağlantı noktasına ulaşması gerekir ve sunucunun proxy bağlantı noktasına ağ geçidi tarafından ulaşılabilir olması gerekir.

Etkinleştirmek için çalışanın `proxy_mode`'ını `tunnel` olarak ayarlayın:

```yaml
proxy_mode: tunnel
```
