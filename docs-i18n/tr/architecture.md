---
translation_status: translated
---

# Mimari { #architecture }

Aşağıdaki şema GPUStack mimarisinin üst düzey görünümünü sağlar.

![gpustack-v2-mimarisi](assets/gpustack-v2-architecture.png)

Aşağıdaki diyagram dahili bileşenleri ve bunların etkileşimlerini detaylandırmaktadır.

![gpustack-v2-bileşenleri](assets/gpustack-v2-components.png)

### Sunucu { #server }

GPUStack sunucusu aşağıdaki bileşenlerden oluşur:

- **API Sunucusu:** İstemcilerin sistemle etkileşime girmesi için RESTful bir arayüz sağlar. Kimlik doğrulama ve yetkilendirme işlemlerini gerçekleştirir.
- **Zamanlayıcı:** Model örneklerini çalışanlara atamaktan sorumludur.
- **Kontrolörler:** Sistemdeki kaynakların durumunu yönetir. Örneğin, istenen sayıda kopyayla eşleşecek şekilde model örneklerinin kullanıma sunulmasını ve ölçeklendirilmesini yönetiyorlar.

### İşçi { #worker }

GPUStack çalışanı aşağıdaki bileşenlerden oluşur:

- **GPUStack Çalışma Zamanı:** GPU cihazlarını algılar ve model örneklerini dağıtmak için konteyner çalışma zamanı ile etkileşime girer.
- **Serving Manager:** Çalışandaki model örneklerinin yaşam döngüsünü yönetir.
- **Metrik Aktarıcı:** Model örnekleri ve performanslarıyla ilgili metrikleri dışa aktarır.

### Yapay Zeka Ağ Geçidi { #ai-gateway }

AI Ağ Geçidi, istemcilerden gelen API isteklerini yönetir. Talepleri, talep edilen modele göre uygun model örneklerine yönlendirir. GPUStack, API yönlendirme ve yük dengeleme için [Higress](https://github.com/alibaba/higress)'i kullanır.

### SQL Veritabanı { #sql-database }

GPUStack sunucusu veri deposu olarak bir SQL veritabanına bağlanır. GPUStack varsayılan olarak Gömülü PostgreSQL kullanır, ancak bunu harici bir veritabanı kullanacak şekilde de yapılandırabilirsiniz. Uyumlu veritabanlarının ve doğrulanmış sürümlerin listesi için [Veritabanı Gereksinimleri](installation/requirements.md#database-requirements)'e bakın.

### Çıkarım Sunucusu { #inference-server }

Çıkarım sunucuları, çıkarım görevlerini gerçekleştiren arka uçlardır. GPUStack, yerleşik çıkarım sunucusu olarak [vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang), [Ascend MindIE](https://www.hiascend.com/en/software/mindie) ve [VoxBox](https://github.com/gpustack/vox-box)'yi destekler. Ayrıca özel çıkarım arka uçları da ekleyebilirsiniz.

### Ray { #ray }

[Ray](https://ray.io), GPUStack'in dağıtılmış vLLM'yi çalıştırmak için kullandığı dağıtılmış bir bilgi işlem çerçevesidir. GPUStack, birden fazla çalışan arasında dağıtılmış vLLM'yi çalıştırmak için isteğe bağlı olarak Ray kümesini önyükler.
