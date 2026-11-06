---
translation_status: translated
---

# Uyumluluk Kontrolü { #compatibility-check }

GPUStack, model dağıtımından önce bir uyumluluk kontrolü gerçekleştirir. Bu kontrol, modelin mevcut GPUStack ortamıyla uyumluluğu hakkında ayrıntılı bilgi sağlar. Aşağıdaki uyumluluk kontrolleri gerçekleştirilir:

## Çıkarım Arka Uç Uyumluluğu { #inference-backend-compatibility }

Seçilen çıkarım arka ucunun işletim sistemi, GPU ve mimari dahil olmak üzere mevcut ortamla uyumlu olup olmadığını kontrol eder.

## Model Uyumluluğu { #model-compatibility }

Modelin seçilen çıkarım arka ucu tarafından desteklenip desteklenmediğini belirler. Buna, desteklenen model formatlarının ve mimarilerinin (ör. `LlamaForCausalLM`, `Qwen3ForCausalLM` vb.) kontrol edilmesi de dahildir. Bu kontrol, yerleşik çıkarım arka uçlarına ve bunların desteklenen modellerine dayanmaktadır. Özel bir arka uç veya arka uç sürümü kullanılıyorsa bu kontrol atlanır.

## Planlanabilirlik Kontrolü { #schedulability-check }

Modelin mevcut ortamda planlanıp planlanamayacağını değerlendirir. Buna RAM ve VRAM gibi mevcut kaynakların yanı sıra yapılandırılmış planlama kurallarının doğrulanması da dahildir.

### Planlama Kuralları { #scheduling-rules }

Planlama kuralları (çalışan seçiciler, GPU seçiciler ve planlama politikaları dahil), bir modelin mevcut ortamda planlanıp planlanamayacağını belirlemek için kullanılır.

### Kaynak Kontrolü { #resource-check }

Kaynak kontrolü, modeli dağıtmak için yeterli sistem kaynaklarının mevcut olmasını sağlar. GPUStack gerekli kaynakları tahmin eder ve bunları ortamdaki mevcut kaynaklarla karşılaştırır. Tahminler aşağıdaki yöntemler kullanılarak gerçekleştirilir:

1. **GGUF modelleri için**: GPUStack, modelin kaynak gereksinimlerini tahmin etmek için [GGUF ayrıştırıcısını](https://github.com/gpustack/gguf-parser-go) kullanır.
2. **Diğer modeller için**: GPUStack, aşağıdaki formülü kullanarak VRAM kullanımını tahmin eder:

$$
\text{VRAM} = \text{WEIGHT\_SIZE} \times 1.2 + \text{FRAMEWORK\_FOOTPRINT}
$$

- `WEIGHT_SIZE` model ağırlıklarının bayt cinsinden boyutunu ifade eder.
- `FRAMEWORK_FOOTPRINT` çerçevenin bellek yükünü temsil eden bir sabittir. Örneğin vLLM, CUDA grafikleri için birkaç gigabaytlık VRAM kullanabilir.
- 1.2x çarpanı ampirik bir tahmindir. Daha fazla ayrıntı için [bu açıklamaya](https://blog.eleuther.ai/transformer-math/#total-inference-memory) bakın.

Bu formül kaba bir tahmin sağlar ve tüm modeller için doğru olmayabilir. Genellikle gerekli VRAM'in alt sınır tahminini yansıtır. Tahmin yetersizse kullanıcılar, çalışanları ve GPU'ları manuel olarak seçerek veya gelişmiş arka uç parametrelerini ayarlayarak ayrıntılı planlama gerçekleştirebilir. Örneğin, vLLM ile kullanıcılar model için GPU tahsisini kontrol etmek için `--tensor-parallel-size` ve `--pipeline-parallel-size` belirtebilirler.

#### Arama tablolarının CPU'ya aktarılması { #offloaded-lookup-tables }

!!! note

    Otomatik arama tablosu tahmini yalnızca desteklenen vLLM ve SGLang mimarilerini, biçimlerini ve boşaltma ayarlarını kapsar. Yerel MindIE, custom arka uçları ve henüz uyarlanmamış vLLM Ascend yolları otomatik olarak kapsanmaz. Otomatik tahminler dağıtım beklentilerinizi karşılamıyorsa [özel kaynak gereksinimleri](#custom-resource-requirements) aracılığıyla bunları ayarlayabilirsiniz.

Engram ve PLE gibi arama tablolarını CPU belleğine aktaran desteklenen modellerde GPUStack, model yapılandırması ve hassasiyet meta verilerinden tablo boyutlarını tahmin eder. Yukarıdaki VRAM formülünü uygulamadan önce aktarılan baytları `WEIGHT_SIZE` değerinden çıkarır ve katılan her çalışanda, yerel genişletilmiş KV önbelleği RAM'ine ek olarak tablolar için RAM ayırır.

RAM tahmini, aynı çalışandaki veri paralelliği kopyalarının tablo depolamasını paylaştığını varsayar ve paralel yerleşimin gerektirdiği bağımsız kopyaları ek pay olmadan hesaba katar. Gerçek bellek kullanımı daha yüksek olabilir. Tablo boyutu veya aktarım davranışı belirlenemiyorsa GPUStack, tam ağırlık boyutuna dayalı VRAM tahminini korur.

#### Özel kaynak gereksinimleri { #custom-resource-requirements }

Otomatik tahminler dağıtımınızın gereksinimleriyle uyuşmuyorsa modelin [kaynak gereksinimi ortam değişkenlerini](../environment-variables.md#model-deployment-configuration) kullanarak RAM ve VRAM gereksinimlerini bayt cinsinden belirtebilirsiniz. Belirtilen RAM değeri, tablo ve yerel önbellek belleği dahil her çalışan için toplam tahminin yerine geçer. Bu değerler planlamada kullanılır; arka ucun gerçek bellek kullanımını yapılandırmaz. RAM ve VRAM birbirinden bağımsız olarak belirtilebilir. Belirtilmeyen kaynaklar için ilgili arka ucun tahmini kullanılmaya devam eder.

Örneğin, her çalışan için 64 GiB RAM ve modelin tamamı için 32 GiB VRAM gereksinimi belirtmek üzere aşağıdaki model ortam değişkenlerini ekleyin (`1 GiB = 1024^3 bytes`). VRAM değeri, GPU başına ayrılan miktarı değil, modelin kullandığı GPU'ların tamamındaki tahmini gereksinimini belirtir.

```text
GPUSTACK_MODEL_RAM_CLAIM=68719476736
GPUSTACK_MODEL_VRAM_CLAIM=34359738368
```
