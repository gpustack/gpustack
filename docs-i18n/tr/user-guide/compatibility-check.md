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
