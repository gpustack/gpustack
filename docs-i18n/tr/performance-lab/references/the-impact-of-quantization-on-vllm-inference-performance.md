---
translation_status: translated
---

# Nicemlemenin vLLM Çıkarım Performansına Etkisi { #the-impact-of-quantization-on-vllm-inference-performance }

Nicemleme, ağırlıkları ve aktivasyonları olağan 32-bit kayan nokta (float32) yerine 8-bit tamsayı (int8) gibi düşük hassasiyetli türlerle göstererek çıkarımın hesaplama ve bellek maliyetini azaltır.

Bu belgenin amacı, vLLM'de yaygın kullanılan [nicemleme tekniklerini](https://docs.vllm.ai/en/latest/features/quantization/index.html) ve bunların çıkarım performansına, özellikle yüksek eş zamanlılıkta işlem hacmine etkisini değerlendirmektir.


## Sonuçlar { #conclusions }

1. **Ağırlık-aktivasyon nicemlemesi** (örneğin FP8), çok az kalite kaybıyla önemli performans artışı sağlar ve çoğu senaryo için önerilir. Statik nicemleme dinamik nicemlemeden yüksek işlem hacmi verir; doğruluk kalibrasyon verisine bağlıdır:

    - **Dinamik FP8**: +14.8% TPS, -14.1% TTFT, -21.8% TPOT
    - **Statik FP8**: +26.7% TPS, -20.4% TTFT, -32.4% TPOT

2. **Çoğu nicemleme yaklaşımı**, VRAM'in sınırlı olduğu senaryolarda işlem hacmini önemli ölçüde artırabilir; deneylerde +46%'ya varan kazanç görülmüştür.

3. **Yalnızca ağırlık nicemlemesi**, VRAM kısıtı olmadığında ters nicemleme yükü nedeniyle TPS, TTFT ve TPOT performansını düşürür.

4. **Yalnızca ağırlık nicemleme yöntemleri arasında** AWQ ve GPTQ en iyi çıkarım performansını verir. bitsandbytes ve GGUF düşük performans veya uyumluluk sorunları nedeniyle önerilmez.

5. Deneylerde AWQ ve GPTQ için **varsayılan vLLM çekirdekleri**, özel çekirdek uygulamalarından daha iyi performans göstermiştir.

6. **KV önbelleği nicemlemesi**, diğer optimizasyonlara göre görece küçük işlem hacmi kazançları sağlar.

## Teknik arka plan { #technical-background }

### Nicemleme türü { #quantization-type }

| Nicemleme türü | Açıklama |
|-------------------|-------------|
| Yalnızca ağırlık | Eğitimden sonra yalnızca ağırlıklar nicemlenir; aktivasyonlar tam hassasiyette kalır |
| Dinamik | Ağırlıklar önceden nicemlenir; aktivasyonlar çıkarım sırasında anlık nicemlenir |
| Statik | Ağırlıklar ve aktivasyonlar temsili veri kümesiyle kalibrasyondan sonra önceden nicemlenir |
| Nicemlemeye duyarlı eğitim | Eğitimde nicemlemeyi benzeterek modelin düşük hassasiyete uyum sağlamasını sağlar |

### Kalibrasyon { #calibration }

Kalibrasyon, nicemleme sırasında float32 aralıklarının hesaplandığı adımdır. Ağırlıkların gerçek aralığı nicemleme sırasında bilindiğinden işlem görece kolaydır. Aktivasyonlar için durum daha belirsizdir ve farklı yaklaşımlar bulunur:

1. **Eğitim sonrası dinamik nicemleme**: her aktivasyonun aralığı çalışma sırasında anlık hesaplanır. Az çabayla iyi sonuç verir; ancak aralıkları her seferinde hesaplama yükü nedeniyle statik nicemlemeden yavaş olabilir. Bazı donanımlarda kullanılamaz.

2. **Eğitim sonrası statik nicemleme**: her aktivasyonun aralığı nicemleme sırasında önceden hesaplanır. Genellikle temsili veriler modelden geçirilir ve aktivasyon değerleri kaydedilir. Uygulamada adımlar şöyledir:
   - Aktivasyon değerlerini kaydetmek için gözlemciler yerleştirilir
   - Kalibrasyon veri kümesinde belirli sayıda ileri geçiş yapılır (yaklaşık 200 örnek yeterlidir)
   - Her hesaplamanın aralığı seçilen kalibrasyon tekniğine göre hesaplanır

3. **Nicemlemeye duyarlı eğitim**: eğitim sonrası statik nicemlemeye benzer şekilde her aktivasyonun aralığı eğitim sırasında hesaplanır. Ancak gözlemciler yerine "sahte nicemleme" işleçleri kullanılır. Bunlar değerleri kaydederken nicemleme hatasını da benzeterek modelin bu hataya uyum sağlamasına izin verir.

### KV önbelleği nicemlemesi { #kv-cache-quantization }

KV önbelleği nicemlemesi, anahtar-değer önbelleğini daha düşük hassasiyetle saklayarak çıkarımın bellek kullanımını azaltır; daha fazla token önbelleklenmesini ve daha yüksek işlem hacmini sağlar. vLLM FP8 türlerini (E4M3 ve E5M2) destekler, ancak INT8 KV önbelleğini desteklemez. 4-bit veya 2-bit KV nicemlemesi de araştırılmıştır; ancak genellikle MMLU puanlarının düşmesi gibi belirgin doğruluk kayıplarına yol açar.

### Nicemleme çekirdekleri { #quantization-kernels }

vLLM, AWQ ve GPTQ gibi yöntemler için birden çok nicemleme çekirdeği sunar. AWQ varsayılan olarak resmî AWQ çekirdeğini, GPTQ modelleri ise ExLlamaV2 çekirdeğini kullanır. Marlin ve Machete gibi ek optimize edilmiş çekirdekler de vardır ve özellikle büyük toplu işlerde performansı artırır.

### Çıkarım kalitesindeki düşüş { #inference-quality-degradation }

Nicemleme çıkarım kalitesini düşürebilir. Bu belge ağırlıklı olarak performans metriklerine odaklansa da kalite etkisini değerlendirmek için aşağıdaki kaynakları sunar:

Qwen3-8B modelinin çeşitli nicemleme yöntemleriyle benchmark sonuçları:

| Quantization | CEVAL | MMLU | GSM8K | HUMANEVAL |
|-------------|-------|------|-------|-----------|
| BF16 | 79.27 | 74.78 | 87.79 | 63.41 |
| FP8-Static | 78.23 | 74.79 | 86.96 | 62.20 |
| FP8-Dynamic | 78.45 | 74.75 | 87.64 | 62.80 |
| INT8-Dynamic | 78.01 | 74.84 | 86.96 | 67.07 |
| INT4-GPTQ | 77.19 | 73.26 | 86.43 | 62.20 |
| INT4-AWQ | 76.15 | 73.59 | 86.96 | 63.41 |

Ayrıntılar için [AngelSlim benchmark sonuçlarına](https://github.com/Tencent/AngelSlim/tree/main?tab=readme-ov-file#-benchmark) bakın.

## Deney kurulumu { #experimental-setup }

- **Model**: Qwen3-8B
- **Donanım**: NVIDIA RTX 4090 24GB / A100 80GB / H100 80GB
- **vLLM sürümü**: v0.9.2
- **Veri kümesi**: ShareGPT
- **Benchmark betiği**:
```bash
# Prepare the ShareGPT dataset
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

# Benchmark on ShareGPT dataset
vllm bench serve --model Qwen/Qwen3-8B --endpoint-type openai-chat --endpoint /v1/chat/completions --dataset-name sharegpt --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json --num-prompts 1000
```

## Deney sonuçları { #experimental-results }

### RTX 4090 { #rtx-4090 }

| Nicemleme yöntemi | İşlem hacmi (tok/s) | Ortalama TTFT (ms) | Ortalama TPOT (ms) |
|---------------------|-------------------|----------------|----------------|
| BF16                | 3869.3            | 15385.7        | 63.45          |
| AWQ                 | 5653.4 (+46.1%)   | 7913.4         | 87.7           |
| AWQ Marlin          | 5536.7 (+43.1%)   | 8133.8         | 90.42          |
| GPTQ (Int8)         | 4918.98 (+27.1%)  | 8415.70        | 96.07          |
| GPTQ Marlin         | 5025.82 (+29.9%)  | 8143.93        | 93.05          |

### A100 80GB { #a100-80gb }


| Nicemleme yöntemi | İşlem hacmi (tok/s) | Ortalama TTFT (ms) | Ortalama TPOT (ms) |
|---------------------|-------------------|----------------|----------------|
| BF16                | 10338.25          | 3412.85        | 200.02         |
| GPTQ-Marlin (Int4)  | 8146.73           | 10336.24       | 261.81         |
| GPTQ (Int4)         | 8129.27           | 10414.74       | 261.64         |
| AWQ                 | 9611.61           | 3950.64        | 249.06         |
| AWQ Marlin          | 8066.03           | 10506.70       | 264.33         |
| GPTQ Marlin (Int8)  | 7119.60           | 12359.22       | 309.37         |
| GPTQ (Int8)         | 7100.46           | 12380.82       | 309.34         |
| bitsandbytes        | 5916.34           | 9115.43        | 252.91         |
| GGUF (Q4_K_M)       | N/A               | N/A            | N/A            |

!!! note
    qwen3 mimarisine sahip GGUF modeli desteklenmez.

### H100 80GB { #h100-80gb }

| Nicemleme yöntemi | İşlem hacmi (tok/s) | Ortalama TTFT (ms) | Ortalama TPOT (ms) |
|---------------------|-------------------|----------------|----------------|
| FP8 Statik          | 16452.52 (+26.7%) | 4116.87 (-20.4%) | 85.57 (-32.4%) |
| FP8 Dinamik         | 15275.64 (+14.8%) | 4445.10 (-14.1%) | 98.94 (-21.8%) |
| Int4-W4A16          | 13605.46          | 5302.14         | 130.54         |
| BF16                | 13305.38          | 5172.78         | 126.55         |
| AWQ                 | 9756.06           | 8794.29         | 209.13         |
| GPTQ                | N/A               | N/A             | N/A            |

!!! note
    GPTQ uygulaması çalışmamaktadır; [vLLM issue kaydına](https://github.com/vllm-project/vllm/issues/20986) bakın.

### KV önbelleği nicemlemesi { #kv-cache-quantization_1 }

| Yapılandırma | İşlem hacmi (tok/s) | Ortalama TTFT (ms) | Ortalama TPOT (ms) |
|---------------|-------------------|----------------|----------------|
| BF16 (Temel yapılandırma) | 13305.38 | 5172.78 | 126.55 |
| BF16 + KV Cache FP8 (E4M3) | 13513.19 (+1.6%) | 5688.39 | 103.21 |
