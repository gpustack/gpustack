---
translation_status: translated
---

# Çıkarım Performansı Ayarlarına Genel Bakış { #inference-performance-tuning-overview }


## Genel bakış { #overview }

vLLM ve SGLang gibi açık kaynak çıkarım motorları çok iyi performans sunar; ancak ayarlanmış ve ayarlanmamış dağıtımlar arasındaki fark düşündüğünüzden büyük olabilir. En etkili doğrulama yöntemi, hedef cihazlarda gerçek trafiğinizle benchmark çalıştırmaktır. Bununla birlikte farklı çıkarım motorları, GPU'lar, modeller ve parametre birleşimleriyle çok sayıda deney yaptık. Bu deneylerden çıkan genel gözlemler, ayrıntılı optimizasyona başlamadan önce yol gösterebilir.

## Gözlemler { #observations }

Aşağıdaki gözlemler mevcut deney kapsamımıza dayanır; testler ve topluluk ilerledikçe güncellenebilir veya tamamlanabilir. Belirli modellerin belirli GPU'lardaki optimizasyon yöntemleri ve sonuçları için ilgili deney belgelerine bakın.

### Çıkarım motorları { #inference-engines }

Çıkarım motoru seçimi kritiktir. Çıkarım optimizasyonu çoğu zaman belirli modeller, nicemleme yöntemleri ve GPU'lar için motora özgü ayrıntılı ayar gerektirir. Dolayısıyla motorun senaryoya göre optimize edilmiş olması büyük fark yaratır. Örneğin A100 GPU'da gpt-oss-20b, vLLM ile SGLang/TensorRT-LLM'den on kattan fazla hızlı çalışır ([ayrıntılar](gpt-oss-20b/a100.md#1-choosing-the-inference-engine)). Ancak bir motorun diğerinden her durumda üstün olduğu söylenemez. Deneylerimizde vLLM, SGLang ve TensorRT-LLM farklı senaryolarda en iyi performansı göstermiştir.

1. **vLLM**, iyi kullanıcı deneyimi, güçlü uyumluluğu ve kapsamlı, zamanında model desteğiyle öne çıkar. Sürekli yüksek çıkarım performansı sunar ve çok çeşitli senaryolarda dengeli, güvenilir bir seçenektir.
2. **SGLang** de iyi kullanıcı deneyimi ve rekabetçi çıkarım performansı sağlar. Uyumluluğu ve model kapsamı vLLM'in biraz gerisinde olsa da yeni model desteği sunma hızı en iyiler arasındadır. Özellikle spekülatif kod çözmede ve Hopper GPU'larda büyük FP8 MoE modellerini çalıştırmada güçlüdür.
3. **TensorRT-LLM**, yüksek düzeyde optimize edilmiş çıkarım performansı sağlar ve bazı deneylerde olağanüstü sonuçlar gösterir. Ancak sağlayıcıya daha sıkı bağlılığı ve yeni modellere görece yavaş destek vermesi nedeniyle kullanım alanı daha özeldir.

### Quantization { #quantization }

1.  Nicemleme, **işlem hacmini en üst düzeye çıkarmak** için kritik bir tekniktir.
2.  **Ağırlık-aktivasyon nicemlemesi** gecikmeyi azaltmaya da yardımcı olur.

### Spekülatif kod çözme { #speculative-decoding }

Spekülatif kod çözme gecikme optimizasyonunda etkilidir. Ancak toplu iş boyutu büyüdükçe etkinliği belirgin biçimde azalır. Bu nedenle **işlem hacmini artırmak için uygun değildir.**

### Paralellik stratejileri { #parallelism-strategies }

Paralellik stratejileri, birden çok GPU'da dağıtık çıkarım için gereklidir.

### Çekirdekler { #kernels }

vLLM/SGLang genellikle donanıma göre makul varsayılanlar seçer. Çoğu senaryoda varsayılan dikkat arka ucu en uygun seçenektir. **DeepGEMM** gibi çekirdek optimizasyonları belirli hassasiyetlere ve GPU'lara yöneliktir. Genellikle varsayılan etkin olsalar da bazen devre dışı bırakmak daha uygundur.

### Genellenebilirlik { #generalizing }

İyi performans gösteren bir dağıtım yapılandırması, farklı Giriş Dizisi Uzunluklarında (ISL) ve Çıkış Dizisi Uzunluklarında (OSL) genellikle iyileşme sağlar; **ancak iyileşme oranları önemli ölçüde değişebilir.**

### Ayrıntılı ayarlama { #deeper-tuning }

Bazı parametreler ISL/OSL, verideki önek tekrarı ve eş zamanlılık gibi gerçek çıkarım isteği kalıplarına göre ayarlanmalıdır. Bunlar şunlardır:

  1.  **En büyük toplu iş boyutu** (eş zamanlılıkla ilişkilidir)
  2.  **Zamanlama yapılandırması** (örneğin API sunucusunu yatay ölçekleme [vLLM], asenkron zamanlama [vLLM], zamanlama temkinliliği [SGLang], zamanlama ilkesi [SGLang]) (çeşitli etkenlerle ilişkilidir)
  3.  **Genişletilmiş KV önbelleği** (dizi uzunluğu ve önek tekrarıyla ilişkilidir)
  4.  **CUDA graph** (eş zamanlılıkla ilişkilidir)
  5.  **Torch compile** (model mimarisi ve eş zamanlılıkla ilişkilidir)
