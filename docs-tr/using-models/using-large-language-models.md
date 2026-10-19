---
translation_status: translated
---

# Büyük Dil Modellerini Kullanma { #using-large-language-models }

**Büyük Dil Modelleri (LLM'ler)**, insan benzeri metinleri anlayabilen ve üretebilen güçlü yapay zeka modelleridir; bu da onları sohbet robotları, içerik oluşturma, kod tamamlama ve daha fazlası gibi uygulamalar için vazgeçilmez kılar.

Bu kılavuzda, GPUStack'te LLM'leri nasıl dağıtacağınızı ve bunlarla nasıl etkileşim kuracağınızı öğreneceksiniz.

## Önkoşullar { #prerequisites }

Başlamadan önce aşağıdakilere sahip olduğunuzdan emin olun:

- Toplamda en az 30 GB VRAM'e sahip, bir veya daha fazla GPU'ya sahip bir Linux makinesi. Yalnızca Linux'u destekleyen vLLM arka ucunu kullanacağız.
- Model dosyalarını indirmek için [Hugging Face](https://huggingface.co/) veya [ModelScope](https://www.modelscope.cn/)'e erişim.
- GPUStack kurulu ve çalışıyor. Değilse, [Hızlı Başlangıç ​​Kılavuzu](../quickstart.md)'na bakın.

## Adım 1: Büyük Dil Modellerini Dağıtın { #step-1-deploy-large-language-models }

### Katalogdan Dağıt { #deploy-from-catalog }

Katalogdaki büyük dil modelleri `LLM` kategorisiyle işaretlenmiştir. Katalogdan büyük bir dil modeli seçtiğinizde, yeterli GPU kaynağınız olduğu ve arka uç kurulumunuzla uyumlu olduğu sürece varsayılan yapılandırmalar çalışmalıdır (örneğin, vLLM arka ucu bir amd64 Linux çalışanı gerektirir).

Burada örnek olarak `Qwen3 0.6B`'in konuşlandırılmasını ele alıyoruz.

Modeli Katalogdan dağıtmak için şu adımları izleyin:

1. GPUStack kullanıcı arayüzünde `Catalog` sayfasına gidin.
2. Model listesi sayfasında, `LLM` ile filtrelemek için açılır menüyü kullanın.
3. Model açıklamasını, maksimum bağlam uzunluğunu ve desteklenen boyutları inceleyin.

![Model Listesi](../assets/using-models/using-large-language-models/model-list.png)

#### vLLM Kullanarak Dağıtım { #deployment-using-vllm }

1. Katalogdan `Qwen3 0.6B`'ı seçin.
2. Model adını ve konfigürasyonunu gerektiği gibi değiştirin.
3. Modeli dağıtmak için `Save` düğmesine tıklayın.

![Modeli Dağıt](../assets/using-models/using-large-language-models/deploy-model.png)

Dağıtımdan sonra model dağıtımının durumunu `Deployments` sayfasından izleyebilir ve çalışmaya başlamasını bekleyebilirsiniz.

## Adım 2: Metin Oluşturma için LLM'leri Kullanın { #step-2-use-the-llms-for-text-generation }

1. GPUStack kullanıcı arayüzünde `Playground` > `Chat` sayfasına gidin.
2. Dağıtılan modelin sağ üstteki `Model` açılır menüsünden seçildiğini doğrulayın.
3. Metin üretimi için bir bilgi istemi sağlayın.
4. İhtiyaçlarınıza göre sağdaki `Parameters`'ı ayarlayın.
5. Metni oluşturmak için `Submit` düğmesine tıklayın.

Oluşturulan düşünce zinciri ve sonuç kullanıcı arayüzünde görüntülenecektir.

![Oluşturuldu](../assets/using-models/using-large-language-models/playground-chat.png)

Bu adımları izleyerek, GPUStack'te yapay zeka destekli metin oluşturma ve doğal dil görevleri için LLM'lerden yararlanabilirsiniz. LLM'nin tüm yeteneklerini keşfetmek için farklı istemler ve ayarlarla denemeler yapın!
