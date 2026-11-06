---
translation_status: translated
---

# Görme Dili Modellerini Kullanmak { #using-vision-language-models }

**Görüntü Dili Modelleri** hem görsel (görüntü) hem de dil (metin) verilerini aynı anda işleyebilir, bu da onları görüntü altyazısı, görsel soru yanıtlama ve daha fazlası gibi çeşitli uygulamalar için çok yönlü araçlar haline getirir. Bu kılavuzda GPUStack'te Görme Dili Modellerini (VLM'ler) nasıl dağıtacağınızı ve bunlarla nasıl etkileşim kuracağınızı öğreneceksiniz.

Bu modelleri GPUStack'te dağıtma ve bunlarla etkileşim kurma prosedürü benzerdir. Temel fark, modelleri dağıtırken ayarlamanız gereken parametrelerdir. Ayarlayabileceğiniz parametreler hakkında daha fazla bilgi için lütfen [Arka Uç Parametreleri](../user-guide/built-in-inference-backends.md#parameters-reference_1) konusuna bakın.

Bu kılavuzda aşağıdaki modellerin dağıtımını ele alacağız:

- **Qwen3-VL**
- **Llama3.2-Vision**
- **Pixtral**
- **Phi3.5-Vision**

## Önkoşullar { #prerequisites }

Başlamadan önce aşağıdakilere sahip olduğunuzdan emin olun:

- Toplamda en az 30 GB VRAM'e sahip, bir veya daha fazla GPU'ya sahip bir Linux makinesi. Yalnızca Linux'u destekleyen vLLM arka ucunu kullanacağız.
- Model dosyalarını indirmek için Hugging Face'e ve [Hugging Face API anahtarına](https://huggingface.co/settings/tokens) erişim.
- Hugging Face'te yukarıdaki modellere erişim izni verildi. [Llama3.2-VL](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) ve [Pixtral](https://huggingface.co/mistralai/Pixtral-12B-2409) kapılı modellerdir ve bunlara erişim talep etmeniz gerekir.

!!! note

    Bu kılavuz boyunca bir adet H100 (80GB) GPU ile donatılmış bir Ubuntu düğümü kullanılmıştır.

## Adım 1: GPUStack'i yükleyin { #step-1-install-gpustack }

GPUStack'i yüklemek için lütfen [Kurulum Belgelerini](../installation/requirements.md) takip edin.

## Adım 2: GPUStack kullanıcı arayüzünde oturum açın { #step-2-log-in-to-gpustack-ui }

Sunucu başladıktan sonra varsayılan yönetici şifresini almak için aşağıdaki komutu çalıştırın:

```bash
docker exec gpustack cat /var/lib/gpustack/initial_admin_password
```

Tarayıcınızı açın ve GPUStack kullanıcı arayüzüne erişmek için `http://your_host_ip` seçeneğine gidin. Giriş yapmak için varsayılan kullanıcı adı olan `admin`'i ve yukarıda aldığınız şifreyi kullanın.

## Adım 3: vLLM ile Görme Dili Modellerini Dağıtın { #step-3-deploy-vision-language-models-with-vllm }

### Qwen3-VL'yi dağıtın { #deploy-qwen3-vl }

1. GPUStack kullanıcı arayüzünde `Deployments` sayfasına gidin.
2. `Deploy Model` düğmesine tıklayın, ardından açılır menüden `Hugging Face`'i seçin.
3. Arama çubuğunda `Qwen/Qwen3-VL-4B-Instruct` ifadesini arayın.
4. `Save` düğmesine tıklayın. Yeterli GPU kaynağınız olduğu sürece varsayılan yapılandırmalar çalışmalıdır.

### Llama3.2-Vision'ı dağıtın { #deploy-llama32-vision }

1. GPUStack kullanıcı arayüzünde `Deployments` sayfasına gidin.
2. `Deploy Model` düğmesine tıklayın, ardından açılır menüden `Hugging Face`'i seçin.
3. Arama çubuğunda `meta-llama/Llama-3.2-11B-Vision-Instruct` ifadesini arayın.
4. Yapılandırmalarda `Advanced` bölümünü genişletin ve `Backend Parameters` bölümüne doğru aşağı kaydırın.
5. `Add Parameter` düğmesine birden çok kez tıklayın ve aşağıdaki parametreleri ekleyin:

- `--enforce-eager`
- `--max-num-seqs=16`
- `--max-model-len=8192`

6. `Save` düğmesine tıklayın.

### Pixtral'ı dağıtın { #deploy-pixtral }

1. GPUStack kullanıcı arayüzünde `Deployments` sayfasına gidin.
2. `Deploy Model` düğmesine tıklayın, ardından açılır menüden `Hugging Face`'i seçin.
3. Arama çubuğunda `mistralai/Pixtral-12B-2409` ifadesini arayın.
4. Yapılandırmalarda `Advanced` bölümünü genişletin ve `Backend Parameters` bölümüne doğru aşağı kaydırın.
5. `Add Parameter` düğmesine birden çok kez tıklayın ve aşağıdaki parametreleri ekleyin:

- `--tokenizer-mode=mistral`
- `--limit-mm-per-prompt=image=4`

6. `Save` düğmesine tıklayın.

### Phi3.5-Vision'ı dağıtın { #deploy-phi35-vision }

1. GPUStack kullanıcı arayüzünde `Deployments` sayfasına gidin.
2. `Deploy Model` düğmesine tıklayın, ardından açılır menüden `Hugging Face`'i seçin.
3. Arama çubuğunda `microsoft/Phi-3.5-vision-instruct` ifadesini arayın.
4. Yapılandırmalarda `Advanced` bölümünü genişletin ve `Backend Parameters` bölümüne doğru aşağı kaydırın.
5. `Add Parameter` düğmesine tıklayın ve aşağıdaki parametreyi ekleyin:

- `--trust-remote-code`

6. `Save` düğmesine tıklayın.

## Adım 4: Görme Dili Modelleriyle Etkileşim Kurun { #step-4-interact-with-vision-language-models }

1. GPUStack kullanıcı arayüzünde `Chat` sayfasına gidin.
2. Sağ üstteki açılır menüden dağıtılan modeli seçin.
3. Giriş metni alanının üstündeki `Upload Image` düğmesine tıklayın ve bir resim yükleyin.
4. Giriş metni alanına bir istem girin. Örneğin, "Resmi tanımlayın."
5. Çıktıyı oluşturmak için `Submit` düğmesine tıklayın.

![oyun alanı-vl](../assets/using-models/using-vision-language-models/playground-vl.png)
