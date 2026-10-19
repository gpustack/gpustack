---
translation_status: translated
---

# Ses Modellerini Kullanma { #using-audio-models }

GPUStack, hem Konuşmadan Metne hem de Metinden Konuşmaya modellerinin çalıştırılmasını destekler. Speech-to-Text modelleri çeşitli dillerdeki ses girişlerini yazılı metne dönüştürürken, Text-to-Speech modelleri yazılı metni doğal ve anlamlı konuşmaya dönüştürür.

Bu kılavuzda, GPUStack'te Konuşmadan Metne ve Metinden Konuşmaya modellerinin dağıtımı ve kullanımı konusunda size yol göstereceğiz.

## Önkoşullar { #prerequisites }

Başlamadan önce aşağıdakilere sahip olduğunuzdan emin olun:

- GPUStack kurulu ve çalışıyor. Değilse, [Hızlı Başlangıç ​​Kılavuzu](../quickstart.md)'na bakın.
- Model dosyalarını indirmek için Hugging Face veya ModelScope'a erişim.

## Konuşmayı Metne Dönüştürme Modelini Çalıştırma { #running-speech-to-text-model }

### Adım 1: Konuşmayı Metne Dönüştürme Modelini Dağıtın { #step-1-deploy-speech-to-text-model }

Modeli Model Kataloğu'ndan dağıtmak için şu adımları izleyin:

1. GPUStack kullanıcı arayüzünde `Model Catalog` sayfasına gidin.
2. Kategori filtresinde `Speech-to-Text`'i seçin, ardından `Whisper-Large-V3-Turbo` modelini seçin.
3. Her şeyi varsayılan olarak bırakın ve modeli dağıtmak için `Save` düğmesine tıklayın.

![Modeli Dağıt](../assets/using-models/using-audio-models/deploy-stt-model.png)

Dağıtımdan sonra model dağıtımının durumunu `Deployments` sayfasından izleyebilirsiniz. Dağıtım başarılı olduktan sonra dağıtımın üç nokta simgesine tıklayın ve modeli Oyun Alanında kullanmaya başlamak için `Open in Playground` öğesini seçin.

![Model Listesi](../assets/using-models/using-audio-models/stt-model-list.png)

### Adım 2: Konuşmayı Metne Çevirme Modeli ile Etkileşim Kurun { #step-2-interact-with-speech-to-text-model }

`Speech to Text` oyun alanında,

1. Bir ses dosyası yüklemek için `Upload` düğmesine tıklayın veya ses kaydetmek için `Microphone` düğmesine tıklayın.
2. Transkripsiyonu oluşturmak için `Generate Text Content` düğmesine tıklayın.

![Oluşturuldu](../assets/using-models/using-audio-models/inference-stt-model.png)

### Adım 3: API aracılığıyla Çıkış Akışı { #step-3-streaming-output-via-api }

Akış transkripsiyonlarını almak için API'yi de kullanabilirsiniz. İşte curl'un kullanıldığı bir örnek:

```bash
# Replace ${SERVER_URL} with your GPUStack server URL and ${YOUR_GPUSTACK_API_KEY} with your API key.
curl ${SERVER_URL}/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -H "Authorization: Bearer ${YOUR_GPUSTACK_API_KEY}" \
  -F model="whisper-large-v3-turbo" \
  -F file="@/path/to/audio-file;type=audio/mpeg" \
  -F language="en" \
  -F stream="true"
```

Bu, kullanılabilir hale geldikçe akış transkripsiyon sonuçlarını döndürecektir.

!!! note

    Akışlı transkripsiyon yalnızca Konuşmadan Metne Dönüştürme modeli `vLLM` arka uçta dağıtıldığında desteklenir. `VoxBox` arka uç akışı desteklemiyor.

## Metinden Konuşmaya Modelini Çalıştırma { #running-text-to-speech-model }

### 1. Adım: Metin-Konuşma Modelini Dağıtın { #step-1-deploy-text-to-speech-model }

Modeli Model Kataloğu'ndan dağıtmak için şu adımları izleyin:

1. GPUStack kullanıcı arayüzünde `Model Catalog` sayfasına gidin.
2. Kategori filtresinde `Text-to-Speech`'i seçin, ardından `Qwen3-TTS-12Hz-1.7B-CustomVoice` modelini seçin.
3. Her şeyi varsayılan olarak bırakın ve modeli dağıtmak için `Save` düğmesine tıklayın.

![Modeli Dağıt](../assets/using-models/using-audio-models/deploy-tts-model.png)

Dağıtımdan sonra model dağıtımının durumunu `Deployments` sayfasından izleyebilirsiniz. Dağıtım başarılı olduktan sonra dağıtımın üç nokta simgesine tıklayın ve modeli Oyun Alanında kullanmaya başlamak için `Open in Playground` öğesini seçin.

![Model Listesi](../assets/using-models/using-audio-models/tts-model-list.png)

### Adım 2: Metinden Konuşmaya Modeliyle Etkileşim Kurun { #step-2-interact-with-text-to-speech-model }

`Text to Speech` oyun alanında,

1. `Voice` açılır menüsünden istediğiniz sesi seçin.
2. (İsteğe bağlı) İstenilen konuşma tarzını oluşturmak üzere modeli yönlendirmek için `Instructions` sağlayın.
3. Konuşmaya dönüştürmek istediğiniz metni girin.
4. Sesi oluşturmak için `Submit` düğmesine tıklayın.

![Oluşturuldu](../assets/using-models/using-audio-models/inference-tts-model.png)

### Adım 3: API aracılığıyla Çıkış Akışı { #step-3-streaming-output-via-api_1 }

Akışlı ses çıkışı almak için API'yi de kullanabilirsiniz. İşte curl'un kullanıldığı bir örnek:

```bash
# Replace ${SERVER_URL} with your GPUStack server URL and ${YOUR_GPUSTACK_API_KEY} with your API key.
curl ${SERVER_URL}/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${YOUR_GPUSTACK_API_KEY}" \
  -d '{
    "model": "qwen3-tts-12hz-1.7b-customvoice",
    "voice": "Vivian",
    "task_type": "CustomVoice",
    "language": "Auto",
    "input": "Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye.",
    "stream": true,
    "response_format": "pcm"
  }' --no-buffer | play -t raw -r 24000 -e signed -b 16 -c 1 -
```

Bu, ses çıkışını doğrudan yayınlayacak ve `play` komutunu kullanarak oynatacaktır. Ses, 16 bit işaretli kodlama ve mono kanal ile 24 kHz örnek hızında PCM formatında yayınlanır.

!!! note

    Konuşma akışı yalnızca Metinden Konuşmaya modeli `vLLM` arka ucunda dağıtıldığında desteklenir. `VoxBox` arka uç akışı desteklemiyor.

## Qwen3-TTS Kullanarak Ses Klonlama { #voice-cloning-using-qwen3-tts }

GPUStack ayrıca Metinden Konuşmaya modelleriyle ses klonlamayı da destekler. İşte nasıl kullanılacağı:

### Adım 1: Ses Klonlama Modelini Dağıtın { #step-1-deploy-voice-cloning-model }

1. GPUStack kullanıcı arayüzünde `Model Catalog` sayfasına gidin.
2. Kategori filtresinde `Text-to-Speech`'i seçin, ardından `Qwen3-TTS-12Hz-1.7B-Base` modelini seçin.
3. Her şeyi varsayılan olarak bırakın ve modeli dağıtmak için `Save` düğmesine tıklayın.

### Adım 2: Oyun Alanında Ses Klonlamayı Kullanın { #step-2-use-voice-cloning-in-playground }

Dağıtım başarılı olduktan sonra dağıtımın üç nokta simgesine tıklayın ve modeli Oyun Alanında kullanmaya başlamak için `Open in Playground` öğesini seçin. Daha sonra şu adımları izleyin:

1. `Reference Audio` alanına bir ses dosyası yükleyin veya klonlama için referans ses sağlamak üzere bir ses URL'si girin. Örneğin, `https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-0115/APS-en_33.wav` URL'sini girebilirsiniz.
2. `Use Speaker Embedding Only (no ICL)` seçeneğini işaretleyin.
3. Sentezlenecek metni girin, örneğin: `Good one. Okay, fine, I'm just gonna leave this sock monkey here. Goodbye.`
4. Klonlanmış sesle konuşmayı oluşturmak için `Submit` düğmesine tıklayın.

![Ses Klonlama](../assets/using-models/using-audio-models/voice-clone.png)
