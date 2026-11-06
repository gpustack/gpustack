---
translation_status: translated
---

# Claude Code ile Entegrasyon { #integrate-with-claude-code }

Claude Code, Anthropic'in ajansal bir kodlama aracıdır. GPUStack'teki model dağıtımları Anthropic API ile uyumlu olduğundan, Claude Code'u GPUStack dağıtımınıza kolayca bağlayabilir ve kod oluşturma görevleri için kullanabilirsiniz. Bu kılavuzda Claude Code'u GPUStack ile entegre etme adımlarını inceleyeceğiz ve Claude'dan Flappy Bird oyunu oluşturmasını isteyerek entegrasyonu test edeceğiz.

## Önkoşullar { #prerequisites }

- Toplamda en az 100 GB VRAM'e sahip bir veya daha fazla GPU
- GPUStack kurulu ve çalışıyor
- Model dosyalarını indirmek için Hugging Face veya ModelScope'a erişim

!!! note

    Bu kılavuzda, FP8 nicemlemeli `Qwen3-Coder-Next` modeline hizmet vermek için tek bir NVIDIA H200 GPU kullanılmıştır.

## Modeli Dağıt { #deploy-the-model }

1. GPUStack kullanıcı arayüzünde **Model Kataloğu** sayfasına gidin.

2. `Qwen3-Coder-Next`'i arayın ve varsayılan yapılandırmayı kullanarak modeli dağıtın.

   ![Model Kataloğu](../assets/integrations/claude-code/catalog.png)

3. Dağıtıldıktan sonra model, Anthropic uyumlu API kullanılarak çıkarım için hazır hale gelir.

   ![Modele Hazır](../assets/integrations/claude-code/model-ready.png)

## API Anahtarı Oluşturun { #create-an-api-key }

1. **Erişim Kontrolü** > **API Anahtarları** sayfasına gidin ve ardından **API Anahtarı Ekle**'ye tıklayın.

2. Bir ad girin ve ardından **Kaydet**'e tıklayın.

3. API anahtarını kopyalayın ve daha sonra kullanmak üzere kaydedin.

   ![API Anahtarı](../assets/integrations/claude-code/api-key.png)

## Claude Code'u Yükleme { #install-claude-code }

Claude Code'u sisteminize yüklemek için [Claude Code belgelerini](https://github.com/anthropics/claude-code) izleyin.

## CC-Switch'i yükleyin (İsteğe bağlı) { #install-cc-switch-optional }

Farklı model sağlayıcılar arasında kolaylıkla geçiş yapmak için CC-Switch veya benzeri araçları kullanabilirsiniz. Bu adım isteğe bağlıdır; Claude Code'u manuel olarak da yapılandırabilir veya seçtiğiniz başka bir aracı kullanabilirsiniz.

[CC-Switch](https://github.com/farion1231/cc-switch)'i belgelerini takip ederek kurun.

## Claude Code'u GPUStack ile Yapılandırma { #configure-claude-code-with-gpustack }

1. CC-Switch'i açın ve aşağıdaki ayarlara sahip özel bir sağlayıcı ekleyin:

   - **Sağlayıcı Adı**: `GPUStack`
   - **API Uç Noktası**: GPUStack sunucu URL'niz
   - **API Anahtarı**: Daha önce oluşturduğunuz API anahtarı

2. Tüm modelleri `qwen3-coder-next` kullanacak şekilde yapılandırın.

   ![CC Anahtar Kurulumu](../assets/integrations/claude-code/cc-switch.png)

## Entegrasyonu Test Edin { #test-the-integration }

1. Yeni bir dizin oluşturun ve Claude Code'u başlatın:

   ```bash
   mkdir flappybird && cd flappybird
   claude
   ```

2. `/status` komutunu kullanarak yapılandırmanın doğru olduğunu doğrulayın.

3. Örnek bir görevi deneyin, örneğin Claude Code'dan Flappy Bird oyunu oluşturmasını isteyin:

   ```
   Write a Flappy Bird game.
   ```

   ![Claude Kod Sonucu](../assets/integrations/claude-code/claude.png)

4. Claude Code tamamlandıktan sonra, sonucu test etmek için oluşturulan HTML dosyasını tarayıcınızda açın.

   ![Flappy Kuş Oyunu](../assets/integrations/claude-code/flappybird.gif)

## Sonuç { #conclusion }

Bu kılavuzda Claude Code'u GPUStack ile başarıyla entegre ettik ve Flappy Bird oyunu oluşturmak için kullandık. Artık Claude Code ile daha karmaşık kodlama görevlerini keşfedebilir ve verimli model sunumu için GPUStack'in gücünden yararlanabilirsiniz.
