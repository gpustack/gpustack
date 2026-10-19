---
translation_status: translated
---

# OpenClaw ile entegrasyon { #integrating-with-openclaw }

[OpenClaw](https://openclaw.ai/), yerel cihazlarda çalışan ve Telegram, WhatsApp, Slack ve Discord gibi ortak kanallar aracılığıyla etkileşimi destekleyen kişisel bir yapay zeka asistanıdır.

## Modeli Dağıt { #deploy-the-model }

Model dağıtımını tamamlamak için lütfen GPUStack belgelerindeki [**Model Dağıtımı**](../user-guide/model-deployment-management.md#deploy-model) bölümüne bakın.

## API Erişim Bilgisi { #api-access-info }

1. GPUStack Web Kullanıcı Arayüzü'nde oturum açın
2. **Routes** sayfasına gidin
3. Hedef modelin sağ tarafındaki menüden **API Erişim Bilgisi** seçeneğini seçin.

![](../assets/integrations/openclaw-01.png)

![](../assets/integrations/openclaw-02.png)

Aşağıdaki bilgileri kaydedin (henüz bir API Anahtarı oluşturulmadıysa, oluşturmak için sayfadaki talimatları izleyin):

* **Erişim URL'si**
* **Model Adı**
* **API Anahtarı**

## OpenClaw'ı yükleyin { #install-openclaw }

Kurulumu tamamlamak için resmi OpenClaw belgelerini takip edin: [https://docs.openclaw.ai/install](https://docs.openclaw.ai/install)

## OpenClaw'da GPUStack'ı yapılandırma { #configure-gpustack-in-openclaw }

1. Etkileşimli yapılandırma sihirbazını başlatın:

   ```bash
   openclaw onboard --install-daemon
   ```

2. **Model / Kimlik Doğrulama Sağlayıcı** seçiminde, **Özel Sağlayıcı**'yi seçin

   ![](../assets/integrations/openclaw-03.png)

3. GPUStack tarafından sağlanan bilgileri istendiği şekilde doldurun:

    * **API Temel URL'si**: Erişim URL'si
    * **API Anahtarı**: API Anahtarı
    * **Model Kimliği**: Model Adı

   ![](../assets/integrations/openclaw-04.png)

Bu adımları tamamladıktan sonra OpenClaw, çıkarım için ilgili modeli çağırmak üzere GPUStack'ı kullanacaktır.

## Kanalları Yapılandır { #configure-channels }

İstenilen iletişim kanalını yapılandırmak için resmi OpenClaw belgelerini izleyin: [https://docs.openclaw.ai/channels](https://docs.openclaw.ai/channels)
