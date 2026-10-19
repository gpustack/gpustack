---
translation_status: translated
---

# n8n ile entegrasyon { #integrating-with-n8n }

[n8n](https://n8n.io/) adil kod modeli altında dağıtılan genişletilebilir bir iş akışı otomasyon aracıdır. Görevleri yönetmek ve otomatikleştirmek için herhangi bir uygulamayı bir API'ye bağlamanıza olanak tanır. Gelişmiş yapay zeka düğümleri sayesinde büyük dil modellerini otomatikleştirilmiş iş akışlarınıza kolayca entegre edebilirsiniz.

## Modeli Dağıt { #deploy-the-model }

Model dağıtımını tamamlamak için lütfen GPUStack belgelerindeki **[Model Dağıtımı](../user-guide/model-deployment-management.md#deploy-model)** bölümüne bakın.

## API Erişim Bilgisi { #api-access-info }

1. GPUStack Web Kullanıcı Arayüzü'nde oturum açın
2. **Routes** sayfasına gidin
3. Hedef modelin sağ tarafındaki menüden **API Erişim Bilgisi** seçeneğini seçin.

![](../assets/integrations/n8n-01.png)
![](../assets/integrations/n8n-02.png)

Aşağıdaki bilgileri kaydedin (henüz bir API Anahtarı oluşturulmadıysa, oluşturmak için sayfadaki talimatları izleyin):

* **Erişim URL'si**
* **API Anahtarı**

## n8n'yi yükleyin { #install-n8n }

Kendi kendine barındırılan kurulumu tamamlamak için resmi n8n belgelerini izleyin veya doğrudan n8n Bulut hizmetini kullanın: [https://docs.n8n.io/hosting/](https://docs.n8n.io/hosting/)

## GPUStack'i n8n'ye entegre etme { #integrating-gpustack-in-n8n }

GPUStack, OpenAI uyumlu bir API sağladığından, yapılandırma için doğrudan n8n'deki OpenAI düğümlerini kullanabilirsiniz:

1. n8n'ye **Kimlik Bilgisi** ekleyin

   ![](../assets/integrations/n8n-03.png)
   ![](../assets/integrations/n8n-04.png)
   ![](../assets/integrations/n8n-05.png)

   Başarılı ekleme sonrasında aşağıdaki gibi görünecektir:

   ![](../assets/integrations/n8n-05-02.png)

2. GPUStack Kimlik Bilgisini Kullanın

   ![](../assets/integrations/n8n-06.png)
   ![](../assets/integrations/n8n-07.png)

## Gelişmiş Yapay Zeka İş Akışları Oluşturun { #build-advanced-ai-workflows }

Güçlü otomatikleştirilmiş süreçler oluşturmak amacıyla çeşitli uygulama düğümlerini yapay zeka ile nasıl birleştireceğinizi öğrenmek için resmi n8n belgelerine bakın: [https://docs.n8n.io/advanced-ai/](https://docs.n8n.io/advanced-ai/)
